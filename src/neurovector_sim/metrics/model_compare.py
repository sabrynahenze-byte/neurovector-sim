# Unified eval pipeline for ANN vs SNN comparison.
# Adapters normalize output differences (ANNs return logits,
# SNNs return spike trains) so all models use the same metrics.

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aihwkit.nn import AnalogLinear, AnalogConv2d


class SOPCounter:
    """Counts synaptic operations (SOPs) and activation sparsity via forward hooks."""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.total_sops = 0
        self.total_spikes = 0
        self.total_possible_activations = 0
        self._register_hooks()

    def _register_hooks(self):
        for layer in self.model.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d, AnalogLinear, AnalogConv2d)):
                # Fan-out = # output connections per input neuron
                if isinstance(layer, (nn.Linear, AnalogLinear)):
                    fan_out = layer.out_features
                else:
                    fan_out = (
                        layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                    )

                self.hooks.append(
                    layer.register_forward_hook(
                        lambda m, inp, out, f=fan_out: self._count_metrics(inp, f)
                    )
                )

    def _count_metrics(self, layer_input, fan_out):
        data = layer_input[0]

        # Threshold at 0.5 so analog activations don't inflate the spike count
        spikes = (data > 0.5).float()
        num_spikes = spikes.sum().item()

        self.total_sops += num_spikes * fan_out
        self.total_spikes += num_spikes
        self.total_possible_activations += data.numel()

    def get_sparsity(self):
        if self.total_possible_activations == 0:
            return 0.0
        return 1.0 - (self.total_spikes / self.total_possible_activations)

    def reset(self):
        self.total_sops = 0
        self.total_spikes = 0
        self.total_possible_activations = 0

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


@dataclass
class EvalOutputs:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    extras: Optional[Dict[str, Any]] = None


def adapt_torch(model_out, targets, loss_fn):
    """For standard PyTorch ANN outputs (just logits)."""
    loss = loss_fn(model_out, targets) if loss_fn else None
    return EvalOutputs(logits=model_out, loss=loss)


def adapt_snn(model_out, targets, loss_fn):
    """For SNN/AIHWKIT outputs. Sums spikes over time to get logits."""
    spk = model_out
    logits = spk.sum(dim=0)
    loss = loss_fn(spk, targets) if loss_fn else None
    return EvalOutputs(logits=logits, loss=loss, extras={"spk": spk})


@torch.no_grad()
def top1_accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def expected_calibration_error(logits, targets, n_bins=15):
    """ECE from logits [B, C]. Lower = better calibrated."""

    probs = F.softmax(logits, dim=1)
    conf, preds = probs.max(dim=1)
    acc = (preds == targets).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)

    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            ece += mask.float().mean() * (conf[mask].mean() - acc[mask].mean()).abs()

    return ece.item()


@torch.no_grad()
def snn_energy_latency_from_spikes(spk):
    """Energy (avg spikes/sample) and latency (avg first-spike step) from spikes [T, B, C]."""

    T, B, C = spk.shape
    energy = spk.sum(dim=(0, 2)).float().mean().item()
    spk_any = spk.sum(dim=2)
    lat = []
    for b in range(B):
        nz = (spk_any[:, b] > 0).nonzero(as_tuple=False)
        lat.append(float(nz[0].item()) if nz.numel() else float(T))
    return energy, sum(lat) / len(lat)


@dataclass
class EvalResult:
    name: str
    loss: Optional[float]
    acc: float
    ece: Optional[float]
    extra: Dict[str, Any]


class ModelEvaluator:
    def __init__(self, model, adapter, name, loss_fn=None, compute_ece=True, is_snn=False):
        self.model = model
        self.adapter = adapter
        self.name = name
        self.loss_fn = loss_fn
        self.compute_ece = compute_ece
        self.is_snn = is_snn
        self.sop_counter = SOPCounter(model) if is_snn else None

    @torch.no_grad()
    def evaluate(self, loader, device):
        self.model.eval()

        if self.sop_counter:
            self.sop_counter.reset()

        total_loss = total_acc = 0.0
        total_samples = batches = 0
        ece_vals, energy_vals, latency_vals = [], [], []

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = self.model(x)
            eo = self.adapter(out, y, self.loss_fn)

            total_acc += top1_accuracy(eo.logits, y)
            total_samples += x.size(0)
            batches += 1

            if eo.loss:
                total_loss += eo.loss.item()
            if self.compute_ece:
                ece_vals.append(expected_calibration_error(eo.logits, y))
            if eo.extras and "spk" in eo.extras:
                e, l = snn_energy_latency_from_spikes(eo.extras["spk"])
                energy_vals.append(e)
                latency_vals.append(l)

        result = EvalResult(
            name=self.name,
            loss=(total_loss / batches) if batches else None,
            acc=(total_acc / batches) if batches else 0.0,
            ece=(sum(ece_vals) / len(ece_vals)) if ece_vals else None,
            extra={},
        )

        if self.is_snn:
            result.extra["avg_sops_per_sample"] = (
                self.sop_counter.total_sops / total_samples
            )
            result.extra["activation_sparsity_pct"] = (
                self.sop_counter.get_sparsity() * 100
            )

            if energy_vals:
                result.extra["energy_proxy_spks"] = sum(energy_vals) / len(energy_vals)
            else:
                # TODO: Consider putting to -1 to signify no metrics
                result.extra["energy_proxy_spks"] = 0

            if latency_vals:
                result.extra["latency_proxy_steps"] = sum(latency_vals) / len(
                    latency_vals
                )
            else:
                result.extra["latency_proxy_steps"] = 0

        return result


def compare_models(evals, loader, device):
    return {k: v.evaluate(loader, device) for k, v in evals.items()}

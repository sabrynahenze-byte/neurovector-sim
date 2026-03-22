############################################################
# This script provides a unified evaluation pipeline to
# compare standard PyTorch classifiers (ANNs) against
# SNNs. Adapters are used to abstract differences in output
# between ANN and SNN models. That is, whereas the output
# of ANNs is typically a single tensor of logits, the output
# of SNNS is a sequence of spikes over time. Models are
# evaluated on standard performance metrics (Accuracy, Loss,
# and Expected Calibration Error) as well as hardware-based
# metrics (SNN only).
############################################################

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------
# SOP Counter for Hardware Efficiency
# ----------------------------------------
class SOPCounter:
    """
    Represents a counter that hooks into model layers to count
    Synaptic Operations (SOPs) and calculate Activation Sparsity.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.total_sops = 0
        self.total_spikes = 0
        self.total_possible_activations = 0
        self._register_hooks()

    def _register_hooks(self):
        for layer in self.model.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                # Fan-out calc (# output connections for each input neuron)
                if isinstance(layer, nn.Linear):
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
        # layer_input[0] shape: [B, C] for ANN, [T, B, C] for SNN
        data = layer_input[0]

        # Threshold at 0.5 so analog activations don't inflate the spike count
        spikes = (data > 0.5).float()
        num_spikes = spikes.sum().item()

        self.total_sops += num_spikes * fan_out
        self.total_spikes += num_spikes
        self.total_possible_activations += data.numel()

    def get_sparsity(self) -> float:
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


# ----------------------------------------
# Common outputs & Adapters
# ----------------------------------------
@dataclass
class EvalOutputs:
    """
    Simple data structure used to unify outputs for metrics computation.
    """

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    extras: Optional[Dict[str, Any]] = (
        None  # Additional properties (e.g. {"spk": [T, B, C], "mem": ...})
    )


class OutputAdapter(Protocol):
    """
    Model-agnostic adapter blueprint.
    """

    def __call__(
        self, model_out: Any, targets: torch.Tensor, loss_fn: Optional[Callable]
    ) -> EvalOutputs: ...


class TorchAdapter:
    """
    The default adapter used to convert PyTorch ANN outputs to EvalOutput values.
    """

    def __call__(
        self, model_out: Any, targets: torch.Tensor, loss_fn: Optional[Callable]
    ) -> EvalOutputs:
        logits = model_out
        loss = loss_fn(logits, targets) if loss_fn is not None else None
        return EvalOutputs(logits=logits, loss=loss)


class SnnTorchAdapter:
    """
    The default adapter used to convert snnTorch SNN outputs to EvalOutput values.
    """

    def __init__(self, reduce: str = "sum"):
        self.reduce = reduce

    def __call__(
        self, model_out: Any, targets: torch.Tensor, loss_fn: Optional[Callable]
    ) -> EvalOutputs:
        if isinstance(model_out, (tuple, list)):
            spk = model_out[0]
            extras = {"spk": spk, "mem": model_out[1] if len(model_out) > 1 else None}
        else:
            spk = model_out
            extras = {"spk": spk}

        if self.reduce == "sum":
            logits = spk.sum(dim=0)
        elif self.reduce == "mean":
            logits = spk.mean(dim=0)
        else:
            logits = spk.max(dim=0).values

        loss = loss_fn(spk, targets) if loss_fn is not None else None
        return EvalOutputs(logits=logits, loss=loss, extras=extras)


class AIHWKITAdapter:
    """
    An adapter for AIHWKIT-based models.

    Supports the following formats for evaluation:
        - logits tensor: [B, C]
        - tuple/list: (logits, *extras)
        - dict: {"logits": ..., ...}
        - snnTorch-like tuple: (spk[T,B,C], mem...) -> reduces to logits (optional)
    """

    def __init__(self, assume_spikes_if_3d: bool = True, spike_reduce: str = "sum"):
        self.assume_spikes_if_3d = assume_spikes_if_3d
        if spike_reduce not in {"sum", "mean", "max"}:
            raise ValueError("spike_reduce must be one of: sum, mean, max")
        self.spike_reduce = spike_reduce

    def __call__(
        self, model_out: Any, targets: torch.Tensor, loss_fn: Optional[Callable]
    ) -> EvalOutputs:
        extras: Dict[str, Any] = {}

        if isinstance(model_out, dict):
            if "logits" not in model_out:
                raise ValueError("AIHWKITAdapter expected dict with key 'logits'.")
            logits = model_out["logits"]
            extras = {k: v for k, v in model_out.items() if k != "logits"}
        elif isinstance(model_out, (tuple, list)):
            if len(model_out) == 0:
                raise ValueError("AIHWKITAdapter got empty tuple/list output.")
            first_out = model_out[0]

            # If first output looks like spikes [T, B, C], then reduce to [B, C]
            if (
                self.assume_spikes_if_3d
                and isinstance(first_out, torch.Tensor)
                and first_out.dim() == 3
            ):
                spk = first_out
                extras["spk"] = spk
                if len(model_out) > 1:
                    extras["mem"] = model_out[1]
                logits = self._reduce_spikes(spk)

                # For snnTorch losses, spikes are usually expected instead of logits
                loss = loss_fn(spk, targets) if loss_fn is not None else None
                return EvalOutputs(logits=logits, loss=loss, extras=extras)

            # Otherwise, assume logits
            logits = first_out
            if len(model_out) > 1:
                extras["raw_out"] = model_out[1:]
        else:
            # Plain tensor output
            logits = model_out

        if not isinstance(logits, torch.Tensor):
            raise ValueError(f"Expected logits tensor, but got {type(logits)}")

        # 3D logits means the time dimension wasn't reduced upstream, so handle this case
        if logits.dim() == 3 and self.assume_spikes_if_3d:
            spk = logits
            extras["spk"] = spk
            logits = self._reduce_spikes(spk)
            loss = loss_fn(spk, targets) if loss_fn is not None else None
            return EvalOutputs(logits=logits, loss=loss, extras=extras)

        if logits.dim() != 2:
            raise ValueError(
                f"Expected logits shape [B, C], but got {tuple(logits.shape)}"
            )

        loss = loss_fn(logits, targets) if loss_fn is not None else None
        return EvalOutputs(logits=logits, loss=loss, extras=extras or None)

    def _reduce_spikes(self, spk: torch.Tensor) -> torch.Tensor:
        if self.spike_reduce == "sum":
            return spk.sum(dim=0)
        if self.spike_reduce == "mean":
            return spk.mean(dim=0)
        return spk.max(dim=0).values


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def expected_calibration_error(
    logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15
) -> float:
    """
    Compute top-label Expected Calibration Error (ECE) from logits.

    Top-label ECE from logits [B, C]. Bins predictions by confidence, then
    sums |conf - acc| weighted by bin fraction. Lower = better calibrated.

    Args:
        logits: Model outputs of shape [B, C], where B is batch size and C
            is the number of classes.
        targets: Ground-truth class indices of shape [B].
        n_bins: Number of equal-width confidence bins.

    Returns:
        Scalar ECE value as a Python float (lower is better calibrated).
    """

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
def snn_energy_latency_from_spikes(spk: torch.Tensor) -> Tuple[float, float]:
    """
    Proxy metrics from a spike tensor [T, B, C].
      - Energy: avg total spikes per sample
      - Latency: avg time step of first spike per sample (T if no spike fires)
    """

    T, B, C = spk.shape
    energy = spk.sum(dim=(0, 2)).float().mean().item()
    spk_any = spk.sum(dim=2)
    lat = []
    for b in range(B):
        nz = (spk_any[:, b] > 0).nonzero(as_tuple=False)
        lat.append(float(nz[0].item()) if nz.numel() else float(T))
    return energy, sum(lat) / len(lat)


# ----------------------------
# Evaluator
# ----------------------------
@dataclass
class EvalResult:
    name: str
    loss: Optional[float]
    acc: float
    ece: Optional[float]
    extra: Dict[str, Any]


class ModelEvaluator:
    def __init__(
        self,
        model: nn.Module,
        adapter: OutputAdapter,
        name: str,
        loss_fn: Optional[Callable] = None,
        compute_ece: bool = True,
        is_snn: bool = False,
    ):
        self.model = model
        self.adapter = adapter
        self.name = name
        self.loss_fn = loss_fn
        self.compute_ece = compute_ece
        self.is_snn = is_snn
        self.sop_counter = SOPCounter(model) if is_snn else None

    @torch.no_grad()
    def evaluate(self, loader, device: torch.device) -> EvalResult:
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


def compare_models(
    evals: Dict[str, ModelEvaluator], loader, device: torch.device
) -> Dict[str, EvalResult]:
    return {k: v.evaluate(loader, device) for k, v in evals.items()}

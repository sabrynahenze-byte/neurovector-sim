############################################################
# poc_aihwkit.py — AIHWKIT-Integrated SNN with ReRAM Simulation
#
# EEE 489 Senior Project | February 2026
#
# AUTHORS:
#   Sabryna H. — AIHWKIT integration
#   Bryan M. — Original SNN + hand-rolled RRAM PoC
#
# MODES:
#   "ideal"    — Standard nn.Linear, no device noise (baseline)
#   "handrole" — Bryan's hand-rolled noise model (comparison)
#   "aihwkit"  — IBM calibrated ReRAM model (primary)
#
# INSTALL:
#   conda install -c conda-forge aihwkit
#   pip install snntorch torch torchvision matplotlib
############################################################

from typing import Callable
import os
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import spikegen
import snntorch.functional as SF

import matplotlib.pyplot as plt

try:
    from aihwkit.nn import AnalogLinear
    from aihwkit.optim import AnalogSGD
    from aihwkit.simulator.configs import InferenceRPUConfig
    from aihwkit.simulator.presets import ReRamESPreset, ReRamSBPreset
    from aihwkit.simulator.presets.devices import (
        ReRamESPresetDevice,
        ReRamArrayOMPresetDevice,
    )
    from aihwkit.simulator.configs.utils import (
        MappingParameter,
        IOParameters,
    )
    AIHWKIT_AVAILABLE = True
except ImportError:
    AIHWKIT_AVAILABLE = False
    print(
        "[WARNING] aihwkit not installed. Only 'ideal' and "
        "'handrole' modes are available.\n"
        "Install with: conda install -c conda-forge aihwkit"
    )


# ============================================================
# CONFIGURATION
# ============================================================

MODE = "aihwkit"        # "ideal" | "handrole" | "aihwkit"

NUM_EPOCHS = 10
BATCH_SIZE = 128
NUM_STEPS = 25
HIDDEN_SIZE = 500
LR = 2e-3

# Hand-rolled RRAM parameters (handrole mode only)
RRAM_NOISE_STD = 0.02
RRAM_DRIFT_PER_EPOCH = 0.01
RRAM_FAIL_PROB = 0.01

# AIHWKIT device preset: "es" (exp-step) or "om" (soft-bounds array)
AIHWKIT_DEVICE_PRESET = "om"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# AIHWKIT RPU CONFIGURATION
# ============================================================

def build_rpu_config(device_preset: str = "om") -> "InferenceRPUConfig":
    """
    Build an InferenceRPUConfig for ReRAM simulation.
    "es" = exponential step (Gong et al., Nat. Commun. 2018)
    "om" = soft-bounds array (Gong & Rasch et al., IEDM 2022)
    """
    rpu_config = InferenceRPUConfig()

    if device_preset == "es":
        rpu_config.device = ReRamESPresetDevice()
    elif device_preset == "om":
        rpu_config.device = ReRamArrayOMPresetDevice()
        rpu_config.device.corrupt_devices_prob = 0.01
    else:
        raise ValueError(f"Unknown device preset: {device_preset}")

    return rpu_config


# ============================================================
# MODELS
# ============================================================

class SNNIdeal(nn.Module):
    """Ideal SNN baseline — no device noise. Input(784) → LIF → LIF → Output(10)."""

    def __init__(self, num_steps: int, hidden_size: int = 500):
        super().__init__()
        self.num_steps = num_steps
        beta = 0.9
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        x_spikes = spikegen.rate(x_flat, num_steps=self.num_steps)

        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        spk2_rec, mem2_rec = [], []

        for step in range(self.num_steps):
            spike_input = x_spikes[step]
            cur1 = self.fc1(spike_input)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)


class SNNHandRolled(nn.Module):
    """
    SNN with manually-implemented RRAM non-idealities (Bryan's PoC).
    Non-idealities: write noise (Gaussian), drift (multiplicative per epoch),
    endurance failure (stuck-at-zero mask).
    """

    def __init__(self, num_steps, hidden_size=500,
                 noise_std=0.02, drift_per_epoch=0.01, fail_prob=0.01):
        super().__init__()
        self.num_steps = num_steps
        self.noise_std = noise_std
        self.drift_per_epoch = drift_per_epoch
        beta = 0.9

        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)

        self.register_buffer("drift_fc1", torch.tensor(1.0))
        self.register_buffer("drift_fc2", torch.tensor(1.0))
        self.register_buffer("mask_fc1", torch.ones_like(self.fc1.weight))
        self.register_buffer("mask_fc2", torch.ones_like(self.fc2.weight))

        if fail_prob > 0.0:
            self.mask_fc1.copy_((torch.rand_like(self.fc1.weight) > fail_prob).float())
            self.mask_fc2.copy_((torch.rand_like(self.fc2.weight) > fail_prob).float())

    @torch.no_grad()
    def step_drift(self):
        """Advance drift by one epoch. Call once per training epoch."""
        self.drift_fc1 *= 1.0 + self.drift_per_epoch
        self.drift_fc2 *= 1.0 + self.drift_per_epoch

    def _perturb(self, w, mask, drift):
        noise = self.noise_std * torch.randn_like(w) * w.abs().clamp(min=1e-3) if self.training else 0.0
        return (w + noise) * drift * mask

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        x_spikes = spikegen.rate(x_flat, num_steps=self.num_steps)

        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        spk2_rec, mem2_rec = [], []

        for step in range(self.num_steps):
            spike_input = x_spikes[step]
            w1 = self._perturb(self.fc1.weight, self.mask_fc1, self.drift_fc1)
            cur1 = F.linear(spike_input, w1, self.fc1.bias)
            spk1, mem1 = self.lif1(cur1, mem1)
            w2 = self._perturb(self.fc2.weight, self.mask_fc2, self.drift_fc2)
            cur2 = F.linear(spk1, w2, self.fc2.bias)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)


class SNNAnalog(nn.Module):
    """
    SNN with AIHWKIT-simulated ReRAM crossbar arrays.
    AnalogLinear replaces nn.Linear; LIF neurons are unchanged.
    Device noise is applied internally by AIHWKIT per the RPUConfig.
    """

    def __init__(self, num_steps: int, hidden_size: int = 500, rpu_config=None):
        super().__init__()
        self.num_steps = num_steps
        beta = 0.9
        self.fc1 = AnalogLinear(28 * 28, hidden_size, bias=True, rpu_config=rpu_config)
        self.fc2 = AnalogLinear(hidden_size, 10, bias=True, rpu_config=rpu_config)
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        x_spikes = spikegen.rate(x_flat, num_steps=self.num_steps)

        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        spk2_rec, mem2_rec = [], []

        for step in range(self.num_steps):
            spike_input = x_spikes[step]
            cur1 = self.fc1(spike_input)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(mode: str, num_steps: int, hidden_size: int):
    """Build model and optimizer for the specified mode."""
    if mode == "ideal":
        model = SNNIdeal(num_steps, hidden_size).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    elif mode == "handrole":
        model = SNNHandRolled(
            num_steps, hidden_size,
            noise_std=RRAM_NOISE_STD,
            drift_per_epoch=RRAM_DRIFT_PER_EPOCH,
            fail_prob=RRAM_FAIL_PROB,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    elif mode == "aihwkit":
        if not AIHWKIT_AVAILABLE:
            raise RuntimeError("aihwkit not installed. Run: conda install -c conda-forge aihwkit")
        rpu_config = build_rpu_config(AIHWKIT_DEVICE_PRESET)
        model = SNNAnalog(num_steps, hidden_size, rpu_config=rpu_config).to(DEVICE)
        optimizer = AnalogSGD(model.parameters(), lr=LR)
        optimizer.regroup_param_groups(model)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return model, optimizer


# ============================================================
# PROXY METRICS
# ============================================================

def batch_energy_latency(spk_rec: torch.Tensor):
    """
    Energy proxy: avg spike count per sample.
    Latency proxy: avg time step of first output spike.
    """
    T, B, C = spk_rec.shape
    energy_per_sample = spk_rec.sum(dim=(0, 2)).mean().item()

    latencies = []
    spk_any = spk_rec.sum(dim=2)
    for b in range(B):
        nz = (spk_any[:, b] > 0).nonzero(as_tuple=False)
        latencies.append(float(nz[0].item()) if nz.numel() > 0 else float(T))

    return energy_per_sample, sum(latencies) / len(latencies)


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_epoch(model, loader, optimizer, loss_fn):
    """Train one epoch. Returns (accuracy, avg_loss)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for data, targets in loader:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        spk_out, mem_out = model(data)
        loss = loss_fn(spk_out, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = spk_out.sum(dim=0).max(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return correct / total if total > 0 else 0.0, running_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn):
    """Evaluate on test set. Returns (accuracy, loss, energy, latency)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    energy_vals, latency_vals = [], []

    for data, targets in loader:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        spk_out, mem_out = model(data)
        loss = loss_fn(spk_out, targets)
        running_loss += loss.item()

        _, predicted = spk_out.sum(dim=0).max(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        energy, latency = batch_energy_latency(spk_out)
        energy_vals.append(energy)
        latency_vals.append(latency)

    return (
        correct / total if total > 0 else 0.0,
        running_loss / len(loader),
        sum(energy_vals) / len(energy_vals),
        sum(latency_vals) / len(latency_vals),
    )


# ============================================================
# DATA LOADING
# ============================================================

def get_mnist_loaders(batch_size: int):
    """Download MNIST and return train/test DataLoaders."""
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])
    data_path = "./data/mnist"
    train_ds = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0),
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_curves(history: dict, mode: str, out_dir: str = "./plots"):
    """Save accuracy, loss, energy, and latency plots."""
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_acc"]) + 1)

    plots = [
        ("accuracy", "Accuracy", [("train_acc", "Train"), ("test_acc", "Test")]),
        ("loss", "Loss", [("train_loss", "Train"), ("test_loss", "Test")]),
        ("energy", "Avg spikes/sample", [("energy", "Energy proxy")]),
        ("latency", "Avg first-spike step", [("latency", "Latency proxy")]),
    ]

    for filename, ylabel, series_list in plots:
        plt.figure()
        for key, label in series_list:
            plt.plot(epochs, history[key], label=label)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Epoch [{mode}]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{filename}_{mode}.png"))
        plt.close()


# ============================================================
# DRIFT SIMULATION (AIHWKIT post-training only)
# ============================================================

def simulate_drift(model, test_loader, loss_fn, drift_times):
    """
    Simulate post-training conductance drift at specified times (seconds).
    Returns dict mapping time → {accuracy, loss, energy, latency}.
    """
    if not AIHWKIT_AVAILABLE:
        print("[SKIP] Drift simulation requires aihwkit.")
        return {}

    results = {}
    for t in drift_times:
        model.drift_analog_weights(t_inference=t)
        acc, loss, energy, latency = eval_epoch(model, test_loader, loss_fn)
        results[t] = {"accuracy": acc, "loss": loss, "energy": energy, "latency": latency}
        print(f"  Drift t={t:>8.0f}s | acc={acc:.4f} | loss={loss:.4f}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print(f"  NeuroVector Sim — Mode: {MODE}")
    print(f"  Device: {DEVICE}")
    if MODE == "aihwkit":
        print(f"  AIHWKIT preset: {AIHWKIT_DEVICE_PRESET}")
    print("=" * 60)

    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)
    model, optimizer = build_model(MODE, NUM_STEPS, HIDDEN_SIZE)

    print("Model architecture:")
    print(model)

    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    history = {
        "train_acc": [], "train_loss": [],
        "test_acc": [], "test_loss": [],
        "energy": [], "latency": [],
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        if MODE == "handrole":
            model.step_drift()

        train_acc, train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        test_acc, test_loss, avg_energy, avg_latency = eval_epoch(model, test_loader, loss_fn)

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        history["energy"].append(avg_energy)
        history["latency"].append(avg_latency)

        print(f"  Train acc={train_acc:.3f} loss={train_loss:.4f} | Test acc={test_acc:.3f} loss={test_loss:.4f}")
        print(f"  Energy={avg_energy:.2f} spk/sample | Latency={avg_latency:.2f} steps\n")

    if MODE == "aihwkit":
        print("-" * 60)
        print("Post-training drift simulation")
        print("-" * 60)
        drift_times = [0, 60, 3600, 86400, 604800, 2592000]
        history["drift"] = simulate_drift(model, test_loader, loss_fn, drift_times)

    plot_curves(history, MODE)
    print(f"Plots saved to ./plots/*_{MODE}.png")

    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/results_{MODE}.json"
    serializable = {k: v for k, v in history.items() if k != "drift"}
    if "drift" in history:
        serializable["drift"] = {str(k): v for k, v in history["drift"].items()}

    with open(report_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {report_path}\nDone!")


if __name__ == "__main__":
    main()

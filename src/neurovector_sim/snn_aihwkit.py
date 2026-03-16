############################################################
# snn_aihwkit.py — Hardware-Aware SNN with AIHWKIT ReRAM Simulation
#
# Author: shenze (EEE489)
#
# Purpose:
#   Full MNIST classifier using AIHWKIT AnalogLinear layers
#   with IBM's hardware-calibrated ReRAM device preset.
#
#   This is the "Hardware-Aware SNN" leg of the three-way comparison:
#     1. Classical ANN        → ann_baseline.py
#     2. Digital SNN          → poc_snntorch.py  (Bryan's, rram_enabled=False)
#     3. Hardware-Aware SNN   → THIS FILE
#
#   Interface is intentionally aligned with poc_snntorch.py so that
#   run_comparison.py can call all three models the same way.
#
# AIHWKIT replaces the hand-rolled _rram_perturb_weight() mechanism
# from poc_snntorch.py. Instead of manually adding Gaussian noise and
# drift factors to weight tensors, we use AnalogLinear, which internally
# applies hardware-calibrated noise during each forward pass based on
# measured HfOx ReRAM device data (Gong & Rasch, IEDM 2022).
#
# Dependencies:
#   pip install aihwkit snntorch
############################################################

import os
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import spikegen
import snntorch.functional as SF

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts and servers
import matplotlib.pyplot as plt

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.presets.devices import ReRamArrayOMPresetDevice


# ============================================================
# Config
# ============================================================
# These defaults match poc_snntorch.py so comparison runs are
# apples-to-apples. Override via run_comparison.py if needed.

NUM_EPOCHS    = 10
BATCH_SIZE    = 128
NUM_STEPS     = 25      # SNN time steps per sample
HIDDEN_SIZE   = 500     # hidden layer width — matches poc_snntorch.py
LR            = 2e-3

# AIHWKIT NOTE: AnalogLinear has known compatibility issues on MPS and
# some CUDA configurations. Pinning to CPU here is intentional and safe.
# If you want to test CUDA support, verify your AIHWKIT install supports
# it before changing this. The digital SNN and ANN run on the best
# available device — only the hardware-aware model is CPU-pinned.
DEVICE = torch.device("cpu")


# ============================================================
# AIHWKIT Device Configuration
# ============================================================

def build_rpu_config() -> InferenceRPUConfig:
    """
    Build the ReRAM device configuration for AIHWKIT.

    Build the ReRAM device configuration for AIHWKIT.

    NOTE on what this experiment actually does:
    Despite InferenceRPUConfig's name, we are NOT doing
    digital-train → map weights → analog-inference.
    We are training the AnalogLinear model directly, with ReRAM
    device noise present during every forward pass throughout training.
    The network learns under hardware noise conditions from the start.

    For the report, use "aligned benchmark conditions" not
    "identical training conditions" — the ANN and digital SNN
    train without analog noise; this model does not.

    ReRamArrayOMPresetDevice is IBM's soft-bounds ReRAM preset fitted
    to measured HfOx device arrays (Gong & Rasch, IEDM 2022). It models:
      - Cycle-to-cycle conductance noise (write variability)
      - Device-to-device variation across the crossbar
      - Asymmetric weight update behavior (soft bounds)

    This replaces the three hand-rolled non-idealities in poc_snntorch.py
    (RRAM_NOISE_STD, RRAM_DRIFT_PER_EPOCH, RRAM_FAIL_PROB) with a single
    physics-grounded device model.
    """
    cfg = InferenceRPUConfig()
    cfg.device = ReRamArrayOMPresetDevice()
    return cfg


# ============================================================
# Hardware-Aware SNN Model
# ============================================================

class HardwareAwareSNN(nn.Module):
    """
    Spiking neural network with AIHWKIT AnalogLinear layers.

    Architecture: Input(784) → AnalogLinear+LIF(500) → AnalogLinear+LIF(10)

    This mirrors the two-layer structure of SNNNet in poc_snntorch.py,
    making the comparison fair. The key difference is that AnalogLinear
    applies hardware-calibrated ReRAM noise internally — we don't need
    to call _rram_perturb_weight() at each time step.

    The LIF (Leaky Integrate-and-Fire) neurons from snntorch are
    layer-agnostic: they only care about the current coming in,
    not where it came from. So AnalogLinear + LIF works the same
    way as nn.Linear + LIF, just with realistic hardware noise on
    the weight reads.
    """

    def __init__(
        self,
        num_steps: int = NUM_STEPS,
        hidden_size: int = HIDDEN_SIZE,
        rpu_config: InferenceRPUConfig = None,
    ):
        super().__init__()
        self.num_steps = num_steps

        # LIF decay rate — same as poc_snntorch.py for fair comparison
        beta = 0.9

        # Use provided config or build a fresh default
        if rpu_config is None:
            rpu_config = build_rpu_config()

        # AnalogLinear is a drop-in for nn.Linear with hardware noise baked in.
        # AIHWKIT applies the ReRAM device model during the forward pass.
        self.fc1 = AnalogLinear(28 * 28, hidden_size, bias=True, rpu_config=rpu_config)
        self.lif1 = snn.Leaky(beta=beta)

        self.fc2 = AnalogLinear(hidden_size, 10, bias=True, rpu_config=rpu_config)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the hardware-aware SNN.

        Returns spk_rec and mem_rec (same signature as SNNNet in
        poc_snntorch.py) so the training/eval functions work unchanged.
        """
        batch_size = x.size(0)

        # Flatten images: (B, 1, 28, 28) → (B, 784)
        x_flat = x.view(batch_size, -1)

        # Rate encoding: convert pixel intensities to spike trains
        # Output shape: (num_steps, B, 784)
        x_spikes = spikegen.rate(x_flat, num_steps=self.num_steps)

        # Initialize membrane potentials at t=0
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()

        spk2_rec = []
        mem2_rec = []

        # Time loop — unroll the SNN for num_steps timesteps
        for step in range(self.num_steps):
            spike_input = x_spikes[step]   # shape: (B, 784)

            # AnalogLinear applies ReRAM noise internally here
            cur1 = self.fc1(spike_input)   # shape: (B, hidden_size)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)          # shape: (B, 10)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        # Stack time steps: shape (num_steps, B, 10)
        spk2_rec = torch.stack(spk2_rec)
        mem2_rec = torch.stack(mem2_rec)

        return spk2_rec, mem2_rec


# ============================================================
# Proxy Metrics
# ============================================================
# Copied from poc_snntorch.py so this module is self-contained.
# run_comparison.py can import this version or Bryan's — same output.

def batch_energy_latency(spk_rec: torch.Tensor):
    """
    Return spike-count energy proxy and first-spike latency proxy.

    Energy proxy: average number of spikes fired per sample across the
    full time window. Fewer spikes = less switching activity = lower power.

    Latency proxy: average time step at which the output layer fires its
    first spike. Earlier firing = faster inference.
    """
    T, B, C = spk_rec.shape

    # Energy: total spikes across time and output classes, averaged over batch
    spike_counts = spk_rec.sum(dim=(0, 2))   # shape: (B,)
    energy_per_sample = spike_counts.mean().item()

    # Latency: first time step with any output spike, per sample
    latencies = []
    spk_sum_over_outputs = spk_rec.sum(dim=2)  # shape: (T, B)
    for b in range(B):
        timeline = spk_sum_over_outputs[:, b]
        nonzero = (timeline > 0).nonzero(as_tuple=False)
        latencies.append(float(nonzero[0].item()) if nonzero.numel() > 0 else float(T))
    latency_per_sample = sum(latencies) / len(latencies)

    return energy_per_sample, latency_per_sample


# ============================================================
# Data Loading
# ============================================================
# Matches poc_snntorch.py's get_mnist_loaders() exactly.

def get_mnist_loaders(batch_size: int):
    """Load MNIST train and test sets with standard normalization."""
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])
    data_path = "./data/mnist"

    train_ds = datasets.MNIST(data_path, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    return train_loader, test_loader


# ============================================================
# Training and Evaluation
# ============================================================
# Same signatures as poc_snntorch.py — run_comparison.py can
# call train_epoch / eval_epoch from either module interchangeably.

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
):
    """Run one training epoch. Returns (accuracy, avg_loss)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data    = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        spk_out, _ = model(data)

        loss = loss_fn(spk_out, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Decision: class with the most spikes over the full time window
        _, predicted = spk_out.sum(dim=0).max(1)
        correct += (predicted == targets).sum().item()
        total   += targets.size(0)

    return correct / total if total > 0 else 0.0, running_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Callable,
):
    """Evaluation pass. Returns (accuracy, avg_loss, avg_energy, avg_latency)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    energy_vals  = []
    latency_vals = []

    for data, targets in loader:
        data    = data.to(device)
        targets = targets.to(device)

        spk_out, _ = model(data)

        loss = loss_fn(spk_out, targets)
        running_loss += loss.item()

        _, predicted = spk_out.sum(dim=0).max(1)
        correct += (predicted == targets).sum().item()
        total   += targets.size(0)

        energy, latency = batch_energy_latency(spk_out)
        energy_vals.append(energy)
        latency_vals.append(latency)

    avg_loss    = running_loss / len(loader)
    acc         = correct / total if total > 0 else 0.0
    avg_energy  = sum(energy_vals)  / len(energy_vals)
    avg_latency = sum(latency_vals) / len(latency_vals)

    return acc, avg_loss, avg_energy, avg_latency


# ============================================================
# Plotting
# ============================================================
# Same structure as poc_snntorch.py's plot_curves().
# run_comparison.py will call its own unified plotter instead,
# but this standalone version is useful for individual runs.

def plot_curves(history: dict, out_dir: str = "./plots/aihwkit"):
    """Save accuracy, loss, energy, and latency plots for this model."""
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_acc"]) + 1)

    for key, ylabel, title in [
        (("train_acc", "test_acc"),   "Accuracy",                 "Accuracy vs Epoch"),
        (("train_loss", "test_loss"), "Loss",                     "Loss vs Epoch"),
        (("energy",),                 "Avg spikes/sample",        "Energy Proxy vs Epoch"),
        (("latency",),                "Avg first-spike step",     "Latency Proxy vs Epoch"),
    ]:
        plt.figure()
        for k in key:
            plt.plot(epochs, history[k], label=k)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"[Hardware-Aware SNN] {title}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{key[0]}.png"))
        plt.close()

    print(f"Plots saved to {out_dir}/")


# ============================================================
# Main — standalone run for testing this module independently
# ============================================================

def main():
    print(f"Using device: {DEVICE}")
    print("Building hardware-aware SNN (AIHWKIT ReRAM preset)...\n")

    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)

    rpu  = build_rpu_config()
    model = HardwareAwareSNN(
        num_steps=NUM_STEPS,
        hidden_size=HIDDEN_SIZE,
        rpu_config=rpu,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    loss_fn   = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    history = {
        "train_acc": [], "train_loss": [],
        "test_acc":  [], "test_loss":  [],
        "energy":    [], "latency":    [],
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_acc, train_loss = train_epoch(model, train_loader, DEVICE, optimizer, loss_fn)
        test_acc, test_loss, avg_energy, avg_latency = eval_epoch(model, test_loader, DEVICE, loss_fn)

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        history["energy"].append(avg_energy)
        history["latency"].append(avg_latency)

        print(
            f"  Train: acc={train_acc:.3f}, loss={train_loss:.4f} | "
            f"Test: acc={test_acc:.3f}, loss={test_loss:.4f}"
        )
        print(f"  Energy={avg_energy:.2f} spikes/sample, Latency={avg_latency:.2f} steps\n")

    plot_curves(history)
    print("Hardware-aware SNN run complete.")


if __name__ == "__main__":
    main()

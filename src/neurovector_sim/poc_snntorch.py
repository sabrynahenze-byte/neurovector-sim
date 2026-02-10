############################################################
# This script implements a minimal PoC with the following
# workflow:
# - Data loading/preprocessing and rate-based spike encoding
# - SNN architecture and neuron dynamics, with the SNN
#   consisting of linear and LIF layers
# - Forward (inference) and backward (training) propagation,
#   where training is done via surrogate gradient descent
# - Simulation of device-level non-idealities (RRAM variability
#   in write noise, conductance drift, and memristor endurance)
# - Performance measurement and evaluation through proxy metrics
############################################################

from typing import Callable

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import spikegen
import snntorch.functional as SF

import matplotlib.pyplot as plt

# ----------------------------------------
# Config
# ----------------------------------------
# Learning variables. Keep small for a quick run.
NUM_EPOCHS = 10  # number of training epochs
BATCH_SIZE = 128  # batch size for DataLoader
NUM_STEPS = 25  # number of time steps in the SNN simulation
HIDDEN_SIZE = 500  # hidden layer width
LR = 2e-3  # learning rate for the Adam optimizer

# Toggle to enable/disable RRAM-like variability. Set to False
# if ideal SNN baseline (free of non-idealities) is desired.
ENABLE_RRAM = True

# RRAM variability hyperparameters initialized to rough PoC-level assumptions.
# Not calibrated to any specific process or hardware model. Instead, these
# exist to show qualitative effects of noise, drift, and endurance.
RRAM_NOISE_STD = 0.02  # the relative Gaussian noise strength on weights
RRAM_DRIFT_PER_EPOCH = 0.01  # a multiplicative drift factor per epoch (~1%/epoch)
RRAM_FAIL_PROB = 0.01  # endurance given as the probability that a memristor will fail (or the proportion of synapses that die out)

# Select optimal device for PyTorch.
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


# ----------------------------------------
# SNN Model with RRAM variability
# ----------------------------------------
class SNNNet(nn.Module):
    """
    Basic fully-connected SNN using surrogate gradients with the following layers:
      - Input: 28x28 pixels (MNIST image size)
      - Hidden: Linear + LIF
      - Output: Linear + LIF

    The linear layers are required in order to provide synaptic weights to represent
    physical RRAM crossbars that can be subject to RRAM variability. This variability is
    introduced by perturbing the weight matrices of the named fc1 and fc2 layers at each
    forward pass (write noise, cell endurance, and conductance drift mask).
    """

    def __init__(
        self,
        num_steps: int,
        hidden_size: int = 500,
        rram_enabled: bool = True, # enable device-level variability
        rram_noise_std: float = 0.02, # write noise
        rram_drift_per_epoch: float = 0.01, # conductance drift
        rram_fail_prob: float = 0.01, # probability that a memristor will fail = 1 - memristor cell endurance
    ):
        """Initialize SNN and optional RRAM variability settings."""
        super().__init__()
        self.num_steps = num_steps

        # LIF decay rate (beta = exp(-dt/tau)). Hard-coded for the PoC.
        beta = 0.9

        # Plain fully-connected (linear) stack with snnTorch Leaky neurons.
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.lif2 = snn.Leaky(beta=beta)

        # RRAM config flags & hyperparameters.
        self.rram_enabled = rram_enabled
        self.rram_noise_std = rram_noise_std
        self.rram_drift_per_epoch = rram_drift_per_epoch

        ### Register non-learnable RRAM variability parameters to perturb W matrices

        # Layer-wise drift factors updated each epoch by step_drift().
        self.register_buffer("drift_factor_fc1", torch.tensor(1.0))
        self.register_buffer("drift_factor_fc2", torch.tensor(1.0))

        # Endurance mask tensors (1 = healthy cell, 0 = dead cell).
        self.register_buffer("endurance_mask_fc1", torch.ones_like(self.fc1.weight))
        self.register_buffer("endurance_mask_fc2", torch.ones_like(self.fc2.weight))

        # Initialize endurance failures (a random subset of weights) if enabled.
        if self.rram_enabled and rram_fail_prob > 0.0:
            self.init_endurance_masks(rram_fail_prob)

    @torch.no_grad()
    def init_endurance_masks(self, fail_prob: float):
        """Randomly kill a fraction of devices (i.e. weights) permanently."""
        if fail_prob <= 0.0:
            self.endurance_mask_fc1.fill_(1.0)
            self.endurance_mask_fc2.fill_(1.0)
            return

        # Bernoulli trial for each weight (1 = healthy, 0 = dead).
        self.endurance_mask_fc1.copy_(
            (torch.rand_like(self.fc1.weight) > fail_prob).float()
        )
        self.endurance_mask_fc2.copy_(
            (torch.rand_like(self.fc2.weight) > fail_prob).float()
        )

    @torch.no_grad()
    def step_drift(self):
        """Update drift factors once per epoch."""
        if not self.rram_enabled:
            return
        
        self.drift_factor_fc1 *= 1.0 + self.rram_drift_per_epoch
        self.drift_factor_fc2 *= 1.0 + self.rram_drift_per_epoch

    def _rram_perturb_weight(self, w: torch.Tensor, layer: str) -> torch.Tensor:
        """Apply noise, drift, and endurance mask to a weight tensor."""
        if not self.rram_enabled:
            return w

        # Select the appropriate mask and drift factor depending on the layer.
        if layer == "fc1":
            mask = self.endurance_mask_fc1
            drift = self.drift_factor_fc1
        elif layer == "fc2":
            mask = self.endurance_mask_fc2
            drift = self.drift_factor_fc2
        else:
            mask = 1.0
            drift = 1.0

        # Add noise in training only.
        if self.training:
            noise = self.rram_noise_std * torch.randn_like(w) * w.abs().clamp(min=1e-3)
        else:
            noise = 0.0

        w_noisy = w + noise
        w_drifted = w_noisy * drift
        w_endurance = w_drifted * mask

        return w_endurance

    def forward(self, x: torch.Tensor):
        """Forward pass through the SNN."""
        batch_size = x.size(0)

        x_flat = x.view(batch_size, -1)

        x_spikes = spikegen.rate(x_flat, num_steps=self.num_steps)

        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()

        spk2_rec = []
        mem2_rec = []

        ### Time Loop
        for step in range(self.num_steps):
            spike_input = x_spikes[step]

            w1 = self._rram_perturb_weight(self.fc1.weight, "fc1")
            cur1 = F.linear(spike_input, w1, self.fc1.bias)

            spk1, mem1 = self.lif1(cur1, mem1)

            w2 = self._rram_perturb_weight(self.fc2.weight, "fc2")
            cur2 = F.linear(spk1, w2, self.fc2.bias)

            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        spk2_rec = torch.stack(spk2_rec)
        mem2_rec = torch.stack(mem2_rec)

        return spk2_rec, mem2_rec


# ----------------------------------------
# Proxy Metrics Computation
# ----------------------------------------
def batch_energy_latency(spk_rec: torch.Tensor):
    """Return spike-count energy proxy and first-spike latency proxy."""
    T, B, C = spk_rec.shape

    spike_counts = spk_rec.sum(dim=(0, 2))  # shape [B]
    energy_per_sample = spike_counts.mean().item()

    latencies = []
    spk_sum_over_outputs = spk_rec.sum(dim=2)  # shape [T,B]
    for b in range(B):
        timeline = spk_sum_over_outputs[:, b]  # shape [T]
        nonzero = (timeline > 0).nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            latencies.append(float(T))
        else:
            latencies.append(float(nonzero[0].item()))
    latency_per_sample = float(sum(latencies) / len(latencies))

    return energy_per_sample, latency_per_sample


# ----------------------------------------
# Training / Evaluation
# ----------------------------------------
def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):
    """Run one training epoch and return accuracy and loss."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        spk_out, mem_out = model(data)

        loss = loss_fn(spk_out, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        spk_sum = spk_out.sum(dim=0)
        _, predicted = spk_sum.max(1)

        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / len(loader)
    acc = correct / total if total > 0 else 0.0
    return acc, avg_loss


@torch.no_grad()
def eval_epoch(model, loader, device, loss_fn):
    """Evaluation loop with energy and latency proxies."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    energy_vals = []
    latency_vals = []

    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)

        spk_out, mem_out = model(data)

        loss = loss_fn(spk_out, targets)
        running_loss += loss.item()

        spk_sum = spk_out.sum(dim=0)
        _, predicted = spk_sum.max(1)

        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        energy, latency = batch_energy_latency(spk_out)
        energy_vals.append(energy)
        latency_vals.append(latency)

    avg_loss = running_loss / len(loader)
    acc = correct / total if total > 0 else 0.0
    avg_energy = sum(energy_vals) / len(energy_vals)
    avg_latency = sum(latency_vals) / len(latency_vals)

    return acc, avg_loss, avg_energy, avg_latency


# ----------------------------------------
# Data Retrieval and Preprocessing
# ----------------------------------------
def get_mnist_loaders(batch_size: int):
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ]
    )
    data_path = "./data/mnist"

    train_ds = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return train_loader, test_loader


# ----------------------------------------
# Plotting
# ----------------------------------------
def plot_curves(history, out_dir: str="./plots"):
    os.makedirs(out_dir, exist_ok=True)

    epochs = range(1, len(history["train_acc"]) + 1)

    ### Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["test_acc"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy.png"))
    plt.close()

    ### Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["test_loss"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    ### Energy Consumption (Proxy Measurement)
    plt.figure()
    plt.plot(epochs, history["energy"], label="Energy proxy (spikes/sample)")
    plt.xlabel("Epoch")
    plt.ylabel("Avg spikes per sample")
    plt.title("Energy Proxy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "energy.png"))
    plt.close()

    ### Inference Latency (Proxy Measurement)
    plt.figure()
    plt.plot(epochs, history["latency"], label="Latency proxy (steps)")
    plt.xlabel("Epoch")
    plt.ylabel("Avg first-spike time (steps)")
    plt.title("Latency Proxy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency.png"))
    plt.close()


# ----------------------------------------
# Main (driver function)
# ----------------------------------------
def main():
    print(f"Using device: {DEVICE}\n")
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)

    model = SNNNet(
        num_steps=NUM_STEPS,
        hidden_size=HIDDEN_SIZE,
        rram_enabled=ENABLE_RRAM,
        rram_noise_std=RRAM_NOISE_STD,
        rram_drift_per_epoch=RRAM_DRIFT_PER_EPOCH,
        rram_fail_prob=RRAM_FAIL_PROB,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))

    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    history = {
        "train_acc": [],
        "train_loss": [],
        "test_acc": [],
        "test_loss": [],
        "energy": [],
        "latency": [],
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}\n")

        model.step_drift()

        train_acc, train_loss = train_epoch(
            model, train_loader, DEVICE, optimizer, loss_fn
        )

        test_acc, test_loss, avg_energy, avg_latency = eval_epoch(
            model, test_loader, DEVICE, loss_fn
        )

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        history["energy"].append(avg_energy)
        history["latency"].append(avg_latency)

        print(
            f"  Train: acc={train_acc:.3f}, loss={train_loss:.4f} | Test: acc={test_acc:.3f}, loss={test_loss:.4f}\n"
        )
        print(f"  Energy proxy (spikes/sample)={avg_energy:.2f}, Latency proxy (steps)={avg_latency:.2f}\n")

    plot_curves(history, out_dir="plots")

    print("PoC complete")

if __name__ == "__main__":
    main()

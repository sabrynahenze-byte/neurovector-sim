############################################################
# poc_aihwkit.py — AIHWKIT SNN Example
#
# Purpose:
#   - Show how IBM AIHWKIT integrates into an SNN architecture.
#   - Provide a clean demonstration for team integration.
#
############################################################

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import spikegen

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.presets.devices import ReRamArrayOMPresetDevice


# ============================================================
# Build RPU config

def build_rpu_config():
    """
    Minimal RRAM device config for demonstration.
    Uses IBM's soft-bounds ReRAM preset.
    """
    cfg = InferenceRPUConfig()
    cfg.device = ReRamArrayOMPresetDevice()
    return cfg


# ============================================================
# AIHWKIT SNN Model

class AIHWKITSNN(nn.Module):
    """
    Spiking network using AIHWKIT AnalogLinear layers.
    Demonstrates:
      - Input flattening
      - Rate-based spike encoding
      - Two AnalogLinear layers
      - LIF dynamics

    All extended behavior (training, metrics, drift, plots)
    is handled elsewhere by the team.
    """

    def __init__(self, num_steps=25, hidden=500, rpu_config=None):
        super().__init__()
        beta = 0.9
        self.num_steps = num_steps

        self.fc1 = AnalogLinear(28*28, hidden, bias=True, rpu_config=rpu_config)
        self.lif1 = snn.Leaky(beta=beta)

        self.fc2 = AnalogLinear(hidden, 10, bias=True, rpu_config=rpu_config)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # Simple rate encoding (team may replace)
        spikes = spikegen.rate(x_flat, num_steps=self.num_steps)

        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()

        spk_out = []

        for t in range(self.num_steps):
            cur1 = self.fc1(spikes[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_out.append(spk2)

        return torch.stack(spk_out)  # shape: [T, B, 10]


# ============================================================
# Quick-test entry function

def demo_run():
    """
    Run to verify AIHWKIT integrates with snnTorch.
    This is NOT training.
    Loads a batch, forwards once, and prints shapes.
    """
    print("Running AIHWKIT SNN demo...")

    # Load one mini-batch of MNIST
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    test_ds = datasets.MNIST("./data/mnist", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    data, _ = next(iter(test_loader))

    # Build model
    rpu = build_rpu_config()
    model = AIHWKITSNN(num_steps=5, hidden=128, rpu_config=rpu)

    # Forward pass
    spk_out = model(data)

    print("Output spike tensor shape:", spk_out.shape)
    print("Demo complete — AIHWKIT integration verified.")


if __name__ == "__main__":
    demo_run()


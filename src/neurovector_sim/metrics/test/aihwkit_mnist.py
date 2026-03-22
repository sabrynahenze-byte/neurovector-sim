############################################################
# poc_aihwkit.py — AIHWKIT SNN Example
#
# Purpose:
#   - Show how IBM AIHWKIT integrates into an SNN architecture.
#   - Provide a clean demonstration for team integration.
#
############################################################

import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import spikegen, surrogate

from aihwkit.optim import AnalogOptimizer
from aihwkit.nn import AnalogLinear, AnalogConv2d
from aihwkit.simulator.configs import InferenceRPUConfig, SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
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

class AIHWKITSNNFC(nn.Module):
    """
    Fast Linear AIHWKIT SNN. 
    Skips convolutions for maximum simulation speed.
    """
    def __init__(self, num_steps=10, beta=0.95, rpu_config=None):
        super().__init__()
        self.num_steps = num_steps

        # Layer 1: 784 (28x28) -> 256 neurons
        self.fc1 = AnalogLinear(784, 256, rpu_config=rpu_config)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # Layer 2: 256 -> 10 (Digits)
        self.fc2 = AnalogLinear(256, 10, rpu_config=rpu_config)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        # Flatten input [Batch, 1, 28, 28] -> [Batch, 784]
        x_flat = x.view(x.size(0), -1)
        
        # Rate encoding
        spikes = spikegen.rate(x_flat, num_steps=self.num_steps)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_out = []

        for t in range(self.num_steps):
            cur1 = self.fc1(spikes[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_out.append(spk2)

        return torch.stack(spk_out)

class AIHWKITSNNConv(nn.Module):
    """
    Analog Spiking CNN using AIHWKIT AnalogConv2d layers.
    Matches the architecture: 32C3 -> 64C3 -> P2 -> 9216 -> 10.
    """

    def __init__(self, num_steps=25, beta=0.95, rpu_config=None):
        super().__init__()
        self.num_steps = num_steps

        # Layer 1: Analog Conv -> LIF
        # 1 input channel, 32 output channels, 3x3 kernel
        self.conv1 = AnalogConv2d(1, 32, kernel_size=3, rpu_config=rpu_config)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # Layer 2: Analog Conv -> LIF
        self.conv2 = AnalogConv2d(32, 64, kernel_size=3, rpu_config=rpu_config)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # Pooling and Flattening
        self.pool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()

        # Output Layer: Analog Linear -> LIF
        # 9216 = 64 channels * 12 * 12 spatial size
        self.fc1 = AnalogLinear(9216, 10, rpu_config=rpu_config)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        # 1. Rate Encoding (Input is [B, 1, 28, 28], output is [T, B, 1, 28, 28])
        # This converts static pixels into a stream of spikes over time
        spikes = spikegen.rate(x, num_steps=self.num_steps)

        # 2. State Initialization
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_out = []

        # 3. Temporal Loop
        for t in range(self.num_steps):
            # Block 1: Conv -> LIF
            cur1 = self.conv1(spikes[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            # Block 2: Conv -> LIF
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Pool and Flatten spatial features
            pooled_spks = self.pool(spk2)
            flattened = self.flat(pooled_spks)

            # Output: Linear -> LIF
            cur3 = self.fc1(flattened)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk_out.append(spk3)

        return torch.stack(spk_out) # shape: [T, B, 10]

# ============================================================
# Model train and save functions

def train_save_aihwkit_cnn(num_epochs=5):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # ConstantStepDevice: trains with realistic analog device noise so the model
    # learns to be robust to it — this is the intended AIHWKIT workflow.
    rpu_config = SingleRPUConfig(device=ConstantStepDevice())

    # Hyperparameters
    batch_size = 512
    # 25 steps gives rate-encoded spike trains enough density to carry useful
    # information. At 10 steps the trains were too sparse for the model to learn from.
    num_steps = 25
    learning_rate = 1e-3

    # No Normalize here — spikegen.rate() expects values in [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    model = AIHWKITSNNConv(num_steps=num_steps, rpu_config=rpu_config).to(device)

    # Resume from the latest checkpoint if one exists. Scans backwards from the
    # last epoch so we always pick up from the furthest point reached.
    start_epoch = 0
    for e in range(num_epochs - 1, -1, -1):
        checkpoint = f'aihwkit_cnn_epoch_{e + 1}.pt'
        if os.path.exists(checkpoint):
            print(f"Resuming from {checkpoint}...")
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
            start_epoch = e + 1
            break

    if start_epoch >= num_epochs:
        print("All epochs already complete — skipping training.")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = AnalogOptimizer(
        optim.Adam,
        model.parameters(),
        lr=1e-3
    )

    # Training Loop
    model.train()
    print(f"Training AIHWKIT Conv SNN on {device} (epochs {start_epoch + 1}–{num_epochs})...")

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass: [T, B, 10]
            spk_rec = model(data)
            
            # Sum spikes over time for CrossEntropy
            output = spk_rec.sum(dim=0)

            loss = criterion(output, targets)
            loss.backward()
            
            # The analog optimizer performs "analog-aware" weight updates
            optimizer.step()
            
            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        checkpoint_name = f'aihwkit_cnn_epoch_{epoch+1}.pt'
        torch.save(model.state_dict(), checkpoint_name)
        print(f"Successfully saved checkpoint: {checkpoint_name}")

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}")

    # Save the Model
    torch.save(model.state_dict(), 'aihwkit_conv_model.pt')
    print("Analog model saved as aihwkit_conv_model.pt")

def train_save_aihwkit_fc(num_epochs=5):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # ConstantStepDevice: trains with realistic analog device noise so the model
    # learns to be robust to it — this is the intended AIHWKIT workflow.
    rpu_config = SingleRPUConfig(device=ConstantStepDevice())

    # Hyperparameters
    batch_size = 512
    # 25 steps gives rate-encoded spike trains enough density to carry useful
    # information. At 10 steps the trains were too sparse for the model to learn from.
    num_steps = 25
    learning_rate = 1e-3

    # No Normalize here — spikegen.rate() expects values in [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    model = AIHWKITSNNFC(num_steps=num_steps, rpu_config=rpu_config).to(device)

    # Resume from the latest checkpoint if one exists. Scans backwards from the
    # last epoch so we always pick up from the furthest point reached.
    start_epoch = 0
    for e in range(num_epochs - 1, -1, -1):
        checkpoint = f'aihwkit_fc_epoch_{e + 1}.pt'
        if os.path.exists(checkpoint):
            print(f"Resuming from {checkpoint}...")
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
            start_epoch = e + 1
            break

    if start_epoch >= num_epochs:
        print("All epochs already complete — skipping training.")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = AnalogOptimizer(torch.optim.Adam, model.parameters(), lr=1e-3)

    # Training Loop
    model.train()
    print(f"Training AIHWKIT FC SNN on {device} (epochs {start_epoch + 1}–{num_epochs})...")

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass: [T, B, 10]
            spk_rec = model(data)
            
            # Sum spikes over time for CrossEntropy
            output = spk_rec.sum(dim=0)

            loss = criterion(output, targets)
            loss.backward()
            
            # The analog optimizer performs "analog-aware" weight updates
            optimizer.step()
            
            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        checkpoint_name = f'aihwkit_fc_epoch_{epoch+1}.pt'
        torch.save(model.state_dict(), checkpoint_name)
        print(f"Successfully saved checkpoint: {checkpoint_name}")

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}")

    # Save the Model
    torch.save(model.state_dict(), 'aihwkit_fc_model.pt')
    print("Analog model saved as aihwkit_fc_model.pt")

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
    # demo_run()
    train_save_aihwkit_cnn()
    # train_save_aihwkit_fc()


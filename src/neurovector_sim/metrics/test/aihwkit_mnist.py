# AIHWKIT hardware-aware FC and Conv SNNs.

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

def build_rpu_config():
    """ReRAM inference config using IBM's soft-bounds preset."""
    cfg = InferenceRPUConfig()
    cfg.device = ReRamArrayOMPresetDevice()
    return cfg


class AIHWKITSNN(nn.Module):
    """Rate-encoded SNN with two AnalogLinear + LIF layers."""

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
    """FC-only AIHWKIT SNN (784 -> 256 -> 10)."""
    def __init__(self, num_steps=10, beta=0.95, rpu_config=None):
        super().__init__()
        self.num_steps = num_steps

        self.fc1 = AnalogLinear(784, 256, rpu_config=rpu_config)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = AnalogLinear(256, 10, rpu_config=rpu_config)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_out = []

        for t in range(self.num_steps):
            cur1 = self.fc1(x_flat)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_out.append(spk2)

        return torch.stack(spk_out)


class AIHWKITSNNConv(nn.Module):
    """AIHWKIT Conv SNN (32C3 -> 64C3 -> P2 -> 9216 -> 10)."""

    def __init__(self, num_steps=25, beta=0.95, rpu_config=None):
        super().__init__()
        self.num_steps = num_steps

        self.conv1 = AnalogConv2d(1, 32, kernel_size=3, rpu_config=rpu_config)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.conv2 = AnalogConv2d(32, 64, kernel_size=3, rpu_config=rpu_config)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.pool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()

        # 9216 = 64 * 12 * 12
        self.fc1 = AnalogLinear(9216, 10, rpu_config=rpu_config)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_out = []

        for t in range(self.num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            pooled_spks = self.pool(spk2)
            flattened = self.flat(pooled_spks)

            cur3 = self.fc1(flattened)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk_out.append(spk3)

        return torch.stack(spk_out) # shape: [T, B, 10]

def train_save_aihwkit_cnn(num_epochs=10):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # ConstantStepDevice injects analog noise during training.
    rpu_config = SingleRPUConfig(device=ConstantStepDevice())

    batch_size = 512
    num_steps = 15
    # lr=1e-3 matches SNN baselines. Lower lr + grad clipping starves
    # ConstantStepDevice of pulses, so weights barely move.
    learning_rate = 1e-3
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False
    )

    model = AIHWKITSNNConv(num_steps=num_steps, rpu_config=rpu_config).to(device)

    # Resume from the latest checkpoint if one exists
    start_epoch = 0
    for e in range(num_epochs - 1, -1, -1):
        checkpoint = f'aihwkit_cnn_epoch_{e + 1}.pt'
        if os.path.exists(checkpoint):
            print(f"Resuming from {checkpoint}...")
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
            start_epoch = e + 1
            break

    if start_epoch >= num_epochs:
        print("All epochs already complete, skipping training.")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = AnalogOptimizer(
        optim.Adam,
        model.parameters(),
        lr=learning_rate
    )

    # ConstantStepDevice's noisy updates can push weights out of a good
    # region if training goes too long, so stop early on val loss.
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3

    # Training Loop
    model.train()
    print(f"Training AIHWKIT Conv SNN on {device} (epochs {start_epoch + 1}-{num_epochs})...")

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            spk_rec = model(data)
            output = spk_rec.sum(dim=0)

            loss = criterion(output, targets)
            loss.backward()

            # max_norm=5.0 caps BPTT gradient spikes across 15 LIF steps
            # without starving ConstantStepDevice of pulses (1.0 did).
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        checkpoint_name = f'aihwkit_cnn_epoch_{epoch+1}.pt'
        torch.save(model.state_dict(), checkpoint_name)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                spk_rec = model(data)
                val_loss += criterion(spk_rec.sum(dim=0), targets).item()
        val_loss /= len(val_loader)
        model.train()

        avg_train = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'aihwkit_conv_model.pt')
            print(f"  Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    print("Best analog conv model saved as aihwkit_conv_model.pt")

def train_save_aihwkit_fc(num_epochs=10):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # ConstantStepDevice injects analog noise during training.
    rpu_config = SingleRPUConfig(device=ConstantStepDevice())

    batch_size = 512
    num_steps = 15
    # See train_save_aihwkit_cnn for lr rationale.
    learning_rate = 1e-3
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False
    )

    model = AIHWKITSNNFC(num_steps=num_steps, rpu_config=rpu_config).to(device)

    # Resume from the latest checkpoint if one exists
    start_epoch = 0
    for e in range(num_epochs - 1, -1, -1):
        checkpoint = f'aihwkit_fc_epoch_{e + 1}.pt'
        if os.path.exists(checkpoint):
            print(f"Resuming from {checkpoint}...")
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
            start_epoch = e + 1
            break

    if start_epoch >= num_epochs:
        print("All epochs already complete, skipping training.")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = AnalogOptimizer(torch.optim.Adam, model.parameters(), lr=learning_rate)

    # Early stopping, see train_save_aihwkit_cnn for rationale.
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3

    # Training Loop
    model.train()
    print(f"Training AIHWKIT FC SNN on {device} (epochs {start_epoch + 1}-{num_epochs})...")

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            spk_rec = model(data)
            output = spk_rec.sum(dim=0)

            loss = criterion(output, targets)
            loss.backward()

            # See train_save_aihwkit_cnn for grad clipping rationale.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        checkpoint_name = f'aihwkit_fc_epoch_{epoch+1}.pt'
        torch.save(model.state_dict(), checkpoint_name)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                spk_rec = model(data)
                val_loss += criterion(spk_rec.sum(dim=0), targets).item()
        val_loss /= len(val_loader)
        model.train()

        avg_train = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'aihwkit_fc_model.pt')
            print(f"  Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    print("Best analog FC model saved as aihwkit_fc_model.pt")

def demo_run():
    """Quick sanity check that AIHWKIT + snnTorch work together."""
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
    print("Demo complete, AIHWKIT integration verified.")


if __name__ == "__main__":
    # demo_run()
    train_save_aihwkit_cnn()
    # train_save_aihwkit_fc()


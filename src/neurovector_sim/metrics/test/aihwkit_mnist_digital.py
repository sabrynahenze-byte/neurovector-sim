# AIHWKIT digitally-trained FC and Conv SNNs.
# Uses InferenceRPUConfig + ReRamArrayOMPresetDevice: device noise
# only hits the forward pass (weight reads), backward pass and
# updates stay in standard float math.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import surrogate

from aihwkit.nn import AnalogLinear, AnalogConv2d
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.presets.devices import ReRamArrayOMPresetDevice


def build_rpu_config():
    """InferenceRPUConfig with ReRamArrayOMPresetDevice for read noise."""
    cfg = InferenceRPUConfig()
    cfg.device = ReRamArrayOMPresetDevice()
    return cfg


class AIHWKITDigitalFC(nn.Module):
    """AIHWKIT FC SNN (784 -> 256 -> 10), trained digitally."""
    def __init__(self, num_steps=15, beta=0.95, rpu_config=None):
        super().__init__()
        self.num_steps = num_steps

        if rpu_config is None:
            rpu_config = build_rpu_config()

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


class AIHWKITDigitalConv(nn.Module):
    """AIHWKIT Conv SNN (32C3 -> 64C3 -> P2 -> 9216 -> 10), trained digitally."""
    def __init__(self, num_steps=15, beta=0.95, rpu_config=None):
        super().__init__()
        self.num_steps = num_steps

        if rpu_config is None:
            rpu_config = build_rpu_config()

        self.conv1 = AnalogConv2d(1, 32, kernel_size=3, rpu_config=rpu_config)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.conv2 = AnalogConv2d(32, 64, kernel_size=3, rpu_config=rpu_config)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.pool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()

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

        return torch.stack(spk_out)


def train_save_digital_fc(num_epochs=10):
    device = torch.device("cpu")

    batch_size = 512
    num_steps = 15
    learning_rate = 1e-3

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    model = AIHWKITDigitalFC(num_steps=num_steps).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    print(f"Training AIHWKIT Digital FC SNN on {device}...")

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            spk_rec = model(data)
            output = spk_rec.sum(dim=0)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train:.4f}")

    torch.save(model.state_dict(), 'aihwkit_digital_fc_model.pt')
    print("Model saved as aihwkit_digital_fc_model.pt")


def train_save_digital_cnn(num_epochs=10):
    device = torch.device("cpu")

    batch_size = 512
    num_steps = 15
    learning_rate = 1e-3

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    model = AIHWKITDigitalConv(num_steps=num_steps).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    print(f"Training AIHWKIT Digital Conv SNN on {device}...")

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            spk_rec = model(data)
            output = spk_rec.sum(dim=0)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train:.4f}")

    torch.save(model.state_dict(), 'aihwkit_digital_conv_model.pt')
    print("Model saved as aihwkit_digital_conv_model.pt")


if __name__ == "__main__":
    train_save_digital_fc()
    train_save_digital_cnn()

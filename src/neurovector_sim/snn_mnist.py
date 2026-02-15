import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader


# Device
device = torch.device("cpu")


# Hyperparameters
batch_size = 64
num_steps = 25
beta = 0.95
num_epochs = 1


# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# SNN Model (Matches Classical 784 → 128 → 64 → 10)
class SNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 1
        self.fc1 = nn.Linear(28 * 28, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # Layer 2
        self.fc2 = nn.Linear(128, 64)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # Output Layer
        self.fc3 = nn.Linear(64, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)

        return torch.stack(spk3_rec)


# Initialize model
model = SNNModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Training Loop
for epoch in range(num_epochs):
    for data, targets in train_loader:
        data = data.view(data.size(0), -1).to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        spk_rec = model(data)
        output = spk_rec.sum(dim=0)

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# Testing
correct = 0
total = 0

with torch.no_grad():
    for data, targets in test_loader:
        data = data.view(data.size(0), -1).to(device)
        targets = targets.to(device)

        spk_rec = model(data)
        output = spk_rec.sum(dim=0)
        _, predicted = torch.max(output, 1)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

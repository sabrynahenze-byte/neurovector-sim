import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader


class SNNFCModel(nn.Module):
    def __init__(self, beta=0.95, num_steps=25):
        super().__init__()

        self.beta = beta
        self.num_steps = num_steps

        self.fc1 = nn.Linear(28 * 28, 256)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(256, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []

        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)
    
class SNNConvModel(nn.Module):
    def __init__(self, beta=0.95, num_steps=25):
        super().__init__()
        self.num_steps = num_steps

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.pool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()

        # 9216 = 64 * 12 * 12
        self.fc1 = nn.Linear(9216, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []

        for _ in range(self.num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            pooled_spks = self.pool(spk2)
            flat_spks = self.flat(pooled_spks)

            cur3 = self.fc1(flat_spks)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)

        return torch.stack(spk3_rec)

def train_save_snn_model(num_epochs=10):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 512
    num_steps = 15
    beta = 0.95

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = SNNFCModel(beta=beta, num_steps=num_steps).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
    model.train()
    print(f"Training SNN FC on {device}...")
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
    model.eval()
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

    torch.save(model.state_dict(), 'snntorch_fc_model.pt')

def train_save_snn_cnn_model(num_epochs=10):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 512
    num_steps = 15
    beta = 0.95

    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = SNNConvModel(beta=beta, num_steps=num_steps).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training Loop
    print(f"Training CSNN on {device}...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            spk_rec = model(data)
            
            output = spk_rec.sum(dim=0)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}")

    # Testing
    model.eval()
    correct = 0
    total = 0

    print("Evaluating...")
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            spk_rec = model(data)
            output = spk_rec.sum(dim=0)
            _, predicted = torch.max(output, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Save the weights
    torch.save(model.state_dict(), 'snntorch_conv_model.pt')
    print("Model saved as snntorch_conv_model.pt")

if __name__ == "__main__":
    train_save_snn_model()
    train_save_snn_cnn_model()
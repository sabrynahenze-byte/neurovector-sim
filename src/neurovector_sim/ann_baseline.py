import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class ClassicalANN(nn.Module):
    def __init__(self, hidden_size=500):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_mnist_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, dataloader, device, optimizer, loss_fn):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return accuracy, avg_loss


def eval_epoch(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = loss_fn(output, target)

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return accuracy, avg_loss, None, None
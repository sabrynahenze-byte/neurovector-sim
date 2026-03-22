import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Simple but effective CNN
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train_save_cnn_model(num_epochs=3):
    model = CNNModel()
    optimizer = optim.Adam(model.parameters())
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True)

    model.train()
    print("Training PyTorch CNN...")
    for epoch in range(num_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs} complete")

    torch.save(model.state_dict(), 'pytorch_conv_model.pt')
    print("✅ Model saved as pytorch_conv_model.pt")


if __name__ == "__main__":
    train_save_cnn_model()
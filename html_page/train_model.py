import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Simple but effective CNN
class SimpleMNIST(nn.Module):
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
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Train for just 3-5 epochs (takes minutes)
model = SimpleMNIST()
optimizer = optim.Adam(model.parameters())
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)

model.train()
for epoch in range(3):
    for data, target in train_loader:
        optimizer.zero_grad()
        loss = nn.functional.nll_loss(model(data), target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} complete")

# Save it
torch.save(model.state_dict(), 'my_trained_mnist.pt')
print("✅ Model trained and saved!")
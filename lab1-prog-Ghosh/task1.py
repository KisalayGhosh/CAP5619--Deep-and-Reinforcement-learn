import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: Custom Dataset Class
# ----------------------------
class HandwrittenDigitsDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = int(float(parts[0]))  
                pixels = list(map(float, parts[1:257]))  
                self.data.append((label, pixels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, pixels = self.data[idx]
        image = torch.tensor(pixels, dtype=torch.float32).reshape(1, 16, 16)  
        return image, torch.tensor(label, dtype=torch.long)

# ----------------------------
# Step 2: Model Architectures
# ----------------------------

# Model 1: Fully Connected Network (FCN)
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))  
        x = self.sigmoid(self.fc2(x))  
        x = self.relu(self.fc3(x))  
        return self.fc4(x)

# Model 2: Locally Connected Network 
class LocallyConnected(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.unfold = nn.Unfold(kernel_size, stride=stride)
        self.linear = nn.Linear(in_channels * kernel_size**2, out_channels)

    def forward(self, x):
        # x shape: (batch, in_channels, H, W)
        batch_size = x.shape[0]
        patches = self.unfold(x)  # Shape: (batch, in_channels*kernel^2, L)
        patches = patches.permute(0, 2, 1)  # Shape: (batch, L, in_channels*kernel^2)
        out = self.linear(patches)  # Shape: (batch, L, out_channels)
        out = out.permute(0, 2, 1)  # Shape: (batch, out_channels, L)
        
        
        h = (x.shape[2] - self.kernel_size) // self.stride + 1
        w = (x.shape[3] - self.kernel_size) // self.stride + 1
        return out.reshape(batch_size, out.shape[1], h, w)

class LocalNet(nn.Module):
    def __init__(self):
        super(LocalNet, self).__init__()
        # Layer 1: Input 16x16 → Output 14x14
        self.lc1 = LocallyConnected(1, 32, kernel_size=3, stride=1)
        # Layer 2: Input 14x14 → Output 12x12
        self.lc2 = LocallyConnected(32, 64, kernel_size=3, stride=1)
        # Layer 3: Input 12x12 → Output 10x10
        self.lc3 = LocallyConnected(64, 128, kernel_size=3, stride=1)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 10 * 10, 10)  # Adjusted for 10x10 spatial output
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.lc1(x))  # Output: (batch, 32, 14, 14)
        x = self.tanh(self.lc2(x))  # Output: (batch, 64, 12, 12)
        x = self.relu(self.lc3(x))  # Output: (batch, 128, 10, 10)
        x = self.flatten(x)
        return self.fc(x)

# Model 3: Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))  
        x = self.pool(x)
        x = self.relu(self.conv2(x))  
        x = self.sigmoid(self.conv3(x))  
        x = self.flatten(x)
        x = self.relu(self.fc1(x))  
        return self.fc2(x)

# ----------------------------
# Step 3: Training Setup
# ----------------------------
def train_model(model, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

       
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.2f}%")

# ----------------------------
# Step 4: Execute Training
# ----------------------------
if __name__ == "__main__":
   
    train_dataset = HandwrittenDigitsDataset("zip_train.txt")
    test_dataset = HandwrittenDigitsDataset("zip_test.txt")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    
    fcn = FCN()
    local_net = LocalNet()
    cnn = CNN()

   
    print("Training Fully Connected Network:")
    train_model(fcn, train_loader, test_loader, epochs=10)

    
    print("\nTraining Locally Connected Network:")
    train_model(local_net, train_loader, test_loader, epochs=10)

    
    print("\nTraining Convolutional Neural Network:")
    train_model(cnn, train_loader, test_loader, epochs=10)
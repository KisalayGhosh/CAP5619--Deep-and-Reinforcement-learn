import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from task1 import HandwrittenDigitsDataset 

# ----------------------------
# Step 1: Load Dataset
# ----------------------------
train_dataset = HandwrittenDigitsDataset("zip_train.txt")
test_dataset = HandwrittenDigitsDataset("zip_test.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# ----------------------------
# Step 2: Fully Connected Network (FCN) with Batch Normalization
# ----------------------------
class FCN_BN(nn.Module):
    def __init__(self):
        super(FCN_BN, self).__init__()
        self.fc1 = nn.Linear(256, 512)
        self.bn1 = nn.BatchNorm1d(512)  
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)  
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)  
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(self.fc1(x)))  
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

# ----------------------------
# Step 3: Convolutional Neural Network (CNN) with Batch Normalization
# ----------------------------
class CNN_BN(nn.Module):
    def __init__(self):
        super(CNN_BN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.bn4 = nn.BatchNorm1d(128)  
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  
        x = self.relu(self.bn3(self.conv3(x)))  
        x = self.flatten(x)
        x = self.relu(self.bn4(self.fc1(x)))  
        return self.fc2(x)

# ----------------------------
# Step 4: Define Training Function
# ----------------------------
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, optimizer_type="adam", momentum=0):
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

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

        # Validation
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
# Step 5: Train FCN with Batch Normalization
# ----------------------------
print("\nTraining FCN with Batch Normalization:")
fcn_bn = FCN_BN()
train_model(fcn_bn, train_loader, test_loader, epochs=10)

# ----------------------------
# Step 6: Train CNN with Batch Normalization
# ----------------------------
print("\nTraining CNN with Batch Normalization:")
cnn_bn = CNN_BN()
train_model(cnn_bn, train_loader, test_loader, epochs=10)

# ----------------------------
# Step 7: Experimenting with Different Batch Sizes
# ----------------------------
batch_sizes = [16, 64, 256]
for batch_size in batch_sizes:
    print(f"\nTraining CNN with Batch Size: {batch_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    cnn_bn = CNN_BN()
    train_model(cnn_bn, train_loader, test_loader, epochs=5)

# ----------------------------
# Step 8: Experimenting with Momentum
# ----------------------------
momentum_values = [0.5, 0.9, 0.99]
for momentum in momentum_values:
    print(f"\nTraining CNN with Momentum: {momentum}")

    cnn_bn = CNN_BN()
    train_model(cnn_bn, train_loader, test_loader, epochs=5, optimizer_type="sgd", momentum=momentum)

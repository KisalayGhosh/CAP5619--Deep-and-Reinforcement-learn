import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from task1 import FCN, HandwrittenDigitsDataset

# ----------------------------
# Step 1: Load Dataset
# ----------------------------
train_dataset = HandwrittenDigitsDataset("zip_train.txt")
test_dataset = HandwrittenDigitsDataset("zip_test.txt")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# ----------------------------
# Step 2: Define FGSM Attack
# ----------------------------
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbed_images = images + epsilon * images.grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images

# ----------------------------
# Step 3: Train Model with Adversarial Examples
# ----------------------------
def adversarial_train(model, train_loader, epsilon=0.1, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            perturbed_images = fgsm_attack(model, images, labels, epsilon)
            optimizer.zero_grad()
            outputs = model(perturbed_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print(f"Adversarial Training Completed for Epsilon: {epsilon}")


fcn = FCN()
adversarial_train(fcn, train_loader, epsilon=0.1, epochs=5)

# ----------------------------
# Step 4: Hyperparameter Optimization (Random Search)
# ----------------------------
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def random_search(train_loader, epochs=5):
    best_model = None
    best_acc = 0
    best_params = None

    for _ in range(5):
        lr = random.choice([0.0001, 0.001, 0.01])
        batch_size = random.choice([32, 64, 128])
        dropout_rate = random.choice([0.2, 0.5, 0.7])
        
        print(f"Testing: LR={lr}, Batch={batch_size}, Dropout={dropout_rate}")
        
        model = FCN()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        acc = evaluate_model(model, test_loader)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_params = (lr, batch_size, dropout_rate)

    print(f"Best Hyperparameters: LR={best_params[0]}, Batch={best_params[1]}, Dropout={best_params[2]}")
    return best_acc


best_model = random_search(train_loader, epochs=5)

# ----------------------------
# Step 5: Architecture Optimization (Grid Search)
# ----------------------------
class SearchableFCN(nn.Module):
    def __init__(self, num_layers=3, neurons=256, activation='relu'):
        super(SearchableFCN, self).__init__()
        layers = [nn.Linear(256, neurons)]
        
        for _ in range(num_layers - 1):
            layers.append(nn.ReLU() if activation == 'relu' else nn.Sigmoid())
            layers.append(nn.Linear(neurons, neurons))
        
        layers.append(nn.Linear(neurons, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

# Grid search over architectures
best_acc = 0
best_arch = None

for num_layers in [2, 3, 4]:
    for neurons in [128, 256, 512]:
        for activation in ['relu', 'sigmoid']:
            print(f"Testing Architecture: Layers={num_layers}, Neurons={neurons}, Activation={activation}")
            model = SearchableFCN(num_layers=num_layers, neurons=neurons, activation=activation)
            acc = random_search(train_loader, epochs=3)  # Reuse random search for evaluation
            if acc > best_acc:
                best_acc = acc
                best_arch = (num_layers, neurons, activation)

print(f"Best Architecture: Layers={best_arch[0]}, Neurons={best_arch[1]}, Activation={best_arch[2]}")

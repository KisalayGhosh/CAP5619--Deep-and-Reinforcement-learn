import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from task1 import FCN, LocalNet, CNN, HandwrittenDigitsDataset  # Import models from Task 1

# ----------------------------
# Step 1: Load Dataset
# ----------------------------
train_dataset = HandwrittenDigitsDataset("zip_train.txt")
test_dataset = HandwrittenDigitsDataset("zip_test.txt")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# ----------------------------
# Step 2: Define Training Function
# ----------------------------
def train_model(model, train_loader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
    return model  


fcn = train_model(FCN(), train_loader, epochs=5)
local_net = train_model(LocalNet(), train_loader, epochs=5)
cnn = train_model(CNN(), train_loader, epochs=5)

# ----------------------------
# Step 3: Ensemble Prediction
# ----------------------------
def ensemble_predict(models, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models:
        model.to(device)
        model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = [F.softmax(model(images), dim=1) for model in models]  # Softmax for probability
            avg_output = torch.mean(torch.stack(outputs), dim=0)  # Average softmax scores
            _, predicted = torch.max(avg_output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"\nEnsemble Test Accuracy: {accuracy:.2f}%")

ensemble_models = [fcn, local_net, cnn]
ensemble_predict(ensemble_models, test_loader)

# ----------------------------
# Step 4: Dropout Regularization (FCN)
# ----------------------------
class FCN_Dropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(FCN_Dropout, self).__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# ----------------------------
# Step 5: Train FCN with Different Dropout Rates
# ----------------------------
dropout_rates = [0.0, 0.5, 0.9]
for rate in dropout_rates:
    print(f"\nTraining Fully Connected Network with Dropout {rate}")
    fcn_dropout = FCN_Dropout(dropout_rate=rate)
    train_model(fcn_dropout, train_loader, epochs=5)

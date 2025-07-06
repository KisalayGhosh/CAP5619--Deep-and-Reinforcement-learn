from dnn_classifier import DNNClassifier, load_data, train_model, evaluate_model
import torch
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load data
train_loader = load_data("embeddings/train_embeddings.npy", "embeddings/train_labels.npy")
val_loader = load_data("embeddings/eval_embeddings.npy", "embeddings/eval_labels.npy")
test_loader = load_data("embeddings/test_embeddings.npy", "embeddings/test_labels.npy")

# Initialize and train
input_dim = 320  # for ESM2-t6-8M 
model = DNNClassifier(input_dim)
train_model(model, train_loader, val_loader, epochs=15)


torch.save(model.state_dict(), "models/best_dnn_model.pth")


evaluate_model(model, test_loader)

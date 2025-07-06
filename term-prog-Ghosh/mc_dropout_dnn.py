# mc_dropout_dnn.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from dnn_classifier import DNNClassifier


def enable_dropout(model):
    """Enable dropout layers during test-time."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def load_data(emb_path, label_path):
    X = torch.tensor(np.load(emb_path), dtype=torch.float32)
    y = torch.tensor(np.load(label_path), dtype=torch.float32).unsqueeze(1)
    return DataLoader(TensorDataset(X, y), batch_size=1, shuffle=False)


def predict_with_uncertainty(model, dataloader, T=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    enable_dropout(model)  

    probs = []
    for _ in range(T):
        outputs = []
        with torch.no_grad():
            for xb, _ in dataloader:
                xb = xb.to(device)
                out = torch.sigmoid(model(xb)).cpu().numpy()
                outputs.append(out)
        probs.append(np.vstack(outputs))

    probs = np.stack(probs, axis=0)  # T x N x 1
    mean_probs = np.mean(probs, axis=0).squeeze()
    std_devs = np.std(probs, axis=0).squeeze()

    return mean_probs, std_devs


def plot_uncertainty(mean_probs, std_devs, save_path="figures/dnn_uncertainty.png"):
    plt.figure(figsize=(8, 6))
    plt.errorbar(range(len(mean_probs)), mean_probs, yerr=std_devs, fmt='o', alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Probability (± std dev)")
    plt.title("Prediction Uncertainty via Monte Carlo Dropout")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

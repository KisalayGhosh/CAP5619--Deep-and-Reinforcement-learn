# tsne_visualization.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os

def run_tsne(embedding_path, label_path, save_path="figures/tsne_plot.png"):
    X = np.load(embedding_path)
    y = np.load(label_path)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    for label in [0, 1]:
        plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], label=f"Class {label}", alpha=0.7)

    plt.legend()
    plt.title("t-SNE Visualization of ESM Embeddings")
    os.makedirs("figures", exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")

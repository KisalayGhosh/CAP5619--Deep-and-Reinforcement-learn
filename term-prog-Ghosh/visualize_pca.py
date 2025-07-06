import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#  embeddings and mutation metadata
data = np.load("mutant_embeddings.npz")
X = data["features"]  # (268, 1280) 
positions = data["positions"]

df = pd.read_csv("mutants.csv")  
if len(df) != len(X):
    raise ValueError("Mismatch between mutants.csv and embeddings")

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=positions, cmap="viridis", s=50, alpha=0.8)
plt.title("PCA of BRCA1 Variant Embeddings (colored by position)")
plt.xlabel("PC1")
plt.ylabel("PC2")
cbar = plt.colorbar(scatter)
cbar.set_label("Mutation Position")
plt.tight_layout()
plt.savefig("figures/brca1_pca_position_colored.png")
plt.show()

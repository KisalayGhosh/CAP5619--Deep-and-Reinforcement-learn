# run_correlation.py

import pandas as pd
import numpy as np
from correlation_analysis import compute_biological_features, compute_correlations, plot_top_correlations

# Load sequences and embeddings
sequences = pd.read_csv("df_tp53_train.csv")["sequence"].tolist()
embeddings = np.load("embeddings/train_embeddings.npy")

# Compute biological features
features_df = compute_biological_features(sequences)

# Compute correlation
correlations = compute_correlations(embeddings, features_df)

#  10 correlations for each feature
for feature in features_df.columns:
    plot_top_correlations(correlations, feature, top_k=10)

print("Correlation analysis completed and plots saved to figures/")

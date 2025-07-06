# run_mc_dropout.py

import torch
import numpy as np
from dnn_classifier import DNNClassifier
from mc_dropout_dnn import load_data, predict_with_uncertainty, plot_uncertainty
import os

os.makedirs("figures", exist_ok=True)

# Load test data
test_loader = load_data("embeddings/test_embeddings.npy", "embeddings/test_labels.npy")

# Initialize and load DNN model
input_dim = 320  # for esm2_t6_8M_UR50D
model = DNNClassifier(input_dim)
model.load_state_dict(torch.load("models/best_dnn_model.pth"))

# Predict with uncertainty
mean_probs, std_devs = predict_with_uncertainty(model, test_loader, T=50)


plot_uncertainty(mean_probs, std_devs)
print("Uncertainty estimation complete. Plot saved to figures/dnn_uncertainty.png")

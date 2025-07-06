# run_xgboost.py

import os
from xgboost_classifier import load_data, train_xgboost, evaluate_xgboost, explain_with_shap

os.makedirs("figures", exist_ok=True)

# Load data
X_train, y_train = load_data("embeddings/train_embeddings.npy", "embeddings/train_labels.npy")
X_val, y_val = load_data("embeddings/eval_embeddings.npy", "embeddings/eval_labels.npy")
X_test, y_test = load_data("embeddings/test_embeddings.npy", "embeddings/test_labels.npy")

# Train model
model = train_xgboost(X_train, y_train, X_val, y_val)

# Evaluate
evaluate_xgboost(model, X_test, y_test)

# with SHAP
explain_with_shap(model, X_test)

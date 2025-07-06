# calibrate_dnn.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from joblib import dump

# Load validation outputs from training phase
val_probs = np.load("outputs/val_probs.npy")
val_labels = np.load("outputs/val_labels.npy").ravel()


# Compute uncalibrated reliability curve
prob_true, prob_pred = calibration_curve(val_labels, val_probs, n_bins=10)

# Platt Scaling (Logistic Regression)
platt = LogisticRegression()
platt.fit(val_probs.reshape(-1, 1), val_labels)
platt_probs = platt.predict_proba(val_probs.reshape(-1, 1))[:, 1]

# Isotonic Regression
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(val_probs, val_labels)
iso_probs = iso.predict(val_probs)

# Save calibration models
dump(platt, "models/platt_scaler.joblib")
dump(iso, "models/isotonic_scaler.joblib")

# Reliability plot
plt.figure(figsize=(8,6))
plt.plot(prob_pred, prob_true, "s-", label="Uncalibrated")
plt.plot(*calibration_curve(val_labels, platt_probs, n_bins=10), label="Platt")
plt.plot(*calibration_curve(val_labels, iso_probs, n_bins=10), label="Isotonic")
plt.plot([0,1], [0,1], "k--", label="Perfect Calibration")
plt.xlabel("Predicted Probability")
plt.ylabel("True Frequency")
plt.title("Reliability Diagram")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/dnn_reliability.png")
plt.show()

# Report Brier scores
print("Brier Score:")
print(f"  Uncalibrated: {brier_score_loss(val_labels, val_probs):.4f}")
print(f"  Platt:        {brier_score_loss(val_labels, platt_probs):.4f}")
print(f"  Isotonic:     {brier_score_loss(val_labels, iso_probs):.4f}")

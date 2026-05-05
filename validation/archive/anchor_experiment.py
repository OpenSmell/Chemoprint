"""
Anchor Calibration with Mixture Anchors and Gaussian Process
"""

import numpy as np
import os
from glob import glob
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from chemoprint import chemoprint_from_smiles

# ------------------------------
# 1. Load data and split by batch (same as before)
# ------------------------------
data_dir = "../gas+sensor+array+drift+dataset/Dataset"

gas_to_smiles = {
    1: "CCO", 2: "C=C", 3: "N", 4: "CC=O", 5: "CC(=O)C", 6: "Cc1ccccc1"
}

chemoprint_by_gas = {}
for label, smiles in gas_to_smiles.items():
    cp = chemoprint_from_smiles(smiles)
    if cp is None:
        raise ValueError(f"Invalid SMILES for gas {label}: {smiles}")
    chemoprint_by_gas[label] = cp

X_A, y_gas_A = [], []
X_B, y_gas_B = [], []

for filepath in sorted(glob(os.path.join(data_dir, "batch*.dat"))):
    batch_num = int(filepath.split('batch')[1].split('.')[0])
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            label = int(parts[0])
            features = []
            for token in parts[1:]:
                _, val = token.split(':')
                features.append(float(val))
            if len(features) != 128:
                continue
            if batch_num <= 5:
                X_A.append(features)
                y_gas_A.append(label)
            else:
                X_B.append(features)
                y_gas_B.append(label)

X_A = np.array(X_A)
y_gas_A = np.array(y_gas_A)
X_B = np.array(X_B)
y_gas_B = np.array(y_gas_B)
print(f"Device A: {len(X_A)} samples, Device B: {len(X_B)} samples")

# ------------------------------
# 2. Helper to get a random mixture reading for a given device
# ------------------------------
def get_mixture_reading(device, gases, weights, X, y_gas):
    """Return sensor reading of mixture (weighted sum of random pure samples)."""
    mix = np.zeros(128)
    for gas, w in zip(gases, weights):
        idx = np.where(y_gas == gas)[0]
        sample = X[np.random.choice(idx)]
        mix += w * sample
    return mix

# ------------------------------
# 3. Create anchor mixtures (pure gases + synthetic mixtures)
# ------------------------------
np.random.seed(42)
n_anchor_mixtures = 30   # number of synthetic mixtures to use as anchors (in addition to pure gases)
n_components_range = (2, 3)   # each mixture has 2 or 3 components

# Collect anchor points for device A and B
anchors_A = []   # list of (sensor_vector, true_chemoprint)
anchors_B = []

# First add the 6 pure gases (using mean sensor reading for each gas)
def gas_mean_reading(X, y_gas, gas):
    idx = np.where(y_gas == gas)[0]
    return X[idx].mean(axis=0)

for gas in range(1, 7):
    sensor_A = gas_mean_reading(X_A, y_gas_A, gas)
    sensor_B = gas_mean_reading(X_B, y_gas_B, gas)
    true_cp = chemoprint_by_gas[gas]
    anchors_A.append((sensor_A, true_cp))
    anchors_B.append((sensor_B, true_cp))

# Now generate synthetic mixtures as anchors
for _ in range(n_anchor_mixtures):
    n_comp = np.random.randint(n_components_range[0], n_components_range[1]+1)
    gases = np.random.choice(range(1, 7), n_comp, replace=False)
    weights = np.random.rand(n_comp)
    weights /= weights.sum()
    # True chemoprint for mixture (weighted average of pure chemoprints)
    true_cp = np.zeros(29)
    for gas, w in zip(gases, weights):
        true_cp += w * chemoprint_by_gas[gas]
    # Sensor reading for device A
    sensor_A = get_mixture_reading('A', gases, weights, X_A, y_gas_A)
    # Sensor reading for device B
    sensor_B = get_mixture_reading('B', gases, weights, X_B, y_gas_B)
    anchors_A.append((sensor_A, true_cp))
    anchors_B.append((sensor_B, true_cp))

print(f"Total anchor points per device: {len(anchors_A)} (6 pure + {n_anchor_mixtures} mixtures)")

# Separate into arrays
X_anchors_A = np.array([a[0] for a in anchors_A])
y_anchors_A = np.array([a[1] for a in anchors_A])
X_anchors_B = np.array([a[0] for a in anchors_B])
y_anchors_B = np.array([a[1] for a in anchors_B])

# ------------------------------
# 4. Dimensionality reduction (PCA) for each device on anchor points
# ------------------------------
# We'll use PCA to reduce to min(10, n_anchors-1) dimensions
n_components = min(10, len(anchors_A)-1)
pca_A = PCA(n_components=n_components)
pca_A.fit(X_anchors_A)
X_anchors_A_low = pca_A.transform(X_anchors_A)

pca_B = PCA(n_components=n_components)
pca_B.fit(X_anchors_B)
X_anchors_B_low = pca_B.transform(X_anchors_B)

# ------------------------------
# 5. Train Gaussian Process for each chemoprint dimension (separate GP per device)
# ------------------------------
def train_gps(X_low, y_chemoprint):
    """Return list of GPs (one per dimension)."""
    gps = []
    for d in range(29):
        # Kernel: RBF + white noise (small noise to help stability)
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X_low, y_chemoprint[:, d])
        gps.append(gp)
    return gps

print("Training GPs for device A...")
gps_A = train_gps(X_anchors_A_low, y_anchors_A)
print("Training GPs for device B...")
gps_B = train_gps(X_anchors_B_low, y_anchors_B)

def predict_gps(gps, pca, X):
    """Predict 29‑dim chemoprint for array X."""
    X_low = pca.transform(X)
    pred = np.zeros((len(X), 29))
    for d, gp in enumerate(gps):
        pred[:, d] = gp.predict(X_low).flatten()
    return pred

# ------------------------------
# 6. Generate test mixtures (unseen)
# ------------------------------
n_test_mixtures = 200
test_mixtures_A = []
test_mixtures_B = []
true_test_cps = []

# Generate a set of mixture compositions (keep them same for both devices)
mixture_compositions = []
for _ in range(n_test_mixtures):
    n_comp = np.random.randint(2, 4)
    gases = np.random.choice(range(1, 7), n_comp, replace=False)
    weights = np.random.rand(n_comp)
    weights /= weights.sum()
    mixture_compositions.append((gases, weights))
    true_cp = np.zeros(29)
    for gas, w in zip(gases, weights):
        true_cp += w * chemoprint_by_gas[gas]
    true_test_cps.append(true_cp)

# For each composition, generate one sample for each device
for gases, weights in mixture_compositions:
    sensor_A = get_mixture_reading('A', gases, weights, X_A, y_gas_A)
    sensor_B = get_mixture_reading('B', gases, weights, X_B, y_gas_B)
    test_mixtures_A.append(sensor_A)
    test_mixtures_B.append(sensor_B)

test_mixtures_A = np.array(test_mixtures_A)
test_mixtures_B = np.array(test_mixtures_B)
true_test_cps = np.array(true_test_cps)

# ------------------------------
# 7. Predict chemoprints for test mixtures
# ------------------------------
pred_A = predict_gps(gps_A, pca_A, test_mixtures_A)
pred_B = predict_gps(gps_B, pca_B, test_mixtures_B)

# ------------------------------
# 8. Evaluate agreement between devices
# ------------------------------
correlations = [np.corrcoef(pred_A[i], pred_B[i])[0,1] for i in range(n_test_mixtures)]
avg_corr = np.mean(correlations)
print(f"\nAverage per‑mixture correlation (Devices A vs B): {avg_corr:.4f}")

r2_overall = r2_score(pred_B.flatten(), pred_A.flatten())
print(f"Overall R² (Device B vs A): {r2_overall:.4f}")

mse = np.mean((pred_A - pred_B)**2)
print(f"Mean squared error: {mse:.4f}")

# ------------------------------
# 9. Also evaluate how well each device predicts true chemoprint of test mixtures
#    (only possible because we know the composition – but this is a sanity check)
# ------------------------------
print("\nSanity check: Agreement with true chemoprint (from composition)")
r2_A_true = r2_score(pred_A.flatten(), true_test_cps.flatten())
r2_B_true = r2_score(pred_B.flatten(), true_test_cps.flatten())
print(f"Device A vs true: R² = {r2_A_true:.4f}")
print(f"Device B vs true: R² = {r2_B_true:.4f}")

# ------------------------------
# 10. Additional: Evaluate GP generalization to pure gases (using mean readings)
# ------------------------------
print("\nGP generalization to pure gases (using mean sensor readings of each gas):")
for gas in range(1, 7):
    # Sensor mean for device A
    sensor_A = gas_mean_reading(X_A, y_gas_A, gas).reshape(1, -1)
    pred_A_gas = predict_gps(gps_A, pca_A, sensor_A)[0]
    true_cp = chemoprint_by_gas[gas]
    err_A = np.mean((pred_A_gas - true_cp)**2)
    print(f"Gas {gas} (Device A) MSE: {err_A:.4f}")
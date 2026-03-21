import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from chemoprint import chemoprint_from_smiles

data_dir = "gas+sensor+array+drift+dataset/Dataset"

# Map gas labels to SMILES (these are the 6 pure compounds in the dataset)
gas_to_smiles = {
    1: "CCO",           # ethanol
    2: "C=C",           # ethylene
    3: "N",             # ammonia
    4: "CC=O",          # acetaldehyde
    5: "CC(=O)C",       # acetone
    6: "Cc1ccccc1"      # toluene
}

# Compute chemoprints for each gas (29‑dimensional)
chemoprint_by_gas = {}
for label, smiles in gas_to_smiles.items():
    cp = chemoprint_from_smiles(smiles)
    if cp is None:
        raise ValueError(f"Invalid SMILES for gas {label}: {smiles}")
    chemoprint_by_gas[label] = cp

# Load all samples
X_list = []
y_list = []   # will store the chemoprint target (29‑dim) for each sample

for filepath in glob(os.path.join(data_dir, "*.dat")):
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            label = int(parts[0])
            # Extract 128 feature values in order
            features = []
            for token in parts[1:]:
                _, val = token.split(':')
                features.append(float(val))
            if len(features) != 128:
                continue   # skip malformed lines
            X_list.append(features)
            y_list.append(chemoprint_by_gas[label])

X = np.array(X_list)          # shape (n_samples, 128)
y = np.array(y_list)          # shape (n_samples, 29)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None   # no need to stratify by gas if we have enough samples
)

# Random Forest regressor – can handle many features well
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate
r2_per_dim = r2_score(y_test, y_pred, multioutput='raw_values')
print("R² per chemoprint dimension:")
print(r2_per_dim)
print(f"Average R² (variance‑weighted): {r2_score(y_test, y_pred, multioutput='variance_weighted'):.4f}")
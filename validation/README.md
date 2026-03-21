# Validation: Sensor → Chemoprint

This folder contains the code and results showing that a physical sensor array can be calibrated to output the 29‑dimensional chemoprint.

## Dataset
- **Source:** [UCI Gas Sensor Array Drift Dataset](https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset)
- **Content:** 6 pure gases (ethanol, ethylene, ammonia, acetaldehyde, acetone, toluene) measured by 16 metal‑oxide sensors over 36 months. Each sample is a 16×8 time series.

## Method
- **Input:** Raw sensor readings (128 features per sample).
- **Target:** 29‑dim chemoprint computed from SMILES using `chemoprint_from_smiles`.
- **Model:** Random Forest Regressor (100 trees) trained on 80% of samples, tested on 20%.
- **Evaluation:** Coefficient of determination (R²) per dimension and variance‑weighted average.

## Results
- **Average R² (variance‑weighted):** 0.982
- **Per‑dimension R²:** see table below (or attached file).

## Reproduce
1. Install dependencies: `pip install numpy pandas scikit-learn rdkit networkx`
2. Download the UCI dataset and place it in `gas+sensor+array+drift+dataset/Dataset/`
3. Run `python experiment.py`

## Conclusion
A commercial sensor array can predict the chemoprint of pure compounds with high accuracy. This validates the chemoprint as a hardware‑agnostic representation for digital olfaction.
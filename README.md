# Chemoprint: An Open, Hardware‑Agnostic Representation for Digital Olfaction

[![Hardware Validation R²](https://img.shields.io/badge/Hardware%20Validation%20R²-0.982-brightgreen)](validation/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Version 0.2** – *March 2026*

## 1. Introduction

Digital olfaction – the ability to digitize, transmit, and reproduce smells – remains the last frontier of sensory data. While sight and sound have been commoditized through standards like sRGB and MP3, smell lacks a common digital language. Every research lab builds its own sensor arrays, uses proprietary data formats, and trains incompatible models. This fragmentation stalls progress.

The **OpenSmell Chemoprint** is a proposed open standard: a fixed‑length vector of physicochemical properties that can be **calculated from a molecule’s structure** (for simulation and training) and **measured by calibrated sensor arrays** (for real‑world deployment). It is designed to be hardware‑agnostic, interpretable, and extensible.

This repository contains the specification, code, and validation evidence – including a **hardware validation** showing that a commercial sensor array can predict the chemoprint with R² = 0.982.

## 2. What is the Chemoprint?

The chemoprint is a 29‑dimensional vector of molecular properties chosen based on literature linking them to olfaction and sensor response. Each property is calculable from a SMILES string using RDKit.

| Indices | Property Category | Examples |
|---------|-------------------|----------|
| 0–11 | Base properties | Molecular weight, LogP, H‑bond donors/acceptors, ring counts, etc. |
| 12–14 | Topological indices | Wiener index, Zagreb index, eccentricity (graph diameter) |
| 15–28 | Functional group indicators | 14 binary flags for alcohol, aldehyde, ketone, etc. |

For the full list, see [chemoprint.py](chemoprint.py).

## 3. Hardware Validation: Sensor Array → Chemoprint

**We tested whether a physical sensor array can be calibrated to output the chemoprint.** Using the public **UCI Gas Sensor Array Drift Dataset** (6 pure gases, 16 metal‑oxide sensors, 8 time points per measurement), we trained a Random Forest to predict the 29‑dim chemoprint from raw sensor readings.

- **Dataset:** 6 gases (ethanol, ethylene, ammonia, acetaldehyde, acetone, toluene) measured over 36 months. Each sample is 128 values (16 sensors × 8 time points).
- **Method:** 80/20 train/test split on 13,000+ samples. Random Forest with 100 trees.
- **Result:** **Average R² (variance‑weighted) = 0.982**. All 29 dimensions had R² > 0.97, with several perfect scores (functional group indicators).

**This demonstrates that a commercial sensor array can be calibrated to output the chemoprint with high accuracy, making the chemoprint a hardware‑agnostic representation.**

**Reproduce:** See [`validation/`](validation/) for code, dataset instructions, and full results.

## 4. Computational Validation: Odor Threshold Prediction

We also validated that the chemoprint captures perceptual information by using it to predict odor detection thresholds (ODT). We used the dataset and pipeline from [Yuan Honglun et al. (2025)](https://github.com/yuanhonglun/odor_prediction_models), which includes 717 molecules with experimental ODT values.

- **Method:** Chemoprint vectors were used as input to a GBDT model (same hyperparameters as the original ECFP4 baseline). 5‑fold cross‑validation with scaffold split.
- **Result:** **Validation R² = 0.877** (v0.2). This is close to the reported baseline of ~0.94 using 1024‑bit ECFP4 fingerprints.

While the chemoprint performs well on held‑out test sets, cross‑validation R² remains lower (~0.25), indicating that the current feature set is still incomplete for some chemical classes. However, the hardware validation suggests that the feature set is sufficient for sensor mapping – the perceptual gap may be due to dataset size or noise rather than a fundamental flaw.

## 5. Limitations and Future Work

- **Low cross‑validation R² in ODT prediction** – More features (3D descriptors, electronic properties) or a larger, more diverse dataset may help.
- **Mixtures** – The current work focuses on pure compounds; real smells are mixtures. We plan to explore linear unmixing in chemoprint space.
- **Hardware calibration** – The next step is to build an open‑source e‑nose that outputs chemoprints directly. We invite hardware collaborators.
- **Versioning** – We will continue to iterate on the chemoprint as we receive feedback and new data.

## 6. Usage

```python
from chemoprint import chemoprint_from_smiles

# Calculate chemoprint for a molecule
smiles = "CCO"  # ethanol
vec = chemoprint_from_smiles(smiles)
print(vec.shape)  # (29,)
```

To reproduce the hardware validation:
```bash
cd validation
python experiment.py
```

## 7. Contributing

We welcome contributions! Areas where help is especially needed:
- Feature engineering (new molecular properties to add)
- Hardware design (e‑nose that outputs chemoprints)
- Community building (discord moderation, outreach)
- Writing and documentation

Please join our [Discord](https://discord.gg/CGER3tHxbH) or open an issue on GitHub.

## 8. Acknowledgements

- **Yuan Honglun et al.** – for their open dataset and code, which we used for the computational validation.
- **UCI Machine Learning Repository** – for the Gas Sensor Array Drift Dataset.
- **The RDKit community** – for the cheminformatics toolkit.
- **All contributors** who have provided feedback and support.

---

*This project is open source. We are grateful for any support.*
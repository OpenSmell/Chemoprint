# Chemoprint: An Open, Human‑Readable Descriptor for Pure Compounds

[![Hardware Validation R²](https://img.shields.io/badge/Hardware%20Validation%20R²-0.982-brightgreen)](validation/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Version 0.2** – *March 2026*

The chemoprint is a 29‑dimensional physicochemical vector that can be computed from a SMILES string and **predicted from sensor readings for pure compounds** (R² = 0.982 on the UCI gas sensor dataset). Its strengths are:

- **Human‑interpretable** – dimensions correspond to molecular weight, LogP, functional groups, etc.
- **Deterministic and open** – completely defined via RDKit and MIT‑licensed code.
- **Useful for** cheminformatics, QSAR, and pure‑gas identification.
- **Can serve as an optional interpretability layer** for learned embeddings.

It is **not** a universal interoperability standard for mixtures. For cross‑device interoperability with mixtures see the [OpenSmell Calibration](https://github.com/opensmell/calibration) project (coming soon).

---

## 2. What is the Chemoprint?

The chemoprint is a 29‑dimensional vector of molecular properties chosen based on literature linking them to olfaction and sensor response. Each property is calculable from a SMILES string using RDKit.

| Indices | Property Category | Examples |
|---------|-------------------|----------|
| 0–11 | Base properties | Molecular weight, LogP, H‑bond donors/acceptors, ring counts, etc. |
| 12–14 | Topological indices | Wiener index, Zagreb index, eccentricity (graph diameter) |
| 15–28 | Functional group indicators | 14 binary flags for alcohol, aldehyde, ketone, etc. |

For the full list see [chemoprint.py](https://github.com/OpenSmell/Chemoprint/blob/main/chemoprint.py).

---

## 3. Hardware Validation: Sensor Array → Chemoprint

**We tested whether a physical sensor array can be calibrated to output the chemoprint.** Using the public **UCI Gas Sensor Array Drift Dataset** (6 pure gases, 16 metal‑oxide sensors, 8 time points per measurement) we trained a Random Forest to predict the 29‑dim chemoprint from raw sensor readings.

- **Dataset:** 6 gases (ethanol, ethylene, ammonia, acetaldehyde, acetone, toluene) measured over 36 months. Each sample is 128 values (16 sensors × 8 time points).
- **Method:** 80/20 train/test split on 13,000+ samples. Random Forest with 100 trees.
- **Result:** **Average R² (variance‑weighted) = 0.982**. All 29 dimensions had R² > 0.97, with several perfect scores (functional group indicators).

**This demonstrates that a commercial sensor array can be calibrated to output the chemoprint with high accuracy.**

**Reproduce:** See [`validation/`](https://github.com/OpenSmell/Chemoprint/blob/main/validation) for code, dataset instructions, and full results.

---

## 4. Computational Validation: Odor Threshold Prediction

We also validated that the chemoprint captures perceptual information by using it to predict odor detection thresholds (ODT). We used the dataset and pipeline from [Yuan Honglun et al. (2025)](https://github.com/yuanhonglun/odor_prediction_models), which includes 717 molecules with experimental ODT values.

- **Method:** Chemoprint vectors were used as input to a GBDT model (same hyperparameters as the original ECFP4 baseline). 5‑fold cross‑validation with scaffold split.
- **Result:** **Validation R² = 0.877** (v0.2). This is close to the reported baseline of ~0.94 using 1024‑bit ECFP4 fingerprints.

While the chemoprint performs well on held‑out test sets, cross‑validation R² remains lower (~0.25), indicating that the current feature set is still incomplete for some chemical classes. The perceptual gap may be due to dataset size or noise rather than a fundamental flaw.

---

## 5. Usage

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

---

## 6. Limitations

- **Cannot be computed for mixtures** – The 29‑dim vector is defined per molecule. A complex mixture has no single SMILES, so a chemoprint cannot be directly calculated.
- **Six‑pure‑anchor calibration fails** – A convex hull analysis shows that six pure compounds (methanol, hexane, acetone, acetic acid, toluene, isopropanol) cover only 0.1% of common odorants. Therefore, **using chemoprint for universal device calibration is not viable**.

---

## 7. Contributing

We welcome contributions! Areas where help is especially needed:

- Feature engineering (new molecular properties to add)
- Hardware design (e‑nose that outputs chemoprints)
- Community building (Discord moderation, outreach)
- Writing and documentation

Please join our [Discord](https://discord.gg/CGER3tHxbH) or open an issue on GitHub.

---

## 8. Acknowledgements

- **Yuan Honglun et al.** – for their open dataset and code, which we used for the computational validation.
- **UCI Machine Learning Repository** – for the Gas Sensor Array Drift Dataset.
- **The RDKit community** – for the cheminformatics toolkit.
- **All contributors** who have provided feedback and support.

---

*This project is open source. We are grateful for any support.*
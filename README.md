# Chemoprint: An Open, Human‑Readable Descriptor for Pure Compounds

[![Hardware Validation R²](https://img.shields.io/badge/Hardware%20Validation%20R²-0.982-brightgreen)](validation/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Version 1.0** – *May 2026*

The chemoprint is a 29‑dimensional physicochemical vector that can be computed from a SMILES string and **predicted from precomputed sensor features for pure compounds** (R² = 0.982 on the UCI gas sensor dataset, using 128-dim feature vectors, not raw sensor readings). Its strengths are:

- **Human‑interpretable** – dimensions correspond to molecular weight, LogP, functional groups, etc.
- **Deterministic and open** – completely defined via RDKit and MIT‑licensed code.
- **Useful for** cheminformatics, QSAR, and pure‑gas identification.
- **Can serve as an optional interpretability layer** for learned embeddings.

It is **not** an interoperability standard for mixtures. For cross‑device work with mixtures see the [encoder](https://github.com/opensmell/encoder) and [session-invariance](https://github.com/opensmell/session-invariance) repos.

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

**This demonstrates that 128-dim precomputed features from a sensor array contain enough information to predict the chemoprint. Direct calibration of raw sensor readings from a commercial array has not yet been tested.**

**Reproduce:** See [`validation/`](https://github.com/OpenSmell/Chemoprint/blob/main/validation) for code, dataset instructions, and full results.

---

## 4. Computational Validation: Odor Threshold Prediction

We also validated that the chemoprint captures perceptual information by using it to predict odor detection thresholds (ODT). We used the dataset and pipeline from [Yuan Honglun et al. (2025)](https://github.com/yuanhonglun/odor_prediction_models), which includes 717 molecules with experimental ODT values.

- **Method:** Chemoprint vectors were used as input to a GBDT model (same hyperparameters as the original ECFP4 baseline). 5‑fold cross‑validation with scaffold split.
- **Result:** **Validation R² = 0.877** (v1.0, held-out test set). This is close to the reported baseline of ~0.94 using 1024‑bit ECFP4 fingerprints.

While the chemoprint performs well on held‑out test sets, cross‑validation R² (5-fold) is 0.403 ± 0.052 on the same dataset (see [Chemoprint-Apps QSAR benchmark](https://github.com/opensmell/chemoprint-apps)). These are different metrics: held-out test set R² measures best-case generalisation, while cross-validated R² measures robustness across all data splits. Both are reported transparently.

---

## 5. Usage

```python
from chemoprint import chemoprint_from_smiles, chemoprint_from_mixture

# Calculate chemoprint for a pure molecule
smiles = "CCO"  # ethanol
vec = chemoprint_from_smiles(smiles)
print(vec.shape)  # (29,)

# Calculate chemoprint for a mixture with known VOCs
# (e.g., coffee's volatile compounds from FooDB)
smiles_list = [
    "CCO",         # ethanol
    "CC(=O)C",     # acetone
    "CC=O",        # acetaldehyde
]
concentrations = [0.5, 0.3, 0.2]  # optional: relative concentrations
mixture_vec = chemoprint_from_mixture(smiles_list, concentrations)
print(mixture_vec.shape)  # (29,)
```

For known food mixtures, if you have the FooDB chemoprints CSV:
```python
from chemoprint import chemoprint_from_foodb

vec = chemoprint_from_foodb("coffee", "path/to/foodb_chemoprints.csv")
```

To reproduce the hardware validation:
```bash
cd validation
python experiment.py
```

---

## 6. Limitations

### What the chemoprint library can do

| Input | Method | Accuracy | Limitations |
|-------|--------|----------|-------------|
| **Pure compound** with SMILES | `chemoprint_from_smiles(smiles)` | Near-perfect (R² ≈ 0.98 on UCI pure gases) | None. If the SMILES is valid, the chemoprint is exact. |
| **Simple mixture** with known VOCs + concentrations | `chemoprint_from_mixture([smiles_list], [concentrations])` | Good, if VOC list is complete | Requires knowing what's in the mixture. FooDB provides this for foods. |
| **Known food mixture** via FooDB lookup | `chemoprint_from_foodb("coffee", "foodb_chemoprints.csv")` | Good for the 44 covered substances | Covered mixtures are those in FooDB with GC-MS data. Not all foods are covered. |
| **Complex mixture** (breath, environmental, unknown sample) | Not computable from structure alone | N/A | You'd need GC-MS to know what's in it. This is the gap the session-invariant encoder aims to fill. |

### When to use the encoder instead

For unknown or complex mixtures (e.g., breath, environmental samples), the chemoprint cannot be computed from structure alone. Use the OpenSmell encoder (sensor → latent → chemoprint) instead. This is under development at `opensmell/encoder`.

### Calibration limitation

Six-pure-anchor calibration fails — a convex hull analysis ([`validation/convex_hull.py`](validation/convex_hull.py)) shows that six pure compounds (methanol, hexane, acetone, acetic acid, toluene, isopropanol) cover only 0.1% of 4,565 common odorants from the GoodScents database. Therefore, **using chemoprint for device calibration is not viable**.

---

## 7. Contributing

We welcome contributions! Areas where help is especially needed:

- Feature engineering (new molecular properties to add)
- Hardware design (e‑nose that outputs chemoprints)
- Community building (Discord moderation, outreach)
- Writing and documentation

Please join our [Discord](https://discord.gg/CGER3tHxbH) or open an issue on GitHub.

---

## Limitations & Future Dimensions

The v1.0 Chemoprint encodes 29 structural physicochemical properties.
It does **not** yet capture:

| Missing property | Why it matters | How to add it |
|-----------------|----------------|---------------|
| **Chirality** | Mirror-image molecules (e.g., R- vs S-carvone) can smell different (spearmint vs caraway) | 30th dimension: chirality index, computable from SMILES using RDKit |
| **Vibrational modes** | If vibrational theory of olfaction is correct, dominant IR absorption frequencies correlate with perceived odor | Future dimensions: compute IR spectra for a reference set; encode peak frequencies |
| **Isotope effects** | Deuterated compounds smell different to some animals | Dimension: isotope ratio or mass shift index |
| **Conformational flexibility** | Flexing molecules may bind multiple receptors | Dimension: rotatable bonds weighted by ring constraints |
| **Mixture interaction terms** | Synergy and suppression between molecules | Future v2.0: learned interaction coefficients from community data |

Each addition is **backward-compatible**. Apps written for v1.0 ignore extra dimensions.

### How to contribute a new dimension

1. Propose a dimension with a computable definition and a hypothesis.
2. Validate on at least one published odor dataset (Pyrfume, GoodScents, Dravnieks).
3. Submit a PR. If it improves predictive power on the QSAR benchmark, it's accepted.
---

## 8. Acknowledgements

- **Yuan Honglun et al.** – for their open dataset and code, which we used for the computational validation.
- **UCI Machine Learning Repository** – for the Gas Sensor Array Drift Dataset.
- **The RDKit community** – for the cheminformatics toolkit.
- **All contributors** who have provided feedback and support.

---

*This project is open source. We are grateful for any support.*
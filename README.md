# Chemoprint: An Open, Hardware‑Agnostic Representation for Digital Olfaction

**Version 0.1**  
*March 2026*

## 1. Introduction

Digital olfaction – the ability to digitize, transmit, and reproduce smells – remains the last frontier of sensory data. While sight and sound have been commoditized through standards like sRGB and MP3, smell lacks a common digital language. Every research lab builds its own sensor arrays, usesp proprietary data formats, and trains incompatible models. This fragmentation stalls progress.

The **OpenSmell Chemoprint** is a proposed open standard: a fixed‑length vector of physicochemical properties that can be **calculated from a molecule’s structure** (for simulation and training) and **measured by calibrated sensor arrays** (for real‑world deployment). It is designed to be hardware‑agnostic, interpretable, and extensible.

This report documents the initial design, experimental validation, and current limitations of chemoprint v0.1 and v0.2, based on publicly available odor threshold data. We invite feedback, critique, and collaboration from the community.

## 2. Background and Related Work

Our work builds on several key contributions:

- **Yuan Honglun  et al. (2025)** – Provided a high‑accuracy (R²=0.94) odor threshold prediction model using ECFP fingerprints and GBDT, along with an open‑source training pipeline and dataset ([GitHub](https://github.com/yuanhonglun/odor_prediction_models), [Zenodo](https://zenodo.org/records/17559514)). We used their code and data as the validation platform for the chemoprint.
- **Haddad et al. (2010)** – Demonstrated that a low‑dimensional physicochemical space can capture olfactory differences.
- **Ihara et al. (2013)** – Linked receptor activation to molecular properties.
- **NASA JPL QSAR work** – Showed that sensor responses can be predicted from molecular descriptors.

The chemoprint is **not** a replacement for molecular fingerprints (ECFP, MACCS) in computational chemistry; those remain essential for similarity searching and virtual screening. Instead, the chemoprint is a **bridge** – a representation that can be both computed and physically measured, enabling interoperability between hardware and software.

## 3. Chemoprint Design (v0.1 and v0.2)

The chemoprint is a vector of **physicochemical properties** chosen based on literature linking them to olfaction and sensor response. Each property is calculable from a SMILES string using RDKit, ensuring reproducibility.

### v0.1 – 26 Dimensions

| Index | Property | Rationale | Reference |
|-------|----------|-----------|-----------|
| 0 | Molecular weight | Affects diffusion, volatility | |
| 1 | Heavy atom count | Rough size proxy | |
| 2 | Rotatable bonds | Flexibility | |
| 3 | Ring count | Cyclic structures | |
| 4 | Aromatic ring count | Planarity, electronics | |
| 5 | Fraction Csp³ | Saturation | |
| 6 | LogP | Hydrophobicity | |
| 7 | TPSA | Polarity | |
| 8 | H‑bond donors | Receptor interactions | |
| 9 | H‑bond acceptors | Receptor interactions | |
| 10 | Net charge | Ionic interactions | |
| 11 | Heteroatom count | Presence of O, N, S, etc. | |
| 12–25 | Functional group indicators (14 binary) | Alcohol, aldehyde, ketone, acid, ester, ether, amines, nitro, thiol, sulfide, aromatic N, halogen | SMARTS patterns |

### v0.2 – Added Topological Indices (29 Dimensions)

To better capture molecular shape and branching, we added three graph‑based descriptors:

| Index | Property | Calculation |
|-------|----------|-------------|
| 12 | Wiener index | Sum of shortest path distances |
| 13 | Zagreb index (M1) | Sum of squares of vertex degrees |
| 14 | Eccentricity | Graph diameter (longest shortest path) |

These were computed using RDKit and NetworkX. The remaining functional group indicators shifted to indices 15–28.

## 4. Experimental Validation

We used the **threshold_data.csv** from Yuan’s repository (717 molecules with experimental odor detection thresholds). The target was `-log10(threshold)` (higher = more detectable).

For each molecule, we generated chemoprint vectors and fed them into Yuan’s **GBDT training pipeline** (5‑fold cross‑validation with scaffold split, hyperparameter tuning). We compared performance to the original ECFP4 baseline (reported R² ≈ 0.94).

### 4.1 Results

| Model | Features | CV Mean R² | Validation R² | Notes |
|-------|----------|------------|---------------|-------|
| GBDT (Yuan) | ECFP4 (1024 bits) | ~0.90 | ~0.94 | Reported baseline |
| GBDT (ours) | Chemoprint v0.1 (26) | 0.224 ± 0.261 | **0.883** | Large CV‑val gap |
| Random Forest | Chemoprint v0.1 | 0.271 ± 0.193 | 0.773 | Slightly better CV |
| MLP | Chemoprint v0.1 | 0.155 ± 0.316 | 0.336 | Poor |
| GBDT | Chemoprint v0.2 (29) | 0.252 ± 0.254 | 0.877 | Mid‑range improved |

**Band‑wise evaluation (v0.2 GBDT)**:

| Tercile | R² | RMSE |
|---------|-----|------|
| Low (easy to detect) | 0.475 | 0.812 |
| Mid | -0.246 | 0.468 |
| High (hard to detect) | 0.519 | 0.541 |

### 4.2 Discussion

- The chemoprint captures **significant signal** – validation R² up to 0.88, close to the ECFP baseline.
- However, **cross‑validation R² remains low** (~0.25), indicating that the model does not generalize well across different subsets of molecules. This suggests that the current feature set is **incomplete** for certain chemical classes, especially those in the mid‑range of detectability.
- Adding topological indices (v0.2) **improved mid‑range performance** (from -0.71 to -0.25), but overall CV only rose slightly. More work is needed.
- The discrepancy between CV and validation may also reflect the **scaffold split** – if the validation set contains molecules that are structurally easier to predict, the high validation score could be misleading. We are investigating this further.

Despite the instability, we believe the chemoprint concept is sound. The fact that a simple 29‑dimensional vector achieves near‑state‑of‑the‑art validation performance is encouraging and suggests that a hardware‑friendly representation is feasible.

## 5. Limitations and Open Questions

- **Low CV R²** – The current chemoprint does not capture all relevant features; additional properties (e.g., 3D conformation, electronic descriptors) may be needed.
- **Data quality** – The dataset, while large, may contain noise and inconsistencies. More diverse data could help.
- **Scalability to mixtures** – This work focused on pure compounds; real smells are mixtures.
- **Hardware calibration** – Mapping real sensor outputs to chemoprints remains to be solved (though IEEE 1451.4 and transfer learning provide a path).

## 6. Next Steps and Call for Collaboration

We are releasing chemoprint as open source (code, specification, and this report) to invite feedback. Immediate goals:

1. **Feature importance analysis** – Identify which properties drive predictions (requires fixing the model export issue).
2. **Chemoprint v0.3** – Incorporate 3D descriptors (e.g., molecular volume, polarizability) or vibrational pseudo‑spectra (EVA/PD‑EVA) to better capture shape and electronic effects.
3. **Sensor mapping** – Research real sensors that can measure the top‑ranked properties (e.g., LogP via polymer‑coated QCM, H‑bonding via functionalized metal oxides). We will publish a sensor wishlist.
4. **Community engagement** – Reach out to experts in olfaction, cheminformatics, and sensor design for guidance and collaboration.
5. **Ethics and credit** – We explicitly acknowledge and thank Yuan et al. for their open dataset and code. All modifications are clearly marked, and we encourage others to build upon their work.

## 7. How to Contribute

- **GitHub repository**: [opensmell/chemoprint](https://github.com/opensmell/chemoprint) (placeholder)
- **Discussion**: Join our [Discord](https://discord.gg/CGER3tHxbH) or open an issue on GitHub.
- **Data**: The original dataset is available from Yuan’s Zenodo. Our generated chemoprint vectors and training logs are in the `outputs/` folder of this repo.

We welcome critique, suggestions, and especially help with sensor calibration and feature engineering.

## Requirements
- Python 3.8+
- RDKit (`conda install -c conda-forge rdkit`)
- NetworkX (`pip install networkx`)

## Usage
```python
from chemoprint import chemoprint_from_smiles

smiles = "CCO"  # ethanol
vec = chemoprint_from_smiles(smiles)
print(vec)
```

---

*This report and the chemoprint standard are works in progress. We are humbled by the complexity of the problem and grateful for the foundational work of the research community. All errors and omissions are our own.*
#!/usr/bin/env python3
"""
OpenSmell Chemoprint Generator v0.1
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdmolops
import networkx as nx

# ----------------------------------------------------------------------
def _has_substruct(mol, smarts):
    pattern = Chem.MolFromSmarts(smarts)
    return 1 if pattern and mol.HasSubstructMatch(pattern) else 0

def _functional_group_features(mol):
    groups = [
        ("alcohol",          "[OH]"),
        ("aldehyde",         "[CX3H1](=O)[#6]"),
        ("ketone",           "[#6][CX3](=O)[#6]"),
        ("carboxylic_acid",  "[CX3](=O)[OX2H]"),
        ("ester",            "[#6][CX3](=O)[OX2][#6]"),
        ("ether",            "[OD2]([#6])[#6]"),
        ("amine_primary",    "[NX3;H2]"),
        ("amine_secondary",  "[NX3;H1]"),
        ("amine_tertiary",   "[NX3;H0]"),
        ("nitro",            "[NX3](=O)=O"),
        ("thiol",            "[SH]"),
        ("sulfide",          "[SX2]"),
        ("aromatic_nitrogen","n"),
        ("halogen",          "[F,Cl,Br,I]"),
    ]
    return [_has_substruct(mol, smarts) for _, smarts in groups]

def _fraction_csp3(mol):
    carbon_sp3 = 0
    total_carbons = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            total_carbons += 1
            if atom.GetHybridization() == Chem.HybridizationType.SP3:
                carbon_sp3 += 1
    if total_carbons == 0:
        return 0.0
    return carbon_sp3 / total_carbons

def _net_charge(mol):
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

# ---------- Topological indices ----------
def _wiener_index(mol):
    """Wiener index = sum of shortest path distances between all atom pairs."""
    dm = rdmolops.GetDistanceMatrix(mol)
    # Sum of upper triangle (excluding diagonal)
    return float(np.sum(np.triu(dm, k=1)))

def _zagreb_index(mol):
    """
    Zagreb index (M1) = sum of squares of vertex degrees.
    Captures branching and is easy to compute.
    """
    zagreb = 0
    for atom in mol.GetAtoms():
        deg = atom.GetDegree()
        zagreb += deg * deg
    return float(zagreb)

def _eccentricity(mol):
    """Maximum distance from any atom to any other (graph diameter)."""
    if mol.GetNumBonds() == 0:
        # Single atom or isolated atoms with no bonds – eccentricity is 0
        return 0.0
    G = nx.Graph()
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    if nx.is_connected(G):
        return float(max(nx.eccentricity(G).values()))
    else:
        # For disconnected molecules, compute largest component's eccentricity
        components = list(nx.connected_components(G))
        max_ecc = 0.0
        for comp in components:
            subgraph = G.subgraph(comp)
            if len(comp) > 1:
                ecc = max(nx.eccentricity(subgraph).values())
            else:
                ecc = 0.0  # isolated atom
            if ecc > max_ecc:
                max_ecc = ecc
        return max_ecc

# ----------------------------------------------------------------------
def chemoprint_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Base properties (indices 0‑11)
    props = [
        Descriptors.MolWt(mol),                             # 0
        Descriptors.HeavyAtomCount(mol),                    # 1
        Descriptors.NumRotatableBonds(mol),                 # 2
        rdMolDescriptors.CalcNumRings(mol),                 # 3
        rdMolDescriptors.CalcNumAromaticRings(mol),         # 4
        _fraction_csp3(mol),                                 # 5
        Descriptors.MolLogP(mol),                            # 6
        Descriptors.TPSA(mol),                               # 7
        Descriptors.NumHDonors(mol),                         # 8
        Descriptors.NumHAcceptors(mol),                      # 9
        _net_charge(mol),                                    # 10
        Descriptors.NumHeteroatoms(mol),                     # 11
    ]

    # Topological indices (indices 12‑14)
    topo = [
        _wiener_index(mol),
        _zagreb_index(mol),
        _eccentricity(mol),
    ]

    # Functional group indicators (indices 15‑28)
    fg_features = _functional_group_features(mol)

    chemoprint = np.array(props + topo + fg_features, dtype=np.float32)
    return chemoprint

def chemoprint_from_mixture(smiles_list, concentrations=None):
    if len(smiles_list) == 0:
        return None
    vectors = []
    weights = []
    for i, smi in enumerate(smiles_list):
        vec = chemoprint_from_smiles(smi)
        if vec is not None:
            vectors.append(vec)
            if concentrations is not None and i < len(concentrations):
                weights.append(concentrations[i])
            else:
                weights.append(1.0)
    if len(vectors) == 0:
        return None
    vectors = np.array(vectors)
    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()
    return (vectors.T * weights).sum(axis=1)

def chemoprint_from_foodb(mixture_name, foobd_csv="foodb_chemoprints.csv"):
    if not os.path.exists(foobd_csv):
        return None
    df = pd.read_csv(foobd_csv, index_col=0)
    if mixture_name in df.index:
        return df.loc[mixture_name].values.astype(np.float32)
    return None

# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_molecules = {
        "ethanol": "CCO",
        "vanillin": "COc1ccc(C=O)c(O)c1",
        "toluene": "Cc1ccccc1",
        "water": "O",
        "acetic acid": "CC(=O)O",
    }

    print("\n" + "="*60)
    print("Chemoprint vectors:")
    print("="*60)
    for name, smi in test_molecules.items():
        vec = chemoprint_from_smiles(smi)
        if vec is not None:
            print(f"\n{name} (SMILES: {smi})")
            print("Chemoprint:", np.array2string(vec, precision=2, separator=', '))
        else:
            print(f"\n{name}: invalid SMILES?")
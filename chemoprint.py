#!/usr/bin/env python3
"""
OpenSmell Chemoprint Generator v0.2 — CLI convenience wrapper.
The actual implementation lives in chemoprint/__init__.py (pip package).
"""
import sys
import numpy as np
from chemoprint import chemoprint_from_smiles

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

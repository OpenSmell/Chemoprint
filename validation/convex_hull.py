"""
Convex hull analysis for Chemoprint anchors.
Evaluates coverage of 6 reference compounds against GoodScents odorants.
Downloads GoodScents data automatically via pyrfume if not found.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Import your chemoprint function (adjust path if needed)
from chemoprint import chemoprint_from_smiles

# ------------------------------------------------------------
# 1. Download GoodScents dataset using pyrfume (if missing)
# ------------------------------------------------------------
def download_goodscents(data_dir):
    """Download GoodScents molecules.csv using pyrfume."""
    file_path = data_dir / "molecules.csv"
    if file_path.exists():
        print(f"GoodScents data already exists at {file_path}")
        return file_path
    
    print("Downloading GoodScents molecules from Pyrfume...")
    try:
        import pyrfume
        df = pyrfume.load_data('goodscents/molecules.csv')
        df.to_csv(file_path, index=False)
        print(f"Downloaded {len(df)} molecules to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error downloading GoodScents: {e}")
        raise

# ------------------------------------------------------------
# 2. Load GoodScents CSV and return SMILES list
# ------------------------------------------------------------
def load_goodscents(csv_path):
    """Load GoodScents CSV, return list of SMILES and names."""
    df = pd.read_csv(csv_path)
    # The column with SMILES is 'IsomericSMILES' (as seen in the user's file)
    if 'IsomericSMILES' not in df.columns:
        raise KeyError("Column 'IsomericSMILES' not found in GoodScents CSV.")
    smiles_list = df['IsomericSMILES'].dropna().tolist()
    name_list = df.get('name', [''] * len(smiles_list)).fillna('').tolist()
    return smiles_list, name_list

# ------------------------------------------------------------
# 3. Compute chemoprints for a list of SMILES
# ------------------------------------------------------------
def compute_chemoprints(smiles_list, name_list=None):
    """Return array of chemoprints (29D) and list of valid names."""
    vectors = []
    valid_names = []
    for i, smi in enumerate(smiles_list):
        cp = chemoprint_from_smiles(smi)
        if cp is not None:
            vectors.append(cp)
            if name_list and i < len(name_list):
                valid_names.append(name_list[i])
            else:
                valid_names.append(str(i))
    return np.array(vectors), valid_names

# ------------------------------------------------------------
# 4. Test if a point is inside convex hull of anchors (linear programming)
# ------------------------------------------------------------
def point_in_convex_hull(point, hull_points):
    """
    point: (29,) array
    hull_points: (n_anchors, 29) array
    Returns True if point is inside convex hull, else False.
    """
    n_anchors = hull_points.shape[0]
    # Equality: point = hull_points.T @ c , and sum(c)=1, c>=0
    A_eq = np.vstack([hull_points.T, np.ones((1, n_anchors))])
    b_eq = np.concatenate([point, [1.0]])
    bounds = [(0, None)] * n_anchors
    c = np.zeros(n_anchors)  # dummy objective
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return res.success and np.allclose(res.x.sum(), 1.0, atol=1e-6)

# ------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------
def main():
    # Define data directory and anchors
    data_dir = Path(__file__).resolve().parent / "goodscents"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    anchors = {
        'methanol': 'CO',
        'hexane': 'CCCCCC',
        'acetone': 'CC(=O)C',
        'acetic_acid': 'CC(=O)O',
        'toluene': 'Cc1ccccc1',
        'isopropanol': 'CC(C)O'
    }
    
    # Ensure GoodScents data is present
    csv_path = download_goodscents(data_dir)
    
    # Load GoodScents molecules
    print("Loading GoodScents molecules...")
    smiles_list, name_list = load_goodscents(csv_path)
    print(f"Loaded {len(smiles_list)} molecules with SMILES.")
    
    # Compute chemoprints for anchors
    anchor_smiles = list(anchors.values())
    anchor_names = list(anchors.keys())
    anchor_vecs, anchor_names_valid = compute_chemoprints(anchor_smiles, anchor_names)
    print(f"Anchor vectors shape: {anchor_vecs.shape}")
    
    # Compute chemoprints for GoodScents (limit to first 2000 for speed? keep all)
    print("Computing chemoprints for GoodScents molecules...")
    gs_vecs, gs_names_valid = compute_chemoprints(smiles_list, name_list)
    print(f"GoodScents vectors shape: {gs_vecs.shape} (valid chemoprints)")
    
    # Test each GoodScents molecule
    print("Testing convex hull inclusion...")
    inside_count = 0
    outside_samples = []
    for i, vec in enumerate(gs_vecs):
        if point_in_convex_hull(vec, anchor_vecs):
            inside_count += 1
        else:
            if len(outside_samples) < 10:
                outside_samples.append(gs_names_valid[i])
    total = len(gs_vecs)
    coverage = inside_count / total * 100
    print(f"\nCoverage: {inside_count} out of {total} molecules ({coverage:.1f}%) are inside the convex hull of the six anchors.")
    if outside_samples:
        print("Example outside molecules:", outside_samples[:5])
    
    # Visualise in 2D using PCA
    print("\nGenerating PCA visualisation...")
    all_vectors = np.vstack([anchor_vecs, gs_vecs])
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_vectors)
    anchor_2d = all_2d[:len(anchor_vecs)]
    gs_2d = all_2d[len(anchor_vecs):]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(gs_2d[:,0], gs_2d[:,1], c='lightblue', s=1, alpha=0.5, label='GoodScents')
    plt.scatter(anchor_2d[:,0], anchor_2d[:,1], c='red', s=50, label='Anchors')
    for i, name in enumerate(anchor_names_valid):
        plt.annotate(name, (anchor_2d[i,0], anchor_2d[i,1]), fontsize=8)
    # Draw convex hull of anchors in 2D (for visual reference only)
    if len(anchor_2d) >= 3:
        hull_2d = ConvexHull(anchor_2d)
        for simplex in hull_2d.simplices:
            plt.plot(anchor_2d[simplex, 0], anchor_2d[simplex, 1], 'r-', linewidth=2)
    plt.title('PCA projection of Chemoprint space')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('convex_hull_visual.png', dpi=150)
    plt.show()
    print("Visualisation saved as convex_hull_visual.png")

if __name__ == "__main__":
    main()
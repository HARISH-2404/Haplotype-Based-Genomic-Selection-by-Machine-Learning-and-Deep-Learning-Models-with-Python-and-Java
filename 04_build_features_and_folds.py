#!/usr/bin/env python
"""STEP 4 - Build design matrices (SNP, haplotype-LD, haplotype-fixed) and the shared CV fold file."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from hgs import config as C, haplotypes as H, cv

if __name__ == "__main__":
    geno = pd.read_csv(C.DATA_PROC / "genotypes_qc_imputed.csv")
    lines = pd.read_csv(C.DATA_RAW / "lines.csv").set_index("line_id").loc[geno["line_id"]].reset_index()
    G = geno.drop(columns="line_id").values
    feats = {"SNP": G}
    for name in ("ld", "fixed"):
        X, _ = H.onehot_haplotypes(np.load(C.DATA_PROC / f"hap_ids_{name}.npy"), np.load(C.DATA_PROC / f"hap_nalleles_{name}.npy"))
        feats[f"HAP_{name.upper()}"] = X
    np.savez_compressed(C.DATA_PROC / "features.npz", **feats)
    for k, v in feats.items():
        print(f"{k:10s} shape={v.shape}")
    folds = cv.make_folds(lines, C.N_FOLDS, C.N_REPEATS, C.SEED)
    folds.to_csv(C.DATA_PROC / "cv_folds.csv", index=False)
    print(folds.head())

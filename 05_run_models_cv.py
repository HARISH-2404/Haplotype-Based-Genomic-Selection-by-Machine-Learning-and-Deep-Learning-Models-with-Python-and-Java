#!/usr/bin/env python
"""STEP 5 - Train / cross-validate all models on all feature sets and traits.
Usage: python 05_run_models_cv.py [--quick]   (quick = 1 repeat, subset of models)"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from hgs import config as C, cv, models

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--models", default="")
    a = ap.parse_args()
    feats = dict(np.load(C.DATA_PROC / "features.npz"))
    folds = pd.read_csv(C.DATA_PROC / "cv_folds.csv")
    pheno = pd.read_csv(C.DATA_RAW / "phenotypes.csv").set_index("line_id").loc[folds["line_id"]]
    tbv = pd.read_csv(C.DATA_TRUTH / "true_breeding_values.csv").set_index("line_id").loc[folds["line_id"]]
    zoo = models.get_models(C.SEED)
    if a.models:
        zoo = {k: v for k, v in zoo.items() if k in a.models.split(",")}
    schemes = [c for c in folds.columns if c.startswith("random_rep")] + ["family_out"]
    if a.quick:
        schemes = schemes[:1] + ["family_out"]
    out_rows, t_start = [], time.time()
    for trait, (_, _, h2, _) in C.TRAITS.items():
        y, tb = pheno[trait].values, tbv[trait].values
        for fs_name, X in feats.items():
            for mname, (fam, est) in zoo.items():
                for sc in schemes:
                    res, _ = cv.run_scheme(X, y, tb, folds, sc, est, h2)
                    res.insert(0, "scheme", "family_out" if sc == "family_out" else "random_kfold")
                    res.insert(1, "repeat", int(sc[-1]) if sc.startswith("random") else 1)
                    res.insert(0, "model", mname); res.insert(1, "family", fam)
                    res.insert(0, "features", fs_name); res.insert(0, "trait", trait)
                    out_rows.append(res)
                print(f"[{time.time() - t_start:7.0f}s] {trait:4s} {fs_name:10s} {mname:12s} done", flush=True)
                pd.concat(out_rows).to_csv(C.RES_TAB / "cv_results_python_folds.csv", index=False)
    print("finished")

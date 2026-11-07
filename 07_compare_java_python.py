#!/usr/bin/env python
"""STEP 7 - Cross-check the Java implementation against the Python implementation (same folds)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hgs import config as C, plotting as P

if __name__ == "__main__":
    J = pd.read_csv(C.RES_TAB / "cv_results_java_folds.csv")
    Y = pd.read_csv(C.RES_TAB / "cv_results_python_folds.csv")
    keys = ["trait", "features", "model", "scheme", "repeat", "fold"]
    M = J.merge(Y, on=keys, suffixes=("_java", "_py"))
    M = M[M.model.isin(["RR-BLUP", "MLP"])]
    out = []
    for m, g in M.groupby("model"):
        a = g.groupby(["trait", "features", "scheme"])[["r_phenotype_java", "r_phenotype_py"]].mean()
        out.append(dict(model=m, n_fold_pairs=len(g),
                        fold_level_correlation=np.corrcoef(g.r_phenotype_java, g.r_phenotype_py)[0, 1],
                        mean_abs_diff_r=float((g.r_phenotype_java - g.r_phenotype_py).abs().mean()),
                        mean_r_java=g.r_phenotype_java.mean(), mean_r_python=g.r_phenotype_py.mean()))
    cmp_ = pd.DataFrame(out).round(4)
    cmp_.to_csv(C.RES_TAB / "java_vs_python_agreement.csv", index=False)
    print(cmp_.to_string(index=False))

    agg = (M.groupby(["model", "scheme", "features", "trait"])[["r_phenotype_java", "r_phenotype_py"]].mean().round(4).reset_index())
    agg.to_csv(C.RES_TAB / "java_vs_python_by_config.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for ax, m in zip(axes, ["RR-BLUP", "MLP"]):
        g = agg[agg.model == m]
        for fs, c in P.PALETTE.items():
            s = g[g.features == fs]
            ax.scatter(s.r_phenotype_py, s.r_phenotype_java, color=c, label=P.LABEL[fs], s=26)
        lo, hi = agg[["r_phenotype_java", "r_phenotype_py"]].min().min() - .03, agg[["r_phenotype_java", "r_phenotype_py"]].max().max() + .03
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6)
        ax.set_xlabel("Python predictive ability (r)"); ax.set_ylabel("Java predictive ability (r)"); ax.set_title(m)
    axes[0].legend(frameon=False, fontsize=7)
    P.save(fig, C.RES_FIG / "fig11_java_vs_python.png")

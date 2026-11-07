"""Cross-validation schemes and accuracy metrics for genomic prediction."""
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


def make_folds(lines, n_folds=5, n_repeats=2, seed=2026):
    """Random repeated K-fold (CV1-like: untested lines, mixed families) and
    leave-families-out K-fold (stringent across-family prediction)."""
    df = lines[["line_id", "family"]].copy()
    n = len(df)
    for r in range(n_repeats):
        fold = np.empty(n, dtype=int)
        for f, (_, te) in enumerate(KFold(n_folds, shuffle=True, random_state=seed + r).split(np.arange(n))):
            fold[te] = f
        df[f"random_rep{r + 1}"] = fold
    fam_ids = {f: i for i, f in enumerate(sorted(df["family"].unique()))}
    df["family_out"] = df["family"].map(fam_ids) % n_folds
    return df


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def run_scheme(X, y, tbv, folds, scheme_col, model, h2):
    """Fit `model` for every fold of `scheme_col`; return per-fold metrics (+ pooled predictions)."""
    rows, pred = [], np.full(len(y), np.nan)
    for f in sorted(folds[scheme_col].unique()):
        te = (folds[scheme_col] == f).values
        tr = ~te
        m = clone(model)
        t0 = time.time()
        m.fit(X[tr], y[tr])
        p = np.asarray(m.predict(X[te])).ravel()
        dt = time.time() - t0
        pred[te] = p
        r = pearson(p, y[te])
        rows.append(dict(fold=int(f), r_phenotype=r, r_tbv=pearson(p, tbv[te]),
                         accuracy=r / np.sqrt(h2) if np.isfinite(r) else np.nan,
                         rmse=float(np.sqrt(np.mean((p - y[te]) ** 2))), fit_seconds=dt))
    return pd.DataFrame(rows), pred

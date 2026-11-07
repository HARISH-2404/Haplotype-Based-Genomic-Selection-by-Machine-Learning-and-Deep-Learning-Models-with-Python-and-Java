import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from .config import PROCESSED, RESULTS, MODELS, RANDOM_SEED
from .data_io import read_marker_matrix
from .models_genomic import MultiKernelRidgeGS

def score(y, pred):
    return {
        "Correlation": float(pearsonr(y, pred).statistic) if np.std(pred) > 0 else np.nan,
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred))
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    snp = read_marker_matrix(PROCESSED / "snp_qc.csv")
    hap = read_marker_matrix(PROCESSED / "hap_qc.csv")
    ph = pd.read_csv(RESULTS / "adjusted_genotype_means.csv")
    ph = ph[ph.Trait == args.trait].dropna().set_index("Genotype")
    ids = snp.index.intersection(hap.index).intersection(ph.index)
    snp, hap = snp.loc[ids], hap.loc[ids]
    y = ph.loc[ids, "Adjusted_Value"].astype(float)

    # Fold-wise preprocessing and kernels: allele frequencies and imputation
    # are learned from training genotypes only.
    cv = KFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(ids), start=1):
        st, sv = snp.iloc[tr], snp.iloc[te]
        ht, hv = hap.iloc[tr], hap.iloc[te]
        yt, yv = y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()

        # Candidate alpha and kernel weights selected within a proper nested
        # CV in a final publication. This small grid is a starter comparison.
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            for ws in [0.0, 0.25, 0.5, 0.75, 1.0]:
                model = MultiKernelRidgeGS(alpha=alpha, snp_weight=ws,
                                           hap_weight=1.0-ws)
                model.fit_with_training_matrices(st, ht, yt)
                # Cross-kernel prediction; transform uses training allele frequencies.
                Ks = model.snp_kernel_.kernel(sv, st)
                Kh = model.hap_kernel_.kernel(hv, ht)
                Ktest = ws * Ks + (1.0-ws) * Kh
                pred = model.y_mean_ + Ktest @ model.dual_coef_
                rows.append({"Fold": fold, "Alpha": alpha,
                             "SNP_weight": ws, "Haplotype_weight": 1.0-ws,
                             **score(yv, pred)})

    cvres = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True, parents=True)
    cvres.to_csv(RESULTS / f"{args.trait}_multikernel_cv_grid.csv", index=False)
    summary = (cvres.groupby(["Alpha", "SNP_weight", "Haplotype_weight"])
               [["Correlation", "RMSE", "MAE"]].mean().reset_index()
               .sort_values("RMSE"))
    summary.to_csv(RESULTS / f"{args.trait}_multikernel_cv_summary.csv", index=False)
    best = summary.iloc[0]
    final = MultiKernelRidgeGS(
        alpha=float(best.Alpha),
        snp_weight=float(best.SNP_weight),
        hap_weight=float(best.Haplotype_weight)
    ).fit_with_training_matrices(snp, hap, y.to_numpy())
    MODELS.mkdir(exist_ok=True, parents=True)
    import joblib
    joblib.dump({"model": final, "trait": args.trait,
                 "snp_features": list(snp.columns),
                 "hap_features": list(hap.columns)},
                MODELS / f"{args.trait}_multikernel_gs.joblib")
    print("Best CV setting:", best.to_dict())
    print("Saved integrated genomic kernel model.")
if __name__ == "__main__":
    main()

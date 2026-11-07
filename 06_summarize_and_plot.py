#!/usr/bin/env python
"""STEP 6 - Summary tables, statistical tests and figures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.base import clone
from hgs import config as C, cv, haplotypes as H, models, plotting as P

FS = ["SNP", "HAP_FIXED", "HAP_LD"]

if __name__ == "__main__":
    R = pd.read_csv(C.RES_TAB / "cv_results_python_folds.csv")
    F = C.RES_FIG; F.mkdir(exist_ok=True, parents=True)

    # ---- 1. summary tables ----
    keys = ["trait", "features", "model", "family", "scheme"]
    per_rep = R.groupby(keys + ["repeat"]).agg(r=("r_phenotype", "mean"), r_tbv=("r_tbv", "mean"),
                                               acc=("accuracy", "mean"), rmse=("rmse", "mean")).reset_index()
    summ = per_rep.groupby(keys).agg(r_mean=("r", "mean"), r_sd=("r", "std"), r_tbv=("r_tbv", "mean"),
                                     accuracy=("acc", "mean"), rmse=("rmse", "mean")).reset_index()
    t_fit = R.groupby(keys)["fit_seconds"].mean().reset_index(name="fit_seconds")
    summ = summ.merge(t_fit, on=keys).round(4)
    summ.to_csv(C.RES_TAB / "cv_summary_python.csv", index=False)

    by_mf = per_rep.groupby(["scheme", "model", "family", "features", "trait"])["r"].mean().reset_index()
    by_mf = by_mf.groupby(["scheme", "model", "family", "features"])["r"].agg(mean="mean", sd="std").reset_index()
    by_mf.round(4).to_csv(C.RES_TAB / "summary_by_model_features.csv", index=False)

    # ---- 2. haplotype vs SNP: gain + paired Wilcoxon over (trait, repeat, fold) ----
    rows = []
    for sc in R.scheme.unique():
        for m in R.model.unique():
            for hs in ["HAP_LD", "HAP_FIXED"]:
                a = R[(R.scheme == sc) & (R.model == m) & (R.features == hs)].set_index(["trait", "repeat", "fold"])["r_phenotype"]
                b = R[(R.scheme == sc) & (R.model == m) & (R.features == "SNP")].set_index(["trait", "repeat", "fold"])["r_phenotype"]
                a, b = a.align(b, join="inner")
                d = (a - b).dropna()
                if len(d) < 5:
                    continue
                try:
                    p = stats.wilcoxon(d).pvalue
                except ValueError:
                    p = np.nan
                rows.append(dict(scheme=sc, model=m, haplotype_definition=hs, mean_r_hap=a.mean(), mean_r_snp=b.mean(),
                                 delta_r=d.mean(), pct_gain=(100 * d.mean() / abs(b.mean())) if abs(b.mean()) >= 0.10 else np.nan, wilcoxon_p=p,
                                 n_folds_compared=len(d)))
    gain = pd.DataFrame(rows).round(4)
    gain.to_csv(C.RES_TAB / "haplotype_vs_snp_gain.csv", index=False)

    # trait-level best configuration
    best = summ[summ.scheme == "random_kfold"].sort_values("r_mean", ascending=False).groupby("trait").head(3)
    best.to_csv(C.RES_TAB / "top3_per_trait_random_kfold.csv", index=False)

    # ---- 3. figures: data / block diagnostics ----
    geno = pd.read_csv(C.DATA_PROC / "genotypes_qc_imputed.csv")
    info = pd.read_csv(C.DATA_PROC / "snp_blocks.csv")
    truth = pd.read_csv(C.DATA_PROC / "true_snp_blocks_qc.csv")["true_block"].values
    lines = pd.read_csv(C.DATA_RAW / "lines.csv").set_index("line_id").loc[geno["line_id"]]
    G = geno.drop(columns="line_id").values
    idx = np.where(info["chr"].values == "A01")[0]
    P.ld_heatmap(H.r2_matrix(G[:, idx]), info["block_ld"].values[idx], truth[idx], "chromosome A01", F / "fig01_ld_heatmap_A01.png")
    P.block_size_hist({"True": np.bincount(truth)[np.bincount(truth) > 0],
                       "LD blocks": np.bincount(info["block_ld"].values)[np.bincount(info["block_ld"].values) > 0],
                       "Fixed 5-SNP": np.bincount(info["block_fixed"].values)[np.bincount(info["block_fixed"].values) > 0]},
                      F / "fig02_block_size_distribution.png")
    ht = pd.read_csv(C.RES_TAB / "haplotype_table_ld.csv")
    P.hap_freq_spectrum(ht["freq"].values, F / "fig03_haplotype_frequency_spectrum.png")
    P.pca_plot(G, lines["family"].tolist(), F / "fig04_pca_population_structure.png")

    # ---- 4. figures: accuracy ----
    for sc in ["random_kfold", "family_out"]:
        P.bar_by_features(by_mf[by_mf.scheme == sc], sc.replace("_", " "), F / f"fig05_accuracy_by_model_{sc}.png")
    for sc in ["random_kfold", "family_out"]:
        sub = summ[summ.scheme == sc].copy()
        sub["cfg"] = sub["trait"] + " | " + sub["features"]
        mat = sub.pivot(index="model", columns="cfg", values="r_mean")
        mat = mat[[f"{t} | {f}" for t in C.TRAITS for f in FS if f"{t} | {f}" in mat.columns]]
        P.heatmap(mat, f"Predictive ability (r) - {sc.replace('_', ' ')}", F / f"fig06_heatmap_{sc}.png", figsize=(11, 4.4))
    P.family_boxplot(R, F / "fig07_model_family_boxplot.png")
    P.runtime_plot(R, F / "fig08_runtime.png")

    # ---- 5. observed-vs-predicted scatter (RR-BLUP, HAP_LD) + CNN training curve ----
    feats = dict(np.load(C.DATA_PROC / "features.npz"))
    folds = pd.read_csv(C.DATA_PROC / "cv_folds.csv")
    ph = pd.read_csv(C.DATA_RAW / "phenotypes.csv").set_index("line_id").loc[folds.line_id]
    tb = pd.read_csv(C.DATA_TRUTH / "true_breeding_values.csv").set_index("line_id").loc[folds.line_id]
    zoo = models.get_models(C.SEED)
    panels = []
    for t, (_, _, h2, _) in C.TRAITS.items():
        _, pred = cv.run_scheme(feats["HAP_LD"], ph[t].values, tb[t].values, folds, "random_rep1", zoo["RR-BLUP"][1], h2)
        panels.append((f"{t} (RR-BLUP, haplotypes)", ph[t].values, pred))
    P.scatter_obs_pred(panels, F / "fig09_observed_vs_predicted.png")
    cnn = clone(zoo["WindowCNN"][1]).fit(feats["HAP_LD"], ph["SCY"].values)
    P.training_curve(cnn.history_, F / "fig10_cnn_training_curve.png", "Window-CNN training (SCY, haplotype features)")

    # ---- 6. text report ----
    pd.set_option("display.width", 200)
    print("\n=== Mean predictive ability by model x features (random k-fold, mean of traits) ===")
    print(by_mf[by_mf.scheme == "random_kfold"].pivot(index="model", columns="features", values="mean").round(3)[FS])
    print("\n=== Family-out ===")
    print(by_mf[by_mf.scheme == "family_out"].pivot(index="model", columns="features", values="mean").round(3)[FS])
    print("\n=== Haplotype (LD) vs SNP ===")
    print(gain[gain.haplotype_definition == "HAP_LD"][["scheme", "model", "mean_r_snp", "mean_r_hap", "delta_r", "pct_gain", "wilcoxon_p"]].to_string(index=False))

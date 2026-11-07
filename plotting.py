"""Publication-style figures for the haplotype genomic selection project."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {"SNP": "#7a7a7a", "HAP_FIXED": "#e0a030", "HAP_LD": "#1b7f5c"}
LABEL = {"SNP": "SNP", "HAP_FIXED": "Haplotype (5-SNP window)", "HAP_LD": "Haplotype (LD blocks)"}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ld_heatmap(R2, block_pred, block_true, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, blk, name in zip(axes, [block_true, block_pred], ["Simulated (true) blocks", "Detected LD blocks"]):
        ax.imshow(R2, cmap="magma_r", vmin=0, vmax=1, origin="upper")
        edges = np.where(np.diff(blk) != 0)[0] + 0.5
        for e in edges:
            ax.axvline(e, color="#00d0ff", lw=0.5, alpha=0.8); ax.axhline(e, color="#00d0ff", lw=0.5, alpha=0.8)
        ax.set_title(f"{name}\n{title}"); ax.set_xlabel("SNP index"); ax.set_ylabel("SNP index")
    save(fig, path)


def block_size_hist(sizes_dict, path):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    mx = max(max(s) for s in sizes_dict.values())
    bins = np.arange(0.5, mx + 1.5)
    for (name, s), col in zip(sizes_dict.items(), ["#444444", "#1b7f5c", "#e0a030"]):
        ax.hist(s, bins=bins, alpha=0.55, label=f"{name} (n={len(s)})", color=col)
    ax.set_xlabel("SNPs per block"); ax.set_ylabel("Number of blocks"); ax.legend(frameon=False)
    ax.set_title("Haplotype block size distribution")
    save(fig, path)


def hap_freq_spectrum(freqs, path):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(freqs, bins=30, color="#1b7f5c")
    ax.set_xlabel("Haplotype allele frequency"); ax.set_ylabel("Count")
    ax.set_title("Haplotype allele frequency spectrum (LD blocks)")
    save(fig, path)


def pca_plot(G, families, path):
    Gc = G - G.mean(0)
    U, S, _ = np.linalg.svd(Gc, full_matrices=False)
    pcs = U * S
    var = S ** 2 / (S ** 2).sum()
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    fams = sorted(set(families))
    cmap = plt.get_cmap("tab10")
    for i, f in enumerate(fams):
        m = np.array(families) == f
        ax.scatter(pcs[m, 0], pcs[m, 1], s=14, color=cmap(i), label=f, alpha=0.85)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title("Population structure (SNP PCA)"); ax.legend(frameon=False, fontsize=6, ncol=2)
    save(fig, path)


def bar_by_features(df, scheme, path, metric="r_phenotype"):
    """df: columns model, features, mean, sd (averaged over traits)."""
    models = list(df["model"].unique())
    fig, ax = plt.subplots(figsize=(9.5, 4))
    w = 0.26
    for k, fs in enumerate(["SNP", "HAP_FIXED", "HAP_LD"]):
        sub = df[df.features == fs].set_index("model").reindex(models)
        ax.bar(np.arange(len(models)) + (k - 1) * w, sub["mean"], w, yerr=sub["sd"], capsize=2,
               color=PALETTE[fs], label=LABEL[fs], error_kw=dict(lw=0.7))
    ax.set_xticks(np.arange(len(models))); ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Predictive ability (Pearson r, mean of 4 traits)")
    ax.set_title(f"Genomic prediction accuracy - {scheme}"); ax.legend(frameon=False)
    ax.axhline(0, color="k", lw=0.5)
    save(fig, path)


def heatmap(mat, title, path, cmap="viridis", fmt="{:.2f}", figsize=(8.5, 4.6)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=40, ha="right")
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5, color="w" if v < np.nanmean(mat.values) else "k")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Predictive ability (r)")
    ax.set_title(title)
    save(fig, path)


def scatter_obs_pred(panels, path):
    """panels: list of (title, obs, pred)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.3))
    for ax, (t, o, p) in zip(np.atleast_1d(axes), panels):
        ax.scatter(p, o, s=8, alpha=0.6, color="#1b7f5c")
        lo, hi = min(o.min(), p.min()), max(o.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.6, ls="--")
        r = np.corrcoef(o, p)[0, 1]
        ax.set_title(f"{t}\nr = {r:.2f}"); ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    save(fig, path)


def family_boxplot(df, path):
    fams = ["linear", "ML", "DL"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, sc in zip(axes, ["random_kfold", "family_out"]):
        pos, data, cols, labs = [], [], [], []
        i = 0
        for fam in fams:
            for fs in ["SNP", "HAP_LD"]:
                v = df[(df.scheme == sc) & (df.family == fam) & (df.features == fs)]["r_phenotype"].dropna().values
                data.append(v); pos.append(i); cols.append(PALETTE[fs]); labs.append(f"{fam}\n{fs.replace('HAP_LD','HAP')}")
                i += 1
            i += 0.6
        bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(0.8)
        ax.set_xticks(pos); ax.set_xticklabels(labs, fontsize=7); ax.set_title(sc.replace("_", " "))
        ax.axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("Predictive ability (r), fold level")
    save(fig, path)


def training_curve(history, path, title):
    h = np.array(history)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(h[:, 0], h[:, 1], label="train MSE"); ax.plot(h[:, 0], h[:, 2], label="validation MSE")
    ax.axvline(h[np.argmin(h[:, 2]), 0], color="k", ls=":", lw=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (standardised y)"); ax.set_title(title); ax.legend(frameon=False)
    save(fig, path)


def runtime_plot(df, path):
    t = df.groupby(["model", "features"])["fit_seconds"].mean().unstack().reindex(columns=["SNP", "HAP_FIXED", "HAP_LD"])
    t = t.sort_values("SNP")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(t))
    for k, fs in enumerate(t.columns):
        ax.barh(y + (k - 1) * 0.27, t[fs], 0.27, color=PALETTE[fs], label=LABEL[fs])
    ax.set_yticks(y); ax.set_yticklabels(t.index); ax.set_xscale("log"); ax.set_xlabel("mean training time per fold (s, log scale)")
    ax.legend(frameon=False, fontsize=7); ax.set_title("Computational cost (single CPU core)")
    save(fig, path)

"""Quality control + imputation for SNP dosage matrices (inbred lines coded 0 / 2)."""
import numpy as np
import pandas as pd


def qc_filter(G, snp_info, min_maf=0.05, max_snp_missing=0.10, max_line_missing=0.20):
    """Filter lines and SNPs. G: (n_lines x n_snps) float with NaN for missing."""
    n, p = G.shape
    line_miss = np.isnan(G).mean(1)
    keep_line = line_miss <= max_line_missing
    G = G[keep_line]
    snp_miss = np.isnan(G).mean(0)
    freq = np.nanmean(G, axis=0) / 2.0
    maf = np.minimum(freq, 1 - freq)
    keep_snp = (snp_miss <= max_snp_missing) & (maf >= min_maf)
    report = dict(lines_in=n, lines_out=int(keep_line.sum()), snps_in=p, snps_out=int(keep_snp.sum()),
                  removed_high_missing=int((snp_miss > max_snp_missing).sum()),
                  removed_low_maf=int((maf < min_maf).sum()),
                  mean_missing_rate=float(np.isnan(G).mean()))
    return keep_line, keep_snp, report


def nn_impute(G):
    """Nearest-neighbour imputation for inbred lines.

    For every line with missing calls, the genotype is copied from the genetically most
    similar line (identity-by-state over co-observed SNPs) that has an observed call.
    Remaining gaps (rare) are filled with the SNP major allele.
    """
    G = G.copy()
    obs = ~np.isnan(G)
    n = G.shape[0]
    Gz = np.where(obs, G, 0.0)
    both = obs.astype(float) @ obs.astype(float).T                    # co-observed counts
    diff = np.abs(Gz[:, None, :] - Gz[None, :, :]) if n * n * G.shape[1] < 3e8 else None
    if diff is not None:
        mism = (diff * (obs[:, None, :] & obs[None, :, :])).sum(2) / 2.0
    else:                                                              # memory-safe fallback
        mism = np.zeros((n, n))
        for i in range(n):
            mism[i] = (np.abs(Gz[i] - Gz) * (obs[i] & obs)).sum(1) / 2.0
    dist = mism / np.maximum(both, 1)
    np.fill_diagonal(dist, np.inf)
    for i in np.where((~obs).any(1))[0]:
        order = np.argsort(dist[i])
        for j in order:
            miss = np.isnan(G[i]) & obs[j]
            if miss.any():
                G[i, miss] = G[j, miss]
            if not np.isnan(G[i]).any():
                break
    still = np.isnan(G)
    if still.any():
        major = np.where(np.nanmean(G, axis=0) >= 1, 2.0, 0.0)
        G[still] = np.broadcast_to(major, G.shape)[still]
    return G

"""LD-based haplotype-block detection and haplotype-allele feature construction."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def r2_matrix(G):
    """Pairwise r^2 between SNP columns of a dosage matrix."""
    Gc = G - G.mean(0)
    sd = Gc.std(0)
    sd[sd == 0] = 1.0
    Z = Gc / sd
    R = (Z.T @ Z) / G.shape[0]
    return np.clip(R, -1, 1) ** 2


def detect_ld_blocks(G, chroms, r2_threshold=0.30, max_block=20):
    """Greedy LD blocks. A SNP is appended to the current block while its *mean* r^2 with the
    SNPs already in the block is >= threshold (and the block is shorter than max_block).
    Returns an int vector (block id per SNP, globally unique) -- singletons are allowed."""
    chroms = np.asarray(chroms)
    block_id = np.zeros(G.shape[1], dtype=int)
    nxt = 0
    for ch in pd.unique(chroms):
        idx = np.where(chroms == ch)[0]
        R2 = r2_matrix(G[:, idx])
        m, i = len(idx), 0
        while i < m:
            j = i + 1
            while j < m and (j - i) < max_block and R2[j, i:j].mean() >= r2_threshold:
                j += 1
            block_id[idx[i:j]] = nxt
            nxt += 1
            i = j
    return block_id


def fixed_window_blocks(chroms, window=5):
    chroms = np.asarray(chroms)
    block_id = np.zeros(len(chroms), dtype=int)
    nxt = 0
    for ch in pd.unique(chroms):
        idx = np.where(chroms == ch)[0]
        for s in range(0, len(idx), window):
            block_id[idx[s:s + window]] = nxt
            nxt += 1
    return block_id


def snp_blocks(n_snp):
    """Every SNP is its own 'block' (SNP-based baseline)."""
    return np.arange(n_snp)


def block_haplotypes(G01, block_id, min_freq=0.03):
    """Identify haplotype alleles within each block.

    G01: (n x p) 0/1 allele matrix (inbred lines). Haplotypes with frequency < min_freq are merged
    into the most similar frequent haplotype (Hamming distance), which stabilises rare-allele estimates.
    Returns (hap_ids [n x B], n_alleles [B], haplotype strings per block)."""
    n = G01.shape[0]
    blocks = np.unique(block_id)
    hap_ids = np.zeros((n, len(blocks)), dtype=int)
    n_alleles, strings = [], []
    for bi, b in enumerate(blocks):
        sub = G01[:, block_id == b].astype(int)
        uniq, inv, cnt = np.unique(sub, axis=0, return_inverse=True, return_counts=True)
        inv = np.asarray(inv).ravel()
        freq = cnt / n
        common = np.where(freq >= min_freq)[0]
        if len(common) == 0:
            common = np.array([int(np.argmax(cnt))])
        remap = np.arange(len(uniq))
        for u in range(len(uniq)):
            if u not in common:
                d = (uniq[common] != uniq[u]).sum(1)
                remap[u] = common[int(np.argmin(d))]
        merged = remap[inv]
        final = {old: new for new, old in enumerate(np.unique(merged))}
        hap_ids[:, bi] = [final[x] for x in merged]
        n_alleles.append(len(final))
        strings.append(["".join(map(str, uniq[o])) for o in sorted(final, key=final.get)])
    return hap_ids, np.array(n_alleles), strings


def onehot_haplotypes(hap_ids, n_alleles, drop_monomorphic=True):
    """One-hot (haplotype-allele dosage) design matrix. Returns (X, column names)."""
    cols, names = [], []
    for b in range(hap_ids.shape[1]):
        if n_alleles[b] < 2 and drop_monomorphic:
            continue
        for k in range(n_alleles[b]):
            cols.append((hap_ids[:, b] == k).astype(float))
            names.append(f"blk{b}_h{k}")
    return np.column_stack(cols), names


def evaluate_blocks(pred_block, true_block, chroms, pos=None):
    """Compare a detected block partition with the simulated truth."""
    pred_block, true_block = np.asarray(pred_block), np.asarray(true_block)
    ari = adjusted_rand_score(true_block, pred_block)
    tb = set(np.where(np.diff(true_block) != 0)[0])
    pb = set(np.where(np.diff(pred_block) != 0)[0])
    tp = len(tb & pb)
    prec = tp / max(len(pb), 1)
    rec = tp / max(len(tb), 1)
    sizes = np.bincount(pred_block)
    sizes = sizes[sizes > 0]
    return dict(n_blocks=int(len(sizes)), mean_size=float(sizes.mean()), median_size=float(np.median(sizes)),
                max_size=int(sizes.max()), singleton_frac=float((sizes == 1).mean()),
                ARI=float(ari), boundary_precision=float(prec), boundary_recall=float(rec),
                boundary_F1=float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0)

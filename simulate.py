"""Simulate a cotton-like inbred population with a known haplotype-block structure.

Why simulate? The pipeline can be validated end-to-end because the true block
boundaries, true haplotype effects and true breeding values are all known.

Genetic model
-------------
* Each chromosome is split into blocks of 4-9 SNPs. Each block has 2-5 ancestral
  haplotype classes (founder haplotypes).
* 20 founders carry one haplotype class per block. 10 biparental families of 30 F5
  recombinant inbred lines (RILs) are produced by mosaicking the two parents with a
  per-block parent-switch probability.
* Phenotype = additive haplotype effects (QTL blocks) + haplotype x haplotype
  epistasis + environmental noise, scaled to the requested line-mean heritability.
"""
import numpy as np
import pandas as pd

from . import config as C


def _chr_names():
    return [f"A{i:02d}" for i in range(1, 14)] + [f"D{i:02d}" for i in range(1, 14)]


def _make_blocks(rng, n_snp, size_range):
    sizes, tot = [], 0
    while tot < n_snp:
        s = int(rng.integers(size_range[0], size_range[1] + 1))
        s = min(s, n_snp - tot)
        sizes.append(s)
        tot += s
    return sizes


def _block_classes(rng, size, n_founders):
    """Return (class_matrix [K x size], founder_class [n_founders])."""
    while True:
        k = int(rng.integers(2, 6))
        base = (rng.random(size) < rng.uniform(0.25, 0.75)).astype(int)
        classes = [base]
        tries = 0
        while len(classes) < k and tries < 50:
            tries += 1
            src = classes[int(rng.integers(0, len(classes)))]
            flip = rng.random(size) < 0.3
            cand = np.where(flip, 1 - src, src)
            if not any((cand == c).all() for c in classes):
                classes.append(cand)
        classes = np.array(classes)
        k = len(classes)
        probs = rng.dirichlet(np.ones(k) * 1.5)
        founder_class = rng.choice(k, size=n_founders, p=probs)
        if len(np.unique(founder_class)) >= 2:
            return classes, founder_class


def simulate(seed=C.SEED):
    rng = np.random.default_rng(seed)
    chr_names = _chr_names()[: C.N_CHR]

    # ---------- genome / block layout ----------
    snp_rows, block_rows = [], []
    block_classes, founder_cls = [], []
    snp_counter, block_counter = 0, 0
    block_chr_start = []                       # (first block index) per chromosome
    for ch in chr_names:
        sizes = _make_blocks(rng, C.SNPS_PER_CHR, C.BLOCK_SIZE_RANGE)
        pos = np.sort(rng.choice(np.arange(10_000, 100_000_000, 1000), C.SNPS_PER_CHR, replace=False))
        block_chr_start.append(block_counter)
        s0 = 0
        for sz in sizes:
            cls, fc = _block_classes(rng, sz, C.N_FOUNDERS)
            block_classes.append(cls)
            founder_cls.append(fc)
            for j in range(sz):
                snp_rows.append((f"Gh_{ch}_{pos[s0 + j]}", ch, int(pos[s0 + j]), block_counter))
            block_rows.append((block_counter, ch, snp_counter, snp_counter + sz))
            snp_counter += sz
            s0 += sz
            block_counter += 1
    snp_info = pd.DataFrame(snp_rows, columns=["snp_id", "chr", "pos", "true_block"])
    true_blocks = pd.DataFrame(block_rows, columns=["block", "chr", "start", "end"])
    n_snp, n_block = len(snp_info), len(true_blocks)
    founder_cls = np.array(founder_cls).T               # founders x blocks
    block_chr = true_blocks["chr"].values

    # ---------- families & RILs ----------
    def divergence(a, b):
        return float((founder_cls[a] != founder_cls[b]).mean())

    fam_rows, hap_class = [], []
    line_idx = 0
    for f in range(C.N_FAMILIES):
        while True:
            a, b = rng.choice(C.N_FOUNDERS, 2, replace=False)
            if divergence(a, b) > 0.45:
                break
        for r in range(C.FAMILY_SIZE):
            parent = int(rng.integers(0, 2))
            row = np.empty(n_block, dtype=int)
            for bi in range(n_block):
                new_chr = bi == 0 or block_chr[bi] != block_chr[bi - 1]
                if new_chr:
                    parent = int(rng.integers(0, 2))
                elif rng.random() < C.SWITCH_PROB:
                    parent = 1 - parent
                row[bi] = founder_cls[(a, b)[parent], bi]
            hap_class.append(row)
            fam_rows.append((f"RIL{line_idx + 1:03d}", f"FAM{f + 1:02d}", f"FND{a + 1:02d}", f"FND{b + 1:02d}"))
            line_idx += 1
    lines = pd.DataFrame(fam_rows, columns=["line_id", "family", "parent1", "parent2"])
    hap_class = np.array(hap_class)                      # lines x blocks (true haplotype class)
    n_lines = len(lines)

    # ---------- SNP matrix (inbred => 0 / 2 dosage) ----------
    G = np.empty((n_lines, n_snp), dtype=float)
    for bi, (cls, row) in enumerate(zip(block_classes, true_blocks.itertuples())):
        G[:, row.start:row.end] = cls[hap_class[:, bi]]
    G = G * 2.0
    err = rng.random(G.shape) < C.GENOTYPING_ERROR
    G[err] = 2.0 - G[err]
    miss = rng.random(G.shape) < C.MISSING_RATE
    G_obs = G.copy()
    G_obs[miss] = np.nan

    # ---------- phenotypes ----------
    qtl_pool = rng.choice(n_block, C.N_QTL_POOL, replace=False)
    n_cls = np.array([len(c) for c in block_classes])
    common = {b: rng.normal(0, 1, n_cls[b]) for b in qtl_pool}
    phen, tbv, effects_rows = {}, {}, []
    epi_pairs = [tuple(rng.choice(qtl_pool, 2, replace=False)) for _ in range(C.N_EPI_PAIRS)]
    for t, (mu, sd, h2, rho) in C.TRAITS.items():
        qtl = rng.choice(qtl_pool, C.N_QTL_PER_TRAIT, replace=False)
        add = np.zeros(n_lines)
        for b in qtl:
            own = rng.normal(0, 1, n_cls[b])
            eff = abs(rho) * np.sign(rho) * common[b] + np.sqrt(1 - rho ** 2) * own
            add += eff[hap_class[:, b]]
            for k in range(n_cls[b]):
                effects_rows.append((t, int(b), k, float(eff[k])))
        epi = np.zeros(n_lines)
        for (b1, b2) in epi_pairs:
            c1, c2 = int(rng.integers(0, n_cls[b1])), int(rng.integers(0, n_cls[b2]))
            epi += rng.normal(0, 1) * ((hap_class[:, b1] == c1) & (hap_class[:, b2] == c2)).astype(float)
        add = (add - add.mean()) / add.std()
        epi = (epi - epi.mean()) / (epi.std() if epi.std() > 0 else 1)
        g = np.sqrt(1 - C.EPI_FRACTION) * add + np.sqrt(C.EPI_FRACTION) * epi
        g = (g - g.mean()) / g.std()
        e = rng.normal(0, 1, n_lines)
        y = np.sqrt(h2) * g + np.sqrt(1 - h2) * e
        phen[t] = mu + sd * y
        tbv[t] = mu + sd * np.sqrt(h2) * g
    pheno = pd.DataFrame(phen).round(3)
    pheno.insert(0, "line_id", lines["line_id"])
    tbv = pd.DataFrame(tbv).round(3)
    tbv.insert(0, "line_id", lines["line_id"])

    geno = pd.DataFrame(G_obs, columns=snp_info["snp_id"])
    geno.insert(0, "line_id", lines["line_id"])
    truth_hap = pd.DataFrame(hap_class, columns=[f"block{b}" for b in range(n_block)])
    truth_hap.insert(0, "line_id", lines["line_id"])
    effects = pd.DataFrame(effects_rows, columns=["trait", "block", "hap_class", "effect"])
    return dict(genotypes=geno, snp_info=snp_info, lines=lines, phenotypes=pheno,
                tbv=tbv, true_blocks=true_blocks, truth_hap=truth_hap, effects=effects)


def save(d):
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    C.DATA_TRUTH.mkdir(parents=True, exist_ok=True)
    d["genotypes"].to_csv(C.DATA_RAW / "genotypes_snp.csv", index=False, float_format="%.0f")
    d["snp_info"][["snp_id", "chr", "pos"]].to_csv(C.DATA_RAW / "snp_info.csv", index=False)
    d["lines"].to_csv(C.DATA_RAW / "lines.csv", index=False)
    d["phenotypes"].to_csv(C.DATA_RAW / "phenotypes.csv", index=False)
    d["snp_info"][["snp_id", "chr", "pos", "true_block"]].to_csv(C.DATA_TRUTH / "true_snp_blocks.csv", index=False)
    d["true_blocks"].to_csv(C.DATA_TRUTH / "true_blocks.csv", index=False)
    d["tbv"].to_csv(C.DATA_TRUTH / "true_breeding_values.csv", index=False)
    d["truth_hap"].to_csv(C.DATA_TRUTH / "true_haplotype_classes.csv", index=False)
    d["effects"].to_csv(C.DATA_TRUTH / "true_haplotype_effects.csv", index=False)

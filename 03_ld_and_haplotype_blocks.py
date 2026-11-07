#!/usr/bin/env python
"""STEP 3 - LD analysis, haplotype-block detection (LD-based and fixed window) and haplotype alleles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from hgs import config as C, haplotypes as H

if __name__ == "__main__":
    geno = pd.read_csv(C.DATA_PROC / "genotypes_qc_imputed.csv")
    info = pd.read_csv(C.DATA_PROC / "snp_info_qc.csv")
    truth = pd.read_csv(C.DATA_PROC / "true_snp_blocks_qc.csv")["true_block"].values
    G = geno.drop(columns="line_id").values
    chroms = info["chr"].values

    # (a) sensitivity of the LD threshold (uses the simulated truth - only possible on simulated data)
    rows = []
    for thr in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        b = H.detect_ld_blocks(G, chroms, thr, C.LD_MAX_BLOCK)
        rows.append(dict(r2_threshold=thr, **H.evaluate_blocks(b, truth, chroms)))
    sens = pd.DataFrame(rows)
    sens.to_csv(C.RES_TAB / "ld_threshold_sensitivity.csv", index=False)
    print(sens.round(3).to_string(index=False))

    # (b) final block definitions
    ld_blocks = H.detect_ld_blocks(G, chroms, C.LD_R2_THRESHOLD, C.LD_MAX_BLOCK)
    fx_blocks = H.fixed_window_blocks(chroms, C.FIXED_WINDOW)
    info["block_ld"], info["block_fixed"] = ld_blocks, fx_blocks
    info.to_csv(C.DATA_PROC / "snp_blocks.csv", index=False)

    comp = pd.DataFrame([
        dict(method=f"LD blocks (r2>={C.LD_R2_THRESHOLD})", **H.evaluate_blocks(ld_blocks, truth, chroms)),
        dict(method=f"Fixed window ({C.FIXED_WINDOW} SNP)", **H.evaluate_blocks(fx_blocks, truth, chroms)),
        dict(method="Single SNP", **H.evaluate_blocks(H.snp_blocks(len(truth)), truth, chroms)),
    ])
    comp.to_csv(C.RES_TAB / "block_definition_comparison.csv", index=False)
    print(comp.round(3).to_string(index=False))

    # (c) haplotype alleles per block
    G01 = (G / 2.0).round().astype(int)
    summ = []
    for name, blk in [("ld", ld_blocks), ("fixed", fx_blocks)]:
        hap_ids, n_all, strings = H.block_haplotypes(G01, blk, C.MIN_HAP_FREQ)
        np.save(C.DATA_PROC / f"hap_ids_{name}.npy", hap_ids)
        np.save(C.DATA_PROC / f"hap_nalleles_{name}.npy", n_all)
        summ.append(dict(definition=name, n_blocks=len(n_all), n_polymorphic_blocks=int((n_all > 1).sum()),
                         total_haplotype_alleles=int(n_all[n_all > 1].sum()), mean_alleles_per_block=float(n_all.mean()),
                         max_alleles=int(n_all.max())))
        # human-readable haplotype table
        rows = []
        for bi, s in enumerate(strings):
            cols = np.where(blk == np.unique(blk)[bi])[0]
            for k, hs in enumerate(s):
                rows.append((f"{name}_blk{bi}", info.loc[cols[0], "chr"], info.loc[cols[0], "pos"], info.loc[cols[-1], "pos"],
                             len(cols), k, hs, float((hap_ids[:, bi] == k).mean())))
        pd.DataFrame(rows, columns=["block", "chr", "start_bp", "end_bp", "n_snp", "hap_allele", "haplotype", "freq"]) \
          .to_csv(C.RES_TAB / f"haplotype_table_{name}.csv", index=False)
    pd.DataFrame(summ).to_csv(C.RES_TAB / "haplotype_summary.csv", index=False)
    print(pd.DataFrame(summ).round(2).to_string(index=False))

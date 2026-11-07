#!/usr/bin/env python
"""STEP 2 - Quality control (call rate, MAF) and nearest-neighbour imputation."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from hgs import config as C, qc

if __name__ == "__main__":
    geno = pd.read_csv(C.DATA_RAW / "genotypes_snp.csv")
    info = pd.read_csv(C.DATA_RAW / "snp_info.csv")
    truth = pd.read_csv(C.DATA_TRUTH / "true_snp_blocks.csv")
    ids = geno["line_id"].values
    G = geno.drop(columns="line_id").values.astype(float)

    keep_line, keep_snp, rep = qc.qc_filter(G, info, C.MIN_MAF, C.MAX_SNP_MISSING, C.MAX_LINE_MISSING)
    G, ids = G[keep_line], ids[keep_line]
    G = G[:, keep_snp]
    info = info[keep_snp].reset_index(drop=True)
    truth = truth[keep_snp].reset_index(drop=True)
    Gi = qc.nn_impute(G)
    rep["missing_after_imputation"] = int(np.isnan(Gi).sum())
    print(json.dumps(rep, indent=2))

    C.DATA_PROC.mkdir(exist_ok=True, parents=True)
    out = pd.DataFrame(Gi, columns=info["snp_id"]); out.insert(0, "line_id", ids)
    out.to_csv(C.DATA_PROC / "genotypes_qc_imputed.csv", index=False, float_format="%.0f")
    info.to_csv(C.DATA_PROC / "snp_info_qc.csv", index=False)
    truth.to_csv(C.DATA_PROC / "true_snp_blocks_qc.csv", index=False)
    (C.RES_TAB / "qc_report.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame([rep]).to_csv(C.RES_TAB / "qc_report.csv", index=False)

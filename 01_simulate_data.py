#!/usr/bin/env python
"""STEP 1 - Simulate genotype (SNP), phenotype and truth files."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hgs import config as C, simulate as S

if __name__ == "__main__":
    d = S.simulate()
    S.save(d)
    print(f"lines={len(d['lines'])} snps={len(d['snp_info'])} true_blocks={len(d['true_blocks'])}")
    print(d["phenotypes"].describe().round(2))
    print("Wrote files to", C.DATA_RAW, "and", C.DATA_TRUTH)

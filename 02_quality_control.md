# 02 · Quality control and imputation (`hgs/qc.py`, Java `QualityControl`)

1. **Line filter** – drop lines with > 20 % missing calls.
2. **SNP filter** – drop SNPs with > 10 % missing calls or minor-allele frequency < 0.05 (monomorphic SNPs vanish here as well).
3. **Imputation** – inbred lines are homozygous, so a missing call is copied from the *genetically closest line that has the call*
   (IBS distance computed over co-observed SNPs). Rare remaining gaps use the major allele.

Why not mean-imputation? A mean (e.g. 0.8) is not a valid homozygous state and would corrupt haplotype strings.

Result for the simulated data: 300 lines, **2,600 → 1,393 SNPs**, 0 missing after imputation (`results/tables/qc_report.json`).

# 08 · Using your own cotton data

Required files (same layout as `data/raw/`):

* `genotypes_snp.csv` – `line_id` + one column per SNP; values 0 / 2 (homozygous), empty = missing. (Heterozygous 1 is allowed for RIL/F2 – haplotype strings then need phasing.)
* `snp_info.csv` – `snp_id,chr,pos` **in the same order as the genotype columns**, sorted by chromosome and position.
* `lines.csv` – `line_id,family,parent1,parent2` (`family` drives the family-out CV).
* `phenotypes.csv` – `line_id` + trait columns (best: BLUEs/BLUPs from your multi-environment model).

From a VCF (example):
```bash
bcftools view -m2 -M2 -v snps raw.vcf.gz | bcftools query -f '%CHROM\t%POS[\t%GT]\n' ...   # then convert GT to 0/2
plink --vcf raw.vcf.gz --allow-extra-chr --make-bed --out cotton
plink --bfile cotton --blocks no-pheno-req --blocks-max-kb 500 --allow-extra-chr --out cotton_blocks   # Gabriel blocks (optional alternative)
```
Phase/impute heterozygous data first (e.g. Beagle) if lines are not fully inbred.
Disable truth-dependent items: `r_tbv`, block-recovery metrics in step 3 (`evaluate_blocks`).

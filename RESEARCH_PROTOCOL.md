# Research protocol and methodological decisions

## Primary objective
Develop and validate genomic prediction methods using SNP and haplotype predictors in cotton, comparing SNP-only, haplotype-only and integrated prediction.

## Experimental unit
The genotype/line is the unit of genomic prediction. Plot-level observations are used to estimate adjusted genotype performance while accounting for design and environment.

## Validation designs
Use at least:
1. Repeated random genotype-level K-fold CV.
2. Family-wise CV when families/pedigrees are known.
3. Forward-generation or independent-population validation when the intended use is prediction of future lines.

Keep all observations/replicates of one genotype in the same fold. Perform marker filtering, imputation, scaling and feature selection using training-fold data only.

## Outcomes
For ML/DL:
- predicted adjusted genotype performance
- Pearson correlation
- RMSE, MAE
- calibration slope and intercept

For genomic mixed/kernel models:
- genomic deviation / predicted genetic value
- report GEBV only where the target and model support that interpretation
- assess prediction bias and validate against independent genetic evaluations where possible

## Cotton-specific checks
Cotton is an allotetraploid crop. Document:
- reference genome and version
- A/D subgenome assignment
- variant caller and ploidy assumptions
- dosage definition at each locus
- homeologous alignment/paralog filtering
- phasing method and haplotype block construction
- whether haplotypes are within-subgenome and disomic
- missingness and minor allele frequency thresholds

Do not assume all markers have 0/1/2 dosage. Some datasets may encode allele counts 0–4 or genotype categories. Adapt kernels accordingly.

## Important modeling caution
SNP markers and haplotypes derived from the same SNPs are statistically redundant. A naive concatenation can double-count genomic information and overfit. Compare separate kernels, tune kernel weights using nested CV, and test on genuinely independent populations. Consider chromosome/block-aware kernels and regularization.

## Multi-environment extension
For multi-environment prediction, use a model that represents genotype-by-environment covariance, e.g. environment-specific genomic kernels or reaction-norm models. Do not simply append environment labels to genotype features and claim a G×E genomic model.

## Reproducibility
Archive data dictionaries, software versions, random seeds, train/test genotype IDs, model parameters, feature definitions and output metrics.

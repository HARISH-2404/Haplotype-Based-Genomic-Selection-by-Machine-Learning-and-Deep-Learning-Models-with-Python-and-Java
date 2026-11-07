# 05 · Cross-validation design (`hgs/cv.py`)

* **random_kfold** — 5-fold × 3 repeats; test lines come from all families (families are represented in training).
* **family_out** — 10 families are split into 5 folds of 2 families; a model never sees the held-out families (mimics predicting a new cross).
* The **same fold file** (`data/processed/cv_folds.csv`) is used by Python and Java so the two implementations are directly comparable.
* All hyper-parameters are tuned **inside the training fold only** (RidgeCV LOO, LassoCV inner 3-fold, MLP/CNN early-stopping split) → no test-set leakage.
* QC and block definition are performed once on all lines (genotypes only, no phenotypes), which is standard practice and does not leak phenotype information.

**Metrics** — predictive ability `r = cor(ŷ, y_test)` per fold, averaged over folds and repeats; accuracy `r/√h²`; `r_tbv = cor(ŷ, true breeding value)` (simulation only); RMSE; training seconds.
**Statistics** — paired Wilcoxon signed-rank test over matched (trait, repeat, fold) pairs for haplotype vs SNP features (`haplotype_vs_snp_gain.csv`).

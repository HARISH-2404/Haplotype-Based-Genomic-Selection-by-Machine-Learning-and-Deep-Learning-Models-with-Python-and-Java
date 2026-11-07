# 00 · Project overview

**Goal.** Test whether *haplotype-based* predictors improve genomic selection (GS) in a cotton (*Gossypium hirsutum*)
breeding population compared with single-SNP predictors, and compare classical, machine-learning (ML) and deep-learning (DL)
models — implemented twice, in **Python** and in **Java**.

## Workflow

```
 01 simulate data ─► 02 QC + imputation ─► 03 LD / haplotype blocks ─► 04 features + CV folds
                                                                              │
        07 Java ↔ Python cross-check ◄── 06 tables + figures ◄── 05 models × features × traits × CV
                         ▲
                 Java pipeline (QC → blocks → features → RR-BLUP / MLP → CV) reads the same raw files + same folds
```

| Step | Python script | Java class(es) | Output |
|------|---------------|----------------|--------|
| 1 Simulate cotton-like RIL population | `01_simulate_data.py` | – | `data/raw/*`, `data/truth/*` |
| 2 SNP QC + nearest-neighbour imputation | `02_quality_control.py` | `QualityControl` | `data/processed/genotypes_qc_imputed.csv` |
| 3 LD blocks / haplotype alleles | `03_ld_and_haplotype_blocks.py` | `LdBlocks`, `Haplotypes` | `results/tables/haplotype_table_*.csv` |
| 4 Design matrices + fold file | `04_build_features_and_folds.py` | (`Haplotypes.oneHot`) | `features.npz`, `cv_folds.csv` |
| 5 Train + cross-validate models | `05_run_models_cv.py` | `RidgeBlup`, `Mlp`, `Main` | `cv_results_*_folds.csv` |
| 6 Summaries, tests, figures | `06_summarize_and_plot.py` | – | `results/tables`, `results/figures` |
| 7 Java vs Python agreement | `07_compare_java_python.py` | – | `java_vs_python_agreement.csv` |

## Feature sets compared
* **SNP** – QC-passed SNP dosage (0/2; inbred lines).
* **HAP_FIXED** – non-overlapping windows of 5 SNPs → each window's haplotype alleles are one-hot encoded.
* **HAP_LD** – LD-based haplotype blocks (greedy, mean r² ≥ 0.10 within block) → haplotype alleles one-hot encoded.

## Models compared
| Family | Models |
|--------|--------|
| Classical linear (GS) | **RR-BLUP / GBLUP-equivalent** (ridge, λ by leave-one-out), **Bayesian ridge**, **LASSO** |
| Machine learning | **SVR (RBF)**, **Random Forest**, **Gradient boosting** (histogram-based) |
| Deep learning | **MLP** (128-64), **Deep MLP** (256-128-64), **Window-CNN** (convolution over marker windows, written from scratch in NumPy with exact back-propagation) ; Java: **MLP** (64-32) from scratch |

## Cross-validation
* `random_kfold` – 5-fold, **3 repeats** (untested lines, all families in training).
* `family_out` – 5 folds, each holding out **2 complete families** (hard, across-family prediction).
* Metrics: predictive ability **r(ŷ, y)** in the test fold; accuracy r/√h²; r with the simulated true breeding value; RMSE; training time.

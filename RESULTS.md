# Results (simulated cotton population, 300 F5 RILs, 4 traits)

All numbers are predictive ability r = cor(predicted, observed) in held-out lines, averaged over folds/repeats/traits.
Tables: `results/tables/`, figures: `results/figures/`.

## 1. Haplotype block detection (vs simulated truth)
| Method | blocks | boundary F1 | ARI |
|---|---|---|---|
| LD blocks (r² ≥ 0.10) | 517 | **0.71** | **0.73** |
| Fixed 5-SNP window | 289 | 0.30 | 0.58 |
| Single SNP | 1393 | 0.45 | 0.00 |

The LD threshold was tuned using the simulated truth (`ld_threshold_sensitivity.csv`); on real data tune it by CV accuracy instead.

## 2. Prediction accuracy, random 5-fold CV ×3 repeats (mean of 4 traits)
| Model | SNP | Hap (5-SNP window) | Hap (LD blocks) |
|---|---|---|---|
| RR-BLUP | 0.491 | 0.501 | 0.508 |
| Bayesian ridge | 0.473 | 0.482 | 0.502 |
| LASSO | 0.479 | 0.392 | 0.517 |
| SVR | 0.483 | 0.485 | 0.484 |
| Random forest | 0.479 | 0.482 | 0.494 |
| Gradient boosting | 0.493 | 0.496 | **0.539** |
| MLP (128-64) | 0.227 | 0.199 | 0.210 |
| Deep MLP | 0.202 | 0.262 | 0.239 |
| Window-CNN (NumPy) | 0.389 | 0.385 | 0.384 |

Haplotype (LD) vs SNP, paired Wilcoxon over fold pairs (`haplotype_vs_snp_gain.csv`): significant gains for RR-BLUP (+0.017, p=0.0004),
Bayesian ridge (+0.029), LASSO (+0.039), random forest (+0.015), gradient boosting (+0.046), all p<0.01; no gain for SVR, MLP, CNN.

## 3. Across-family prediction (2 families held out)
Best: LASSO 0.32 (SNP) / 0.32 (hap), gradient boosting 0.27 → **0.31** with haplotypes, RR-BLUP 0.24 → 0.25. Haplotype gains are in the same direction
but not significant (n = 5 folds × 4 traits). DL models are near zero (MLP 0.02–0.05, Deep MLP ≤ 0.03, CNN 0.07–0.13).

## 4. Java vs Python
RR-BLUP: fold-level correlation **0.9995**, mean |Δr| = 0.004 (same data, same folds). Java MLP (64-32, from scratch) is a different network from
sklearn's (128-64) and agrees only loosely (fold-level correlation 0.31) – treat the two MLPs as separate models, not replicates.
Java RR-BLUP mean r: SNP 0.364, HAP_LD 0.382, HAP_FIXED 0.371 (random_rep1 + family-out folds).

## 5. Honest reading
* Gains from haplotypes are real but **modest** (≈ +3 to +9 % relative for linear/tree models) in this simulation, whose QTL act at the haplotype level with 20 % epistasis.
* With 240 training lines, the neural networks do **not** beat ridge/boosting; deep learning needs far more lines or marker data to pay off.
* Simulated data favour haplotype models by construction; real-data results will differ.
* Fixed windows can hurt (LASSO) because they cut across true blocks.

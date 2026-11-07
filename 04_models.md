# 04 · Models (`hgs/models.py`, `hgs/cnn_numpy.py`; Java `RidgeBlup`, `Mlp`)

| Model | Implementation | Notes |
|-------|----------------|-------|
| RR-BLUP / GBLUP | scikit-learn `RidgeCV` (alphas 1…10⁶, efficient LOO) ; Java: dual-form ridge, Cholesky | Ridge on centred markers ≡ RR-BLUP; the dual form is GBLUP with G ∝ XX′. |
| Bayesian ridge | `BayesianRidge` | Estimates the shrinkage from the data. |
| LASSO | `LassoCV` (3-fold) | Sparse marker/haplotype effects. |
| SVR | RBF kernel, standardised y | Non-linear kernel. |
| Random forest | 100 trees, max_features = 0.2 | Captures interactions. |
| Gradient boosting | `HistGradientBoostingRegressor` | Depth 3, 150 iterations, lr 0.05. |
| MLP | sklearn, (128, 64), L2 α = 1, early stopping | DL baseline. |
| Deep MLP | sklearn, (256, 128, 64) | Deeper network. |
| Window-CNN | NumPy, 1-D convolution (kernel = stride = 8 columns, 8 filters) → Dense 32 → 1; Adam, dropout 0.2, L2, early stopping | Verified with a finite-difference gradient check (`python/tests/test_cnn_gradients.py`). |
| Java MLP | from scratch, (64, 32), mini-batch Adam, L2, early stopping | Validated against the Python MLP trend. |

Why a NumPy CNN? The project must run on any machine without GPU/TensorFlow/PyTorch. Because kernel = stride the convolution is a reshape + matrix product and the
gradients are exact. For GPU-scale work (thousands of lines, > 100 k SNPs) port the same architecture to PyTorch or TensorFlow.

# 07 · Java implementation
Classes (package `org.hgs`): `Dataset` (CSV I/O), `QualityControl`, `LdBlocks`, `Haplotypes`, `RidgeBlup` (dual-form ridge, Cholesky, leave-one-out λ with an
almost-unpenalised intercept), `Mlp` (ReLU, Adam, L2, early stopping), `Main` (CV driver reading `data/processed/cv_folds.csv`), `SelfTest`.
Run: `cd java && ./test.sh && ./run.sh` (needs JDK ≥ 11, no Maven required; `pom.xml` provided). Run Python steps 1–4 first so `cv_folds.csv` exists.
Then `python python/scripts/07_compare_java_python.py`.

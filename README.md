# Haplotype-Based Genomic Selection Using Machine Learning and Deep Learning

A research-oriented starter platform for cotton genomic prediction using Python and Java.

## Scientific scope

The platform compares:
- SNP-only predictors
- Haplotype-only predictors
- Integrated SNP + haplotype predictors

It includes:
- Input validation and genotype/haplotype dosage QC
- Phenotype adjustment for replicated field trials
- SNP and haplotype genomic relationship kernels
- Kernel-ridge genomic prediction (GBLUP-like kernel model)
- Machine-learning models: Ridge, Elastic Net, Random Forest, SVR, Extra Trees, HistGradientBoosting, optional XGBoost/LightGBM
- Deep-learning MLP using PyTorch
- Cross-validation and metrics
- Prediction of new lines
- FastAPI model-serving endpoint
- Java Spring Boot client that submits prediction requests to Python

**Scientific terminology:** Kernel-ridge genomic prediction can estimate genomic values when the target is an appropriately adjusted genotype value and the model is specified as a genomic random-effect model. Generic ML/DL models trained on adjusted phenotypes produce genomic predictions / predicted adjusted performance, not automatically unbiased GEBVs. Report these separately. Validate breeding values against suitable genetic-model benchmarks.

## Important data assumptions

1. SNP dosage file: one row per genotype, first column `Genotype`, remaining columns numeric dosage.
2. Haplotype dosage file: one row per genotype, first column `Genotype`, remaining columns numeric dosage. Use one column per haplotype allele/category, not arbitrary integer labels.
3. Phenotype file: long format with `Genotype, Environment, Replication, Trait, Value`.
4. This starter assumes disomic dosage 0/1/2 for the basic kernel implementation. Cotton is allotetraploid; use validated subgenome-specific marker calls and correct dosage/ploidy coding. Do not coerce 0–4 dosages into 0–2.
5. Haplotype columns must have documented block identity and allele identity. For phased haplotypes, dosage should count copies of each haplotype at the relevant block.
6. Training and validation must be split by genotype/family as appropriate. Never split replicate plots from the same genotype across train and test.

## Repository structure

```text
haplotype_genomic_selection/
├── data/
│   ├── raw/
│   ├── processed/
│   └── templates/
├── models/
├── results/
├── python/
│   ├── config.py
│   ├── data_io.py
│   ├── preprocessing.py
│   ├── phenotype.py
│   ├── kernels.py
│   ├── models_ml.py
│   ├── models_dl.py
│   ├── validation.py
│   ├── train.py
│   ├── predict.py
│   └── api.py
├── java/
│   └── ...
├── requirements.txt
└── .gitignore
```

## 1. Setup (Ubuntu / WSL)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip openjdk-17-jdk maven git
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd haplotype_genomic_selection
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

PyTorch installation can vary by CPU/GPU. The requirements file includes a CPU-compatible installation note; if pip cannot resolve it, install PyTorch using the official selector at https://pytorch.org/get-started/locally/.

## 2. Prepare data

Copy your files into `data/raw/`:
- `snp_dosage.csv`
- `haplotype_dosage.csv`
- `field_phenotypes.csv`

Templates are in `data/templates/`. Replace the example rows with your data.

Run:
```bash
python -m python.preprocessing
python -m python.phenotype
```

The phenotype module creates adjusted genotype means per trait using a fixed-effects linear model as a starter. For complex designs, incomplete blocks, spatial effects or multi-environment MET analysis, replace this with an appropriate mixed model and retain standard errors/BLUEs/BLUPs.

## 3. Train and compare models

```bash
python -m python.train --trait Yield --folds 5 --repeats 3
python -m python.train_genomic --trait Yield --folds 5
```

Results are saved under `results/`. The script compares SNP, haplotype and integrated feature sets and runs grouped cross-validation. If family IDs are available, add them to the split strategy; random genotype folds can overestimate accuracy in related populations.

## 4. Predict new breeding lines

Prepare:
- `new_snp_dosage.csv`
- `new_haplotype_dosage.csv`

Columns and marker definitions must match the training data. Run:
```bash
python -m python.predict --trait Yield --model RandomForest --feature-set Integrated
python -m python.predict_genomic --trait Yield
```

The prediction workflow refits on all labelled training genotypes and predicts new lines. New-line GEBVs from the genomic kernel model are relative genetic deviations on the training target scale; predicted adjusted performance includes the fitted mean. Do not interpret an ML predicted phenotype as a GEBV without an appropriate genetic model.

## 5. Start Python prediction API

After training:
```bash
uvicorn python.api:app --host 0.0.0.0 --port 8000
```

## 6. Run Java client

```bash
cd java
mvn spring-boot:run
```

POST JSON to `http://localhost:8080/predict` with the format described in `java/README.md`. Java forwards the request to the Python API. Python performs model inference; Java manages application integration. This keeps one canonical model implementation and avoids silently reimplementing Python algorithms in Java.

## 7. Model interpretation and reporting

Report at least:
- Pearson correlation between observed and predicted values
- RMSE and MAE
- Regression slope/intercept (calibration)
- Fold/repeat variability and confidence intervals where possible
- Population/family structure of training and validation sets
- Trait heritability and phenotype-adjustment method
- Number of genotypes, SNPs, haplotype blocks and retained haplotypes
- Dosage/ploidy assumptions and missingness thresholds

A high correlation alone does not prove accurate GEBVs. Use independent validation, family-wise validation, forward-generation validation or external population validation when possible.

## Citation and reproducibility

Record software versions, random seeds, reference genome, variant caller, SNP filtering thresholds, phasing method, haplotype block method, phenotype model and all train/test partitions. Add relevant citations for algorithms and software in the final research manuscript.

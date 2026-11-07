from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

SNP_FILE = RAW / "snp_dosage.csv"
HAP_FILE = RAW / "haplotype_dosage.csv"
PHENO_FILE = RAW / "field_phenotypes.csv"

RANDOM_SEED = 2026
DOSAGE_MIN = 0
DOSAGE_MAX = 2  # Change only after confirming ploidy/dosage definition.

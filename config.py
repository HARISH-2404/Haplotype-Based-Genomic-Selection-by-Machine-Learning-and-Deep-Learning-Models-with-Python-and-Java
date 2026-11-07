"""Central configuration: paths, random seed, simulation and analysis parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # repository root
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_TRUTH = ROOT / "data" / "truth"
RES_TAB = ROOT / "results" / "tables"
RES_FIG = ROOT / "results" / "figures"
RES_LOG = ROOT / "results" / "logs"

SEED = 2026

# ---- simulation (cotton-like inbred population: 10 biparental F5 RIL families) ----
N_CHR = 26                    # Gossypium hirsutum: A01-A13 + D01-D13
SNPS_PER_CHR = 100
BLOCK_SIZE_RANGE = (4, 9)     # true simulated block size (SNPs)
N_FOUNDERS = 20
N_FAMILIES = 10
FAMILY_SIZE = 30              # => 300 lines
SWITCH_PROB = 0.15            # parent switch probability between adjacent blocks (F5 recombination)
MISSING_RATE = 0.015
GENOTYPING_ERROR = 0.005

TRAITS = {                    # name: (mean, sd, line-mean heritability, genetic correlation with common factor)
    "SCY": (120.0, 25.0, 0.40, 0.50),   # seed cotton yield (g/plant)
    "BW":  (3.6,   0.45, 0.60, 0.50),   # boll weight (g)
    "LP":  (36.0,  2.2,  0.75, 0.40),   # lint percentage (%)
    "FL":  (28.5,  1.6,  0.55, -0.30),  # 2.5% span fibre length (mm)
}
N_QTL_POOL = 60
N_QTL_PER_TRAIT = 35
N_EPI_PAIRS = 6
EPI_FRACTION = 0.20           # share of genetic variance that is epistatic (haplotype x haplotype)

# ---- quality control ----
MIN_MAF = 0.05
MAX_SNP_MISSING = 0.10
MAX_LINE_MISSING = 0.20

# ---- haplotype definition ----
LD_R2_THRESHOLD = 0.10          # tuned on simulated truth (see results/tables/ld_threshold_sensitivity.csv)
LD_MAX_BLOCK = 20
FIXED_WINDOW = 5
MIN_HAP_FREQ = 0.03           # rarer haplotypes are merged into the nearest common haplotype

# ---- cross-validation ----
N_FOLDS = 5
N_REPEATS = 3

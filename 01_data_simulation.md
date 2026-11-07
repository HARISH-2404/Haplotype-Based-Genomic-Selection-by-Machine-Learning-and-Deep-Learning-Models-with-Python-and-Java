# 01 · Data simulation (`hgs/simulate.py`)

Real cotton GS data are rarely public, so the repository **creates its own data** with a known truth. This lets every
step be validated (e.g. "did LD-block detection recover the real blocks?").

**Genome.** 26 chromosomes (A01–A13, D01–D13) × 100 SNPs = 2,600 SNPs, named `Gh_<chr>_<bp>`.
Each chromosome is cut into *true* haplotype blocks of 4–9 SNPs (411 blocks). Each block has 2–5 ancestral haplotype classes.

**Population.** 20 founders → 10 biparental families (random founder pairs, ≥45 % of blocks divergent) × 30 **F5 RILs** = **300 inbred lines**.
Each RIL is a mosaic of its two parents; the parent switches between adjacent blocks with probability 0.15 (≈ F5 recombination). Genotypes are
coded 0/2 (homozygous). 0.5 % genotyping errors and 1.5 % missing calls are added.

**Traits (line-mean basis).**

| Trait | Mean ± SD | h² |
|-------|-----------|----|
| SCY – seed cotton yield (g/plant) | 120 ± 25 | 0.40 |
| BW – boll weight (g) | 3.6 ± 0.45 | 0.60 |
| LP – lint percentage (%) | 36 ± 2.2 | 0.75 |
| FL – 2.5 % span length (mm) | 28.5 ± 1.6 | 0.55 |

Genetic value = additive **haplotype-class effects** at 35 QTL blocks per trait (drawn from a pool of 60 shared QTL blocks, so traits are
correlated) + haplotype×haplotype **epistasis** (6 pairs, 20 % of genetic variance) → scaled to the target h², plus Gaussian error.

**Files** — `data/raw/` : `genotypes_snp.csv`, `snp_info.csv`, `lines.csv`, `phenotypes.csv`.
`data/truth/` (never used for model fitting): `true_blocks.csv`, `true_snp_blocks.csv`, `true_breeding_values.csv`, `true_haplotype_classes.csv`, `true_haplotype_effects.csv`.

> To use **real data**, replace the four files in `data/raw/` with the same column layout (see `08_real_data_guide.md`) and skip step 1.
> Steps that use the truth (block-recovery metrics, `r_tbv`) then need to be disabled.

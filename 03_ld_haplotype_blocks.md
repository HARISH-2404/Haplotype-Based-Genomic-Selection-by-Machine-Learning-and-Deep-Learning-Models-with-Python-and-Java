# 03 · LD and haplotype blocks (`hgs/haplotypes.py`, Java `LdBlocks`, `Haplotypes`)

**LD measure.** Pairwise r² between SNPs of the same chromosome.

**LD block algorithm (greedy).** Start a block at the first unassigned SNP; add the next SNP while its *mean r² with all SNPs already
in the block* ≥ threshold (default 0.10) and the block has < 20 SNPs; otherwise close the block. Single SNPs that are not in LD with their
neighbour become singleton blocks.

**Threshold choice.** On simulated data the truth is known, so the threshold is tuned by *boundary F1* and *adjusted Rand index* (`ld_threshold_sensitivity.csv`).
On real data, run the sensitivity analysis and pick the threshold by cross-validated prediction accuracy, or use PLINK/Haploview (Gabriel et al.) blocks.

**Haplotype alleles.** Within each block, each line's 0/1 SNP string is its haplotype. Haplotypes with frequency < 3 % are merged into the
Hamming-nearest common haplotype (stabilises estimates). The result is a lines × blocks matrix of haplotype-allele ids.

**Encoding for models.** One-hot (haplotype-allele dosage): each haplotype allele becomes a 0/1 column (1 if the line carries it).
A block with *k* alleles contributes *k* columns, so linear models estimate a separate effect for every haplotype allele — multi-allelic
information that a biallelic SNP cannot carry.

**Baselines.** `HAP_FIXED` (5-SNP windows) ignores LD structure; `SNP` treats every marker independently.

import numpy as np
from .data_io import load_all
from .config import PROCESSED, DOSAGE_MIN, DOSAGE_MAX

def qc_matrix(X, name):
    X = X.copy()
    # Remove all-missing columns and monomorphic columns.
    keep = X.notna().any(axis=0) & (X.nunique(dropna=True) > 1)
    X = X.loc[:, keep]
    if X.shape[1] == 0:
        raise ValueError(f"No polymorphic markers remain in {name}")
    vals = X.to_numpy(dtype=float)
    observed = vals[~np.isnan(vals)]
    if observed.size and (observed.min() < DOSAGE_MIN or observed.max() > DOSAGE_MAX):
        raise ValueError(
            f"{name} dosage outside configured range [{DOSAGE_MIN},{DOSAGE_MAX}]. "
            "Check cotton ploidy/subgenome dosage before proceeding."
        )
    return X

def main():
    snp, hap, pheno = load_all()
    snp = qc_matrix(snp, "SNP")
    hap = qc_matrix(hap, "haplotype")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    snp.to_csv(PROCESSED / "snp_qc.csv", index_label="Genotype")
    hap.to_csv(PROCESSED / "hap_qc.csv", index_label="Genotype")
    pheno.to_csv(PROCESSED / "phenotypes_raw.csv", index=False)
    print(f"SNP matrix: {snp.shape}; haplotype matrix: {hap.shape}")
    print(f"Phenotype rows: {len(pheno)}")
    print("QC complete. Missing-value imputation is learned inside each CV training fold.")

if __name__ == "__main__":
    main()

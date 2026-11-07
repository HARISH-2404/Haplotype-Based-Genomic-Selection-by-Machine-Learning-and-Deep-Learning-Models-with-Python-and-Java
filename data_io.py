import pandas as pd
from .config import SNP_FILE, HAP_FILE, PHENO_FILE

def read_marker_matrix(path):
    df = pd.read_csv(path)
    if "Genotype" not in df.columns:
        raise ValueError(f"{path}: first column must be named Genotype")
    if df["Genotype"].duplicated().any():
        raise ValueError(f"{path}: duplicate genotype IDs")
    ids = df["Genotype"].astype(str)
    X = df.drop(columns="Genotype").apply(pd.to_numeric, errors="coerce")
    X.index = ids
    if X.shape[1] == 0:
        raise ValueError(f"{path}: no marker columns found")
    return X

def load_all():
    snp = read_marker_matrix(SNP_FILE)
    hap = read_marker_matrix(HAP_FILE)
    pheno = pd.read_csv(PHENO_FILE)
    required = {"Genotype", "Environment", "Replication", "Trait", "Value"}
    missing = required - set(pheno.columns)
    if missing:
        raise ValueError(f"Phenotype file missing columns: {sorted(missing)}")
    pheno["Genotype"] = pheno["Genotype"].astype(str)
    pheno["Value"] = pd.to_numeric(pheno["Value"], errors="coerce")
    if not snp.index.equals(hap.index):
        if set(snp.index) != set(hap.index):
            raise ValueError("SNP and haplotype files must contain identical genotype IDs")
        hap = hap.loc[snp.index]
    unknown = set(pheno["Genotype"]) - set(snp.index)
    if unknown:
        raise ValueError(f"Phenotypes contain unknown genotype IDs: {list(unknown)[:5]}")
    return snp, hap, pheno

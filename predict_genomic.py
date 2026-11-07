import argparse
import joblib
import pandas as pd
from .config import RAW, MODELS, RESULTS
from .data_io import read_marker_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", required=True)
    args = ap.parse_args()
    bundle = joblib.load(MODELS / f"{args.trait}_multikernel_gs.joblib")
    snp = read_marker_matrix(RAW / "new_snp_dosage.csv")
    hap = read_marker_matrix(RAW / "new_haplotype_dosage.csv")
    if set(snp.index) != set(hap.index):
        raise ValueError("New SNP and haplotype IDs differ.")
    hap = hap.loc[snp.index]
    if list(snp.columns) != bundle["snp_features"]:
        raise ValueError("SNP features/order differ from training.")
    if list(hap.columns) != bundle["hap_features"]:
        raise ValueError("Haplotype features/order differ from training.")
    result = bundle["model"].predict_components(snp, hap)
    out = pd.DataFrame({"Genotype": snp.index, **result})
    out = out.sort_values("genomic_deviation", ascending=False)
    RESULTS.mkdir(exist_ok=True, parents=True)
    out.to_csv(RESULTS / f"{args.trait}_new_line_genomic_predictions.csv", index=False)
    print(out.head(20))
if __name__ == "__main__":
    main()

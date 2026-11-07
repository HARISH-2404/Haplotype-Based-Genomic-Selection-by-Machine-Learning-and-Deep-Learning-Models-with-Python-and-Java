import argparse
import joblib
import pandas as pd
from .config import RAW, PROCESSED, MODELS, RESULTS
from .data_io import read_marker_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", required=True)
    parser.add_argument("--model", default="RandomForest")
    parser.add_argument("--feature-set", choices=["SNP","Haplotype","Integrated"],
                        default="Integrated")
    args = parser.parse_args()

    snp_new = read_marker_matrix(RAW / "new_snp_dosage.csv")
    hap_new = read_marker_matrix(RAW / "new_haplotype_dosage.csv")
    if set(snp_new.index) != set(hap_new.index):
        raise ValueError("New SNP and haplotype files must contain same line IDs.")
    hap_new = hap_new.loc[snp_new.index]

    if args.feature_set == "SNP":
        Xnew = snp_new
    elif args.feature_set == "Haplotype":
        Xnew = hap_new
    else:
        Xnew = pd.concat([snp_new, hap_new], axis=1)

    path = MODELS / f"{args.trait}_{args.feature_set}_{args.model}.joblib"
    bundle = joblib.load(path)
    expected = bundle["features"]
    missing = set(expected) - set(Xnew.columns)
    if missing:
        raise ValueError(f"New data missing {len(missing)} trained features.")
    Xnew = Xnew.loc[:, expected]
    pred = bundle["estimator"].predict(Xnew)
    out = pd.DataFrame({"Genotype": Xnew.index, "Predicted_adjusted_performance": pred})
    RESULTS.mkdir(exist_ok=True, parents=True)
    out.to_csv(RESULTS / f"{args.trait}_{args.feature_set}_{args.model}_new_predictions.csv",
               index=False)
    print(out.sort_values("Predicted_adjusted_performance", ascending=False).head(20))

if __name__ == "__main__":
    main()

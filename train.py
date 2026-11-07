import argparse
import joblib
import pandas as pd
from .config import PROCESSED, RESULTS, MODELS, RANDOM_SEED
from .data_io import read_marker_matrix
from .validation import feature_sets, run_cv
from .models_ml import get_models, get_optional_models
from .models_dl import TorchRegressor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    snp = read_marker_matrix(PROCESSED / "snp_qc.csv")
    hap = read_marker_matrix(PROCESSED / "hap_qc.csv")
    ph = pd.read_csv(RESULTS / "adjusted_genotype_means.csv")
    ph = ph[ph.Trait == args.trait].dropna(subset=["Adjusted_Value"]).set_index("Genotype")
    ids = snp.index.intersection(hap.index).intersection(ph.index)
    snp, hap, y = snp.loc[ids], hap.loc[ids], ph.loc[ids, "Adjusted_Value"]
    sets = feature_sets(snp, hap)
    RESULTS.mkdir(exist_ok=True, parents=True)
    MODELS.mkdir(exist_ok=True, parents=True)

    all_metrics, all_predictions = [], []
    for set_name, X in sets.items():
        metrics_df, pred_df = run_cv(X, y, args.folds, args.repeats, RANDOM_SEED)
        metrics_df.insert(0, "FeatureSet", set_name)
        pred_df.insert(0, "FeatureSet", set_name)
        all_metrics.append(metrics_df)
        all_predictions.append(pred_df)

        # Fit final models on all labelled genotypes for later inference.
        candidates = {**get_models(RANDOM_SEED), **get_optional_models(RANDOM_SEED)}
        for model_name, estimator in candidates.items():
            estimator.fit(X, y)
            joblib.dump(
                {"estimator": estimator, "features": list(X.columns),
                 "trait": args.trait, "feature_set": set_name},
                MODELS / f"{args.trait}_{set_name}_{model_name}.joblib"
            )
        dl = TorchRegressor().fit(X, y)
        joblib.dump(
            {"estimator": dl, "features": list(X.columns),
             "trait": args.trait, "feature_set": set_name},
            MODELS / f"{args.trait}_{set_name}_MLP.joblib"
        )

    m = pd.concat(all_metrics, ignore_index=True)
    p = pd.concat(all_predictions, ignore_index=True)
    m.to_csv(RESULTS / f"{args.trait}_cv_metrics.csv", index=False)
    p.to_csv(RESULTS / f"{args.trait}_cv_predictions.csv", index=False)
    print(m.groupby(["FeatureSet", "Model"])[["Correlation","RMSE","MAE"]].mean()
            .sort_values("RMSE"))
    print("Training and CV complete.")

if __name__ == "__main__":
    main()

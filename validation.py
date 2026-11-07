import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from .models_ml import get_models, get_optional_models
from .models_dl import TorchRegressor

def metrics(y, pred):
    r = pearsonr(y, pred).statistic if np.std(pred) > 0 and np.std(y) > 0 else np.nan
    slope, intercept = np.polyfit(pred, y, 1) if np.std(pred) > 0 else (np.nan, np.nan)
    return {
        "Correlation": r,
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "Calibration_slope_observed_on_predicted": float(slope),
        "Calibration_intercept": float(intercept)
    }

def feature_sets(snp, hap):
    common = snp.index.intersection(hap.index)
    snp, hap = snp.loc[common], hap.loc[common]
    return {
        "SNP": snp,
        "Haplotype": hap,
        "Integrated": pd.concat([snp, hap], axis=1)
    }

def run_cv(X, y, folds=5, repeats=2, seed=2026, include_dl=True):
    rows, pred_rows = [], []
    for rep in range(repeats):
        cv = KFold(n_splits=folds, shuffle=True, random_state=seed + rep)
        models = {**get_models(seed + rep), **get_optional_models(seed + rep)}
        if include_dl:
            models["MLP"] = None
        for model_name, model in models.items():
            all_obs, all_pred = [], []
            for fold, (tr, te) in enumerate(cv.split(X), start=1):
                Xtr, Xte = X.iloc[tr], X.iloc[te]
                ytr, yte = y.iloc[tr], y.iloc[te]
                if model_name == "MLP":
                    estimator = TorchRegressor(seed=seed + rep + fold)
                else:
                    estimator = model
                estimator.fit(Xtr, ytr)
                pred = estimator.predict(Xte)
                all_obs.extend(yte.tolist())
                all_pred.extend(np.asarray(pred).tolist())
                for gid, obs, pr in zip(Xte.index, yte, pred):
                    pred_rows.append({"Repeat": rep+1, "Fold": fold,
                                      "Model": model_name, "Genotype": gid,
                                      "Observed": obs, "Predicted": float(pr)})
            result = metrics(np.asarray(all_obs), np.asarray(all_pred))
            rows.append({"Repeat": rep+1, "Model": model_name, **result})
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)

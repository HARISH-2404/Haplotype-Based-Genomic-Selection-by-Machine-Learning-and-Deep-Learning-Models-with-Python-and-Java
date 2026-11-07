from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR

def get_models(seed=2026):
    return {
        "Ridge": make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), Ridge(alpha=10.0)),
        "ElasticNet": make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                                    ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000)),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                  max_features=0.5, random_state=seed, n_jobs=-1)),
        "ExtraTrees": make_pipeline(SimpleImputer(strategy="median"),
            ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                max_features=0.5, random_state=seed, n_jobs=-1)),
        "SVR_RBF": make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                                 SVR(C=10, epsilon=0.1, kernel="rbf")),
        "HistGradientBoosting": make_pipeline(SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=seed))
    }

def get_optional_models(seed=2026):
    models = {}
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8,
                         objective="reg:squarederror", random_state=seed, n_jobs=-1)
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMRegressor(n_estimators=500, learning_rate=0.03,
                          num_leaves=15, random_state=seed, verbosity=-1)
        )
    except ImportError:
        pass
    return models

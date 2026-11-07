"""Model zoo: classical genomic-prediction models, machine learning and deep learning."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LassoCV, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .cnn_numpy import WindowCNNRegressor


class Centered(BaseEstimator):
    """Column-centering transformer (marker-effect models use un-scaled, centred codes)."""
    def fit(self, X, y=None):
        self.mu_ = np.asarray(X).mean(0)
        return self

    def transform(self, X):
        return np.asarray(X) - self.mu_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class ScaledTargetSVR(BaseEstimator, RegressorMixin):
    """RBF-SVR with the response standardised (epsilon/C then have a trait-independent meaning)."""
    def __init__(self, C=3.0, epsilon=0.1):
        self.C, self.epsilon = C, epsilon

    def fit(self, X, y):
        self.m_, self.s_ = np.mean(y), np.std(y) or 1.0
        self.svr_ = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=self.C, epsilon=self.epsilon, gamma="scale"))
        self.svr_.fit(X, (y - self.m_) / self.s_)
        return self

    def predict(self, X):
        return self.svr_.predict(X) * self.s_ + self.m_


def get_models(seed=0, fast=False):
    """Return {name: (family, estimator)}."""
    models = {
        "RR-BLUP":  ("linear",   make_pipeline(Centered(), RidgeCV(alphas=np.logspace(0, 6, 25)))),
        "BayesRidge": ("linear", make_pipeline(Centered(), BayesianRidge(max_iter=300))),
        "LASSO":    ("linear",   make_pipeline(Centered(), LassoCV(cv=3, alphas=20, max_iter=3000, random_state=seed))),
        "SVR":      ("ML",       ScaledTargetSVR()),
        "RandomForest": ("ML",   RandomForestRegressor(n_estimators=100, max_features=0.2, min_samples_leaf=3,
                                                       n_jobs=1, random_state=seed)),
        "GradBoost": ("ML",      HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3,
                                                            l2_regularization=1.0, early_stopping=False, random_state=seed)),
        "MLP":      ("DL",       make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1.0,
                                  early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                                  max_iter=400, learning_rate_init=1e-3, random_state=seed))),
        "DeepMLP":  ("DL",       make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 128, 64), alpha=1.0,
                                  early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                                  max_iter=400, learning_rate_init=1e-3, random_state=seed))),
        "WindowCNN": ("DL",      WindowCNNRegressor(k=8, filters=8, hidden=32, l2=1e-2, epochs=200, patience=25, seed=seed)),
    }
    return models

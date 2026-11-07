import numpy as np
import pandas as pd

class DosageKernel:
    def __init__(self):
        self.p_ = None
        self.keep_ = None
        self.scale_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, train_ids=None):
        X = X.copy()
        self.feature_names_ = list(X.columns)
        if train_ids is not None:
            Xfit = X.loc[list(train_ids)]
        else:
            Xfit = X
        self.p_ = Xfit.mean(axis=0, skipna=True).to_numpy(dtype=float) / 2.0
        self.keep_ = np.isfinite(self.p_) & (self.p_ > 0) & (self.p_ < 1)
        self.p_ = self.p_[self.keep_]
        self.feature_names_ = list(np.asarray(self.feature_names_)[self.keep_])
        if len(self.p_) == 0:
            raise ValueError("No polymorphic features in training fold.")
        self.scale_ = 2.0 * np.sum(self.p_ * (1.0 - self.p_))
        return self

    def transform(self, X: pd.DataFrame):
        X = X.loc[:, self.feature_names_].copy()
        a = X.to_numpy(dtype=float)
        means = 2.0 * self.p_
        inds = np.where(np.isnan(a))
        a[inds] = np.take(means, inds[1])
        return a - means

    def kernel(self, X1, X2=None):
        A = self.transform(X1)
        B = A if X2 is None else self.transform(X2)
        return (A @ B.T) / self.scale_

def build_kernel(X_train, X_all=None):
    # Convenience function. Fit allele frequencies on training rows only.
    obj = DosageKernel().fit(X_train)
    if X_all is None:
        X_all = X_train
    K = obj.kernel(X_all)
    return K, obj

"""Window-CNN regressor written from scratch in NumPy (no deep-learning framework needed).

Architecture ("haplotype-window CNN")
    input  : standardised SNP / haplotype columns, split into non-overlapping windows of k columns
    conv   : one shared 1-D convolution (kernel = stride = k) learns F local "haplotype motifs" per window
    dense  : flatten -> Dense(D, ReLU) -> Dense(1)
Training: mini-batch Adam, L2 weight decay, inverted dropout on the conv output, early stopping on an
internal validation split. Because kernel == stride the convolution is a reshape + matrix product, so the
back-propagation below is exact (verified by finite-difference gradient checking in tests/test_cnn_gradients.py).
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class WindowCNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, k=8, filters=8, hidden=32, l2=1e-2, lr=2e-3, epochs=300, batch=32,
                 patience=30, dropout=0.2, val_frac=0.15, seed=0):
        self.k, self.filters, self.hidden, self.l2, self.lr = k, filters, hidden, l2, lr
        self.epochs, self.batch, self.patience, self.dropout = epochs, batch, patience, dropout
        self.val_frac, self.seed = val_frac, seed

    # ---------- helpers ----------
    def _prep(self, X):
        X = (X - self.mu_) / self.sd_
        pad = (-X.shape[1]) % self.k
        if pad:
            X = np.pad(X, ((0, 0), (0, pad)))
        return X.reshape(X.shape[0], -1, self.k)

    def _init(self, M, rng):
        k, F, D = self.k, self.filters, self.hidden
        self.P_ = dict(
            W1=rng.normal(0, np.sqrt(2.0 / k), (k, F)), b1=np.zeros(F),
            W2=rng.normal(0, np.sqrt(2.0 / (M * F)), (M * F, D)), b2=np.zeros(D),
            W3=rng.normal(0, np.sqrt(2.0 / D), (D, 1)), b3=np.zeros(1))

    def _forward(self, Xw, P, train=False, rng=None):
        n, M, _ = Xw.shape
        Z1 = np.einsum("nmk,kf->nmf", Xw, P["W1"]) + P["b1"]
        A1 = np.maximum(Z1, 0)
        A1f = A1.reshape(n, -1)
        mask = None
        if train and self.dropout > 0:
            mask = (rng.random(A1f.shape) >= self.dropout) / (1 - self.dropout)
            A1f = A1f * mask
        Z2 = A1f @ P["W2"] + P["b2"]
        A2 = np.maximum(Z2, 0)
        out = (A2 @ P["W3"] + P["b3"]).ravel()
        return out, (Xw, Z1, A1f, Z2, A2, mask)

    def _grads(self, y, out, cache, P):
        Xw, Z1, A1f, Z2, A2, mask = cache
        n = len(y)
        d_out = ((out - y) / n)[:, None]
        G = {}
        G["W3"] = A2.T @ d_out + self.l2 * P["W3"]
        G["b3"] = d_out.sum(0)
        dZ2 = (d_out @ P["W3"].T) * (Z2 > 0)
        G["W2"] = A1f.T @ dZ2 + self.l2 * P["W2"]
        G["b2"] = dZ2.sum(0)
        dA1f = dZ2 @ P["W2"].T
        if mask is not None:
            dA1f = dA1f * mask
        dZ1 = dA1f.reshape(Z1.shape) * (Z1 > 0)
        G["W1"] = np.einsum("nmk,nmf->kf", Xw, dZ1) + self.l2 * P["W1"]
        G["b1"] = dZ1.sum((0, 1))
        return G

    # ---------- sklearn API ----------
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu_, self.sd_ = X.mean(0), X.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        self.ym_, self.ys_ = y.mean(), y.std() or 1.0
        Xw, yz = self._prep(X), (y - self.ym_) / self.ys_
        n = len(yz)
        perm = rng.permutation(n)
        nv = max(int(n * self.val_frac), 1)
        va, tr = perm[:nv], perm[nv:]
        self._init(Xw.shape[1], rng)
        P = self.P_
        m = {k: np.zeros_like(v) for k, v in P.items()}
        v = {k: np.zeros_like(x) for k, x in P.items()}
        best, best_P, wait, t = np.inf, {k: x.copy() for k, x in P.items()}, 0, 0
        self.history_ = []
        for ep in range(self.epochs):
            order = rng.permutation(tr)
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                out, cache = self._forward(Xw[idx], P, True, rng)
                G = self._grads(yz[idx], out, cache, P)
                t += 1
                for kname in P:
                    m[kname] = 0.9 * m[kname] + 0.1 * G[kname]
                    v[kname] = 0.999 * v[kname] + 0.001 * G[kname] ** 2
                    mh, vh = m[kname] / (1 - 0.9 ** t), v[kname] / (1 - 0.999 ** t)
                    P[kname] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
            tr_loss = float(np.mean((self._forward(Xw[tr], P)[0] - yz[tr]) ** 2))
            va_loss = float(np.mean((self._forward(Xw[va], P)[0] - yz[va]) ** 2))
            self.history_.append((ep + 1, tr_loss, va_loss))
            if va_loss < best - 1e-5:
                best, best_P, wait = va_loss, {k: x.copy() for k, x in P.items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.P_ = best_P
        return self

    def predict(self, X):
        Xw = self._prep(np.asarray(X, float))
        return self._forward(Xw, self.P_)[0] * self.ys_ + self.ym_

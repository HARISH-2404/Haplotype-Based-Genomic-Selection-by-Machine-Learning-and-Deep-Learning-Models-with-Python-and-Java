"""Finite-difference gradient check for the NumPy WindowCNN (run: python tests/test_cnn_gradients.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from hgs.cnn_numpy import WindowCNNRegressor

rng = np.random.default_rng(1)
X = rng.normal(size=(12, 20)); y = rng.normal(size=12)
m = WindowCNNRegressor(k=4, filters=3, hidden=5, l2=0.05, dropout=0.0)
m.mu_, m.sd_ = np.zeros(20), np.ones(20)
Xw = m._prep(X); m._init(Xw.shape[1], rng); P = m.P_

def loss(P):
    out, _ = m._forward(Xw, P)
    reg = 0.5 * m.l2 * sum((P[k] ** 2).sum() for k in ("W1", "W2", "W3"))
    return 0.5 * np.mean((out - y) ** 2) + reg

out, cache = m._forward(Xw, P)
G = m._grads(y, out, cache, P)
worst = 0.0
for name in P:
    flat = P[name].ravel()
    for i in rng.choice(flat.size, min(6, flat.size), replace=False):
        old = flat[i]; flat[i] = old + 1e-6; lp = loss(P); flat[i] = old - 1e-6; lm = loss(P); flat[i] = old
        num = (lp - lm) / 2e-6; ana = G[name].ravel()[i]
        worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
print("max relative gradient error:", worst)
assert worst < 1e-5, "gradient check failed"
print("PASS")

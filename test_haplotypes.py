"""Unit tests for QC / haplotype utilities.  Run:  python tests/test_haplotypes.py   (or: pytest python/tests)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from hgs import haplotypes as H, qc


def test_ld_blocks_split_uncorrelated_regions():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 200) * 2.0
    b = rng.integers(0, 2, 200) * 2.0
    G = np.column_stack([a, a, a, b, b, b])                 # two perfectly correlated triplets
    blk = H.detect_ld_blocks(G, ["c1"] * 6, r2_threshold=0.5)
    assert blk.tolist() == [0, 0, 0, 1, 1, 1], blk


def test_rare_haplotypes_are_merged():
    base = np.zeros((100, 4), dtype=int)
    base[:50, :] = 1                                         # two common haplotypes 0000 / 1111
    base[0, :] = [1, 1, 1, 0]                                # one rare line -> merged into 1111
    ids, n_all, strings = H.block_haplotypes(base, np.zeros(4, dtype=int), min_freq=0.05)
    assert n_all[0] == 2 and len(strings[0]) == 2


def test_onehot_shape():
    ids = np.array([[0, 1], [1, 0], [0, 0]])
    X, names = H.onehot_haplotypes(ids, np.array([2, 2]))
    assert X.shape == (3, 4) and X.sum(1).tolist() == [2, 2, 2]


def test_nn_impute_fills_everything_and_copies_closest_line():
    G = np.array([[0, 0, 2, 2.0], [0, 0, 2, np.nan], [2, 2, 0, 0]])
    out = qc.nn_impute(G)
    assert not np.isnan(out).any() and out[1, 3] == 2.0


def test_fixed_windows():
    assert H.fixed_window_blocks(["a"] * 7 + ["b"] * 3, 5).tolist() == [0] * 5 + [1] * 2 + [2] * 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)

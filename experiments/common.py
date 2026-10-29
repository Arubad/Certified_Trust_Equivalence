import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tec.mlp import MLP  # noqa: E402
from tec.data import load_digits, make_toxic  # noqa: E402
from tec.compress import magnitude_prune, prune_finetune, quantize, low_rank, distill, size_bits  # noqa: E402
from tec.certify import margin_score  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def save_csv(df, name):
    p = os.path.join(RESULTS, name)
    df.to_csv(p, index=False)
    print(f"[saved] {os.path.relpath(p)}")


def fig_path(name):
    return os.path.join(RESULTS, name)


def load_task(task, seed=0, spurious=0.8):
    """Returns train (X,y,g), pool (X,y,g). The pool is later split at random
    into calibration and test sets by each experiment."""
    rng = np.random.default_rng(seed)
    if task == "digits":
        X, y = load_digits(return_X_y=True); X = X / 16.0
        idx = rng.permutation(len(X)); ntr = int(0.5 * len(X))
        tr, po = idx[:ntr], idx[ntr:]
        return (X[tr], y[tr], np.zeros(ntr, int)), (X[po], y[po], np.zeros(len(po), int))
    Xtr, ytr, gtr = make_toxic(6000, spurious=spurious, seed=seed)
    Xpo, ypo, gpo = make_toxic(12000, spurious=0.5, seed=seed + 100)   # unbiased pool
    return (Xtr, ytr, gtr), (Xpo, ypo, gpo)


def train_teacher(task, Xtr, ytr, seed=0, hidden=256):
    sizes = [Xtr.shape[1], hidden, hidden, int(ytr.max()) + 1]
    return MLP(sizes, seed=seed).fit(Xtr, ytr, epochs=40, lr=1e-3, seed=seed)


def compress_all(teacher, Xtr, ytr, task, seed=0):
    """Standard operating points used throughout."""
    base = size_bits(teacher)
    h = 64 if task == "digits" else 64
    out = {
        "prune80": magnitude_prune(teacher, 0.8),
        "prune90": magnitude_prune(teacher, 0.9),
        "pruneFT90": prune_finetune(teacher, 0.9, Xtr, ytr, seed=seed),
        "quant3b": quantize(teacher, 3),
        "quant4b": quantize(teacher, 4),
        "lowrank25": low_rank(teacher, 0.25),
        "distill25": distill(teacher, Xtr, ytr, [Xtr.shape[1], h, h, int(ytr.max()) + 1], seed=seed),
    }
    sizes = {k: (size_bits(m, bits=3) if k == "quant3b" else size_bits(m, bits=4) if k == "quant4b" else size_bits(m)) / base
             for k, m in out.items()}
    return out, sizes


def disagreement(teacher, model, X):
    return (teacher.predict(X) != model.predict(X)).astype(int)


def split_pool(n, n_cal, rng):
    idx = rng.permutation(n)
    return idx[:n_cal], idx[n_cal:]

"""Post-training compression operators on an MLP.

* ``magnitude_prune(model, sparsity)``  -- global unstructured magnitude pruning
* ``quantize(model, bits)``             -- symmetric uniform weight quantisation
                                           (simulated: weights snapped to a grid)
* ``low_rank(model, rank_frac)``        -- truncated SVD of each weight matrix
* ``distill(teacher, X, student_sizes)``-- train a smaller student on teacher
                                           soft labels (KD, Hinton et al. 2015)
* ``prune_finetune``                    -- prune then fine-tune with masks fixed

All operators return a *new* model; the original is untouched. ``size_ratio``
reports the compressed size relative to the teacher in bits.
"""
from __future__ import annotations

import numpy as np

from .mlp import MLP


def magnitude_prune(model: MLP, sparsity: float) -> MLP:
    m = model.copy()
    allw = np.concatenate([np.abs(w[k != 0]).ravel() for w, k in zip(m.W, m.masks)])
    thr = np.quantile(allw, sparsity) if sparsity > 0 else -1
    for i in range(len(m.W)):
        m.masks[i] = (np.abs(m.W[i]) > thr).astype(float) * m.masks[i]
        m.W[i] *= m.masks[i]
    return m


def prune_finetune(model: MLP, sparsity: float, X, Y, epochs=5, lr=5e-4, seed=0) -> MLP:
    m = magnitude_prune(model, sparsity)
    m.fit(X, Y, epochs=epochs, lr=lr, seed=seed)
    return m


def quantize(model: MLP, bits: int, per_channel: bool = True) -> MLP:
    m = model.copy()
    levels = 2 ** (bits - 1) - 1
    for i in range(len(m.W)):
        W = m.W[i]
        scale = (np.abs(W).max(axis=0, keepdims=True) if per_channel else np.abs(W).max()) / levels
        scale = np.where(scale == 0, 1.0, scale)
        m.W[i] = np.clip(np.round(W / scale), -levels, levels) * scale
    return m


def low_rank(model: MLP, rank_frac: float) -> MLP:
    m = model.copy()
    for i in range(len(m.W)):
        W = m.W[i]
        r = max(1, int(round(rank_frac * min(W.shape))))
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        m.W[i] = (U[:, :r] * s[:r]) @ Vt[:r]
    m._rank_frac = rank_frac
    return m


def distill(teacher: MLP, X, Y, student_sizes, epochs=30, T=2.0, lr=1e-3, seed=0, alpha=0.9) -> MLP:
    """KD: loss = alpha*KL(teacher_T || student_T) + (1-alpha)*CE(hard). We
    implement the mixture by interpolating targets."""
    from .mlp import _softmax
    soft = _softmax(teacher.logits(X) / T)
    hard = np.eye(student_sizes[-1])[Y]
    targets = alpha * soft + (1 - alpha) * hard
    s = MLP(student_sizes, seed=seed)
    s.fit(X, Y, epochs=epochs, lr=lr, seed=seed, soft_targets=targets, kd_T=T)
    return s


def size_bits(model: MLP, bits=32):
    """Effective storage of the model in bits (pruned weights excluded; low
    rank stored as factors)."""
    rf = getattr(model, "_rank_frac", None)
    if rf is not None:
        tot = 0
        for W in model.W:
            r = max(1, int(round(rf * min(W.shape))))
            tot += r * (W.shape[0] + W.shape[1])
        return tot * bits + sum(b.size for b in model.b) * bits
    return model.n_params() * bits


def compression_suite(teacher: MLP, X_tr, Y_tr, seed=0):
    """Return a dict name -> (model, size_ratio) covering the usual operating
    points of every compression family."""
    base = size_bits(teacher)
    out = {}
    for s in [0.5, 0.8, 0.9, 0.95]:
        m = magnitude_prune(teacher, s); out[f"prune{int(s*100)}"] = (m, size_bits(m) / base)
    for s in [0.8, 0.9, 0.95]:
        m = prune_finetune(teacher, s, X_tr, Y_tr, seed=seed); out[f"pruneFT{int(s*100)}"] = (m, size_bits(m) / base)
    for b in [8, 4, 3, 2]:
        m = quantize(teacher, b); out[f"quant{b}b"] = (m, size_bits(m, bits=b) / base)
    for r in [0.5, 0.25, 0.1]:
        m = low_rank(teacher, r); out[f"lowrank{int(r*100)}"] = (m, size_bits(m) / base)
    h = teacher.sizes[1]
    for frac in [0.5, 0.25, 0.1]:
        sizes = [teacher.sizes[0]] + [max(4, int(h * frac))] * (len(teacher.sizes) - 2) + [teacher.sizes[-1]]
        m = distill(teacher, X_tr, Y_tr, sizes, seed=seed); out[f"distill{int(frac*100)}"] = (m, size_bits(m) / base)
    return out

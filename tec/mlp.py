"""A small, dependency-free multilayer perceptron (numpy) with:

* softmax cross-entropy training with Adam,
* soft-label (knowledge-distillation) training,
* direct access to weight matrices so that pruning / quantisation /
  low-rank factorisation can be applied post hoc,
* temperature-scaled probabilities and a parameter counter.

Keeping the model in numpy makes every compression operator transparent and
reproducible on any machine without a deep-learning framework.
"""
from __future__ import annotations

import numpy as np


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes, seed=0, activation="relu"):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W, self.b = [], []
        for a, b_ in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2 / a), (a, b_)))
            self.b.append(np.zeros(b_))
        self.masks = [np.ones_like(w) for w in self.W]
        self.activation = activation
        self.temperature = 1.0

    # ------------------------------------------------------------------
    def _act(self, z):
        return np.maximum(z, 0) if self.activation == "relu" else np.tanh(z)

    def _dact(self, z, a):
        return (z > 0).astype(float) if self.activation == "relu" else 1 - a ** 2

    def forward(self, X, keep=False):
        a = X
        zs, acts = [], [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ (W * self.masks[i]) + b
            zs.append(z)
            a = self._act(z) if i < len(self.W) - 1 else z
            acts.append(a)
        if keep:
            return zs, acts
        return a  # logits

    def logits(self, X):
        return self.forward(X)

    def predict_proba(self, X):
        return _softmax(self.logits(X) / self.temperature)

    def predict(self, X):
        return self.logits(X).argmax(1)

    # ------------------------------------------------------------------
    def fit(self, X, Y, epochs=30, lr=1e-3, batch=64, weight_decay=1e-4, seed=0,
            soft_targets=None, kd_T=2.0, verbose=False):
        """Y: int labels. If soft_targets (N x C probs) is given the loss is
        KD cross-entropy against them at temperature kd_T (Hinton et al.)."""
        rng = np.random.default_rng(seed)
        C = self.sizes[-1]
        Yoh = np.eye(C)[Y]
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        n = len(X)
        for ep in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                xb = X[idx]
                zs, acts = self.forward(xb, keep=True)
                if soft_targets is not None:
                    p = _softmax(zs[-1] / kd_T)
                    grad = (p - soft_targets[idx]) / len(idx) * (kd_T ** 2)
                else:
                    p = _softmax(zs[-1])
                    grad = (p - Yoh[idx]) / len(idx)
                step += 1
                for i in reversed(range(len(self.W))):
                    gW = acts[i].T @ grad + weight_decay * self.W[i]
                    gb = grad.sum(0)
                    gW *= self.masks[i]
                    mW[i] = b1 * mW[i] + (1 - b1) * gW; vW[i] = b2 * vW[i] + (1 - b2) * gW ** 2
                    mb[i] = b1 * mb[i] + (1 - b1) * gb; vb[i] = b2 * vb[i] + (1 - b2) * gb ** 2
                    mhat = mW[i] / (1 - b1 ** step); vhat = vW[i] / (1 - b2 ** step)
                    self.W[i] -= lr * mhat / (np.sqrt(vhat) + eps)
                    self.b[i] -= lr * (mb[i] / (1 - b1 ** step)) / (np.sqrt(vb[i] / (1 - b2 ** step)) + eps)
                    self.W[i] *= self.masks[i]
                    if i > 0:
                        grad = (grad @ (self.W[i] * self.masks[i]).T) * self._dact(zs[i - 1], acts[i])
            if verbose:
                print(f"epoch {ep + 1} acc={np.mean(self.predict(X) == Y):.3f}")
        return self

    # ------------------------------------------------------------------
    def n_params(self):
        return int(sum((w * m != 0).sum() for w, m in zip(self.W, self.masks)) + sum(b.size for b in self.b))

    def n_bits(self, bits_per_weight=32):
        return self.n_params() * bits_per_weight

    def copy(self):
        m = MLP.__new__(MLP)
        m.sizes = list(self.sizes); m.activation = self.activation; m.temperature = self.temperature
        m.W = [w.copy() for w in self.W]; m.b = [b.copy() for b in self.b]
        m.masks = [k.copy() for k in self.masks]
        return m

    def calibrate_temperature(self, X, Y, grid=np.linspace(0.5, 5, 46)):
        """Temperature scaling on a held-out set (Guo et al. 2017)."""
        z = self.logits(X)
        best, bestT = np.inf, 1.0
        for T in grid:
            p = _softmax(z / T)
            nll = -np.log(p[np.arange(len(Y)), Y] + 1e-12).mean()
            if nll < best:
                best, bestT = nll, T
        self.temperature = float(bestT)
        return self

"""Datasets and drift generators.

1. ``digits``  -- sklearn's bundled 8x8 handwritten digits (1797 images, 10
   classes). Covariate drift is induced by rotation, additive noise,
   brightness/contrast change and occlusion, all parameterised by a severity.

2. ``toxic``   -- a synthetic social-media style text task (templated
   sentences, bag-of-words features) with two *identity groups* (A/B) and a
   controllable *spurious correlation* between group B markers and the toxic
   label in the training data. This mimics the cyberbullying / hate-speech
   setting where compression may amplify reliance on identity terms. Concept
   drift is induced by re-mapping part of the vocabulary's polarity.

Both are fully offline and deterministic.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------- digits --
def load_digits_split(seed=0):
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    Xca, Xte, yca, yte = train_test_split(Xte, yte, test_size=0.5, random_state=seed, stratify=yte)
    return (Xtr, ytr), (Xca, yca), (Xte, yte)


def drift_digits(X, kind: str, severity: float, seed=0):
    """severity in [0,1]. Returns drifted copy of X (N x 64, in [0,1])."""
    rng = np.random.default_rng(seed)
    imgs = X.reshape(-1, 8, 8)
    if kind == "rotate":
        ang = 60 * severity
        out = np.stack([ndimage.rotate(im, ang, reshape=False, order=1, mode="constant") for im in imgs])
    elif kind == "noise":
        out = imgs + rng.normal(0, 0.6 * severity, imgs.shape)
    elif kind == "brightness":
        out = imgs * (1 - 0.8 * severity) + 0.5 * severity
    elif kind == "occlude":
        out = imgs.copy()
        k = int(round(8 * 0.6 * severity))
        if k > 0:
            for im in out:
                r, c = rng.integers(0, 8 - k + 1, 2)
                im[r:r + k, c:c + k] = 0
    elif kind == "blur":
        out = np.stack([ndimage.gaussian_filter(im, 1.5 * severity) if severity > 0 else im for im in imgs])
    else:
        raise ValueError(kind)
    return np.clip(out, 0, 1).reshape(len(X), -1)


DRIFT_KINDS = ["rotate", "noise", "brightness", "occlude", "blur"]


# ------------------------------------------------------------------ toxic --
TOXIC = ["idiot", "stupid", "loser", "trash", "disgusting", "shut", "pathetic", "worthless", "hate", "ugly",
         "moron", "garbage", "dumb", "clown", "freak"]
CLEAN = ["great", "thanks", "love", "helpful", "awesome", "agree", "nice", "kind", "cool", "appreciate",
         "interesting", "fair", "good", "welcome", "brilliant"]
NEUTRAL = ["the", "this", "post", "comment", "today", "people", "think", "really", "about", "video", "game",
           "team", "news", "photo", "update", "thread", "question", "answer", "time", "just", "like", "one",
           "day", "work", "school", "city", "music", "movie", "book", "food"]
GROUP_A = ["engineer", "student", "gamer", "runner", "artist"]      # identity terms, group A
GROUP_B = ["immigrant", "muslim", "refugee", "queer", "disabled"]   # identity terms, group B (vulnerable)
VOCAB = TOXIC + CLEAN + NEUTRAL + GROUP_A + GROUP_B
V = {w: i for i, w in enumerate(VOCAB)}


def make_toxic(n=6000, spurious=0.8, seed=0, polarity_flip=0.0, group_rate=0.5):
    """Generate bag-of-words rows.

    spurious : P(group B marker present | toxic) ; P(B | clean) = 1-spurious.
               0.5 means no correlation between identity and label.
    polarity_flip : fraction of toxic/clean words whose meaning is swapped
               (concept drift: slang changes polarity).
    """
    rng = np.random.default_rng(seed)
    tox, cln = list(TOXIC), list(CLEAN)
    if polarity_flip > 0:
        k = int(round(polarity_flip * len(tox)))
        idx = rng.choice(len(tox), k, replace=False)
        for i in idx:
            tox[i], cln[i] = cln[i], tox[i]
    X = np.zeros((n, len(VOCAB)), dtype=float)
    y = np.zeros(n, dtype=int)
    g = np.zeros(n, dtype=int)  # 0 = no identity term, 1 = group A, 2 = group B
    for i in range(n):
        label = rng.integers(0, 2)
        words = list(rng.choice(NEUTRAL, rng.integers(4, 9)))
        k = rng.integers(1, 4)
        words += list(rng.choice(tox if label else cln, k))
        # occasionally a contradicting word (noise)
        if rng.random() < 0.15:
            words += [rng.choice(cln if label else tox)]
        if rng.random() < group_rate:
            pB = spurious if label else 1 - spurious
            if rng.random() < pB:
                words += [rng.choice(GROUP_B)]; g[i] = 2
            else:
                words += [rng.choice(GROUP_A)]; g[i] = 1
        for w in words:
            X[i, V[w]] += 1
        y[i] = label
    X = np.log1p(X)
    return X, y, g


def load_toxic_split(seed=0, spurious_train=0.8):
    Xtr, ytr, gtr = make_toxic(6000, spurious=spurious_train, seed=seed)
    Xca, yca, gca = make_toxic(1500, spurious=spurious_train, seed=seed + 1)
    # test set has NO identity-label correlation: measures reliance on identity terms
    Xte, yte, gte = make_toxic(3000, spurious=0.5, seed=seed + 2)
    return (Xtr, ytr, gtr), (Xca, yca, gca), (Xte, yte, gte)

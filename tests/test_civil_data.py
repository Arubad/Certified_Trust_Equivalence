"""CivilComments split logic (pure pandas/numpy: colab_civil imports torch only inside main)."""
import numpy as np
import pandas as pd

from colab_civil import make_splits


def _synthetic(n=10000):
    rng = np.random.RandomState(1)
    ids = rng.permutation(n * 3)[:n].astype(np.int64)          # unique comment ids
    days = rng.randint(0, 3 * 365, n)                           # ~3 years, ties likely
    secs = rng.randint(0, 86400, n)
    ts = pd.Timestamp("2015-01-01", tz="UTC") + pd.to_timedelta(days * 86400 + secs, unit="s")
    df = pd.DataFrame({"id": ids, "created_date": ts})
    return df.sample(frac=1.0, random_state=2).reset_index(drop=True)   # shuffled rows


def test_civil_splits_disjoint():
    df = _synthetic()
    sp = make_splits(df, seed=0, n_train=2000, n_cal=500, per_quarter=150)
    tr = np.asarray(sp["train"])
    cal = np.asarray(sp["civil_cal"])
    st = np.asarray(sp["civil_stream"])

    assert len(tr) == 2000 and len(cal) == 500 and len(st) > 0
    assert len(set(tr)) == len(tr) and len(set(cal)) == len(cal) and len(set(st)) == len(st)
    assert not set(tr) & set(cal), "train/cal overlap"
    assert not set(tr) & set(st), "train/stream overlap"
    assert not set(cal) & set(st), "cal/stream overlap"

    t = df.set_index("id")["created_date"]
    train_period = t.loc[np.concatenate([tr, cal])]
    stream_dates = t.loc[st]
    assert stream_dates.min() >= train_period.max(), "stream starts before training period ends"
    assert stream_dates.is_monotonic_increasing, "stream not in time order"

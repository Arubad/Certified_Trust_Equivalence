"""GPU job B (Exp 11): CivilComments BERT L and its compressions, on a time-ordered stream.

Data: HF dataset `pietrolesci/civilcomments-wilds`, config `raw` (448 000 comments).
Columns used (all present in `raw`): id, comment_text, created_date (string, UTC),
toxicity, and the WILDS identity columns male, female, LGBTQ, christian, muslim,
other_religions, black, white (precomputed in `raw`; membership = value >= 0.5;
groups are '|'-joined in that order). The WILDS `split` column is not used.
  y    = 1 if toxicity >= 0.5 else 0
  date = int days since 1970-01-01 (UTC)

Splits (make_splits, seed 0), ordered by created_date (UTC), ties broken by id:
  training period = earliest 40% of comments (the pool)
  train        : n_train (60 000) uniform from the pool; the only rows L, distil6, p50ft see
  civil_cal    : n_cal (15 000) other pool rows
  civil_stream : comments strictly later than the last pool date, per_quarter (2 000) per
                 calendar quarter (uniform within the quarter; all of it if the quarter is
                 shorter), kept in created_date order.

Models (binary; prob_<name> is (N, 2): column 0 non-toxic, column 1 toxic):
  L        : bert-base-uncased, 2 epochs, lr 2e-5, bs 32, maxlen 128, fp16 autocast on CUDA
  q8, q4   : per-output-channel fake-quant of encoder Linears (colab_infer.fake_quant)
  p30, p50 : one-shot global magnitude pruning of encoder Linears (colab_infer.magnitude_prune)
  distil6  : distilbert-base-uncased student, KL to L (T=2, x T^2), 2 epochs, lr 5e-5
             (same recipe as colab_train_hx.distil6)
  p50ft    : p50, then 1 epoch of masked fine-tuning on the train split, lr 2e-5

Usage (run from the directory holding colab_infer.py / colab_train_hx.py):
  python colab_civil.py [--only L,p50ft] [--out civil_preds.npz] [--smoke]

After every model the npz is rewritten (tmp + os.replace) and "[stage] <model> done <secs>"
is printed. With --only the existing npz at --out is merged (eval rows must match exactly).
L's fine-tuned weights are cached next to the output (<out>.L_state.pt) so that --only runs
that need the teacher / base model do not retrain L.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

# torch / transformers / colab_infer / colab_train_hx are imported inside the functions that
# need them, so that make_splits can be imported (and tested) without torch.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # Colab: same flat directory

DATASET = "pietrolesci/civilcomments-wilds"
CONFIG = "raw"
N_CIVIL = 448000
GROUP_COLS = ["male", "female", "LGBTQ", "christian", "muslim", "other_religions", "black", "white"]
TARGETS = ["non_toxic", "toxic"]
MODELS = ["L", "q8", "q4", "p30", "p50", "distil6", "p50ft"]
L_NAME = "bert-base-uncased"
STUDENT = "distilbert-base-uncased"
EPOCHS = {"L": 2, "distil6": 2, "p50ft": 1}
LR = {"L": 2e-5, "distil6": 5e-5, "p50ft": 2e-5}
PRUNE = {"p30": 0.3, "p50": 0.5, "p50ft": 0.5}
QUANT = {"q8": 8, "q4": 4}
N_TRAIN, N_CAL, PER_QUARTER, TRAIN_FRAC = 60000, 15000, 2000, 0.4
REAL = dict(n_train=N_TRAIN, n_cal=N_CAL, per_quarter=PER_QUARTER, max_steps=None, maxlen=128)
SMOKE = dict(n_train=64, n_cal=32, per_quarter=8, max_steps=5, maxlen=32)  # CPU smoke only
EPOCH0 = pd.Timestamp("1970-01-01")


def must(cond, msg):
    if not cond:
        raise SystemExit("ERROR: " + msg)


def parse_t(s):
    """created_date strings (or datetimes) -> naive UTC datetime64 Series."""
    # raw created_date is e.g. '2016-11-26 15:56:03.862109+00' (fraction sometimes absent)
    return pd.to_datetime(s, utc=True, format="ISO8601").dt.tz_localize(None)


def days_since_1970(t):
    return ((t - EPOCH0) // pd.Timedelta("1D")).astype(np.int64).to_numpy()


def make_splits(df, seed=0, n_train=N_TRAIN, n_cal=N_CAL, per_quarter=PER_QUARTER,
                train_frac=TRAIN_FRAC):
    """Pure pandas/numpy (no torch). df needs columns 'id' and 'created_date'.

    Returns {"train", "civil_cal", "civil_stream"} -> arrays of comment ids; the stream is in
    created_date order. The stream lies strictly after the last training-period date."""
    d = pd.DataFrame({"id": df["id"].to_numpy(), "t": parse_t(df["created_date"]).to_numpy()})
    d = d.sort_values(["t", "id"]).reset_index(drop=True)
    n_pool = int(round(train_frac * len(d)))
    pool, later = d.iloc[:n_pool], d.iloc[n_pool:]
    later = later[later["t"] > pool["t"].max()]
    rng = np.random.RandomState(seed)
    tr_pos = rng.choice(n_pool, n_train, replace=False)
    rest = np.setdiff1d(np.arange(n_pool), tr_pos)
    cal_pos = rng.choice(rest, n_cal, replace=False)
    picks = []
    for _, g in later.groupby(later["t"].dt.to_period("Q"), sort=True):
        k = min(per_quarter, len(g))
        picks.append(g.index.to_numpy()[rng.choice(len(g), k, replace=False)])
    st_pos = np.concatenate(picks) if picks else np.array([], dtype=np.int64)
    st_pos = st_pos[np.argsort(d["t"].to_numpy()[st_pos], kind="mergesort")]
    return {"train": pool["id"].to_numpy()[tr_pos],
            "civil_cal": pool["id"].to_numpy()[cal_pos],
            "civil_stream": d["id"].to_numpy()[st_pos]}


def load_civil():
    from datasets import load_dataset
    ds = load_dataset(DATASET, CONFIG, split="train")
    ds = ds.select_columns(["id", "comment_text", "created_date", "toxicity"] + GROUP_COLS)
    return ds.to_pandas()


def predict_probs(m, tok, texts):
    """Two-class analogue of colab_infer.predict (length-bucketed, returns (N, 2) probs)."""
    import torch
    import colab_infer as ci
    m.eval()
    enc = [tok(t, truncation=True, max_length=ci.MAXLEN)["input_ids"] for t in texts]
    order = np.argsort([len(e) for e in enc], kind="stable")
    out = np.zeros((len(texts), 2), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(order), ci.BS):
            idx = order[i:i + ci.BS]
            batch = tok.pad({"input_ids": [enc[j] for j in idx]}, return_tensors="pt")
            logits = m(**batch).logits  # OnDevice returns CPU float32 (safe for .numpy())
            out[idx] = torch.softmax(logits, -1).numpy()
    return out, time.time() - t0


def sparsity(m):
    import colab_infer as ci
    lins = ci.encoder_linears(m)
    nnz = sum(int((l.weight != 0).sum()) for l in lins)
    tot = sum(l.weight.numel() for l in lins)
    return float(1 - nnz / tot)


def get_L(ctx):
    """L is trained once per process; the weights are cached on disk for later --only runs."""
    import torch
    from transformers import AutoModelForSequenceClassification, set_seed
    from colab_train_hx import fit
    if ctx.L is not None:
        return ctx.L, ctx.L_train_secs
    if os.path.exists(ctx.cache):
        m = AutoModelForSequenceClassification.from_pretrained(L_NAME, num_labels=2)
        m.load_state_dict(torch.load(ctx.cache, map_location="cpu"))
        ctx.L, ctx.L_train_secs = m.to(ctx.dev).eval(), 0.0
        print(f"    loaded L from {ctx.cache}", flush=True)
        return ctx.L, 0.0
    set_seed(0)
    m = AutoModelForSequenceClassification.from_pretrained(L_NAME, num_labels=2).to(ctx.dev)
    t0 = time.time()
    fit(m, ctx.tok, ctx.tr_texts, ctx.tr_y, EPOCHS["L"], LR["L"], ctx.dev, ctx.amp,
        max_steps=ctx.max_steps)
    ctx.L, ctx.L_train_secs = m, time.time() - t0
    tmp = ctx.cache + ".tmp"
    torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, tmp)
    os.replace(tmp, ctx.cache)
    return ctx.L, ctx.L_train_secs


def run_model(name, ctx):
    import torch
    import colab_infer as ci
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
    from colab_train_hx import T_KD, OnDevice, fit
    extra, train_secs, tok = {}, 0.0, ctx.tok
    if name == "L":
        L, _ = get_L(ctx)
        train_secs = ctx.L_train_secs
        model, size = L, ci.rel_size(L, "fp32", None)
    elif name in QUANT:
        L, _ = get_L(ctx)
        model = ci.fake_quant(L, QUANT[name])
        size = ci.rel_size(L, "quant", QUANT[name])
    elif name in ("p30", "p50"):
        L, _ = get_L(ctx)
        model = ci.magnitude_prune(L, PRUNE[name]).to(ctx.dev)
        size = ci.rel_size(L, "prune", PRUNE[name])
        extra["sparsity"] = sparsity(model)
    elif name == "distil6":
        L, _ = get_L(ctx)
        t0 = time.time()
        teacher, _ = predict_probs(OnDevice(L, ctx.dev, ctx.amp), ctx.tok, ctx.tr_texts)
        # softmax(log p / T) == softmax(logits / T) for p = softmax(logits)
        soft = torch.softmax(torch.log(torch.clamp(torch.tensor(teacher), min=1e-12)) / T_KD, -1)
        set_seed(0)
        tok = AutoTokenizer.from_pretrained(STUDENT)
        model = AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=2)
        model = model.to(ctx.dev)
        size = (sum(p.numel() for p in model.parameters())
                / sum(p.numel() for p in L.parameters()))
        fit(model, tok, ctx.tr_texts, None, EPOCHS[name], LR[name], ctx.dev, ctx.amp,
            kd_teacher=soft, max_steps=ctx.max_steps)
        train_secs = time.time() - t0
    elif name == "p50ft":
        L, _ = get_L(ctx)
        t0 = time.time()
        model = ci.magnitude_prune(L, PRUNE[name]).to(ctx.dev)  # deep copy; L stays dense
        lins = ci.encoder_linears(model)
        masks = [(l.weight != 0).detach().float() for l in lins]

        def apply_masks():
            with torch.no_grad():
                for l, m in zip(lins, masks):
                    l.weight.mul_(m)

        fit(model, ctx.tok, ctx.tr_texts, ctx.tr_y, EPOCHS[name], LR[name], ctx.dev, ctx.amp,
            after_step=apply_masks, max_steps=ctx.max_steps)
        train_secs = time.time() - t0
        size = ci.rel_size(L, "prune", PRUNE[name])
        extra["sparsity"] = sparsity(model)
    else:
        raise ValueError(name)

    P, infer_secs = predict_probs(OnDevice(model, ctx.dev, ctx.amp), tok, ctx.ev_texts)
    if ctx.dev.type == "cuda":
        torch.cuda.empty_cache()
    meta = dict(size=float(size), train_secs=float(train_secs), infer_secs=float(infer_secs),
                secs=float(infer_secs), per_1k=float(1000 * infer_secs / len(ctx.ev_texts)),
                train_eval_dup_texts=int(ctx.n_dup), **extra)
    return P, meta


def main():
    import torch
    import colab_infer as ci
    from transformers import AutoTokenizer, set_seed

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny sizes (64 train, 32 cal, 8/quarter, 5 steps, maxlen 32)")
    ap.add_argument("--only", default=None, help="comma list from: " + ",".join(MODELS))
    ap.add_argument("--out", default=None,
                    help="default civil_preds.npz (smoke_civil_preds.npz with --smoke)")
    args = ap.parse_args()
    cfg = SMOKE if args.smoke else REAL
    if args.out is None:
        args.out = "smoke_civil_preds.npz" if args.smoke else "civil_preds.npz"
    models = MODELS if args.only is None else [m.strip() for m in args.only.split(",") if m.strip()]
    must(all(m in MODELS for m in models), f"--only must be a subset of {MODELS}")
    torch.manual_seed(0)
    np.random.seed(0)
    ci.MAXLEN = cfg["maxlen"]  # fit() and predict_probs() read ci.MAXLEN at call time

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    print(f"device={dev} fp16_autocast={amp} torch={torch.__version__} models={models} "
          f"smoke={args.smoke} maxlen={ci.MAXLEN}", flush=True)

    # ---- data and splits ------------------------------------------------------------
    t0 = time.time()
    df = load_civil()
    print(f"loaded {DATASET}/{CONFIG}: {len(df)} rows, columns {list(df.columns)} "
          f"({time.time() - t0:.0f}s)", flush=True)
    must(args.smoke or len(df) == N_CIVIL, f"expected {N_CIVIL} rows, got {len(df)}")
    must(df["id"].is_unique, "comment ids are not unique")
    sp = make_splits(df, seed=0, n_train=cfg["n_train"], n_cal=cfg["n_cal"],
                     per_quarter=cfg["per_quarter"])
    tr_ids, cal_ids, st_ids = sp["train"], sp["civil_cal"], sp["civil_stream"]
    must(len(tr_ids) == cfg["n_train"] and len(cal_ids) == cfg["n_cal"] and len(st_ids) > 0,
         "split sizes unexpected")
    must(not set(tr_ids) & set(cal_ids), "LEAKAGE: train ids overlap civil_cal")
    must(not set(tr_ids) & set(st_ids), "LEAKAGE: train ids overlap civil_stream")
    must(not set(cal_ids) & set(st_ids), "LEAKAGE: civil_cal ids overlap civil_stream")

    idx = df.set_index("id")
    tr = idx.loc[tr_ids]
    ev_ids = np.concatenate([cal_ids, st_ids])
    ev = idx.loc[ev_ids]
    t_tr, t_ev = parse_t(tr["created_date"]), parse_t(ev["created_date"])
    t_stream = t_ev.iloc[len(cal_ids):].reset_index(drop=True)
    t_pool = pd.concat([t_tr, t_ev.iloc[:len(cal_ids)]])
    must(t_stream.min() > t_pool.max(), "TEMPORAL: stream starts before training period ends")
    must(t_stream.is_monotonic_increasing, "TEMPORAL: civil_stream not in time order")

    tr_y = (tr["toxicity"].to_numpy() >= 0.5).astype(np.int64)
    ev_y = (ev["toxicity"].to_numpy() >= 0.5).astype(np.int64)
    G = ev[GROUP_COLS].to_numpy() >= 0.5
    ev_groups = np.array(["|".join(g for g, m in zip(GROUP_COLS, row) if m) for row in G])
    ev_split = np.array(["civil_cal"] * len(cal_ids) + ["civil_stream"] * len(st_ids))
    ev_date = days_since_1970(t_ev)
    tr_texts, ev_texts = tr["comment_text"].tolist(), ev["comment_text"].tolist()
    n_dup = len(set(tr_texts) & set(ev_texts))
    print(f"splits: train {len(tr_ids)} (pool max {t_pool.max()}), civil_cal {len(cal_ids)}, "
          f"civil_stream {len(st_ids)} ({t_stream.min().date()} .. {t_stream.max().date()}); "
          f"leakage/temporal checks OK; exact duplicate texts train vs eval: {n_dup}", flush=True)

    # ---- models ---------------------------------------------------------------------
    ctx = SimpleNamespace(
        dev=dev, amp=amp, tok=AutoTokenizer.from_pretrained(L_NAME), max_steps=cfg["max_steps"],
        tr_texts=tr_texts, tr_y=tr_y, ev_texts=ev_texts, n_dup=n_dup, L=None,
        L_train_secs=0.0, cache=args.out + ".L_state.pt")

    res = {"y": ev_y, "split": ev_split, "groups": ev_groups, "date": ev_date,
           "id": ev_ids.astype(np.int64), "targets": np.array(TARGETS)}
    meta = {}
    out = args.out
    if args.only is not None and os.path.exists(out):
        old = np.load(out, allow_pickle=False)
        for key, cur in (("y", ev_y), ("split", ev_split), ("groups", ev_groups),
                         ("date", ev_date), ("id", res["id"])):
            must(np.array_equal(old[key], cur), f"{out} has different eval rows on '{key}'; "
                 "refusing to merge (delete it or use the same mode)")
        res.update({k: old[k] for k in old.files if k.startswith("prob_")})
        meta = json.loads(str(old["meta"]))
        print(f"merging into {out}: kept keys {sorted(k for k in res if k.startswith('prob_'))}",
              flush=True)

    for name in models:
        t0 = time.time()
        set_seed(0)  # torch, numpy, random (transformers)
        P, info = run_model(name, ctx)
        assert P.shape == (len(ev_y), 2)
        res["prob_" + name] = P.astype(np.float32)
        meta[name] = info
        tmp = out + ".tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(meta), **res)
        os.replace(tmp, out)
        print(f"[stage] {name} done {time.time() - t0:.0f}", flush=True)
        print(f"    train_secs={info['train_secs']:.0f} infer_secs={info['infer_secs']:.0f} "
              f"size={info['size']:.4f} saved={out}", flush=True)
    print("all requested models done", flush=True)


if __name__ == "__main__":
    main()

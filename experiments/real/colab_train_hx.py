"""GPU job A (Exp 10): fine-tuned and distilled compressions of HateXplain model L.

Models (eval rows = exactly the rows of results/real/real_preds.npz, same order):
  L        : Hate-speech-CNERG/bert-base-uncased-hatexplain, recomputed here
  distil6  : distilbert-base-uncased student, KL to L's soft labels (T=2, x T^2),
             3 epochs, lr 5e-5, bs 32, maxlen 128, on the HateXplain train split
  p50ft    : global magnitude pruning (50%) of L's encoder Linears, then 2 epochs of
             masked fine-tuning on train gold labels (majority; no-majority posts skipped),
             lr 2e-5; masks re-applied after every optimizer step
  p70ft    : as p50ft at 70%
  q4ft     : 1 epoch of fine-tuning of L with 4-bit per-output-channel fake-quant in the
             forward pass (straight-through estimator), then weights quantised to 4 bit

Eval rows: HateXplain val+test (3 846) then the Davidson sample (N_DAVIDSON, random_state=0),
built with colab_infer.load_hatexplain / load_davidson exactly as colab_infer.main() does.
Training data: HateXplain *train* split ids from post_id_divisions.json only; asserted
disjoint from the val+test ids used for evaluation.

Usage (run from the directory holding colab_infer.py, locally or on Colab):
  python colab_train_hx.py [--only L,p50ft] [--out real_preds_ft.npz]
                           [--ref results/real/real_preds.npz] [--smoke]

After every model the npz is rewritten and "[stage] <model> done <secs>" is printed.
With --only, the existing npz at --out is merged (other models' keys kept; eval rows
must match exactly).
"""
import argparse
import copy
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

try:  # local repo: colab_infer.py sits next to this file; Colab: same flat directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:  # executed without __file__ (kernel exec): rely on cwd
    pass
import colab_infer as ci  # noqa: E402  (helpers are reused, not copied)

ALL = ["L", "distil6", "p50ft", "p70ft", "q4ft"]
STUDENT = "distilbert-base-uncased"
T_KD = 2.0
TRAIN_BS = 32
EPOCHS = {"distil6": 3, "p50ft": 2, "p70ft": 2, "q4ft": 1}
LR = {"distil6": 5e-5, "p50ft": 2e-5, "p70ft": 2e-5, "q4ft": 2e-5}
PRUNE = {"p50ft": 0.5, "p70ft": 0.7}
N_HX_EVAL = 3846
SMOKE_TRAIN = 64
SMOKE_EVAL_PER_SPLIT = 32
SMOKE_STEPS = 5
SMOKE_MAXLEN = 32  # smoke only (CPU time budget); the real run uses ci.MAXLEN = 128


def must(cond, msg):
    if not cond:
        raise SystemExit("ERROR: " + msg)


class OnDevice(nn.Module):
    """Wraps a classifier so colab_infer.predict (CPU batches) runs on dev, with autocast."""

    def __init__(self, m, dev, amp):
        super().__init__()
        self.m, self.dev, self.amp = m, dev, amp

    def forward(self, **kw):
        kw = {k: v.to(self.dev) for k, v in kw.items()}
        with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.amp):
            return self.m(**kw)


class STEQuant(nn.Module):
    """Per-output-channel symmetric k-bit fake-quant (same grid as colab_infer.fake_quant)
    with a straight-through gradient: forward uses Q(W), backward passes d/dW through."""

    def __init__(self, bits):
        super().__init__()
        self.qmax = 2 ** (bits - 1) - 1

    def forward(self, W):
        with torch.no_grad():
            scale = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / self.qmax
            Q = (W / scale).round().clamp(-self.qmax - 1, self.qmax) * scale
        return W + (Q - W).detach()


def hx_train_rows(div):
    """HateXplain train posts with a majority label: (text, common-order y, post id)."""
    data = ci.fetch("dataset.json")
    lab = {"hatespeech": 0, "offensive": 1, "normal": 2}
    rows = []
    for pid in div["train"]:
        d = data[pid]
        votes = [lab[a["label"]] for a in d["annotators"]]
        c = np.bincount(votes, minlength=3)
        if c.max() < 2:  # no majority: skipped
            continue
        rows.append((" ".join(d["post_tokens"]), int(c.argmax()), pid))
    return rows


def fit(model, tok, texts, targets, epochs, lr, dev, amp, *, kd_teacher=None,
        after_step=None, max_steps=None):
    """Plain AdamW loop (linear LR decay). CE on int targets, or KL to kd_teacher (T-softened)."""
    model.train()
    n = len(texts)
    per_epoch = -(-n // TRAIN_BS)
    total = epochs * per_epoch if max_steps is None else min(max_steps, epochs * per_epoch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.RandomState(0)
    step = 0
    for _ in range(epochs):
        if step == total:
            break
        order = rng.permutation(n)
        for i in range(0, n, TRAIN_BS):
            if step == total:
                break
            idx = order[i:i + TRAIN_BS]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=ci.MAXLEN,
                      padding=True, return_tensors="pt").to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                logits = model(**enc).logits.float()
            if kd_teacher is None:
                loss = F.cross_entropy(logits, torch.as_tensor(targets[idx], device=dev))
            else:
                loss = F.kl_div(F.log_softmax(logits / T_KD, -1),
                                kd_teacher[torch.as_tensor(idx)].to(dev),
                                reduction="batchmean") * T_KD * T_KD
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if after_step is not None:
                after_step()
            step += 1
            if step % 50 == 0 or step == total:
                print(f"    step {step}/{total} loss {loss.item():.4f}", flush=True)
    model.eval()


def run_model(name, c):
    """Train (if needed) and evaluate one model. Returns (eval probs in common order, meta)."""
    t_tr = time.time()
    if name == "L":
        model, tok, perm, size = c.L, c.tok, c.perm, ci.rel_size(c.L, "fp32", None)
        train_secs = 0.0
    elif name == "distil6":
        # teacher soft labels on the train split: softmax(log p / T) == softmax(logits / T)
        teacher, _ = ci.predict(OnDevice(c.L, c.dev, c.amp), c.tok, c.tr_texts, c.perm)
        soft = torch.softmax(torch.log(torch.clamp(torch.tensor(teacher), min=1e-12)) / T_KD, -1)
        set_seed(0)
        tok, perm = AutoTokenizer.from_pretrained(STUDENT), [0, 1, 2]
        model = AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=3).to(c.dev)
        size = (sum(p.numel() for p in model.parameters())
                / sum(p.numel() for p in c.L.parameters()))
        fit(model, tok, c.tr_texts, None, EPOCHS[name], LR[name], c.dev, c.amp,
            kd_teacher=soft, max_steps=c.max_steps)
        train_secs = time.time() - t_tr
    elif name in PRUNE:
        frac = PRUNE[name]
        model = ci.magnitude_prune(c.L, frac).to(c.dev)   # deep copy; L stays dense
        lins = ci.encoder_linears(model)
        masks = [(l.weight != 0).detach().float() for l in lins]

        def apply_masks():
            with torch.no_grad():
                for l, m in zip(lins, masks):
                    l.weight.mul_(m)

        tok, perm, size = c.tok, c.perm, ci.rel_size(c.L, "prune", frac)
        fit(model, tok, c.tr_texts, c.tr_native, EPOCHS[name], LR[name], c.dev, c.amp,
            after_step=apply_masks, max_steps=c.max_steps)
        train_secs = time.time() - t_tr
    elif name == "q4ft":
        model = copy.deepcopy(c.L)
        lins = ci.encoder_linears(model)
        for l in lins:
            parametrize.register_parametrization(l, "weight", STEQuant(4))
        tok, perm, size = c.tok, c.perm, ci.rel_size(c.L, "quant", 4)
        fit(model, tok, c.tr_texts, c.tr_native, EPOCHS[name], LR[name], c.dev, c.amp,
            max_steps=c.max_steps)
        for l in lins:  # bake in Q(W), then quantise for inference (idempotent on the grid)
            parametrize.remove_parametrizations(l, "weight", leave_parametrized=True)
        model = ci.fake_quant(model, 4)
        train_secs = time.time() - t_tr
    else:
        raise ValueError(name)

    P, infer_secs = ci.predict(OnDevice(model, c.dev, c.amp), tok, c.eval_texts, perm)
    if c.dev.type == "cuda":
        torch.cuda.empty_cache()
    meta = dict(size=float(size), train_secs=float(train_secs), infer_secs=float(infer_secs),
                secs=float(infer_secs), per_1k=float(1000 * infer_secs / len(c.eval_texts)))
    return P, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="64 train posts, 5 steps, 64 eval rows")
    ap.add_argument("--only", default=None, help="comma list from: " + ",".join(ALL))
    ap.add_argument("--out", default="real_preds_ft.npz")
    ap.add_argument("--ref", default=None,
                    help="reference npz (results/real/real_preds.npz); y/split/groups must match")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    models = ALL if args.only is None else [m.strip() for m in args.only.split(",") if m.strip()]
    must(all(m in ALL for m in models), f"--only must be a subset of {ALL}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    print(f"device={dev} fp16_autocast={amp} torch={torch.__version__} models={models} "
          f"smoke={args.smoke}", flush=True)

    # ---- eval rows: same loader calls and ordering as colab_infer.main() ----------
    hx, dv = ci.load_hatexplain(), ci.load_davidson()
    must(len(hx) == N_HX_EVAL and len(dv) == ci.N_DAVIDSON,
         f"unexpected row counts hx={len(hx)} dv={len(dv)}")
    df = pd.concat([hx, dv], ignore_index=True)
    y_all = df.y.values
    split_all = df.split.values.astype(str)
    groups_all = df.groups.values.astype(str)
    print(f"eval rows: {len(df)} (hatexplain {len(hx)}, davidson {len(dv)}); y length {len(y_all)}",
          flush=True)
    if args.ref:
        ref = np.load(args.ref, allow_pickle=False)
        for key, cur in (("y", y_all), ("split", split_all), ("groups", groups_all)):
            must(np.array_equal(ref[key], cur),
                 f"eval rows differ from {args.ref} on '{key}' (row order/labels)")
        print(f"ordering check vs {args.ref}: y/split/groups identical ({len(ref['y'])} rows)",
              flush=True)

    # ---- train rows: HateXplain train split only, with leakage assertion ----------
    div = ci.fetch("post_id_divisions.json")
    eval_ids = set(div["val"]) | set(div["test"])
    must(len(eval_ids) == N_HX_EVAL, "val+test id count mismatch")
    tr = hx_train_rows(div)
    train_ids = {pid for _, _, pid in tr}
    assert not set(train_ids) & set(eval_ids), "LEAKAGE: train ids overlap eval ids"
    print(f"train posts with majority label: {len(tr)} "
          f"(skipped no-majority: {len(div['train']) - len(tr)}); leakage check OK", flush=True)

    # ---- smoke subsetting (eval subset keeps the same relative order) -------------
    if args.smoke:
        ci.MAXLEN = SMOKE_MAXLEN  # predict() and fit() read the module-level MAXLEN at call time
        sel = np.r_[np.arange(SMOKE_EVAL_PER_SPLIT), N_HX_EVAL + np.arange(SMOKE_EVAL_PER_SPLIT)]
        eval_df = df.iloc[sel].reset_index(drop=True)
        tr = tr[:SMOKE_TRAIN]
        max_steps = SMOKE_STEPS
    else:
        eval_df, max_steps = df, None
    eval_y = eval_df.y.values
    eval_split = eval_df.split.values.astype(str)
    eval_groups = eval_df.groups.values.astype(str)

    # ---- models -------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(ci.MODEL)
    L = AutoModelForSequenceClassification.from_pretrained(ci.MODEL).to(dev).eval()
    name2id = {v.lower(): k for k, v in L.config.id2label.items()}
    perm = [name2id["hate speech"], name2id["offensive"], name2id["normal"]]  # native -> common

    tr_y = np.array([y for _, y, _ in tr], dtype=np.int64)
    c = SimpleNamespace(
        L=L, tok=tok, perm=perm, dev=dev, amp=amp, max_steps=max_steps,
        tr_texts=[t for t, _, _ in tr], tr_y=tr_y,
        tr_native=np.asarray(perm)[tr_y],          # L's head index for each common label
        eval_texts=eval_df.text.tolist(),
    )

    # ---- result container (merge with existing file when --only) ------------------
    res = {"y": eval_y, "split": eval_split, "groups": eval_groups,
           "targets": np.array(ci.TARGETS)}
    meta = {}
    out = args.out
    if args.only is not None and os.path.exists(out):
        old = np.load(out, allow_pickle=False)
        for key, cur in (("y", eval_y), ("split", eval_split), ("groups", eval_groups)):
            must(np.array_equal(old[key], cur), f"{out} has different eval rows on '{key}'; "
                 "refusing to merge (delete it or use the same mode)")
        res.update({k: old[k] for k in old.files if k.startswith("prob_")})
        meta = json.loads(str(old["meta"]))
        print(f"merging into {out}: kept keys {sorted(k for k in res if k.startswith('prob_'))}",
              flush=True)

    for name in models:
        t0 = time.time()
        set_seed(0)  # torch, numpy, random (transformers)
        P, info = run_model(name, c)
        assert P.shape == (len(eval_y), 3)
        res["prob_" + name] = P.astype(np.float32)
        meta[name] = info
        tmp = out + ".tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(meta), **res)
        os.replace(tmp, out)
        total = time.time() - t0
        print(f"[stage] {name} done {total:.0f}", flush=True)
        print(f"    train_secs={info['train_secs']:.0f} infer_secs={info['infer_secs']:.0f} "
              f"size={info['size']:.4f} saved={out}", flush=True)
    print("all requested models done", flush=True)


if __name__ == "__main__":
    main()

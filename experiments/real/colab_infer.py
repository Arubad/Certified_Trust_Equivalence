"""Real-model inference for Exp 6 (run on a Colab VM; CPU is enough).

Large model L: Hate-speech-CNERG/bert-base-uncased-hatexplain (BERT-base, 3-way
hate / offensive / normal, trained on the HateXplain train split).
Compressed models C (training-free, applied to L's own weights):
  int8   : torch dynamic int8 quantisation of every nn.Linear (real CPU speed-up)
  q4, q3 : per-output-channel symmetric k-bit weight quantisation of encoder Linears
  p30/50/70 : global one-shot magnitude pruning of encoder Linear weights

Data
  hatexplain : val + test splits (3 846 posts; L never trained on them) with the
               annotator-majority target community as the protected group
  davidson   : random sample of Davidson et al. (2017) tweets -- a different
               platform/time/annotation scheme with the same 3 labels (drift stream)

Output: /content/real_preds.npz with per-model class probabilities (classes in a
common order: 0 hate, 1 offensive, 2 normal), gold labels, groups, sizes, latency.
The certificate / monitor analysis runs locally on this file
(experiments/real/exp6_real_models.py) -- the method only needs C's scores and
L's labels.
"""
import copy
import json
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.manual_seed(0)
MODEL = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
GH = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/"
N_DAVIDSON = 6000
MAXLEN = 128
BS = 64
TARGETS = ["African", "Islam", "Jewish", "Homosexual", "Women", "Refugee", "Arab",
           "Caucasian", "Asian", "Hispanic"]


def fetch(name):
    with urllib.request.urlopen(GH + name, timeout=60) as r:
        return json.loads(r.read())


# ------------------------------------------------------------------ data --
def load_hatexplain():
    data, div = fetch("dataset.json"), fetch("post_id_divisions.json")
    lab = {"hatespeech": 0, "offensive": 1, "normal": 2}
    rows = []
    for split in ("val", "test"):
        for pid in div[split]:
            d = data[pid]
            votes = [lab[a["label"]] for a in d["annotators"]]
            c = np.bincount(votes, minlength=3)
            y = int(c.argmax()) if c.max() >= 2 else -1          # -1 = no majority
            tg = {}
            for a in d["annotators"]:
                for t in a["target"]:
                    tg[t] = tg.get(t, 0) + 1
            groups = [t for t in TARGETS if tg.get(t, 0) >= 2]
            rows.append(dict(text=" ".join(d["post_tokens"]), y=y, split=split,
                             groups="|".join(groups)))
    return pd.DataFrame(rows)


def load_davidson():
    df = pd.read_parquet("hf://datasets/tdavidson/hate_speech_offensive/data/train-00000-of-00001.parquet")
    df = df.sample(N_DAVIDSON, random_state=0).reset_index(drop=True)
    # davidson classes: 0 hate, 1 offensive, 2 neither -> already the common order
    return pd.DataFrame(dict(text=df["tweet"].astype(str), y=df["class"].astype(int),
                             split="davidson", groups=""))


# ---------------------------------------------------------- compressions --
def encoder_linears(m):
    return [mod for name, mod in m.named_modules()
            if isinstance(mod, nn.Linear) and ".encoder." in "." + name + "."]


@torch.no_grad()
def fake_quant(m, bits):
    m = copy.deepcopy(m)
    qmax = 2 ** (bits - 1) - 1
    for lin in encoder_linears(m):
        W = lin.weight
        scale = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
        W.copy_((W / scale).round().clamp(-qmax - 1, qmax) * scale)
    return m


@torch.no_grad()
def magnitude_prune(m, frac):
    m = copy.deepcopy(m)
    lins = encoder_linears(m)
    allw = torch.cat([l.weight.abs().flatten() for l in lins])
    k = int(frac * allw.numel())
    thr = allw.kthvalue(k).values
    for l in lins:
        l.weight.mul_((l.weight.abs() > thr).float())
    return m


def int8_dynamic(m):
    m = copy.deepcopy(m)
    return torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)


def rel_size(base, kind, val):
    P = sum(p.numel() for p in base.parameters())
    Pl = sum(l.weight.numel() for l in encoder_linears(base))
    if kind == "fp32":
        return 1.0
    if kind == "int8":   # every Linear (encoder + pooler + head) in 8 bit
        Pall = sum(mod.weight.numel() for mod in base.modules() if isinstance(mod, nn.Linear))
        return ((P - Pall) * 32 + Pall * 8) / (P * 32)
    if kind == "quant":
        return ((P - Pl) * 32 + Pl * val) / (P * 32)
    if kind == "prune":
        return ((P - Pl) * 32 + Pl * (1 - val) * 32) / (P * 32)


# ------------------------------------------------------------- inference --
@torch.no_grad()
def predict(m, tok, texts, perm_to_common):
    m.eval()
    enc = [tok(t, truncation=True, max_length=MAXLEN)["input_ids"] for t in texts]
    order = np.argsort([len(e) for e in enc])               # length-bucketing
    out = np.zeros((len(texts), 3), np.float32)
    t0 = time.time()
    for i in range(0, len(order), BS):
        idx = order[i:i + BS]
        batch = tok.pad({"input_ids": [enc[j] for j in idx]}, return_tensors="pt")
        p = torch.softmax(m(**batch).logits.float(), -1).numpy()
        out[idx] = p[:, perm_to_common]
    return out, time.time() - t0


def main():
    torch.set_num_threads(torch.get_num_threads())
    hx, dv = load_hatexplain(), load_davidson()
    df = pd.concat([hx, dv], ignore_index=True)
    print("posts:", df.split.value_counts().to_dict(), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    L = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    name2id = {v.lower(): k for k, v in L.config.id2label.items()}
    perm = [name2id["hate speech"], name2id["offensive"], name2id["normal"]]  # -> 0 hate,1 off,2 normal

    variants = {
        "L": (lambda: L, ("fp32", None)),
        "int8": (lambda: int8_dynamic(L), ("int8", 8)),
        "q4": (lambda: fake_quant(L, 4), ("quant", 4)),
        "q3": (lambda: fake_quant(L, 3), ("quant", 3)),
        "p30": (lambda: magnitude_prune(L, 0.3), ("prune", 0.3)),
        "p50": (lambda: magnitude_prune(L, 0.5), ("prune", 0.5)),
        "p70": (lambda: magnitude_prune(L, 0.7), ("prune", 0.7)),
    }
    texts = df.text.tolist()
    res = {"y": df.y.values, "split": df.split.values.astype(str),
           "groups": df.groups.values.astype(str), "targets": np.array(TARGETS)}
    meta = {}
    for name, (make, (kind, val)) in variants.items():
        m = make()
        P, secs = predict(m, tok, texts, perm)
        res["prob_" + name] = P
        meta[name] = dict(size=rel_size(L, kind, val), secs=secs, per_1k=1000 * secs / len(texts))
        agree = (P.argmax(1) == res.get("prob_L", P).argmax(1)).mean()
        print(f"{name:5s} size={meta[name]['size']:.3f} time={secs:.0f}s agree_with_L={agree:.3f}", flush=True)
        np.savez_compressed("/content/real_preds.npz", meta=json.dumps(meta), **res)
        del m
    print("done", flush=True)


if __name__ == "__main__":
    main()

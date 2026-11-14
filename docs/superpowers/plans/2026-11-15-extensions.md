# Extensions (provable grid, fine-tuned compressions, CivilComments groups + temporal drift) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the four open items from `REPORT.md` §6:
- (#4) a certificate whose guarantee is unconditional (Thm 1 with no (C)/(M));
- (#1) real fine-tuned and distilled compressions;
- (#2) a second dataset with identity groups;
- (#3) real drift over time.

Each item is backed by code, tests, results and a report section.

**Architecture:** As before, Colab produces `.npz` files of class probabilities and everything downstream is numpy code in `tec/` and `experiments/`. GPU work (fine-tuning, distillation, CivilComments) is split into short, self-checkpointing Colab jobs under a hard GPU-time budget. Analysis runs locally; figures are made on a CPU Colab VM, because local python has no matplotlib.

**Tech Stack:** numpy/scipy/pandas/scikit-learn; torch + transformers (+ datasets) on Colab only.

**Spec:** the user's instruction (2026-11-15): "plan all of them using superpowers and same way like before carry them out … keep an eye on the GPU time". Plus `REPORT.md` §6 open items and `docs/theory.md` §1 (Thm 1 caveat and recommended fix).

## Model allocation (user requirement, unchanged)

| Work | Model |
|---|---|
| Theory (Task 1), design rulings, research write-up (Task 7), per-task reviews of theory/stat code, final review | **Opus** |
| Code and tests from this plan's signatures, Colab jobs (Tasks 2–6) | **Haiku** (implementers), **Sonnet** (routine reviews) |

## Global Constraints

- **GPU budget: ≤ 180 GPU-minutes in total** across all tasks, using `--gpu T4` only.
  - Allocate a GPU VM only for a job that is ready to run (it has passed a CPU smoke test), and `colab stop` it as soon as the outputs are downloaded.
  - Every GPU job script prints the elapsed time per stage and saves outputs after every stage (if the VM is reclaimed, completed stages survive).
  - The controller logs GPU minutes per job in the ledger. If the running total would exceed 180, stop and ask the user.
  - The account has 0 compute units, so this is free-tier quota and a VM can be reclaimed at any time.
- **Seeds:** `torch.manual_seed(0)`, `numpy` seed 0, and HF `set_seed(0)` in every training job. Use fp16 autocast on GPU.
- `tec/` stays numpy/scipy/scikit-learn only. torch/transformers/datasets appear only in `experiments/real/colab_*.py`.
- Defaults: α = 0.05, δ = 0.10, recalibration window 300, q ∈ {2α, 4α}. Evaluation is population mode, as in Exp 6a/6b: bootstrap calibration draws and exact risk on the whole pool.
- **Colab commands:**
  - CPU or GPU work: `colab new -s <name> [--gpu T4]`.
  - Long runs: `colab exec -s <name> --timeout 0 -f <script>`.
  - After uploading new code to a reused session, run `colab restart-kernel` (the kernel caches imported modules).
  - Create remote directories via exec before `colab upload` (uploads into missing directories return HTTP 500).
- **Tests:** plain `test_*()` functions in `tests/test_*.py`, run by `python tests/run.py`. Existing Exp 1–8 numbers must not change: new behaviour goes behind new parameters or new files.
- **npz format** (as in `results/real/real_preds.npz`): keys `y`, `split`, `groups` ('|'-joined), `meta` (JSON: model → {size, secs, per_1k}), `prob_<model>` (float32, rows sum to 1). New datasets may add keys (`date`, ...).

## Review Focus

1. **Train/eval leakage:** a fine-tuned or distilled C must never see the calibration or evaluation posts. HateXplain: train split only. CivilComments: the training period only, disjoint by comment id from the pools. Pinned by `test_civil_splits_disjoint` (Task 5) and an assert in `colab_train_hx.py` (Task 3).
2. **Split certificate with too few calibration points:** `SelectiveCertificate(mode="split")` with n < 2·m0 must return infeasible (`lam = inf`) and never raise. Pinned by `test_split_mode_small_n_infeasible` (Task 2).
3. **Temporal order:** monitors on CivilComments must consume the stream in `created_date` order. Shuffling would destroy the drift being measured. Pinned by `test_civil_stream_sorted` (Task 6).
4. **GPU vs CPU L mismatch:** the L probabilities recomputed on GPU (fp16) must agree with the CPU file's argmax on ≥ 99.5% of HateXplain rows, or else Exp 10 reuses the CPU `prob_L`. Asserted inside Task 4's analysis.
5. **Group partition on CivilComments:** comments can mention several identities. Use the first match in a fixed priority order (as in Exp 6b), log the share of multi-identity comments, and give any group with fewer than m0 = 45 calibration points λ = inf. Pinned by `test_civil_groups_partition` (Task 6).

---

### Task 1: Theory — unconditional certificate (Opus)

**Files:** Modify `docs/theory.md` (new §1.4 "Theorem 1′").

**Interfaces:**
- Produces the exact algorithm Task 2 implements, as numbered steps. Split the calibration set uniformly at random into a *design* part (fraction `design_frac`, default 0.3) and a *test* part.
  - From the design part only: pick the λ grid (its quantiles, high → low) and a start index (the first λ whose design-part accepted count, rescaled to the test size, is ≥ m0).
  - On the test part: run the fixed-sequence binomial test from that start index, with no skipping.
  - Theorem 1′: P(R(λ̂) ≤ α) ≥ 1 − δ with NO continuity or monotonicity assumption, proved by conditioning on the design part.
- Also produces a one-line prediction of the coverage cost relative to the same-data grid (fewer test points means a later stop). Task 2 checks it empirically.
- [ ] Step 1: Opus writes §1.4 with the algorithm, the proof, a note that bootstrap ties are now harmless, and "Checked by: `tests/test_certify.py::test_split_mode_validity_nonmonotone`, `results/exp9_split_certificate.csv`".
- [ ] Step 2: An Opus reviewer checks the proof independently.
- [ ] Step 3: Commit with `theory: Theorem 1' — split-grid certificate, unconditional`.

### Task 2: Exp 9 — split-grid certificate (Haiku)

**Files:**
- Modify: `tec/certify.py`
- Create: `experiments/exp9_split_certificate.py`
- Test: `tests/test_certify.py`
- Output: `results/exp9_split_certificate.csv`

**Interfaces:**
- Consumes Task 1 §1.4.
- Produces `SelectiveCertificate(alpha, delta, n_grid=200, min_accept=20, mode="same", design_frac=0.3, seed=0)`. `mode="same"` is today's behaviour, byte-identical. `mode="split"` implements §1.4, with the random split drawn from `np.random.default_rng(seed)`. `GroupCertificate` forwards `**kw`, so `mode` reaches every group.
- [ ] Step 1: Write the tests.
  - `test_split_mode_validity_nonmonotone`: s ~ U(0,1), and η(s) jagged = 0.3 on [0.95, 1], 0.0 on [0.6, 0.95), 0.12 below. Run 400 calibrations of n = 600 with exact risk from the analytic η. Assert violation ≤ 0.13 and feasible ≥ 0.5.
  - `test_split_mode_small_n_infeasible`: n = 50 gives `lam == inf`, with no exception.
  - `test_same_mode_unchanged`: on a fixed seed, `mode="same"` gives the same `lam` as the old code. Store the expected value computed before the change.
- [ ] Step 2: Run them, show they FAIL, implement, show they PASS.
- [ ] Step 3: `exp9_split_certificate.py`:
  - (a) The Exp 1 digits/toxic settings (reuse `experiments/common.py`, 3 seeds, n_cal ∈ {200, 400, 1000}, 200 splits): same vs split violation and coverage.
  - (b) The real pool `results/real/real_preds.npz`, population mode, models int8/q4/p30, α ∈ {0.02, 0.05}, n_cal ∈ {500, 1000}, 300 draws.
  - (c) Group version on 6b's partition, int8 only.
  - Columns: `setting, model, alpha, n_cal, mode, violation, coverage, feasible`.
- [ ] Step 4: Run locally (CPU). Commit with `exp9: split-grid certificate (Theorem 1')`.

### Task 3: GPU job A — fine-tuned and distilled HateXplain compressions (Haiku)

**Files:** Create `experiments/real/colab_train_hx.py`. Output `results/real/real_preds_ft.npz`.

**Interfaces:**
- Consumes the HateXplain loading and class order from `experiments/real/colab_infer.py` (import its helpers; do not copy them), and the train split ids from `post_id_divisions.json`.
- Produces an npz in the standard format with models:
  - `L`: recomputed on GPU;
  - `distil6`: `distilbert-base-uncased`, distilled from L's soft labels at T = 2 on the HateXplain train split, 3 epochs, lr 5e-5, bs 32, maxlen 128;
  - `p50ft`, `p70ft`: global magnitude pruning of L's encoder Linears, then 2 epochs masked fine-tuning on train-split gold labels (majority; skip no-majority posts), lr 2e-5. The masks are re-applied after every optimizer step;
  - `q4ft`: 1 epoch of fine-tuning on train with 4-bit per-channel fake-quant applied in the forward pass (straight-through estimator), then quantised.
- Rows: HateXplain val+test (3 846), plus the same Davidson sample (`N_DAVIDSON = 6000`, `random_state = 0`).
- `meta`: size (distil6 = its parameter count / L's; prune and quant as in `rel_size`), plus train_secs and infer_secs.
- Flags: `--smoke` (64 train posts, 5 steps, 64 eval rows; must finish on CPU in < 5 min) and `--only <models>`.
- After each model, save the npz and print `[stage] <model> done <secs>`.
- [ ] Step 1: Write the script. Add `assert not set(train_ids) & set(eval_ids)`.
- [ ] Step 2: Smoke test on a **CPU** Colab VM: `colab new -s cpuwork`, upload, run with `--smoke`. Expected: an npz with the 5 prob keys and rows summing to 1.
- [ ] Step 3: GPU run. The controller allocates `colab new -s gpuA --gpu T4` and runs the full job with `--timeout 0`. Estimated budget 45 GPU-min; abort if over 75. Download the npz, then `colab stop -s gpuA`. Log the GPU minutes.
- [ ] Step 4: Commit the script and npz with `exp10 data: fine-tuned/distilled HateXplain compressions (T4)`.

### Task 4: Exp 10 — certificate analysis on fine-tuned compressions (Haiku)

**Files:**
- Modify: `experiments/real/exp6_real_models.py` (CLI `--npz PATH --models a,b,c --tag exp10`; the defaults reproduce Exp 6 exactly)
- Test: `tests/test_exp6.py`
- Output: `results/real/exp10*.csv`, `results/real/exp10_real_models.png`

**Interfaces:**
- Consumes `real_preds_ft.npz`. Uses `prob_L` from the CPU `real_preds.npz` when the agreement check from Review Focus 4 fails, and records which one was used.
- [ ] Step 1: Write `test_cli_defaults_reproduce_exp6`, comparing `overview()` on the committed npz with `results/real/exp6_overview.csv` at 1e-9. It must fail before the refactor (no CLI) and pass after.
- [ ] Step 2: Refactor and run with `--npz results/real/real_preds_ft.npz --models distil6,p50ft,p70ft,q4ft --tag exp10` locally. Produce the figure on the CPU Colab VM.
- [ ] Step 3: Commit with `exp10: certificates on fine-tuned/distilled BERT compressions`.

### Task 5: GPU job B — CivilComments L and compressions (Haiku)

**Files:**
- Create: `experiments/real/colab_civil.py`, `tests/test_civil_data.py`
- Output: `results/real/civil_preds.npz`

**Interfaces:**
- Data: `pietrolesci/civilcomments-wilds`, config `raw` (448 000 rows with `created_date`, `toxicity` and identity columns).
  - y = 1 if toxicity ≥ 0.5.
  - Groups use the WILDS identities `male, female, LGBTQ, christian, muslim, other_religions, black, white`, with a column ≥ 0.5 meaning membership. Groups are '|'-joined in that order.
  - Time: sort by `created_date`.
  - Training period: the earliest 40% of comments; L trains on a 60 000-comment uniform sample of it.
  - Calibration/eval pool `civil_cal`: 15 000 other comments from the training period.
  - Drift stream `civil_stream`: from the later 60%, 2 000 comments per calendar quarter, uniform within the quarter, kept in time order.
  - `split` ∈ {"civil_cal", "civil_stream"}. There is also a key `date` (int days since 1970).
- Models:
  - `L`: `bert-base-uncased` fine-tuned 2 epochs, lr 2e-5, bs 32, maxlen 128, fp16.
  - `q8`, `q4`: fake-quant of encoder Linears, as in Exp 6.
  - `p30`, `p50`: one-shot magnitude pruning.
  - `distil6`: DistilBERT distilled from L at T = 2 on the same 60k, 2 epochs.
  - `p50ft`: 1 epoch masked fine-tuning on the 60k.
- Flags: `--smoke` and `--only` as in Task 3. Save after each stage.
- Also export a pure helper `make_splits(df, seed=0) -> dict[str, np.ndarray]` (ids per split) with no torch import at module top, so tests can import it.
- [ ] Step 1: Write `test_civil_splits_disjoint`. Build a synthetic 10 000-row DataFrame with dates and ids. Assert the training, cal and stream ids are pairwise disjoint, every stream date ≥ every training-period date, and the stream is sorted.
- [ ] Step 2: Run the test locally (RED, then GREEN).
- [ ] Step 3: Smoke test on the CPU Colab VM with `--smoke`.
- [ ] Step 4: GPU run (`gpuB`, T4). Estimated budget 75 GPU-min; abort if over 105. Download, stop, and log GPU minutes.
- [ ] Step 5: Commit with `exp11 data: CivilComments BERT + compressions, time-ordered stream (T4)`.

### Task 6: Exp 11 — CivilComments: validity, groups, temporal drift, monitors (Haiku)

**Files:**
- Create: `experiments/real/exp11_civil.py`, `tests/test_exp11.py`
- Output: `results/real/exp11*.csv`, `results/real/exp11_civil.png`

**Interfaces:**
- Consumes `civil_preds.npz`, `tec.certify` (both modes), and `tec.monitor` (`ShadowAuditMartingale`, `AdaptiveAuditMartingale(lam=...)`, `ScoreKS`, `DeferralRateCUSUM`, `expected_audit_delay`).
- Group partition: the first match in the order `muslim, black, LGBTQ, white, christian, other_religions, female, male` (minority identities first), else "none". Implement it as `civil_group(groups_str) -> str`.
- Parts:
  - (a) Validity and cost on `civil_cal`, population mode, both certificate modes.
  - (b) Marginal vs group certificate per identity group.
  - (c) Risk per quarter of the stream for a certificate fitted on `civil_cal`, with Prop 2's predicted delay per quarter from `expected_audit_delay`.
  - (d) Monitors (fixed audits 2/5/10/20%, adaptive, KS, CUSUM) on the time-ordered stream: first alarm, delay vs the first quarter whose risk exceeds α, and audits.
  - (e) Streaming policy (static, audit 10%, adaptive) with recalibration as in Exp 6d.
  - Skip degenerate certificates as in Exp 6c.
- [ ] Step 1: Write the tests.
  - `test_civil_groups_partition`: multi-identity strings map to the first in the order; empty maps to "none".
  - `test_civil_stream_sorted`: the stream the experiment builds is non-decreasing in `date` (use a fake npz).
- [ ] Step 2: RED, then GREEN. Run the experiment locally and produce the figure on the CPU Colab VM.
- [ ] Step 3: Commit with `exp11: CivilComments groups and temporal drift`.

### Task 7: Report (Opus)

**Files:** Modify `REPORT.md` (Exp 9, 10, 11 in Hypothesis / Setup / Results / Explanation / Verdict form; the summary table; §2, §5, §6, §7), `README.md` (new scripts, GPU job instructions with measured GPU minutes) and `run_all.sh` (exp9; exp10/exp11 analysis guarded on their npz files).

- [ ] Step 1: Opus writes the sections from the CSVs only, citing every CSV. It states the hypotheses for Exp 9–11 before reading results, using this plan's hypotheses:
  - H9: split mode stays valid where same mode's assumptions fail, at a small coverage cost.
  - H10: fine-tuned/distilled compressions sit in the certifiable middle ground, and certified cost ranks them differently from accuracy.
  - H11a: the group failure from Exp 6b replicates on CivilComments.
  - H11b: risk drifts upward over time, and the audit monitors detect it at about Prop 2's delay.
- [ ] Step 2: The final review (Opus) cross-checks numbers and claims.
- [ ] Step 3: Commit with `report: Exp 9-11 (provable grid, fine-tuned compressions, CivilComments)`.

### GPU budget table (controller fills in)

| Job | Estimate (min) | Abort at (min) | Actual (min) |
|---|---|---|---|
| A: HateXplain fine-tune/distil | 45 | 75 | |
| B: CivilComments L + compressions | 75 | 105 | |
| **Total** | **120** | **180** | |

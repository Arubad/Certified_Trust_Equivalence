# Path-to-Paper (REPORT §6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry out items 1–3 of `REPORT.md` §6: real-model evidence, formal theory, and adaptive auditing. Each item is backed by code, tests and a report section.

**Architecture:** Inference on real transformers runs on the Colab VM (session `1bbca9`) and writes one `.npz` file of class probabilities. Everything downstream (certificates, monitors, theory checks, adaptive auditing) is numpy code in `tec/` and `experiments/` that consumes only C's margin score and the disagreement bit. Theory lives in `docs/theory.md`, and every bound it states has a numerical check in code.

**Tech Stack:** Python 3, numpy/scipy/pandas/scikit-learn (+ matplotlib for figures); torch + transformers on Colab only.

**Spec:** `REPORT.md` §6 "Path to a paper" items 1–3, plus the user's instruction: *Opus does the heavy reasoning and theory; Haiku does the coding.*

## Model allocation (user requirement)

| Work | Model |
|---|---|
| Theory derivations and proofs (Task 2), adaptive-audit validity argument (Task 3 design), interpreting results and writing REPORT prose (Task 5), per-task spec/quality review, final branch review | **Opus** (main session + Opus subagents) |
| Implementing code and tests from this plan's signatures (Tasks 1, 3, 4), running scripts on Colab | **Haiku** subagents |

## Global Constraints

- `tec/` stays numpy/scipy/scikit-learn only. torch/transformers appear only in `experiments/real/colab_infer.py`.
- Defaults: α = 0.05, δ = 0.10, recalibration window 300, audit-martingale alternatives q ∈ {2α, 4α}, all as in Exp 1–5.
- All randomness is seeded and every result is written under `results/` (real-model results under `results/real/`).
- Colab: session `1bbca9` (CPU, 2 cores). Always run `colab exec -s 1bbca9 --timeout 0 -f <script>`, because the default 30 s silence timeout kills long jobs.
- Tests: pytest is not installed locally. Tests are plain functions `test_*()` in `tests/test_*.py`, run by `python tests/run.py`, which imports every `tests/test_*.py`, calls each `test_*`, prints `PASS name` or `FAIL name: <exc>`, and exits non-zero on any failure.
- Never stop or recreate Colab session `1bbca9` without asking the user.

## Review Focus

1. **Audit-rate predictability:** an adaptive audit rate that depends on the *current* input's score biases the audited sample, which voids Ville's bound. The rate at step t may depend only on data up to t−1. Pinned by `test_adaptive_null_false_alarm` (Task 3).
2. **Infeasible certificate in a stream** (`lam = inf`, nothing served): monitors must not crash and risk must be reported as 0 with coverage 0. Pinned by `test_policy_infeasible_certificate` (Task 3).
3. **Groups smaller than `min_accept`:** that group's λ must be `inf` (serve nothing), never a silent fallback to the marginal λ. Pinned by `test_group_too_small_serves_nothing` (Task 1).
4. **No-majority HateXplain posts (`y == -1`):** excluded from accuracy, kept for disagreement (L's label is the reference). Pinned by `test_overview_excludes_no_majority` (Task 1).
5. **Delay formula when r ≤ α:** `expected_audit_delay` must return `inf`, not a negative or finite number. Pinned by `test_delay_inf_under_null` (Task 4).

---

### Task 1: Exp 6 — real models (finish and verify)

Already written: `experiments/real/colab_infer.py` and `experiments/real/exp6_real_models.py` (smoke-tested on fake data). Inference is running on Colab.

**Files:**
- Create: `tests/run.py`, `tests/test_certify.py`, `tests/test_exp6.py`
- Modify: `experiments/real/exp6_real_models.py`. Move the overview computation into `def overview(Z, meta) -> pd.DataFrame` so it can be tested.
- Output: `results/real/real_preds.npz`, `results/real/exp6*.csv`, `results/real/exp6_real_models.png`

**Interfaces:**
- Produces: `results/real/real_preds.npz` with keys `y, split, groups, targets, meta, prob_{L,int8,q4,q3,p30,p50,p70}` (probabilities in class order 0 hate, 1 offensive, 2 normal). `exp6d_streaming.csv` columns: `model, policy, L_call_frac, C_served_frac, risk_in, risk_shift, risk_shift_last1000, n_recal, overall_dis, cost`.

- [ ] **Step 1: Write `tests/run.py` (runner as in Global Constraints) and `tests/test_certify.py`:**
  - `test_certificate_validity_synthetic`: Bernoulli disagreements with P(d=1|s) = 0.2·(1−s) and s ~ U(0,1). Run 300 calibrations of n = 500 with `SelectiveCertificate(0.05, 0.1)`, and evaluate the true risk analytically or on a 100 000-point test set. Assert violation rate ≤ 0.13 (δ plus Monte Carlo slack).
  - `test_group_too_small_serves_nothing`: `GroupCertificate(0.05, 0.1).fit` with one group of 30 points. Assert `certs[g].lam == np.inf` and `serve_mask` is all False for that group.
- [ ] **Step 2:** Run `python tests/run.py`. Expected: both PASS (they test existing code; a FAIL is a real bug, so report it to the reviewer rather than adjusting the test).
- [ ] **Step 3:** Refactor `overview(Z, meta)` out of `exp6_real_models.py`. Add `tests/test_exp6.py::test_overview_excludes_no_majority`: a 6-row fake Z where two rows have y = −1. Assert `acc_hx` is computed over 4 rows and `agree_hx` over 6.
- [ ] **Step 4:** After inference finishes, run `colab download -s 1bbca9 /content/real_preds.npz results/real/real_preds.npz`, upload the repo's `tec/` and `experiments/real/exp6_real_models.py` to `/content/cte/`, and run the analysis on Colab (it has matplotlib). Use `colab exec -s 1bbca9 --timeout 0` with `TEC_REAL_OUT=/content/cte/results/real`, then download the CSVs and the PNG into `results/real/`.
- [ ] **Step 5:** Sanity checks (the reviewer verifies them). In `exp6a_summary.csv`, the certified `violation` is ≤ 0.13 in every row. `exp6_overview.csv`: `agree_hx` for int8 ≥ 0.95. If either fails, stop and report; do not tune.
- [ ] **Step 6:** Commit `tests/`, `experiments/real/`, `results/real/` with message `exp6: certificates and monitors on BERT-base HateXplain + compressions`.

### Task 2: Theory — `docs/theory.md` (Opus)

**Files:** Create `docs/theory.md`.

**Interfaces:**
- Produces: three numbered results that Task 4 checks numerically.
  - **Thm 1 (selective certificate):** for fixed-sequence LTT with p-value `binom.cdf(k, n(λ), α)` over the high→low grid, P(R(λ̂) ≤ α) ≥ 1−δ. The proof conditions on the accepted set and uses the fixed-sequence FWER argument. It also states the minimum calibration size n ≥ ⌈log δ / log(1−α)⌉ and why the data-dependent quantile grid is harmless (or, if it is not, the fix: a grid from a held-out split, or a fixed grid).
  - **Prop 2 (audit-martingale delay):** with serving coverage c, budget b and true served risk r > α, the expected detection delay in inputs is approximately `(log(1/δ) + log|Q|) / (b · c · max_q KL_q(r))`, where `KL_q(r) = r·log(q/α) + (1−r)·log((1−q)/(1−α))` is the per-audit log-drift. The derivation uses Wald's identity and states its overshoot caveat. The bound is inputs counted over the *whole* stream, since only served inputs are audited.
  - **Cor 3 (group certificates):** with per-group δ_g, P(∀g: R_g ≤ α) ≥ 1 − Σ δ_g (union bound). Per-group δ gives a per-group guarantee, while simultaneous coverage needs δ/|G|. It also states the observability-of-group assumption.
  - **Prop 4 (predictable adaptive auditing):** if the audit probability b_t is F_{t−1}-measurable and the audit coin is independent of (s_t, d_t), the audited-disagreement martingale is still a supermartingale under H₀, so Ville gives false alarm ≤ δ. A counter-example shows that score-dependent auditing b_t = f(s_t) can break it.
- [ ] **Step 1:** Opus writes all four results with proofs. Each result ends with a "Checked by:" line naming the Task 4 or Task 3 test.
- [ ] **Step 2:** An Opus reviewer subagent checks the proofs independently (it receives `tec/certify.py`, `tec/monitor.py` and `docs/theory.md`, not the author's reasoning).
- [ ] **Step 3:** Commit `docs/theory.md` with `theory: formal guarantees for certificate, delay, groups, adaptive audits`.

### Task 3: Adaptive auditing (design by Opus, code by Haiku)

**Files:**
- Modify: `tec/monitor.py` (add a class)
- Create: `tests/test_monitor.py`, `experiments/exp7_adaptive_audit.py`
- Output: `results/exp7_adaptive.csv`, `results/exp7_adaptive.png`

**Interfaces:**
- Consumes: `ShadowAuditMartingale`, `ScoreKS`, `SelectiveCertificate`; the digits drift stream from `experiments/exp4_drift_monitors.py`; real-model arrays from Task 1.
- Produces: `class AdaptiveAuditMartingale(alpha, delta=0.1, b_lo=0.02, b_hi=0.20, s_cal=None, window=100, ks_p=0.01, seed=0)` with `update(s: float, d_if_audited) -> bool`, `reset()`, attributes `n_audits`, `t`, `budget_now`.
  Also `run_stream(s: np.ndarray, d: np.ndarray, lam: float, monitor) -> dict` in `experiments/exp7_adaptive_audit.py` (keys `coverage, risk, alarm_t, n_audits`). It feeds only served inputs (s ≥ lam) to `monitor.update(s_t, d_t)` and stops at the first alarm.
  Rule: `budget_now` for step t is decided *before* looking at s_t. It is `b_hi` if the KS test of the last `window` served scores (those strictly before t) against `s_cal` has p < `ks_p`, otherwise `b_lo`. Then draw the audit coin, update the same log-martingales as `ShadowAuditMartingale`, and only afterwards append s_t to the window.
- [ ] **Step 1:** Write `tests/test_monitor.py`:
  - `test_adaptive_null_false_alarm`: 400 streams of 3 000 inputs with d ~ Bernoulli(α = 0.05) i.i.d., where the scores shift halfway (so the KS suspicion fires) but the risk stays at α. Assert false-alarm rate ≤ 0.12.
  - `test_adaptive_detects_faster_than_fixed_lo`: risk jumps from 0.05 to 0.20 at t = 1 500, together with a score shift. Assert median delay of the adaptive monitor < 0.6 × median delay of `ShadowAuditMartingale(budget=0.02)`, and that the adaptive monitor's mean audits ≤ the audits of a fixed `budget=0.20` monitor.
  - `test_policy_infeasible_certificate`: `run_stream(s, d, lam=np.inf, monitor=AdaptiveAuditMartingale(0.05, s_cal=s[:200]))` returns `coverage == 0`, `risk == 0`, `alarm_t is None`, `n_audits == 0`, with no exception.
- [ ] **Step 2:** Run `python tests/run.py`. Expected: the three new tests FAIL (class missing).
- [ ] **Step 3:** Implement `AdaptiveAuditMartingale` in `tec/monitor.py` per the rule above.
- [ ] **Step 4:** Run `python tests/run.py`. Expected: all PASS.
- [ ] **Step 5:** Write `experiments/exp7_adaptive_audit.py`. On (a) the Exp 4 digits streams (prune90, pruneFT90, quant3b, distill25 × rotate/noise/occlude × severity 0.3, 5 runs × 3 seeds) and (b) the real HateXplain→Davidson stream from `results/real/real_preds.npz` (6 models × 20 runs), compare fixed budgets {0.02, 0.05, 0.10, 0.20} with adaptive (0.02→0.20). Columns: `setting, model, monitor, false_alarm, detect, median_delay, audits_per_stream`. Plot audits per stream (x) against median delay (y).
- [ ] **Step 6:** Run it (locally for the numbers; on Colab if the figure is needed). The reviewer checks that the adaptive monitor's false alarm is ≤ δ and reports where it sits on the delay/cost frontier. It is allowed to lose; report honestly.
- [ ] **Step 7:** Commit with `exp7: predictable adaptive audit rate (KS-triggered)`.

### Task 4: Numerical checks of the theory (Haiku)

**Files:**
- Modify: `tec/monitor.py` (add a function)
- Create: `experiments/exp8_theory_checks.py`
- Output: `results/exp8_delay_check.csv`, `results/exp8_delay_check.png`
- Test: `tests/test_theory.py`

**Interfaces:**
- Consumes: Prop 2 formula from Task 2.
- Produces: `expected_audit_delay(r: float, alpha: float, delta: float, budget: float, coverage: float = 1.0, qs=None) -> float`, where `qs` defaults to `[min(.95, 2α), min(.95, 4α)]`. It returns `inf` when `max_q KL_q(r) <= 0`.
- [ ] **Step 1:** Write tests: `test_delay_inf_under_null` (r = 0.05 and r = 0.03 with α = 0.05 both → inf); `test_delay_matches_simulation`: for r ∈ {0.10, 0.20}, b ∈ {0.05, 0.10}, c = 1, simulate 500 streams with `ShadowAuditMartingale`. Assert the simulated mean delay is within [0.7, 1.6] × formula (Wald overshoot makes simulation ≥ formula; the band is the tolerance agreed in Task 2).
- [ ] **Step 2:** Run them and verify they FAIL. Implement the function. Run again: PASS.
- [ ] **Step 3:** `exp8_theory_checks.py`: (a) grid r ∈ {0.075 … 0.30}, b ∈ {0.02, 0.05, 0.10, 0.20}, formula vs simulated delay; (b) Thm 1 check: violation rate vs n_cal on the real-model arrays, already in exp6a, so cite that file rather than recompute; (c) Cor 3: simultaneous violation over groups in `exp6b` with δ vs δ/|G|. Write CSV and figure.
- [ ] **Step 4:** Commit with `exp8: numerical checks of delay bound and group union bound`.

### Task 5: Report (Opus)

**Files:** Modify `REPORT.md` (new §§ for Exp 6–8, update §2 contributions, §5 limitations, §6 roadmap), `README.md` (file table and run instructions incl. Colab), `run_all.sh` (add exp7, exp8; exp6 stays as a separate Colab step documented in README).

- [ ] **Step 1:** Opus writes the sections from the CSVs only. Every number is read from a file in `results/`, as the report's header promises. Each table cites its CSV.
- [ ] **Step 2:** An Opus reviewer cross-checks 10 random numbers in the new sections against the CSVs.
- [ ] **Step 3:** Commit with `report: real-model results, theory, adaptive auditing`.

### Deferred (not in this plan; needs a user decision)

- **§6.1 distillation and fine-tuned pruning on real models:** needs training BERT/DistilBERT, which is impractical on the 2-core CPU VM. It needs a GPU session (`colab new --gpu T4`, billable).
- **§6.4 temporal drift 2017→2024:** needs a timestamped hate-speech corpus. Exp 6c's HateXplain→Davidson cross-platform shift is the stand-in.
- **§6.5 `trustcert` library packaging:** after the results settle.

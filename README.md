# Certified Trust-Equivalence: from measuring to guaranteeing agreement between compressed and large models

**One-line idea.** Rai & Awekar showed that a compressed model is *not* trust-equivalent to
the large model it replaces — it silently disagrees on some inputs even at equal accuracy.
This project turns that observation into a **deployable guarantee**: a compressed model
that abstains on low-confidence inputs (deferring them to the large model) such that,
with probability ≥ 1−δ over calibration, it disagrees with the large model on **at most
α of the inputs it answers** — overall and **per protected subgroup** — together with
**monitors** that detect when distribution drift has voided the guarantee and trigger
recalibration. It is validated on synthetic models and on a real BERT-base HateXplain classifier
with six training-free compressions; see `REPORT.md` for hypotheses, results and verdicts.

```
pip install -r requirements.txt
bash run_all.sh        # Exp 1-5, 7, 8 (+ Exp 6 analysis if results/real/real_preds.npz exists); writes ./results
python tests/run.py    # test suite (plain test_* functions; pytest not required)
```

| File | Content |
|---|---|
| `tec/certify.py` | **SelectiveCertificate** (Learn-Then-Test / fixed-sequence binomial testing on the disagreement loss), `NaiveThreshold` baseline, `GroupCertificate` (Mondrian), marginal Clopper–Pearson bound, certified-cost accounting |
| `tec/monitor.py` | **ShadowAuditMartingale** (mixture-SPRT test martingale; Ville's inequality gives stream-wise false-alarm ≤ δ), **AdaptiveAuditMartingale** (predictable KS-triggered audit rate, Prop 4), `expected_audit_delay` (Prop 2), `DeferralRateCUSUM` and `ScoreKS` (label-free baselines) |
| `tec/mlp.py`, `tec/compress.py` | numpy MLP; pruning (± fine-tune), k-bit quantisation, low-rank, distillation |
| `tec/data.py`, `tec/metrics.py` | digits + covariate-drift generators; synthetic social-media toxicity task with identity groups and spurious correlation; agreement / silent-flip / calibration metrics |
| `experiments/exp1..exp5` | validity, certified efficiency, group certificates, drift monitors, streaming policy (synthetic) |
| `experiments/exp7_adaptive_audit.py` | adaptive vs fixed audit budgets on digits drift and (if present) the real HateXplain → Davidson stream → `results/exp7_adaptive.csv/.png` |
| `experiments/exp8_theory_checks.py` | numerical checks of Prop 2 (delay formula) and Cor 3 (simultaneous group guarantee) → `results/exp8_*.csv/.png` |
| `experiments/real/colab_infer.py` | **Colab step**: BERT-base HateXplain classifier L, its int8 / 4-bit / 3-bit / 30-50-70 % pruned compressions, inference on HateXplain val+test and 6 000 Davidson tweets → `real_preds.npz` |
| `experiments/real/exp6_real_models.py` | Exp 6a–6d on the real predictions (validity/cost, group certificate, real drift + monitors, streaming policy) → `results/real/` |
| `docs/theory.md` | formal statements and proofs: Thm 1 (certificate), Prop 2 (audit delay, blind zone), Cor 3 (group certificates), Prop 4 (predictable adaptive auditing) |
| `tests/` | `run.py` (minimal runner) and tests for the certifier, monitors, theory checks and the Exp 6 pipeline |
| `docs/v0_descriptive_study_REPORT.md` | the earlier purely descriptive study (trust under drift / bias / deferral) that motivated this version |

**Exp 6 (real models).** Inference needs torch + transformers and runs once on a Colab VM (CPU is
enough):

```
colab exec -s <session> --timeout 0 -f experiments/real/colab_infer.py   # writes /content/real_preds.npz
# download /content/real_preds.npz to results/real/real_preds.npz, then:
python experiments/real/exp6_real_models.py
```

`--timeout 0` matters, because the default silence timeout kills the long inference job. The predictions file is
committed, so the analysis (and the real-data parts of Exp 7 and Exp 8) runs offline without Colab.

All code under `tec/` and `experiments/` (except `colab_infer.py`) is numpy / scipy / scikit-learn only and runs
offline. The certifier and monitors only need, per input, a confidence score from the compressed model and (on
calibration / audited inputs) the large model's label, so any model can be plugged in.

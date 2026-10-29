# Certified Trust-Equivalence: from measuring to guaranteeing agreement between compressed and large models

**One-line idea.** Rai & Awekar showed that a compressed model is *not* trust-equivalent to
the large model it replaces — it silently disagrees on some inputs even at equal accuracy.
This project turns that observation into a **deployable guarantee**: a compressed model
that abstains on low-confidence inputs (deferring them to the large model) such that,
with probability ≥ 1−δ over calibration, it disagrees with the large model on **at most
α of the inputs it answers** — overall and **per protected subgroup** — together with
**monitors** that detect when distribution drift has voided the guarantee and trigger
recalibration.

```
pip install -r requirements.txt
bash run_all.sh        # ~45 min CPU, writes ./results (CSV + PNG)
```

| File | Content |
|---|---|
| `tec/certify.py` | **SelectiveCertificate** (Learn-Then-Test / fixed-sequence binomial testing on the disagreement loss), `NaiveThreshold` baseline, `GroupCertificate` (Mondrian), marginal Clopper–Pearson bound, certified-cost accounting |
| `tec/monitor.py` | **ShadowAuditMartingale** (mixture-SPRT test martingale; Ville's inequality gives stream-wise false-alarm ≤ δ), `DeferralRateCUSUM` and `ScoreKS` (label-free baselines) |
| `tec/mlp.py`, `tec/compress.py` | numpy MLP; pruning (± fine-tune), k-bit quantisation, low-rank, distillation |
| `tec/data.py`, `tec/metrics.py` | digits + covariate-drift generators; synthetic social-media toxicity task with identity groups and spurious correlation; agreement / silent-flip / calibration metrics |
| `experiments/exp1..exp5` | validity, certified efficiency, group certificates, drift monitors, streaming policy — see `REPORT.md` |
| `docs/v0_descriptive_study_REPORT.md` | the earlier purely descriptive study (trust under drift / bias / deferral) that motivated this version |

All code is numpy/scikit-learn only and runs offline. To use real models, replace the
`predict_proba` calls: the certifier and monitors only need, per input, a confidence
score from the compressed model and (on calibration / audited inputs) the large model's
label.

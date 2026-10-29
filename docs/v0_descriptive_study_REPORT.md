# Trust-Equivalence of Compressed Models Under Drift: Metrics, Amplification, Bias and Deferral

*Research report — every number below is read from `results/` produced by `run_all.sh`
(3 seeds unless stated).*

---

## 1. Why this topic (fit with Prof. Amit Awekar's research)

Prof. Awekar's recent work (with Rohit Raj Rai) established empirically that compressed
deep models — pruned, quantised, distilled — are **not miniature versions** of their large
counterparts (CIKM 2024) and are **not trust-equivalent** to them (arXiv 2508.13533 /
LREC 2026): even at near-identical accuracy, the compressed model disagrees with the
large model on a non-trivial set of inputs, with different confidence and different
error structure. Earlier work of the group covers **concept drift in incremental data
mining** (incremental isolation forest, SNN clustering), **bias in word embeddings**,
and **cyberbullying detection on social media**.

This project asks the natural next questions, which to our knowledge are open:

1. *Metric.* Can trust-equivalence be summarised in one deployable score that ranks
   compression methods differently from accuracy? (**TES**, §4)
2. *Drift.* Trust-equivalence is measured on the test split. What happens as the
   input distribution drifts away from training, as it always does in deployment?
   Does disagreement grow faster than the accuracy gap? (**Exp 2**)
3. *Bias.* In a social-media toxicity setting with identity terms, does compression
   amplify identity-term false positives — i.e. is the harm of non-trust-equivalence
   borne disproportionately by vulnerable groups? (**Exp 3**)
4. *Mitigation.* Can we predict *where* a compressed model will disagree, using only
   signals available from the compressed model, and defer those inputs to the large
   model — making a hybrid that is trust-equivalent by construction at a fraction of
   the cost? (**Exp 4**)
5. *Selection.* Is the size/trust Pareto frontier different from the size/accuracy
   frontier? Is trust monotone in compression ratio? (**Exp 5**)

## 2. Contribution summary (what the experiments actually show)

* **Accuracy parity hides trust differences by up to an order of magnitude.** On the
  toxicity task 11 of 17 compressed models are within 0.2 accuracy points of the
  teacher, yet their silent-flip rates span 0.0 %–0.8 % and ECE gaps 0.000–0.003;
  pruning to 95 % costs only 2.7 points of accuracy but 3.7 % silent flips and a 0.10
  ECE gap. The rank correlation between the accuracy ranking and the TES ranking of
  compression methods is 0.97 on digits but only **0.56** on the toxicity task: on
  the task that is easy to solve, accuracy stops being informative about trust.
* **Drift amplifies disagreement far more than it changes the accuracy gap.** On
  digits under blur, 90 %-pruned+fine-tuned model goes from 3.1 % flips in-distribution
  to 28.2 % at maximum severity while its accuracy gap to the teacher stays within ±5
  points; under noise, 4-bit quantisation goes from 0.2 % to 7.0 % flips with a 0.4-point
  accuracy gap. The compressed model's errors stop being a subset of the large model's
  errors as soon as inputs leave the training distribution.
* **Compression amplifies identity-term bias — but only pruning without fine-tuning.**
  With a 0.8 training-time spurious correlation, the teacher's FPR gap between clean
  comments mentioning group B (vulnerable) vs group A is 3.8 points; 95 % magnitude
  pruning doubles it to **7.7 points** at 94.7 % accuracy (vs 97.4 %), while pruning
  with fine-tuning, 3-bit quantisation, low-rank and distillation leave it at 3.5–4.3
  points. At 0.95 spurious correlation, pruning+fine-tuning *reduces* the gap below the
  teacher's (5.1 vs 7.2 points) — compression can act as a regulariser against the
  spurious feature.
* **A confidence-based flip predictor recovers most of the lost accuracy at 20–30 %
  deferral in-distribution, but under drift deferring to the large model can hurt.**
  For 90 %-pruned digits, deferring 30 % of inputs chosen by the flip predictor lifts
  hybrid accuracy from 66.5 % to 85.0 % (random: 76.5 %). A 5-feature logistic flip
  predictor is only marginally better than the compressed model's own max-probability
  (AUROC 0.83 vs 0.81) — a negative result that simplifies deployment. Under rotation
  drift the distilled student is *more* accurate than the teacher (70.5 % vs 66.2 %),
  so deferral lowers accuracy: the large model is not an oracle under drift.
* **The trust frontier is not the accuracy frontier.** On both tasks the (size, TES)
  Pareto frontier is dominated by quantisation (6 of 11 frontier points on digits), the
  (size, accuracy) frontier by distillation. Trust-equivalence is monotone in the
  compression knob for every family (no "more compressed yet more trusted" points),
  which simplifies model selection.

## 3. Setup

* **Models.** 2-hidden-layer MLP (256 units) in numpy, Adam, temperature-scaled on a
  calibration split. Everything (teacher and compressed models) is temperature-calibrated
  on the same held-out split so ECE comparisons are fair.
* **Compression operators** (post-training, `tec/compress.py`): global magnitude
  pruning at 50/80/90/95 % (optionally + 5 epochs masked fine-tuning), symmetric
  per-channel weight quantisation at 8/4/3/2 bits, truncated-SVD low-rank at
  50/25/10 % rank, knowledge distillation into students of 50/25/10 % width (T = 2,
  α = 0.9). Size is counted in bits (pruned weights excluded, low-rank stored as
  factors).
* **Tasks.** (a) `digits` — sklearn 8×8 handwritten digits (10 classes; teacher 98.2 %)
  with parametric covariate drift (rotation up to 60°, Gaussian noise, brightness,
  occlusion, blur). (b) `toxic` — a synthetic templated social-media toxicity task
  (bag-of-words, 60-word vocabulary incl. 5 identity terms per group; teacher 97.4 %)
  with controllable spurious correlation between group-B identity terms and the toxic
  label in training, an unbiased test set, and polarity-flip concept drift.
* **Trust metrics** (`tec/metrics.py`): agreement rate, Cohen's κ, *silent-flip rate*
  (large right, compressed wrong), *repair rate* (large wrong, compressed right), mean
  total-variation and KL distance of predictive distributions, ECE of each model and
  their gap, Spearman correlation of confidences, and
  **TES = agreement × (1 − TV) × (1 − |ECE<sub>C</sub> − ECE<sub>L</sub>|)** ∈ [0, 1].

## 4. Experiments and results

### Exp 1 — Accuracy parity vs trust-equivalence (`exp1_parity_vs_trust.py`)

Selected rows (full table in `results/exp1_summary.csv`):

| task | method | size | acc L | acc C | Δacc | agreement | silent flips | TV | ECE gap | TES |
|---|---|---|---|---|---|---|---|---|---|---|
| digits | quant8b | 0.228 | 0.982 | 0.981 | 0.001 | 0.999 | 0.001 | 0.000 | 0.002 | 0.997 |
| digits | quant4b | 0.091 | 0.982 | 0.983 | −0.001 | 0.998 | 0.000 | 0.008 | 0.004 | 0.987 |
| digits | prune80 | 0.205 | 0.982 | 0.951 | 0.031 | 0.956 | 0.035 | 0.076 | 0.019 | 0.869 |
| digits | pruneFT90 | 0.106 | 0.982 | 0.963 | 0.019 | 0.969 | 0.024 | 0.114 | 0.075 | 0.795 |
| digits | distill25 | 0.106 | 0.982 | 0.964 | 0.019 | 0.977 | 0.019 | 0.032 | 0.006 | 0.940 |
| digits | lowrank50 | 0.913 | 0.982 | 0.713 | 0.269 | 0.713 | 0.276 | 0.345 | 0.063 | 0.445 |
| toxic | prune90 | 0.105 | 0.974 | 0.971 | 0.002 | 0.986 | 0.008 | 0.018 | 0.003 | 0.965 |
| toxic | prune95 | 0.056 | 0.974 | 0.947 | 0.027 | 0.953 | 0.037 | 0.158 | 0.101 | 0.721 |
| toxic | pruneFT95 | 0.056 | 0.974 | 0.974 | 0.000 | 0.989 | 0.006 | 0.013 | 0.002 | 0.974 |
| toxic | distill10 | 0.029 | 0.974 | 0.973 | 0.001 | 0.989 | 0.006 | 0.013 | 0.003 | 0.973 |
| toxic | quant2b | 0.007 | 0.974 | 0.962 | 0.011 | 0.971 | 0.020 | 0.038 | 0.003 | 0.932 |

Same-accuracy pairs tell the story: *pruneFT90* and *distill25* on digits have the
identical 1.9-point accuracy gap, but pruneFT90 has 3.6× the TV distance and 12× the
ECE gap (its confidences are wrong even where its labels are right). Rank correlation
(accuracy-gap ranking vs TES ranking): digits 0.97, toxic 0.56.

![](results/exp1_parity_vs_trust.png)

### Exp 2 — Trust-equivalence under drift (`exp2_drift_curves.py`)

Flip rate at severity 0 → max, and the accuracy gap at max severity (digits; teacher
accuracy at max severity in last column):

| drift | method | flips @0 | flips @max | Δacc @max | acc L @max |
|---|---|---|---|---|---|
| blur | pruneFT90 | 0.031 | 0.282 | −0.047 | 0.461 |
| blur | quant4b | 0.002 | 0.087 | 0.010 | 0.461 |
| blur | distill25 | 0.023 | 0.294 | 0.084 | 0.461 |
| noise | pruneFT90 | 0.031 | 0.281 | 0.015 | 0.535 |
| noise | quant4b | 0.002 | 0.070 | 0.004 | 0.535 |
| noise | distill25 | 0.023 | 0.324 | 0.038 | 0.535 |
| brightness | quant4b | 0.002 | 0.167 | −0.025 | 0.281 |
| rotate | distill25 | 0.023 | 0.122 | −0.032 | 0.140 |
| occlude | distill25 | 0.023 | 0.259 | −0.007 | 0.355 |
| toxic polarity flip 0→50 % | prune90 | 0.014 | 0.054 | −0.001 | 0.458 |

Findings. (i) Flip rate grows 5–30× under drift while the accuracy gap stays within a
few points and often changes sign — the two models fail on *different* inputs. (ii)
Quantisation is the most drift-stable family (it perturbs every weight slightly rather
than removing capacity). (iii) Distillation and fine-tuned pruning, which *look*
trust-equivalent in-distribution (2–3 % flips), diverge the most under drift, because
they re-learned a different function that merely agrees on the training manifold.
(iv) Concept drift on the toxicity task produces small absolute flips because both
models collapse to ~46 % accuracy together; the informative regime is covariate drift.

![](results/exp2_drift_curves.png)

### Exp 3 — Does compression amplify identity-term bias? (`exp3_subgroup_disparity.py`)

FPR on *clean* comments mentioning a group-A vs group-B identity term (test set has no
identity–label correlation; training set has `spurious`):

| spurious (train) | method | acc | FPR clean A | FPR clean B | **gap B−A** | silent flips A | silent flips B |
|---|---|---|---|---|---|---|---|
| 0.5 (none) | teacher | 0.974 | 0.031 | 0.029 | −0.002 | 0 | 0 |
| 0.5 | prune95 | 0.960 | 0.046 | 0.060 | 0.013 | 0.019 | 0.026 |
| 0.8 | teacher | 0.974 | 0.010 | 0.048 | 0.038 | 0 | 0 |
| 0.8 | prune80 | 0.973 | 0.014 | 0.049 | 0.035 | 0.003 | 0.003 |
| 0.8 | **prune95** | 0.947 | 0.043 | 0.120 | **0.077** | 0.032 | **0.050** |
| 0.8 | pruneFT95 | 0.974 | 0.007 | 0.048 | 0.041 | 0.006 | 0.002 |
| 0.8 | quant3b | 0.973 | 0.011 | 0.046 | 0.035 | 0.002 | 0.003 |
| 0.8 | distill10 | 0.973 | 0.006 | 0.049 | 0.043 | 0.005 | 0.003 |
| 0.95 | teacher | 0.970 | 0.001 | 0.073 | 0.072 | 0 | 0 |
| 0.95 | prune95 | 0.954 | 0.022 | 0.100 | 0.078 | 0.028 | 0.033 |
| 0.95 | **pruneFT95** | 0.973 | 0.002 | 0.053 | **0.051** | 0.002 | 0.001 |

Findings. (i) The teacher inherits the training bias (gap 3.8 → 7.2 points as spurious
correlation rises). (ii) Aggressive pruning without fine-tuning **doubles** the gap and
concentrates silent flips on group B (5.0 % vs 3.2 %): the first weights to be pruned are
the small, distributed ones that encode the genuine lexical signal, leaving the strong
identity-term weights — the compressed model relies *more* on the shortcut. (iii) Every
other family preserves the teacher's bias level, and fine-tuned pruning at high spurious
correlation *reduces* it. This yields a concrete, testable recommendation for the
cyberbullying-detection setting: never deploy one-shot pruned models without a subgroup
silent-flip audit.

![](results/exp3_fpr_gap.png)

### Exp 4 — Flip prediction and selective deferral (`exp4_flip_predictor_deferral.py`)

Flip predictor: logistic regression on 5 features of the *compressed* model only
(max-prob, margin, entropy, logit norm, input distance to training mean), trained on the
calibration split. Hybrid accuracy when the top-k % inputs by predicted flip risk are
sent to the large model (digits, 3 seeds):

| model / drift | flip rate | AUROC predictor | AUROC max-prob | hybrid @0 % | @10 % | @20 % | @30 % | @50 % | random @30 % | large model |
|---|---|---|---|---|---|---|---|---|---|---|
| prune90 / none | 0.332 | 0.833 | 0.809 | 0.665 | 0.739 | 0.795 | 0.850 | 0.937 | 0.765 | 0.982 |
| prune90 / rotate 0.3 | 0.545 | 0.709 | 0.695 | 0.370 | 0.413 | 0.447 | 0.491 | 0.557 | 0.452 | 0.662 |
| quant3b / none | 0.026 | 0.970 | 0.971 | 0.961 | 0.981 | 0.982 | 0.982 | 0.982 | 0.967 | 0.982 |
| quant3b / noise 0.6 | 0.129 | 0.901 | 0.906 | 0.790 | 0.783 | 0.794 | 0.803 | 0.806 | 0.788 | 0.808 |
| distill25 / none | 0.023 | 0.977 | 0.975 | 0.964 | 0.982 | 0.982 | 0.982 | 0.982 | 0.969 | 0.982 |
| distill25 / rotate 0.3 | 0.187 | 0.863 | 0.863 | **0.705** | 0.698 | 0.686 | 0.680 | 0.662 | 0.683 | 0.662 |

Findings. (i) In-distribution, 10 % deferral makes 3-bit quantised and 25 %-width
distilled models exactly trust-equivalent to the teacher (0.982). (ii) The learned flip
predictor beats max-probability only marginally (+0.02 AUROC): disagreement is almost
entirely explained by the compressed model's own low margin, so the simplest possible
gate suffices — a useful negative result. (iii) Under drift the flip predictor degrades
gracefully (AUROC 0.70–0.90) **but** the premise of deferral breaks: for distillation
under rotation, the student beats the teacher and every deferral step *lowers* accuracy.
A deferral policy therefore needs a drift detector (the group's incremental
anomaly-detection work is the natural plug-in) to decide which model is the reference.

![](results/exp4_deferral.png)

### Exp 5 — Size / trust Pareto frontier (`exp5_pareto.py`)

Fine sweeps per family (digits: prune 0–98 %, quant 16–2 bit, rank 100–2 %, student
width 256–4). Frontier membership counts:

| task | (size, TES) frontier | (size, accuracy) frontier | TES non-monotone steps | acc non-monotone steps |
|---|---|---|---|---|
| digits | quant 6, distill 4, prune 1 | distill 4, quant 2 | 0 / 36 | 0 / 36 |
| toxic | quant 6, distill 1, prune 1 | distill 2 | 0 / 36 | 0 / 36 |

Representative points (digits): 4-bit quant — size 0.091, acc 0.983, TES 0.987;
distill-64 — size 0.106, acc 0.967, TES 0.933; prune-90 — size 0.106, acc 0.665,
TES 0.320; 3-bit quant — size 0.050, acc 0.965, TES 0.941; distill-32 — size 0.041,
acc 0.958, TES 0.892.

Findings. At equal size, quantisation is 2–5 TES points more trust-equivalent than
distillation even where distillation is more accurate; the practitioner who selects by
accuracy will pick the less trustworthy model. Trust is monotone in every knob, so a
single calibration sweep per family suffices for selection.

![](results/exp5_pareto.png)

## 5. Limitations

* Small MLPs and small/synthetic data. The toxicity task is near-linearly separable,
  which is why every family except one-shot pruning is near-perfectly trust-equivalent
  on it in-distribution; the bias and drift findings are the informative part.
* Post-training compression only (no quantisation-aware training, no structured
  pruning); real transformer compression (the group's setting) may reorder families.
* The drift generators are synthetic; real drift (new slang, new platforms) is needed.
* TES weights its three terms equally; a learned or task-specific weighting is future work.

## 6. From here to a paper (concrete plan)

1. **Scale to the group's existing setup**: BERT-family models on hate-speech /
   cyberbullying datasets (Davidson, HateXplain, the group's own cross-platform
   cyberbullying data) with identity-term annotations; compression via Hugging Face
   `optimum` (8/4-bit), movement pruning and TinyBERT/DistilBERT students. All metric
   code in `tec/metrics.py` applies unchanged to any pair of probability matrices.
2. **Temporal drift for real**: train on 2017 posts, test on 2019–2024 posts (new slang
   = natural polarity flip); report flip-rate growth per compression family — the
   drift-amplification result of Exp 2 on real data is the headline figure.
3. **Subgroup silent-flip audit** as a deployment checklist: Exp 3 on HateXplain target
   groups; test whether the "pruning amplifies shortcut reliance" mechanism holds in
   transformers via attribution (integrated gradients on identity tokens before/after
   compression).
4. **Drift-aware deferral**: combine the max-prob gate with an incremental isolation
   forest on embeddings (the group's SIGMOD/CODS 2024 drift detector) to decide when the
   large model should stop being the reference — a system paper angle.
5. **Theory sketch**: relate silent-flip rate to the margin distribution of the large
   model and the weight-perturbation norm of the compression (a Lipschitz argument gives
   flips ≤ P(margin < L·‖ΔW‖)); check empirically that Exp 2's drift amplification is
   explained by margins shrinking under drift.

Target venues: CIKM / WSDM / ECIR (information-retrieval–flavoured ML, where the
group publishes), or the ACL/EMNLP findings track for the bias angle; a workshop paper
on drift-aware deferral at a NeurIPS/ICLR workshop on trustworthy or efficient ML.

## 7. Reproduction

`bash run_all.sh` (≈25 min CPU). Each experiment is independent and seeds are fixed.

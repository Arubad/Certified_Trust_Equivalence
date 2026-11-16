#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# Synthetic experiments (each script adds experiments/ and the repo root to sys.path,
# so they import `common` and `tec` when run from the repo root).
for e in exp1_certificate_validity exp2_certified_efficiency exp3_group_certificate exp4_drift_monitors exp5_streaming_policy exp7_adaptive_audit exp8_theory_checks; do
  echo "=============== $e ==============="
  python experiments/$e.py
done
# Exp 6 analysis (real BERT-base + compressions) and Exp 9 (split-grid certificate; its part (b)
# uses the HateXplain predictions). Inference is a separate Colab step (see README); these run
# only if the predictions file is present.
if [ -f results/real/real_preds.npz ]; then
  echo "=============== exp6_real_models ==============="
  python experiments/real/exp6_real_models.py
  echo "=============== exp9_split_certificate ==============="
  python experiments/exp9_split_certificate.py
else
  echo "skip exp6, exp9: results/real/real_preds.npz missing (run experiments/real/colab_infer.py on Colab first)"
fi
# Exp 10 analysis (fine-tuned / distilled compressions; GPU job A = experiments/real/colab_train_hx.py).
if [ -f results/real/real_preds_ft.npz ]; then
  echo "=============== exp10 (exp6_real_models --tag exp10) ==============="
  python experiments/real/exp6_real_models.py --npz results/real/real_preds_ft.npz \
    --models distil6,p50ft,p70ft,q4ft --tag exp10
else
  echo "skip exp10: results/real/real_preds_ft.npz missing (run experiments/real/colab_train_hx.py on a GPU first)"
fi
# Exp 11 analysis (CivilComments; GPU job B = experiments/real/colab_civil.py).
if [ -f results/real/civil_preds.npz ]; then
  echo "=============== exp11_civil ==============="
  python experiments/real/exp11_civil.py --npz results/real/civil_preds.npz --out results/real
else
  echo "skip exp11: results/real/civil_preds.npz missing (run experiments/real/colab_civil.py on a GPU first)"
fi

#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# Synthetic experiments (each script adds experiments/ and the repo root to sys.path,
# so they import `common` and `tec` when run from the repo root).
for e in exp1_certificate_validity exp2_certified_efficiency exp3_group_certificate exp4_drift_monitors exp5_streaming_policy exp7_adaptive_audit exp8_theory_checks; do
  echo "=============== $e ==============="
  python experiments/$e.py
done
# Exp 6 analysis (real BERT-base + compressions). Inference is a separate Colab step
# (see README); the analysis runs only if its predictions file is present.
if [ -f results/real/real_preds.npz ]; then
  echo "=============== exp6_real_models ==============="
  python experiments/real/exp6_real_models.py
else
  echo "skip exp6: results/real/real_preds.npz missing (run experiments/real/colab_infer.py on Colab first)"
fi

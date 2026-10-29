#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
for e in exp1_certificate_validity exp2_certified_efficiency exp3_group_certificate exp4_drift_monitors exp5_streaming_policy; do
  echo "=============== $e ==============="
  python experiments/$e.py
done

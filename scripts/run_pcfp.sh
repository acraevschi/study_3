#!/usr/bin/env bash
# PCFP experiment: every stage after `data` (run that first), logged. `ldl_tune` precedes
# `select` (the LDL selector uses the frozen settings; in the default repeated-random design
# `select` only writes each draw's training samples). The artifact audit runs before any
# outer-test LDL fit and again at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/pcfp_v2.yaml}
LOG=outputs/$(basename "$CFG" .yaml)_run.log
for st in splits ldl_tune select audit ldl evaluate outcomes typology audit; do
  echo "[$(date -u +%FT%TZ)] start $st" | tee -a "$LOG"
  .venv/bin/python -m morph_ldl.cli "$st" --config "$CFG" >> "$LOG" 2>&1
  echo "[$(date -u +%FT%TZ)] end $st" | tee -a "$LOG"
done

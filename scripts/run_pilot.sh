#!/usr/bin/env bash
# Full pilot: every stage after `data`, `splits` and `ldl_tune` (run those first), logged.
# The artifact audit runs before any outer-test LDL fit and again at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/pilot.yaml}
LOG=outputs/$(basename "$CFG" .yaml)_run.log
for st in select audit ldl selector evaluate outcomes audit; do
  echo "[$(date -u +%FT%TZ)] start $st" | tee -a "$LOG"
  .venv/bin/python -m morph_ldl.cli "$st" --config "$CFG" >> "$LOG" 2>&1
  echo "[$(date -u +%FT%TZ)] end $st" | tee -a "$LOG"
done

#!/usr/bin/env bash
# ==============================================================================
# study_3 Comprehensive Multi-Tier Opaque-Box E2E Test Suite Runner
# ==============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================================================="
echo " Starting study_3 Opaque-Box E2E Test Suite"
echo " Project Root: $PROJECT_ROOT"
echo " Date: $(date)"
echo "=============================================================================="

# Check Python environment
echo ""
echo "--- Environment Check ---"
python3 --version
pytest --version
Rscript -e "cat(sprintf('R version: %s\n', R.version.string))"

echo ""
echo "--- Running Pytest Suite Across All Tiers ---"
pytest tests/ -v \
  --tb=short \
  --durations=10 \
  -rA \
  "$@"

EXIT_CODE=$?

echo ""
echo "=============================================================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo " [SUCCESS] All executable tests executed and passed successfully!"
else
  echo " [FAILURE/PENDING] Some tests failed or were skipped pending milestone artifacts."
fi
echo "=============================================================================="

exit $EXIT_CODE

#!/usr/bin/env bash
# Run the QGIS-free unit test suite for the BAMBI QGIS plugin.
#
# The tests stub out the qgis package (see tests/conftest.py), so they run in
# any plain Python environment with the test dependencies installed —
# typically inside the Docker image: docker compose run --rm tests

REPORTS_DIR="reports"
mkdir -p "$REPORTS_DIR/coverage/unit"
RUN_TS=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

# Each tier keeps its coverage data in its own subdirectory (a plain
# `.coverage`, NOT `.coverage.<tier>` — that collides with coverage's
# parallel-file naming and pytest-cov's erase would wipe the siblings).
# run_coverage_combine.sh merges the per-tier files into an overall figure.
export COVERAGE_FILE="${COVERAGE_FILE:-$REPORTS_DIR/coverage/unit/.coverage}"

echo "============================================"
echo " BAMBI Plugin — Unit Tests"
echo " $RUN_TS"
echo "============================================"

# --cov-fail-under is a regression ratchet: raise it as coverage grows
# (see DECOUPLING_PLAN.md and EXCHANGE_FORMAT_PLAN.md §12.6), never lower it to
# make a build pass. Raised to 26 by Phase 4 (see EXCHANGE_FORMAT_PLAN.md §12.6).
pytest tests \
    --junitxml="$REPORTS_DIR/pytest.xml" \
    --cov=bambi_wildlife_detection \
    --cov-report=term \
    --cov-fail-under=26 \
    2>&1 | tee "$REPORTS_DIR/pytest.txt"
exit "${PIPESTATUS[0]}"

#!/usr/bin/env bash
# Run the QGIS-free unit test suite for the BAMBI QGIS plugin.
#
# The tests stub out the qgis package (see tests/conftest.py), so they run in
# any plain Python environment with the test dependencies installed —
# typically inside the Docker image: docker compose run --rm tests

REPORTS_DIR="reports"
mkdir -p "$REPORTS_DIR"
RUN_TS=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

echo "============================================"
echo " BAMBI Plugin — Unit Tests"
echo " $RUN_TS"
echo "============================================"

pytest tests \
    --junitxml="$REPORTS_DIR/pytest.xml" \
    --cov=bambi_wildlife_detection \
    --cov-report=term \
    2>&1 | tee "$REPORTS_DIR/pytest.txt"
exit "${PIPESTATUS[0]}"

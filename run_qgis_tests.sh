#!/usr/bin/env bash
# Real-QGIS smoke tests: construct the plugin's Qt widgets inside a headless
# QGIS and round-trip the project configuration through a real QgsProject.
#
# Requires the QGIS image: docker compose run --rm qgis-tests

REPORTS_DIR="reports"
mkdir -p "$REPORTS_DIR/coverage/qgis"
RUN_TS=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

# Own coverage subdir for the combined report (see run_coverage_combine.sh).
# Constructing the widgets exercises thousands of GUI lines the QGIS-free
# unit suite cannot reach, so this is a meaningful contributor to the total.
export COVERAGE_FILE="${COVERAGE_FILE:-$REPORTS_DIR/coverage/qgis/.coverage}"

echo "============================================"
echo " BAMBI Plugin — Real-QGIS Smoke Tests"
echo " $RUN_TS"
echo "============================================"

pytest tests_qgis -v -p no:cacheprovider \
    --junitxml="$REPORTS_DIR/pytest-qgis.xml" \
    --cov=bambi_wildlife_detection \
    --cov-report=term \
    2>&1 | tee "$REPORTS_DIR/pytest-qgis.txt"
exit "${PIPESTATUS[0]}"

#!/usr/bin/env bash
# Merge the per-tier coverage data into one combined report.
#
# Each tier writes reports/coverage/<tier>/.coverage (a plain data file in
# its own subdir). This stages copies under coverage's parallel-file naming
# and combines whatever tiers are present. Standalone:
#     docker compose run --rm coverage-combine
# (usually the last step of run_all_tests.sh).

COV_DIR="reports/coverage"
STAGE="$COV_DIR/.stage"
export COVERAGE_FILE="$COV_DIR/.coverage"

rm -rf "$STAGE"
mkdir -p "$STAGE"

found=0
for tier in unit integration qgis; do
    src="$COV_DIR/$tier/.coverage"
    if [ -f "$src" ]; then
        # Stage under a parallel-file name so `coverage combine <dir>` picks
        # it up; staged copies are disposable so a tier's pytest-cov erase
        # can never clobber another tier's data.
        cp "$src" "$STAGE/.coverage.$tier"
        found=$((found + 1))
        echo "  + $tier"
    else
        echo "  - $tier (not run)"
    fi
done

if [ "$found" -eq 0 ]; then
    echo "No per-tier coverage data found under $COV_DIR/*/.coverage"
    echo "Run the tiers first (e.g. bash run_all_tests.sh)."
    rm -rf "$STAGE"
    exit 1
fi

echo
echo "============================================"
echo " BAMBI Plugin — Combined Coverage"
echo "============================================"

rm -f "$COVERAGE_FILE"
coverage combine "$STAGE"        # merges the staged .coverage.* files
coverage report --precision=1 2>&1 | tee "$COV_DIR/combined.txt"
coverage html -d "$COV_DIR/html" >/dev/null 2>&1
coverage xml -o "$COV_DIR/coverage.xml" >/dev/null 2>&1
rm -rf "$STAGE"

echo
echo "HTML report: $COV_DIR/html/index.html"

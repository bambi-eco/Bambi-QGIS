#!/usr/bin/env bash
# Run the BAMBI pipeline integration tests (real dataset flight + real DEM).
#
# Requires the heavy integration image: docker compose run --rm integration
# First run downloads ~12 GB (flight 6) + DEM tiles into the bambi-test-data
# volume; later runs reuse the cache and are much faster.

REPORTS_DIR="reports"
mkdir -p "$REPORTS_DIR"
RUN_TS=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

echo "============================================"
echo " BAMBI Plugin — Integration Tests"
echo " $RUN_TS"
echo "============================================"

# Headless X server for moderngl/alfspy rendering (DISPLAY=:99 is set in the
# image). Skip if one is already running.
if ! pgrep -x Xvfb > /dev/null 2>&1; then
    Xvfb :99 -screen 0 1280x1024x24 &
    XVFB_PID=$!
    trap 'kill $XVFB_PID 2>/dev/null' EXIT
    sleep 2
fi

pytest tests_integration -v \
    --junitxml="$REPORTS_DIR/pytest-integration.xml" \
    --cov=bambi_wildlife_detection \
    --cov-report=term \
    2>&1 | tee "$REPORTS_DIR/pytest-integration.txt"
exit "${PIPESTATUS[0]}"

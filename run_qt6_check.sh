#!/usr/bin/env bash
# Qt6 / QGIS 4 compatibility check for the BAMBI QGIS plugin.
#
# Runs the same pyqt5_to_pyqt6.py dry run that plugins.qgis.org applies to
# every uploaded plugin version, inside the official
# ghcr.io/qgis/pyqgis4-checker image (see docker-compose.yml). Run the
# script without --dry_run to auto-apply the fixes — but review the diff:
# it resolves ambiguous enum members by guess and can pick the wrong enum.

PLUGIN_DIR="bambi_wildlife_detection"
REPORTS_DIR="reports"
LOG_FILE="$REPORTS_DIR/pyqt6_checker.log"

mkdir -p "$REPORTS_DIR"
RUN_TS=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

echo "============================================"
echo " BAMBI Plugin — Qt6 Compatibility Check"
echo " $RUN_TS"
echo "============================================"

pyqt5_to_pyqt6.py --dry_run --logfile "$LOG_FILE" "$PLUGIN_DIR"
STATUS=$?

# The log holds one line per incompatibility after a header line.
FINDINGS=$(tail -n +2 "$LOG_FILE" 2>/dev/null | grep -c .)

echo ""
if [ "$STATUS" -ne 0 ] && [ "$FINDINGS" -eq 0 ]; then
    # e.g. a SyntaxError while parsing a file (a UTF-8 BOM is enough) —
    # the traceback is printed above.
    echo "FAIL: checker crashed with exit code $STATUS before completing."
    exit 1
elif [ "$FINDINGS" -ne 0 ]; then
    echo "FAIL: $FINDINGS Qt6 incompatibilities found:"
    tail -n +2 "$LOG_FILE"
    echo "Full log saved to $LOG_FILE"
    exit 1
fi
echo "PASS: No Qt6 incompatibilities detected."

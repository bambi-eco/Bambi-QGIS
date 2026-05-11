#!/usr/bin/env bash
# Run security and linting checks on the BAMBI QGIS plugin.

PLUGIN_DIR="bambi_wildlife_detection"
REPORTS_DIR="reports"
FAILED=0

mkdir -p "$REPORTS_DIR"
RUN_TS=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

echo "============================================"
echo " BAMBI Plugin — Security & Linting Checks"
echo " $RUN_TS"
echo "============================================"

# ---- 1. Bandit: security vulnerability scan ----
echo ""
echo "=== 1/3  Bandit (security vulnerabilities) ==="
# Medium+ severity to skip informational noise.
# B603/B607 (subprocess) are expected in the dependency manager.
bandit -r "$PLUGIN_DIR" --severity-level medium 2>&1 | tee "$REPORTS_DIR/bandit.txt"
if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    BANDIT_STATUS="PASS"
else
    BANDIT_STATUS="FAIL"
    FAILED=1
fi

# ---- 2. detect-secrets: secret / credential scan ----
echo ""
echo "=== 2/3  detect-secrets (secrets scan) ==="
detect-secrets scan "$PLUGIN_DIR" > "$REPORTS_DIR/detect-secrets.json" 2>&1
if [ $? -eq 0 ]; then
    SECRETS_MSG=$(python3 - <<'EOF'
import json, sys
with open('reports/detect-secrets.json') as f:
    data = json.load(f)
count = sum(len(v) for v in data.get('results', {}).values())
if count:
    print(f"FAIL: {count} potential secret(s) found:")
    for path, items in data['results'].items():
        for item in items:
            print(f"  {path}:{item['line_number']} [{item['type']}]")
    sys.exit(1)
print("PASS: No secrets detected.")
EOF
)
    SECRETS_EXIT=$?
    echo "$SECRETS_MSG"
    if [ $SECRETS_EXIT -ne 0 ]; then
        SECRETS_STATUS="FAIL"
        FAILED=1
    else
        SECRETS_STATUS="PASS"
    fi
else
    cat "$REPORTS_DIR/detect-secrets.json"
    SECRETS_STATUS="FAIL"
    FAILED=1
fi

# ---- 3. flake8: style and linting ----
echo ""
echo "=== 3/3  flake8 (style & linting) ==="
flake8 "$PLUGIN_DIR" 2>&1 | tee "$REPORTS_DIR/flake8.txt"
if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    FLAKE8_STATUS="PASS"
else
    FLAKE8_STATUS="FAIL"
    FAILED=1
fi

# ---- Summary ----
echo ""
echo "============================================"
{
    echo "Run: $RUN_TS"
    echo "Bandit:         $BANDIT_STATUS"
    echo "detect-secrets: $SECRETS_STATUS"
    echo "flake8:         $FLAKE8_STATUS"
    if [ "$FAILED" -eq 0 ]; then
        echo "Overall:        PASS"
    else
        echo "Overall:        FAIL"
    fi
} | tee "$REPORTS_DIR/summary.txt"
echo "============================================"
echo "Reports saved to $REPORTS_DIR/"

if [ "$FAILED" -eq 0 ]; then
    exit 0
else
    exit 1
fi

#!/usr/bin/env bash
# Qt 6 widget tests: the plugin's GUI modules imported, its windows built and
# its widgets sent mouse, wheel and key events under PyQt6 - without QGIS
# (qgis.core / qgis.gui are stand-ins, see tests_qt6/conftest.py).
#
# Needs the unit test dependencies and PyQt6:
#     pip install PyQt6 PyQt6-QScintilla
# The tests are skipped where PyQt6 is not installed.

REPORTS_DIR="reports"
mkdir -p "$REPORTS_DIR"
RUN_TS=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

echo "============================================"
echo " BAMBI Plugin - Qt 6 Widget Tests"
echo " $RUN_TS"
echo "============================================"

# A process of its own: qgis.PyQt stands for PyQt6 here, which must not meet
# the stub of the unit tests or a real QGIS.
# The Python that has PyQt6, not whichever "pytest" is first on the PATH.
QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}" \
"${PYTHON:-python3}" -m pytest tests_qt6 -v -p no:cacheprovider \
    --junitxml="$REPORTS_DIR/pytest-qt6.xml" \
    2>&1 | tee "$REPORTS_DIR/pytest-qt6.txt"
exit "${PIPESTATUS[0]}"

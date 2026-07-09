#!/usr/bin/env bash
# Run all three test tiers and print the COMBINED coverage.
#
# Each tier runs in its own container (they need mutually incompatible
# environments — the unit suite is deliberately QGIS-free, the integration
# image carries torch/alfspy, the smoke tests need real QGIS), then a small
# container merges their coverage data into one figure.
#
# Run from the repo root on the HOST (it drives docker compose):
#     bash run_all_tests.sh
# or the equivalent from PowerShell:
#     docker compose run --rm tests;
#     docker compose run --rm integration;
#     docker compose run --rm qgis-tests;
#     docker compose run --rm coverage-combine
#
# The individual services still work on their own for running one tier.

# Clear stale coverage from previous runs (each tier recreates its subdir).
rm -rf reports/coverage
mkdir -p reports/coverage

status=0

echo "########## 1/3  Unit tests ##########"
docker compose run --rm tests || status=1

echo "########## 2/3  Integration tests ##########"
docker compose run --rm integration || status=1

echo "########## 3/3  Real-QGIS smoke tests ##########"
docker compose run --rm qgis-tests || status=1

echo "########## Combined coverage ##########"
docker compose run --rm coverage-combine || status=1

if [ "$status" -ne 0 ]; then
    echo
    echo "NOTE: one or more test tiers reported failures (see logs above)."
    echo "The combined coverage above still reflects every tier that ran."
fi
exit "$status"

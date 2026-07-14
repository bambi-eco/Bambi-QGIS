#!/usr/bin/env python3
"""Build the QGIS plugin zip for upload to the plugin repository.

Packages bambi_wildlife_detection/ from the working tree, excluding byte-compiled
caches and editor/test artefacts that the plugin repository validator rejects.
"""
import configparser
import fnmatch
import sys
import zipfile
from pathlib import Path

PLUGIN = "bambi_wildlife_detection"
ROOT = Path(__file__).parent
SRC = ROOT / PLUGIN

EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".git"}
EXCLUDE_FILES = ["*.pyc", "*.pyo", "*.pyd", "*.so", "*~", ".DS_Store", "Thumbs.db", "*.orig", "*.rej"]


def is_excluded(path: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return True
    return any(fnmatch.fnmatch(path.name, pat) for pat in EXCLUDE_FILES)


def main() -> int:
    if not SRC.is_dir():
        print(f"error: {SRC} not found", file=sys.stderr)
        return 1

    cfg = configparser.ConfigParser()
    cfg.read(SRC / "metadata.txt", encoding="utf-8")
    version = cfg["general"]["version"]

    out = ROOT / f"{PLUGIN}-{version}.zip"
    files = sorted(p for p in SRC.rglob("*") if p.is_file() and not is_excluded(p.relative_to(ROOT)))

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())

    # The plugin repository rejects archives containing byte-compiled caches.
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    bad = [n for n in names if "__pycache__" in n or n.endswith((".pyc", ".pyo"))]
    if bad:
        print(f"error: {len(bad)} cache entries leaked into the zip, e.g. {bad[0]}", file=sys.stderr)
        return 1
    tops = {n.split("/")[0] for n in names}
    if tops != {PLUGIN}:
        print(f"error: zip must contain exactly one top-level dir, got {sorted(tops)}", file=sys.stderr)
        return 1

    print(f"{out.name}: {len(names)} files, {out.stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

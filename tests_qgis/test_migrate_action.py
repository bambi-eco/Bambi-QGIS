# -*- coding: utf-8 -*-
"""The "Migrate 5.x…" action on the dock widget (EXCHANGE_FORMAT_PLAN.md §10, Phase 1).

The migration logic itself is headless and covered by ``tests/test_migration.py``;
what needs a real QGIS is the widget wiring — that the button appears exactly
when there is something to migrate, and disappears once there is not.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from bambi_wildlife_detection.core import migration, store


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


@pytest.fixture
def legacy_folder(tmp_path):
    root = str(tmp_path / "flight")
    _write(os.path.join(root, "detections_t", "detections.txt"),
           "# frame x1 y1 x2 y2 confidence class_id\n"
           "1 10.00 20.00 30.00 40.00 0.9000 0\n"
           "1 50.00 60.00 70.00 80.00 0.8000 1\n")
    return root


def test_button_is_hidden_without_a_folder(dock):
    assert not dock.migrate_btn.isVisibleTo(dock)


def test_button_appears_for_a_legacy_folder(dock, legacy_folder):
    dock.target_folder_edit.setText(legacy_folder)
    dock._on_target_folder_changed()
    assert dock.migrate_btn.isVisibleTo(dock)


def test_button_stays_hidden_for_an_empty_folder(dock, tmp_path):
    empty = str(tmp_path / "empty")
    os.makedirs(empty, exist_ok=True)
    dock.target_folder_edit.setText(empty)
    dock._on_target_folder_changed()
    assert not dock.migrate_btn.isVisibleTo(dock)


def test_button_disappears_once_migrated(dock, legacy_folder):
    dock.target_folder_edit.setText(legacy_folder)
    dock._on_target_folder_changed()
    assert dock.migrate_btn.isVisibleTo(dock)

    migration.migrate_project(legacy_folder)
    dock._refresh_migrate_button()
    assert not dock.migrate_btn.isVisibleTo(dock)


def test_migration_writes_a_store_the_widget_can_see(dock, legacy_folder):
    dock.target_folder_edit.setText(legacy_folder)
    migration.migrate_project(legacy_folder, log_fn=dock.log)

    conn = store.open_store(
        store.stage_path(legacy_folder, store.DETECTIONS, "t"),
        store.DETECTIONS, "t")
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM detections").fetchone()["n"] == 2
    conn.close()
    assert "Vocabulary" in dock.log_text.toPlainText()

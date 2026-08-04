# -*- coding: utf-8 -*-
"""Phase 0 de-risking spikes for the 6.0 store (EXCHANGE_FORMAT_PLAN.md §12.3).

Two questions have to be answered before any stage depends on the layout:

1. Does an **aspatial GeoPackage** — payload tables with no geometry, announced
   via ``gpkg_contents`` with ``data_type='attributes'`` — actually open in
   QGIS? If not, §2.1's fallback to plain ``.sqlite`` applies and the container
   boilerplate goes away.
2. What happens when a stage file is **rewritten while a layer is loaded from
   it**? This decides whether Phase 5 reads the store directly or via exported
   layers (§13, Q1).

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import sqlite3

import pytest

from qgis.core import QgsProviderRegistry, QgsVectorLayer

from bambi_wildlife_detection.core import store


@pytest.fixture
def detections_file(tmp_path):
    """A detections stage file holding three rows."""
    path = store.stage_path(str(tmp_path), store.DETECTIONS, "t")
    conn = store.open_store(path, store.DETECTIONS, "t")
    conn.executemany(
        "INSERT INTO detections (frame, x1, y1, x2, y2, confidence, source_id) "
        "VALUES (?, ?, ?, ?, ?, ?, 1)",
        [(1, 10.0, 20.0, 30.0, 40.0, 0.9),
         (1, 50.0, 60.0, 70.0, 80.0, 0.8),
         (2, 11.0, 21.0, 31.0, 41.0, 0.7)])
    conn.commit()
    conn.close()
    return path


def _layer(path, table):
    return QgsVectorLayer(f"{path}|layername={table}", table, "ogr")


# ---------------------------------------------------------------------------
# Spike 1 — does an aspatial GeoPackage open at all?
# ---------------------------------------------------------------------------

def test_aspatial_geopackage_layer_is_valid(detections_file):
    layer = _layer(detections_file, "detections")
    assert layer.isValid(), (
        "aspatial GeoPackage did not open in QGIS — §2.1 fallback to plain "
        ".sqlite applies")


def test_aspatial_layer_exposes_its_rows(detections_file):
    layer = _layer(detections_file, "detections")
    assert layer.featureCount() == 3


def test_aspatial_layer_exposes_its_columns(detections_file):
    layer = _layer(detections_file, "detections")
    names = {field.name() for field in layer.fields()}
    assert {"frame", "x1", "y1", "x2", "y2", "confidence",
            "species_id", "source_id", "attributes"} <= names


def test_attribute_values_round_trip(detections_file):
    layer = _layer(detections_file, "detections")
    frames = sorted(feature["frame"] for feature in layer.getFeatures())
    assert frames == [1, 1, 2]


def test_registered_tables_are_discoverable_as_sublayers(detections_file):
    """The Browser panel lists what gpkg_contents announces."""
    meta = QgsProviderRegistry.instance().providerMetadata("ogr")
    sublayers = meta.querySublayers(detections_file)
    assert "detections" in {sub.name() for sub in sublayers}


@pytest.mark.parametrize("kind,table", [
    (store.TRACKS, "tracks"),
    (store.LABELS, "label_tracks"),
    (store.GEOREFERENCED, "detections_geo"),
    (store.FOV, "fov_polygons"),
    (store.SEGMENTATION, "segments"),
])
def test_every_stage_kind_opens_in_qgis(tmp_path, kind, table):
    path = store.stage_path(str(tmp_path), kind, "t")
    store.open_store(path, kind, "t").close()
    assert _layer(path, table).isValid()


def test_project_store_opens_in_qgis(tmp_path):
    path = store.project_path(str(tmp_path))
    store.open_store(path, store.PROJECT).close()
    layer = _layer(path, "species")
    assert layer.isValid()
    assert layer.featureCount() == 12  # 3 base classes + 9 seeded species


# ---------------------------------------------------------------------------
# Spike 2 — rewriting a stage while a layer is loaded (§13, Q1)
#
# These record *Linux* behaviour and prove nothing about Windows, which locks a
# file that a QGIS layer holds open — confirmed from the existing pain with
# added layers and GeoTIFF exports. The conclusion drawn from that (§11) is that
# the plugin must never hand a stage file path to QGIS as a layer source; these
# tests document what happens if it does anyway.
# ---------------------------------------------------------------------------

def test_stage_rewrite_while_layer_loaded_is_permitted_on_linux(detections_file):
    """POSIX permits it; Windows does not — hence the memory-layer rule (§11).

    Note what this test is *not*: evidence that file-backed layers are safe. It
    only records that the permissive platform is permissive. The rule in §11
    rests on Windows behaviour, which this tier cannot exercise.
    """
    layer = _layer(detections_file, "detections")
    assert layer.isValid()

    conn = store.open_store(detections_file, store.DETECTIONS, "t")
    try:
        with store.transaction(conn):
            conn.execute("DELETE FROM detections")
            conn.execute(
                "INSERT INTO detections (frame, source_id) VALUES (99, 1)")
    except sqlite3.OperationalError as exc:  # pragma: no cover — the bad path
        pytest.fail(f"stage rewrite blocked by a loaded QGIS layer: {exc}")
    finally:
        conn.close()


def test_reloading_shows_the_rewritten_rows(detections_file):
    layer = _layer(detections_file, "detections")
    assert layer.featureCount() == 3

    conn = store.open_store(detections_file, store.DETECTIONS, "t")
    with store.transaction(conn):
        conn.execute("DELETE FROM detections")
        conn.execute("INSERT INTO detections (frame, source_id) VALUES (99, 1)")
    conn.close()

    layer.reload()
    assert layer.featureCount() == 1
    assert [feature["frame"] for feature in layer.getFeatures()] == [99]


def test_opening_a_stage_file_takes_no_exclusive_lock(detections_file):
    """Opening must not need an exclusive lock, or every read would contend.

    ``PRAGMA journal_mode`` briefly requires one, which is why it is issued once
    at creation rather than on every connect (``core/gpkg.py``).
    """
    layer = _layer(detections_file, "detections")
    assert layer.isValid()

    first = store.open_store(detections_file, store.DETECTIONS, "t")
    second = store.open_store(detections_file, store.DETECTIONS, "t")
    assert first.execute(
        "SELECT COUNT(*) AS n FROM detections").fetchone()["n"] == 3
    assert second.execute(
        "SELECT COUNT(*) AS n FROM detections").fetchone()["n"] == 3
    first.close()
    second.close()


def test_stage_file_deletion_while_layer_loaded_is_permitted_on_linux(
        detections_file):
    """`rm` of a stage file — unlocked here, blocked on Windows (§7)."""
    import os

    layer = _layer(detections_file, "detections")
    assert layer.isValid()
    os.remove(detections_file)
    assert not os.path.exists(detections_file)


def test_store_leaves_no_journal_sidecars(tmp_path):
    """DELETE journal mode: the stage file is the whole stage file.

    WAL would leave ``-wal``/``-shm`` next to it — invisible state a user
    copying "the .gpkg" would drop, and unusable on a network share (§2.1).
    """
    import os

    path = store.stage_path(str(tmp_path), store.DETECTIONS, "t")
    conn = store.open_store(path, store.DETECTIONS, "t")
    conn.execute("INSERT INTO detections (frame, source_id) VALUES (1, 1)")
    conn.commit()
    conn.close()

    folder = os.path.dirname(path)
    assert sorted(os.listdir(folder)) == ["detections.gpkg"]

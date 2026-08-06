# -*- coding: utf-8 -*-
"""Separate result per species (§8.2).

Pooling several species into one density raster or one abundance figure answers
a different question from reporting them apart, so the split is a choice. What
matters here is that the strata are the species that actually have something to
count, that one stratum's outputs cannot overwrite the next one's, and that a
species too thin to fit does not take the others down with it.
"""
import json
import os

import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import detection_store, store, track_store


@pytest.fixture
def processor():
    return BambiProcessor()


@pytest.fixture
def survey(tmp_path):
    """Two species with geo-referenced detections, and several with none."""
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()

    boxes = [
        {"frame": 0, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "1"},
        {"frame": 1, "x1": 12.0, "y1": 22.0, "x2": 32.0, "y2": 42.0,
         "confidence": 0.8, "source_class": "1"},
        {"frame": 2, "x1": 50.0, "y1": 60.0, "x2": 70.0, "y2": 80.0,
         "confidence": 0.7, "source_class": "5"},
    ]
    detection_store.record_detections(root, "t", boxes)

    project = store.open_store(store.project_path(root), store.PROJECT)
    source_id = project.execute(
        "SELECT source_id FROM detection_sources").fetchone()["source_id"]
    project.executemany(
        "INSERT INTO class_mapping (source_id, source_class, species_id) "
        "VALUES (?, ?, ?)", [(source_id, "1", 1), (source_id, "5", 5)])
    project.commit()
    project.close()

    detection_store.record_detections(root, "t", boxes)

    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_georeference(root, "t", [
        {"detection_id": i,
         "gx1": 500000.0 + n * 10, "gy1": 5300000.0 + n * 10, "gz1": 400.0,
         "gx2": 500010.0 + n * 10, "gy2": 5300010.0 + n * 10, "gz2": 400.0}
        for n, i in enumerate(ids)])
    track_store.record_tracks(root, "t", [
        {"track_id": 1, "detection_id": ids[0]},
        {"track_id": 1, "detection_id": ids[1]},
        {"track_id": 2, "detection_id": ids[2]},
    ])
    return root


def _config(root, **extra):
    config = {"target_folder": root, "detection_camera": "T",
              "tracking_camera": "T", "target_epsg": 32633,
              "density_source": "detections", "density_cell_size": 5.0,
              "density_bandwidth": 25.0}
    config.update(extra)
    return config


# ---------------------------------------------------------------------------
# Which strata are run
# ---------------------------------------------------------------------------

def test_off_by_default_there_is_one_run(processor, survey):
    assert processor.analytics_strata(_config(survey), "detections") == [
        ("", None)]


def test_the_current_filter_travels_into_the_single_run(processor, survey):
    config = _config(survey, analytics_species_ids=[1])
    assert processor.analytics_strata(config, "detections") == [("", [1])]


def test_one_stratum_per_species_with_something_to_count(processor, survey):
    config = _config(survey, analytics_per_species=True)
    strata = processor.analytics_strata(config, "detections")
    assert [name for name, _ids in strata] == ["roe deer", "chamois"]


def test_species_with_nothing_to_count_are_skipped(processor, survey):
    """Otherwise every project produces an empty result per unused species."""
    config = _config(survey, analytics_per_species=True)
    names = [name for name, _ in
             processor.analytics_strata(config, "detections")]
    assert "unknown" not in names
    assert "animal" not in names


def test_the_skipped_species_are_reported(processor, survey):
    messages = []
    processor.analytics_strata(
        _config(survey, analytics_per_species=True), "detections",
        messages.append)
    assert any("No detections for" in m for m in messages)


def test_the_species_filter_narrows_the_strata(processor, survey):
    """Both controls apply: tick one species, get one result."""
    config = _config(survey, analytics_per_species=True,
                     analytics_species_ids=[1])
    strata = processor.analytics_strata(config, "detections")
    assert [name for name, _ in strata] == ["roe deer"]


def test_each_stratum_carries_exactly_one_species(processor, survey):
    config = _config(survey, analytics_per_species=True)
    for _name, ids in processor.analytics_strata(config, "detections"):
        assert len(ids) == 1


def test_nothing_to_count_at_all_is_an_error(processor, tmp_path):
    root = str(tmp_path / "empty")
    os.makedirs(root, exist_ok=True)
    store.open_store(store.project_path(root), store.PROJECT).close()
    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.9, "source_class": "0"}])

    with pytest.raises(RuntimeError, match="nothing to produce"):
        processor.analytics_strata(
            _config(root, analytics_per_species=True), "detections")


def test_a_project_without_a_store_says_what_to_do(processor, tmp_path):
    with pytest.raises(RuntimeError, match="Migrate"):
        processor.analytics_strata(
            _config(str(tmp_path / "legacy"), analytics_per_species=True),
            "detections")


# ---------------------------------------------------------------------------
# The filenames the strata write to
# ---------------------------------------------------------------------------

def test_species_names_become_filename_fragments():
    assert BambiProcessor.species_slug("roe deer") == "roe-deer"
    assert BambiProcessor.species_slug("Rotwild/Hirsch") == "rotwild-hirsch"
    assert BambiProcessor.species_slug("  ") == "species"


def test_two_species_cannot_collide_on_one_filename():
    red = BambiProcessor.species_slug("red deer")
    roe = BambiProcessor.species_slug("roe deer")
    assert red != roe


def test_a_density_run_writes_one_raster_per_species(processor, survey):
    pytest.importorskip("rasterio")
    processor.run_density_heatmap(
        _config(survey, analytics_per_species=True))

    written = sorted(os.listdir(os.path.join(survey, "analytics_t")))
    assert "density_detections_roe-deer.tif" in written
    assert "density_detections_chamois.tif" in written
    assert "density_detections.tif" not in written


def test_the_result_records_which_species_it_counted(processor, survey):
    pytest.importorskip("rasterio")
    processor.run_density_heatmap(
        _config(survey, analytics_per_species=True))

    path = os.path.join(survey, "analytics_t",
                        "density_detections_roe-deer.json")
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["species"] == "roe deer"


def test_a_pooled_run_still_writes_the_plain_name(processor, survey):
    pytest.importorskip("rasterio")
    processor.run_density_heatmap(_config(survey))

    written = os.listdir(os.path.join(survey, "analytics_t"))
    assert "density_detections.tif" in written


def test_the_pooled_raster_is_not_one_of_the_species(processor, survey):
    """Pooling is a different question, not the first species' answer."""
    pytest.importorskip("rasterio")
    processor.run_density_heatmap(_config(survey))
    path = os.path.join(survey, "analytics_t", "density_detections.json")
    with open(path, encoding="utf-8") as fh:
        pooled = json.load(fh)

    processor.run_density_heatmap(
        _config(survey, analytics_per_species=True))
    path = os.path.join(survey, "analytics_t",
                        "density_detections_roe-deer.json")
    with open(path, encoding="utf-8") as fh:
        roe = json.load(fh)

    assert pooled["species"] is None
    assert pooled["n_points"] > roe["n_points"]


# ---------------------------------------------------------------------------
# The filter must not be quietly dropped
# ---------------------------------------------------------------------------

def test_a_filter_matching_nothing_is_refused_not_ignored(processor, survey):
    """The legacy text files know nothing about species, so falling back to
    them would answer with every animal under one species' name."""
    config = _config(survey, analytics_species_ids=[999])
    with pytest.raises(RuntimeError, match="match the selected species"):
        processor._collect_analytics_points(config, "detections")


def test_the_pooled_run_may_still_fall_back(processor, survey):
    """Without a filter there is nothing to drop, so 5.x projects keep
    working."""
    points, _suffix = processor._collect_analytics_points(
        _config(survey), "detections")
    assert points


def test_provenance_does_not_survive_into_a_legacy_run(processor, survey,
                                                       tmp_path):
    """It described the previous run, not this one."""
    processor._collect_analytics_points(_config(survey), "detections")
    assert processor._last_analytics_provenance is not None

    legacy = str(tmp_path / "legacy")
    os.makedirs(os.path.join(legacy, "tracks_t"), exist_ok=True)
    with open(os.path.join(legacy, "tracks_t", "tracks.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("00000000,1,1.0,2.0,3.0,4.0,5.0,6.0,0.9,0,0\n")

    try:
        processor._collect_analytics_points(_config(legacy), "tracks")
    except Exception:  # noqa: BLE001 — the reset is what is under test
        pass
    assert processor._last_analytics_provenance is None


def test_the_coverage_map_records_no_population_filter(processor, survey):
    """It counts frames, so a species filter would describe nothing."""
    import inspect

    source = inspect.getsource(BambiProcessor.run_coverage_map)
    assert "population_filter" not in source.split("No population filter")[-1]


def test_population_estimation_filters_the_perpendicular_tracks(processor,
                                                                survey):
    """Without this every species in a per-species run gets the same tracks,
    and so the same numbers under a different name."""
    import inspect

    source = inspect.getsource(BambiProcessor._population_project_table)
    assert "analytics_species_ids" in source
    assert "class_id" in source


def test_the_population_estimate_records_its_own_filter(processor):
    """Read off the config, not off whichever analytic ran last."""
    import inspect

    source = inspect.getsource(
        BambiProcessor._run_population_estimation_once)
    filter_block = source.split('result["population_filter"]')[-1]
    assert "_last_analytics_provenance" not in filter_block
    assert "analytics_species_ids" in filter_block


def test_population_estimation_uses_its_own_camera(processor, survey):
    """It has pop_camera; asking tracking_camera would count the other
    camera's animals."""
    config = _config(survey, analytics_per_species=True,
                     pop_camera="T", tracking_camera="W")
    strata = processor.analytics_strata(config, "tracks", None,
                                        camera_key="pop_camera")
    assert [name for name, _ in strata] == ["roe deer", "chamois"]


def test_the_default_camera_key_still_follows_the_source(processor, survey):
    config = _config(survey, analytics_per_species=True)
    by_source = processor.analytics_strata(config, "detections")
    assert [name for name, _ in by_source] == ["roe deer", "chamois"]

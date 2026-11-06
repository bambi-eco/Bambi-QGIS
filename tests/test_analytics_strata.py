# -*- coding: utf-8 -*-
"""Survey strata and pooled points (core.survey, §8.2).

The survey analytics work on the flights ticked on the Survey Analytics tab,
not on the flight selected for processing. Species are chosen by *name* -
ids are assigned per flight - and every result lands in the survey folder.
Pooling several species into one result answers a different question from
reporting them apart, so the split is a choice: the strata are the species
that actually have something to count, one stratum's outputs cannot
overwrite the next one's, and a species too thin to fit does not take the
others down with it.
"""
import json
import os

import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import detection_store, store, track_store
from bambi_wildlife_detection.core.survey import Survey, SurveyFlight


@pytest.fixture
def processor():
    return BambiProcessor()


def _flight(root):
    """Two species with geo-referenced detections, and several with none."""
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


@pytest.fixture
def survey(tmp_path):
    root = str(tmp_path / "flight_a")
    os.makedirs(root)
    return _flight(root)


@pytest.fixture
def second(tmp_path):
    root = str(tmp_path / "flight_b")
    os.makedirs(root)
    return _flight(root)


def _config(*folders, species=None, per_species=False, epsg=32633, **extra):
    survey = Survey(
        folder=os.path.join(os.path.dirname(folders[0]), "survey"),
        modality="t",
        flights=[SurveyFlight(name=os.path.basename(f), target_folder=f,
                              epsg=epsg) for f in folders],
        species=species, per_species=per_species)
    config = {"survey": survey.to_dict(), "target_epsg": epsg,
              "density_source": "detections", "density_cell_size": 5.0,
              "density_bandwidth": 25.0}
    config.update(extra)
    return config


def _results(config):
    return Survey.from_config(config).analytics_folder


# ---------------------------------------------------------------------------
# Which strata are run
# ---------------------------------------------------------------------------

def test_off_by_default_there_is_one_run(processor, survey):
    assert processor.survey_strata(_config(survey), "detections") == [
        ("", None)]


def test_the_current_filter_travels_into_the_single_run(processor, survey):
    config = _config(survey, species=["roe deer"])
    assert processor.survey_strata(config, "detections") == [
        ("", ["roe deer"])]


def test_one_stratum_per_species_with_something_to_count(processor, survey):
    config = _config(survey, per_species=True)
    strata = processor.survey_strata(config, "detections")
    assert [name for name, _names in strata] == ["roe deer", "chamois"]


def test_species_with_nothing_to_count_are_skipped(processor, survey):
    """Otherwise every survey produces an empty result per unused species."""
    names = [name for name, _ in processor.survey_strata(
        _config(survey, per_species=True), "detections")]
    assert "unknown" not in names
    assert "animal" not in names


def test_the_skipped_species_are_reported(processor, survey):
    messages = []
    processor.survey_strata(
        _config(survey, per_species=True), "detections", messages.append)
    assert any("No detections for" in m for m in messages)


def test_the_species_filter_narrows_the_strata(processor, survey):
    """Both controls apply: tick one species, get one result."""
    config = _config(survey, per_species=True, species=["roe deer"])
    strata = processor.survey_strata(config, "detections")
    assert strata == [("roe deer", ["roe deer"])]


def test_nothing_to_count_at_all_is_an_error(processor, tmp_path):
    root = str(tmp_path / "empty")
    os.makedirs(root, exist_ok=True)
    store.open_store(store.project_path(root), store.PROJECT).close()
    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.9, "source_class": "0"}])

    with pytest.raises(RuntimeError, match="nothing to produce"):
        processor.survey_strata(_config(root, per_species=True),
                                "detections")


# ---------------------------------------------------------------------------
# The survey itself
# ---------------------------------------------------------------------------

def test_no_flight_is_a_clear_error(processor, tmp_path):
    config = {"survey": Survey(folder=str(tmp_path)).to_dict()}
    with pytest.raises(RuntimeError, match="No flight is selected"):
        processor.run_density_heatmap(config)


def test_flights_in_different_crs_cannot_be_pooled(processor, survey,
                                                   second):
    config = _config(survey, second)
    config["survey"]["flights"][1]["epsg"] = 32632
    with pytest.raises(RuntimeError, match="different target CRSs"):
        processor.run_density_heatmap(config)


def test_a_missing_flight_folder_is_named(processor, survey, tmp_path):
    config = _config(survey, str(tmp_path / "gone"))
    with pytest.raises(RuntimeError, match="gone"):
        processor.run_density_heatmap(config)


def test_two_flights_are_pooled(processor, survey, second):
    survey_obj = Survey.from_config(_config(survey, second))
    one, _ = processor._survey_points(
        Survey.from_config(_config(survey)), "detections")
    both, provenance = processor._survey_points(survey_obj, "detections")
    assert len(both) == 2 * len(one)
    assert [f["flight"] for f in provenance["flights"]] == [
        "flight_a", "flight_b"]


def test_a_species_a_flight_does_not_know_is_not_an_error(
        processor, survey, second):
    """A name missing from one flight means that flight has none of it."""
    points, provenance = processor._survey_points(
        Survey.from_config(_config(survey, second)), "detections",
        ["wolverine"])
    assert points == []
    assert all(f["n_points"] == 0 for f in provenance["flights"])


def test_results_go_to_the_survey_folder(processor, survey, second):
    pytest.importorskip("rasterio")
    config = _config(survey, second)
    processor.run_density_heatmap(config)
    folder = _results(config)
    assert os.path.isfile(os.path.join(folder, "density_detections.tif"))
    assert not os.path.isdir(os.path.join(survey, "analytics_t"))
    with open(os.path.join(folder, "density_detections.json"),
              encoding="utf-8") as fh:
        stats = json.load(fh)
    assert stats["flights"] == ["flight_a", "flight_b"]
    per_flight = stats["population_filter"]["flights"]
    assert stats["n_points"] == sum(f["n_points"] for f in per_flight)
    assert per_flight[0]["n_points"] == per_flight[1]["n_points"] > 0


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
    config = _config(survey, per_species=True)
    processor.run_density_heatmap(config)

    written = sorted(os.listdir(_results(config)))
    assert "density_detections_roe-deer.tif" in written
    assert "density_detections_chamois.tif" in written
    assert "density_detections.tif" not in written


def test_the_result_records_which_species_it_counted(processor, survey):
    pytest.importorskip("rasterio")
    config = _config(survey, per_species=True)
    processor.run_density_heatmap(config)

    path = os.path.join(_results(config), "density_detections_roe-deer.json")
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["species"] == "roe deer"


def test_the_pooled_raster_is_not_one_of_the_species(processor, survey):
    """Pooling is a different question, not the first species' answer."""
    pytest.importorskip("rasterio")
    config = _config(survey)
    processor.run_density_heatmap(config)
    with open(os.path.join(_results(config), "density_detections.json"),
              encoding="utf-8") as fh:
        pooled = json.load(fh)

    config = _config(survey, per_species=True)
    processor.run_density_heatmap(config)
    with open(os.path.join(_results(config),
                           "density_detections_roe-deer.json"),
              encoding="utf-8") as fh:
        roe = json.load(fh)

    assert pooled["species"] is None
    assert pooled["n_points"] > roe["n_points"]


# ---------------------------------------------------------------------------
# The filter must not be quietly dropped
# ---------------------------------------------------------------------------

def test_a_filter_matching_nothing_is_refused_not_ignored(processor, survey):
    with pytest.raises(RuntimeError, match="match the selected species"):
        processor.run_density_heatmap(_config(survey, species=["wolverine"]))


def test_provenance_does_not_survive_a_run_that_counted_nothing(processor,
                                                                survey):
    """It described the previous run, not this one."""
    pytest.importorskip("rasterio")
    processor.run_density_heatmap(_config(survey))
    assert processor._last_analytics_provenance is not None

    processor._last_analytics_provenance = None
    with pytest.raises(RuntimeError):
        processor.run_density_heatmap(_config(survey, species=["wolverine"]))
    assert processor._last_analytics_provenance is None


def test_a_flight_without_a_store_is_told_to_migrate(processor, tmp_path):
    """The legacy files are not read, so "no store" is the whole answer."""
    legacy = str(tmp_path / "legacy")
    os.makedirs(os.path.join(legacy, "tracks_t"), exist_ok=True)
    with open(os.path.join(legacy, "tracks_t", "tracks.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("00000000,1,1.0,2.0,3.0,4.0,5.0,6.0,0.9,0,0")

    with pytest.raises(FileNotFoundError, match="Migrate"):
        processor._survey_points(Survey.from_config(_config(legacy)),
                                 "tracks")


def test_the_coverage_map_records_no_population_filter(processor):
    """It counts frames, so a species filter would describe nothing."""
    import inspect

    source = inspect.getsource(BambiProcessor.run_coverage_map)
    assert "population_filter" not in source.split("No population filter")[-1]


def test_population_estimation_filters_the_perpendicular_tracks(processor):
    """Without this every species in a per-species run gets the same tracks,
    and so the same numbers under a different name."""
    import inspect

    source = inspect.getsource(BambiProcessor._population_project_table)
    assert "species_ids" in source
    assert "class_id" in source


def test_the_population_estimate_records_its_own_filter(processor):
    """Read off this run's species, not off whichever analytic ran last."""
    import inspect

    source = inspect.getsource(
        BambiProcessor._run_population_estimation_once)
    filter_block = source.split('result["population_filter"]')[-1]
    assert "_last_analytics_provenance" not in filter_block
    assert "names" in filter_block


def test_population_strata_follow_the_tracks(processor, survey):
    strata = processor.survey_strata(_config(survey, per_species=True),
                                     "tracks")
    assert [name for name, _ in strata] == ["roe deer", "chamois"]

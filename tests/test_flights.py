# -*- coding: utf-8 -*-
"""Flights and stored configuration (EXCHANGE_FORMAT_PLAN.md §10.2).

The constraint under test throughout is that one flight is one target folder.
It is not a convention: the stage files are per folder and modality with no
flight dimension, so two flights sharing a folder would overwrite each other's
detections.
"""
import os

import pytest

from bambi_wildlife_detection.core import config_schema, flights, store
from bambi_wildlife_detection.core.flights import FlightError


# ---------------------------------------------------------------------------
# Configuration in the store
# ---------------------------------------------------------------------------

def test_config_round_trips(tmp_path):
    root = str(tmp_path)
    flights.save_config(root, {"Input/TargetFolder": root,
                               "Detection/Confidence": 0.5})
    stored = flights.load_config(root)
    assert stored["Input/TargetFolder"] == root
    assert stored["Detection/Confidence"] == "0.5"


def test_saving_replaces_the_previous_configuration(tmp_path):
    root = str(tmp_path)
    flights.save_config(root, {"a": "1", "b": "2"})
    flights.save_config(root, {"a": "3"})
    assert flights.load_config(root) == {"a": "3"}


def test_loading_a_folder_without_a_store(tmp_path):
    assert flights.load_config(str(tmp_path / "nothing")) == {}
    assert not flights.has_config(str(tmp_path / "nothing"))


def test_none_is_stored_as_empty(tmp_path):
    root = str(tmp_path)
    flights.save_config(root, {"Input/AirDataPath": None})
    assert flights.load_config(root)["Input/AirDataPath"] == ""


def test_config_survives_a_reopen(tmp_path):
    """The folder is self-describing: the settings live with the outputs."""
    root = str(tmp_path)
    flights.save_config(root, {"Detection/Confidence": 0.7})
    conn = store.open_store(store.project_path(root), store.PROJECT)
    conn.close()
    assert flights.load_config(root)["Detection/Confidence"] == "0.7"


# ---------------------------------------------------------------------------
# The readers and writers config_schema already accepts
# ---------------------------------------------------------------------------

def test_readers_round_trip_every_entry_kind(tmp_path):
    """The whole schema goes through the store, not a hand-picked subset."""
    widget_values = {entry.key: entry.default for entry in
                     config_schema.CONFIG_ENTRIES}
    widget_values.update({
        "Input/TargetFolder": "/some/where",
        "Detection/Confidence": 0.75,
        "Detection/AllFrames": True,
    })

    values = {}
    write_str, write_double, write_bool = flights.config_writers(values)
    config_schema.save_config_entries(
        widget_values, write_str, write_double, write_bool)

    flights.save_config(str(tmp_path), values)
    read_str, read_double, read_bool = flights.config_readers(
        flights.load_config(str(tmp_path)))

    assert read_str("Input/TargetFolder") == "/some/where"
    assert read_double("Detection/Confidence") == pytest.approx(0.75)
    assert read_bool("Detection/AllFrames") is True


def test_readers_fall_back_to_defaults(tmp_path):
    read_str, read_double, read_bool = flights.config_readers({})
    assert read_str("missing", "fallback") == "fallback"
    assert read_double("missing", 1.5) == 1.5
    assert read_bool("missing", True) is True


def test_readers_tolerate_unparsable_values():
    read_str, read_double, read_bool = flights.config_readers(
        {"a": "not-a-number", "b": "maybe"})
    assert read_double("a", 2.0) == 2.0
    assert read_bool("b") is False


def test_an_empty_string_reads_as_the_default():
    read_str, _read_double, read_bool = flights.config_readers({"a": ""})
    assert read_str("a", "fallback") == "fallback"
    assert read_bool("a", True) is True


# ---------------------------------------------------------------------------
# One flight, one folder
# ---------------------------------------------------------------------------

def test_adding_a_flight(tmp_path):
    result = flights.add_flight([], "Meadow", str(tmp_path / "a"))
    assert result == [{"name": "Meadow", "target_folder": str(tmp_path / "a")}]


def test_two_flights_cannot_share_a_folder(tmp_path):
    """They would overwrite each other's detections (§10.2)."""
    folder = str(tmp_path / "a")
    existing = flights.add_flight([], "Meadow", folder)
    with pytest.raises(FlightError, match="already uses this folder"):
        flights.add_flight(existing, "Forest", folder)


def test_the_folder_check_ignores_case_and_separators(tmp_path):
    folder = str(tmp_path / "a")
    existing = flights.add_flight([], "Meadow", folder)
    with pytest.raises(FlightError, match="already uses this folder"):
        flights.add_flight(existing, "Forest", folder + os.sep)


def test_a_flight_needs_a_folder():
    with pytest.raises(FlightError, match="needs a target folder"):
        flights.add_flight([], "Meadow", "  ")


def test_a_flight_needs_a_name(tmp_path):
    with pytest.raises(FlightError, match="needs a name"):
        flights.add_flight([], "  ", str(tmp_path / "a"))


def test_flight_names_must_be_unique(tmp_path):
    existing = flights.add_flight([], "Meadow", str(tmp_path / "a"))
    with pytest.raises(FlightError, match="already exists"):
        flights.add_flight(existing, "meadow", str(tmp_path / "b"))


def test_names_are_stripped(tmp_path):
    result = flights.add_flight([], "  Meadow  ", str(tmp_path / "a"))
    assert result[0]["name"] == "Meadow"


# ---------------------------------------------------------------------------
# Renaming and re-pointing
# ---------------------------------------------------------------------------

def test_renaming_a_flight(tmp_path):
    existing = flights.add_flight([], "Meadow", str(tmp_path / "a"))
    renamed = flights.rename_flight(existing, 0, "North Meadow")
    assert renamed[0]["name"] == "North Meadow"
    assert renamed[0]["target_folder"] == existing[0]["target_folder"]


def test_renaming_to_its_own_name_is_allowed(tmp_path):
    existing = flights.add_flight([], "Meadow", str(tmp_path / "a"))
    assert flights.rename_flight(existing, 0, "Meadow")[0]["name"] == "Meadow"


def test_renaming_onto_another_flight_is_refused(tmp_path):
    existing = flights.add_flight([], "Meadow", str(tmp_path / "a"))
    existing = flights.add_flight(existing, "Forest", str(tmp_path / "b"))
    with pytest.raises(FlightError, match="already exists"):
        flights.rename_flight(existing, 1, "Meadow")


def test_renaming_does_not_mutate_the_input(tmp_path):
    existing = flights.add_flight([], "Meadow", str(tmp_path / "a"))
    flights.rename_flight(existing, 0, "Renamed")
    assert existing[0]["name"] == "Meadow"


def test_repointing_a_flight(tmp_path):
    existing = flights.add_flight([], "Meadow", str(tmp_path / "a"))
    moved = flights.set_folder(existing, 0, str(tmp_path / "c"))
    assert moved[0]["target_folder"] == str(tmp_path / "c")


def test_repointing_onto_another_flights_folder_is_refused(tmp_path):
    existing = flights.add_flight([], "Meadow", str(tmp_path / "a"))
    existing = flights.add_flight(existing, "Forest", str(tmp_path / "b"))
    with pytest.raises(FlightError, match="already uses this folder"):
        flights.set_folder(existing, 1, str(tmp_path / "a"))


def test_an_unknown_index_is_rejected():
    with pytest.raises(FlightError, match="No such flight"):
        flights.rename_flight([], 0, "x")


# ---------------------------------------------------------------------------
# Names and groups
# ---------------------------------------------------------------------------

def test_default_name_uses_the_parent_of_a_qgis_folder():
    """The default target folder is <input>/qgis, so 'qgis' identifies nothing."""
    assert flights.default_name("/data/flight_6/qgis") == "flight_6"


def test_default_name_uses_the_folder_when_it_is_meaningful():
    assert flights.default_name("/data/meadow_survey") == "meadow_survey"


def test_default_name_of_nothing():
    assert flights.default_name("") == "Flight"


def test_group_name_follows_the_flight_name():
    """Renaming a flight renames its group, so nothing stale is left behind."""
    assert flights.group_name({"name": "North Meadow"}) == "North Meadow"


def test_group_name_of_an_unnamed_flight():
    assert flights.group_name({"name": "  "}) == "Flight"


def test_find_by_folder(tmp_path):
    existing = flights.add_flight([], "Meadow", str(tmp_path / "a"))
    existing = flights.add_flight(existing, "Forest", str(tmp_path / "b"))
    assert flights.find_by_folder(existing, str(tmp_path / "b")) == 1
    assert flights.find_by_folder(existing, str(tmp_path / "z")) is None


# ---------------------------------------------------------------------------
# Adding a flight to a multi-project analysis (§10.2)
# ---------------------------------------------------------------------------

def test_dem_metadata_comes_from_the_flights_own_config(tmp_path):
    """The point of storing config with the outputs: no path to remember."""
    root = str(tmp_path)
    flights.save_config(root, {"Input/DemMetadataPath": "/data/dem.json"})
    assert flights.dem_metadata_path(root) == "/data/dem.json"


def test_dem_metadata_falls_back_to_the_mesh_sibling(tmp_path):
    root = str(tmp_path)
    mesh = tmp_path / "dem.glb"
    mesh.write_text("x")
    (tmp_path / "dem.json").write_text("{}")
    flights.save_config(root, {"Input/DemPath": str(mesh)})
    assert flights.dem_metadata_path(root).endswith("dem.json")


def test_dem_metadata_is_empty_when_the_sibling_is_missing(tmp_path):
    root = str(tmp_path)
    flights.save_config(root, {"Input/DemPath": str(tmp_path / "dem.glb")})
    assert flights.dem_metadata_path(root) == ""


def test_dem_metadata_of_a_flight_without_config(tmp_path):
    assert flights.dem_metadata_path(str(tmp_path / "nothing")) == ""


def test_analysis_entries_describe_each_flight(tmp_path):
    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    flights.save_config(a, {"Input/DemMetadataPath": "/data/a.json"})
    entries = flights.analysis_entries(
        [{"name": "Meadow", "target_folder": a},
         {"name": "Forest", "target_folder": b}])

    assert [e["name"] for e in entries] == ["Meadow", "Forest"]
    assert entries[0]["dem"] == "/data/a.json"
    assert entries[1]["dem"] == ""


def test_analysis_entries_exclude_the_active_flight(tmp_path):
    """The analytics tools add the current project separately."""
    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    entries = flights.analysis_entries(
        [{"name": "Meadow", "target_folder": a},
         {"name": "Forest", "target_folder": b}], exclude=a)
    assert [e["name"] for e in entries] == ["Forest"]


def test_analysis_entries_skip_flights_without_a_folder():
    assert flights.analysis_entries([{"name": "Ghost", "target_folder": ""}]) == []


def test_analysis_entries_of_an_empty_project():
    assert flights.analysis_entries([]) == []

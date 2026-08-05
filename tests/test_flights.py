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


# ---------------------------------------------------------------------------
# What a new flight starts with (§10.2)
# ---------------------------------------------------------------------------

RECORDINGS = ["Input/ThermalVideoPaths", "Input/RgbVideoPaths",
              "Input/AirdataPath", "Input/DemPath", "Input/CorrectionPath",
              "Input/ThermalCalibrationPath"]


def _current():
    values = {entry.key: entry.default for entry in config_schema.CONFIG_ENTRIES}
    values.update({key: "/previous/flight/file" for key in RECORDINGS})
    values["Detection/Confidence"] = 0.77
    values["Input/TargetCrs"] = "EPSG:32633"
    return values


@pytest.mark.parametrize("copy_configuration", [True, False])
def test_recordings_are_always_cleared(copy_configuration):
    """A new flight is new data; inheriting paths would process it twice."""
    values = flights.new_flight_values(_current(), copy_configuration)
    for key in RECORDINGS:
        assert values[key] == "", key


def test_copying_keeps_the_processing_settings():
    values = flights.new_flight_values(_current(), copy_configuration=True)
    assert values["Detection/Confidence"] == 0.77


def test_defaults_reset_the_processing_settings():
    values = flights.new_flight_values(_current(), copy_configuration=False)
    entry = [e for e in config_schema.CONFIG_ENTRIES
             if e.key == "Detection/Confidence"][0]
    assert values["Detection/Confidence"] == entry.default


def test_the_crs_is_a_setting_not_a_recording():
    """Blanking it on every new flight would only have to be retyped."""
    assert not flights.is_recording_input("Input/TargetCrs")
    values = flights.new_flight_values(_current(), copy_configuration=True)
    assert values["Input/TargetCrs"] == "EPSG:32633"


def test_the_target_folder_is_not_treated_as_a_recording():
    """The new flight sets it itself."""
    assert not flights.is_recording_input("Input/TargetFolder")


def test_the_legacy_toggle_follows_the_configuration_choice():
    assert not flights.is_recording_input("Input/WriteLegacyTextOutputs")


@pytest.mark.parametrize("key", RECORDINGS)
def test_recording_inputs_are_recognised(key):
    assert flights.is_recording_input(key)


def test_non_input_keys_are_never_recordings():
    assert not flights.is_recording_input("Detection/Confidence")
    assert not flights.is_recording_input("ALFS/TileSize")


def test_every_schema_key_gets_a_value():
    """The form is fully written, so nothing keeps the old flight's value."""
    values = flights.new_flight_values({}, copy_configuration=False)
    assert set(values) == {e.key for e in config_schema.CONFIG_ENTRIES}


@pytest.mark.parametrize("kind,expected", [
    ("str", ""), ("bool", False), ("bool01", False),
    ("int_str", 0), ("int_double", 0), ("double", 0.0),
])
def test_empty_values_match_the_widget_kind(kind, expected):
    """A checkbox cannot be handed an empty string."""
    entry = config_schema.ConfigEntry("Scope/Key", kind, None)
    assert flights._entry_default(entry) == expected
    assert type(flights._entry_default(entry)) is type(expected)


def test_declared_defaults_win_over_the_typed_empty():
    entry = [e for e in config_schema.CONFIG_ENTRIES
             if e.key == "Detection/Confidence"][0]
    assert flights._entry_default(entry) == entry.default


# ---------------------------------------------------------------------------
# Removing a flight
# ---------------------------------------------------------------------------

def test_remove_flight_drops_only_that_one():
    listed = [{"name": "A", "target_folder": "/a"},
              {"name": "B", "target_folder": "/b"},
              {"name": "C", "target_folder": "/c"}]
    assert [f["name"] for f in flights.remove_flight(listed, 1)] == \
        ["A", "C"]


def test_remove_flight_leaves_the_original_alone():
    listed = [{"name": "A", "target_folder": "/a"},
              {"name": "B", "target_folder": "/b"}]
    flights.remove_flight(listed, 0)
    assert len(listed) == 2


def test_remove_flight_rejects_an_unknown_index():
    listed = [{"name": "A", "target_folder": "/a"}]
    for index in (-1, 1, 99):
        with pytest.raises(flights.FlightError):
            flights.remove_flight(listed, index)


def test_removing_the_only_flight_empties_the_list():
    assert flights.remove_flight(
        [{"name": "A", "target_folder": "/a"}], 0) == []


def test_the_folder_becomes_free_again_after_removal():
    """The refusal to share folders must not outlive the flight."""
    listed = [{"name": "A", "target_folder": "/a"}]
    remaining = flights.remove_flight(listed, 0)
    flights.validate_folder("/a", remaining)  # no raise
    assert flights.add_flight(remaining, "A2", "/a")


def test_the_name_becomes_free_again_after_removal():
    listed = [{"name": "A", "target_folder": "/a"},
              {"name": "B", "target_folder": "/b"}]
    remaining = flights.remove_flight(listed, 0)
    assert flights.add_flight(remaining, "A", "/c")


# -- which flight is active afterwards --------------------------------------

def test_removing_an_earlier_flight_keeps_the_active_one():
    assert flights.active_after_removal(3, 0, 2) == 1


def test_removing_a_later_flight_leaves_the_active_index_alone():
    assert flights.active_after_removal(3, 2, 0) == 0


def test_removing_the_active_flight_falls_back_to_its_neighbour():
    """Not to the first, so deleting several in a row walks the list."""
    assert flights.active_after_removal(3, 1, 1) == 1
    assert flights.active_after_removal(2, 1, 1) == 0


def test_removing_the_last_flight_reports_none_active():
    assert flights.active_after_removal(1, 0, 0) == -1


def test_removing_the_final_flight_in_the_list_stays_in_range():
    assert flights.active_after_removal(3, 2, 2) == 1

# -*- coding: utf-8 -*-
"""One survey, one folder (core.survey).

Without a folder of its own a survey writes to ``bambi_survey`` next to the
QGIS project - and, while the project has not been saved, next to the first
flight. Saving the project for the first time therefore moved the default:
the results written until then stayed behind and the tools went on in a new,
empty folder. :func:`folder_to_keep` is the rule that stops that.
"""
import os

from bambi_wildlife_detection.core import survey


def _folder(tmp_path, *parts, filled=False):
    folder = os.path.join(str(tmp_path), *parts)
    os.makedirs(folder)
    if filled:
        with open(os.path.join(folder, "census.csv"), "w") as handle:
            handle.write("x")
    return folder


# -- where the default is ----------------------------------------------------

def test_the_default_is_next_to_the_project():
    home = os.path.join(os.sep, "data", "projects")
    assert survey.default_folder(home, [os.path.join(os.sep, "f", "qgis")]) \
        == os.path.join(home, "bambi_survey")


def test_an_unsaved_project_uses_the_first_flights_parent():
    flight = os.path.join(os.sep, "data", "meadow", "qgis")
    assert survey.default_folder("", [flight, "/data/forest/qgis"]) == \
        os.path.join(os.sep, "data", "meadow", "bambi_survey")


def test_a_flight_without_a_folder_is_skipped():
    flight = os.path.join(os.sep, "data", "forest", "qgis")
    assert survey.default_folder("", ["", flight]) == \
        os.path.join(os.sep, "data", "forest", "bambi_survey")


def test_a_trailing_separator_does_not_change_the_parent():
    flight = os.path.join(os.sep, "data", "meadow", "qgis") + os.sep
    assert survey.default_folder("", [flight]) == \
        os.path.join(os.sep, "data", "meadow", "bambi_survey")


def test_nothing_to_go_by_gives_no_default():
    assert survey.default_folder("", []) == ""
    assert survey.default_folder("", [""]) == ""


# -- what counts as in use ---------------------------------------------------

def test_a_folder_with_something_in_it_holds_results(tmp_path):
    assert survey.holds_results(_folder(tmp_path, "s", filled=True))


def test_an_empty_or_missing_folder_does_not(tmp_path):
    assert not survey.holds_results(_folder(tmp_path, "empty"))
    assert not survey.holds_results(os.path.join(str(tmp_path), "missing"))
    assert not survey.holds_results("")


def test_a_file_is_not_a_survey_folder(tmp_path):
    path = os.path.join(str(tmp_path), "file.txt")
    with open(path, "w") as handle:
        handle.write("x")
    assert not survey.holds_results(path)


# -- the rule ----------------------------------------------------------------

def test_a_folder_with_results_is_kept_when_the_default_moves(tmp_path):
    used = _folder(tmp_path, "meadow", "bambi_survey", filled=True)
    moved = os.path.join(str(tmp_path), "projects", "bambi_survey")
    assert survey.folder_to_keep(used, moved) == used


def test_an_empty_folder_is_not_kept(tmp_path):
    used = _folder(tmp_path, "meadow", "bambi_survey")
    moved = os.path.join(str(tmp_path), "projects", "bambi_survey")
    assert survey.folder_to_keep(used, moved) == ""


def test_the_same_folder_needs_no_keeping(tmp_path):
    used = _folder(tmp_path, "meadow", "bambi_survey", filled=True)
    assert survey.folder_to_keep(used, used) == ""
    assert survey.folder_to_keep(used, used + os.sep) == ""
    assert survey.folder_to_keep(
        used, os.path.join(used, os.pardir, "bambi_survey")) == ""


def test_it_is_kept_when_the_default_is_gone_altogether(tmp_path):
    """Every flight unticked in an unsaved project: there is no default."""
    used = _folder(tmp_path, "meadow", "bambi_survey", filled=True)
    assert survey.folder_to_keep(used, "") == used


def test_nothing_used_so_far_keeps_nothing(tmp_path):
    assert survey.folder_to_keep("", os.path.join(str(tmp_path), "x")) == ""

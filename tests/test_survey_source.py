# -*- coding: utf-8 -*-
"""What a survey counts - animals or boxes - is one setting for every tool.

The census, the heat map and distance sampling each had a selector, with
different defaults: the census counted animals (tracks), the other two boxes
(detections). The same survey was three different things depending on the
tool, and a density of boxes is not a density of animals.
"""
import pytest

from bambi_wildlife_detection.core import survey as survey_core
from bambi_wildlife_detection.core.survey import (
    Survey, SurveyError, SurveyFlight, chosen_source)


def _config(source=None, **keys):
    survey = {} if source is None else {"source": source}
    return dict(keys, survey=survey)


def test_animals_are_the_default():
    assert Survey(folder="x").source == "tracks"
    assert chosen_source({}) == "tracks"
    assert chosen_source(None) == "tracks"
    assert chosen_source(_config()) == "tracks"


@pytest.mark.parametrize("key", ["census_source", "density_source",
                                 "ds_source"])
@pytest.mark.parametrize("source", survey_core.SOURCES)
def test_every_tool_follows_the_survey(key, source):
    assert chosen_source(_config(source), key) == source


def test_a_caller_may_name_a_source_for_one_run():
    """A script, say; the dock never does."""
    config = _config("tracks", ds_source="detections")
    assert chosen_source(config, "ds_source") == "detections"
    assert chosen_source(config, "census_source") == "tracks"
    assert chosen_source(_config("detections", ds_source="Tracks"),
                         "ds_source") == "tracks"


@pytest.mark.parametrize("junk", ["", None, "boxes", 3])
def test_anything_else_falls_back_on_the_survey(junk):
    assert chosen_source(_config("detections", ds_source=junk),
                         "ds_source") == "detections"
    assert chosen_source(_config(junk), "ds_source") == "tracks"


def test_the_setting_travels_with_the_survey(tmp_path):
    folder = tmp_path / "flight"
    folder.mkdir()
    survey = Survey(folder=str(tmp_path / "survey"), source="detections",
                    flights=[SurveyFlight(name="a", target_folder=str(folder))])
    again = Survey.from_dict(survey.to_dict())
    assert again.source == "detections"
    assert again.validate() is again
    # a survey description from before the setting
    old = survey.to_dict()
    del old["source"]
    assert Survey.from_dict(old).source == "tracks"


def test_an_unknown_source_is_refused(tmp_path):
    folder = tmp_path / "flight"
    folder.mkdir()
    survey = Survey(folder=str(tmp_path / "survey"), source="boxes",
                    flights=[SurveyFlight(name="a", target_folder=str(folder))])
    with pytest.raises(SurveyError, match="Unknown survey source"):
        survey.validate()


def test_the_selector_names_what_is_counted():
    assert survey_core.SOURCE_LABELS == {
        "tracks": "Animals (tracks)", "detections": "Boxes (detections)"}

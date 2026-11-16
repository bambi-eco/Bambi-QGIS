# -*- coding: utf-8 -*-
"""The Compare Visits step of the BAMBI processor (see
:mod:`core.survey_trend`).

Like the other survey analytics it works on a *survey* - the flights ticked
on the Survey Analytics tab, one camera, one output folder. Here the flights
are those of several visits to the same ground. Mixed into
:class:`BambiProcessor`.
"""

import os
from typing import Any, Dict

from .core import quiet
from .core.survey import chosen_source


class SurveyTrendMixin:
    """The survey visit by visit, and what changed between the visits."""

    def run_survey_trend(self, config: Dict[str, Any], progress_fn=None,
                         log_fn=None, cancel_check=None):
        """Divide the survey's flights into visits by their date and compare
        the visits.

        Writes into ``<survey>/analytics_{m}/``:

        * ``trend_{source}.json`` - everything: the visits, what each
          counted per species, the changes from visit to visit, the trends;
        * ``trend_{source}_visits.csv`` - one row per visit and species.

        Settings (``config``): ``trend_gap_days`` - flights flown no more
        than this many days after the one before are one visit (0 = the
        same day) - and ``trend_ground``: ``"seen"``, all the ground each
        visit saw, or ``"common"``, only the ground every visit saw.
        """
        from .bambi_processing import CancelledException
        from .core import habitat, survey_trend

        survey = self._survey(config)
        source = chosen_source(config, "trend_source")
        gap = config.get("trend_gap_days", 0)
        ground = str(config.get("trend_ground") or survey_trend.SEEN)
        if log_fn:
            log_fn(f"Comparing the visits ({survey.camera_label}, {source}) "
                   f"of {self._flights_note(survey)}...")
        area = self._survey_area(survey, log_fn)
        if progress_fn:
            progress_fn(2)

        flights = []
        for index, flight in enumerate(survey.flights):
            if cancel_check and cancel_check():
                raise CancelledException("Comparing the visits cancelled")
            span = survey_trend.flight_span(flight.target_folder,
                                            survey.modality)
            found = self._flight_groups(survey, flight, source, log_fn)
            records = self._census_records(survey, flight, source, area,
                                           log_fn, found)
            seen = habitat.monitored_geometry(
                flight.target_folder, survey.modality, area)
            if log_fn and seen is None:
                log_fn(f"{flight.name}: no {survey.camera_label} "
                       "field-of-view footprints and no survey area - the "
                       "ground this flight saw is not known")
            try:
                weather = habitat.weather_values(flight.target_folder)
            except Exception as exc:  # noqa: BLE001 - the weather is extra
                quiet.problem(f"The weather of {flight.name} could not be "
                              "read: left out of the comparison")
                if log_fn:
                    log_fn(f"{flight.name}: the weather could not be read "
                           f"({exc}) - left out")
                weather = {}
            flights.append({
                "name": flight.name,
                "start": None if span is None else span[0],
                "end": None if span is None else span[1],
                "records": records, "seen": seen,
                "weather": {key: value for key, value in weather.items()
                            if value is not None},
            })
            if progress_fn:
                progress_fn(int(5 + 75 * (index + 1) / len(survey.flights)))

        try:
            document = survey_trend.analyse(
                flights, area, gap_days=gap, ground=ground,
                species_order=survey.species_names())
        except survey_trend.TrendError as exc:
            raise RuntimeError(str(exc)) from exc
        if progress_fn:
            progress_fn(90)

        document.update({
            "source": source,
            "camera": survey.camera_label,
            "crs": f"EPSG:{survey.epsg}" if survey.epsg else None,
            "survey_area": None if area is None else area.describe(),
            "species_filter": survey.species,
            "notes": self._trend_notes(document, source),
        })
        folder = survey.ensure_analytics_folder()
        stem = f"trend_{source}"
        json_path = survey_trend.write_json(
            document, os.path.join(folder, f"{stem}.json"))
        survey_trend.write_table(
            survey_trend.flat_rows(document),
            os.path.join(folder, f"{stem}_visits.csv"))

        if log_fn:
            for line in survey_trend.describe(document):
                log_fn(line)
            log_fn(f"Visits written to {json_path}")
        if progress_fn:
            progress_fn(100)
        return document

    @staticmethod
    def _trend_notes(document: dict, source: str) -> str:
        from .core import survey_trend

        gap = document["gap_days"]
        if gap:
            notes = (f"A visit is the flights flown no more than {gap} "
                     "day(s) after the one before. ")
        else:
            notes = "A visit is the flights flown on the same day. "
        notes += (
            "Flights less than "
            f"{survey_trend.SAME_NIGHT_HOURS:g} hours apart are always one "
            "visit, so a night survey does not end at midnight. ")
        if document["basis"] == survey_trend.BASIS_GROUND:
            notes += (
                "A density is the animals of a visit over the ground its "
                "flights looked at, summed over the flights: two flights "
                "over the same ground give the mean of the two. ")
            if document["ground"] == survey_trend.COMMON:
                notes += (
                    "Only the ground every visit saw is compared, and only "
                    "the animals that stood on it. ")
        else:
            notes += (
                "The ground a flight saw is not known for every flight (no "
                "field-of-view footprints, no survey area), so the counts "
                "are compared as they are: that is fair only where every "
                "visit covered the same ground. ")
        notes += (
            "The interval of a count takes a group of animals as one "
            "sighting (gamma interval of Fay and Feuer; the exact Poisson "
            "interval for animals seen one by one). The change between two "
            "visits is tested exactly (conditional binomial test), with "
            "clumped animals worth fewer sightings than there are animals. "
            "The trend is a constant rate of change fitted to the counts "
            "(Poisson regression, the ground looked at as exposure); its "
            "interval allows for the groups and, from three visits on, for "
            "visits that lie further from the line than counting explains. "
            "These are counts of the animals seen: canopy, season, time of "
            "day and weather change how many of the animals present are "
            "seen, so compare visits that were flown alike. ")
        if source != "tracks":
            notes += (
                "Boxes are counted here, not animals: an animal seen on "
                "many frames counts many times, so the intervals are far "
                "too narrow and only the direction of a change means "
                "anything. ")
        return notes.strip()

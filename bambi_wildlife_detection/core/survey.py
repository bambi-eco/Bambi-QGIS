# -*- coding: utf-8 -*-
"""A survey: several processed flights analysed together.

The Survey Analytics are not tied to the flight selected on the processing
tabs. The user ticks the processed flights that make up the survey, picks
the camera, and every analytic - census summary, track inventory, density,
coverage, distance sampling, population estimation - works on all of them:

* **one output folder** for the survey (``<survey folder>/analytics_{m}``),
  which also holds the survey area (``survey_area/survey_area.gpkg``);
* **species by name**, because species ids are assigned per flight -
  roe deer may be 3 in one flight and 7 in the next. A name missing from a
  flight is not an error: that flight simply has none of it;
* **one CRS**: pooled points, rasters and areas need the flights to share
  their target CRS, which is checked rather than assumed.

The survey description travels in the processing config as
``config["survey"]`` (see :meth:`Survey.to_dict`), built by the dock from the
flight list and each flight's own stored configuration.
"""

import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

MODALITIES = ("t", "w")
CAMERA_LABELS = {"t": "Thermal", "w": "RGB"}

#: What the survey counts: each tracked animal once, or every box.
TRACKS = "tracks"
DETECTIONS = "detections"
SOURCES = (TRACKS, DETECTIONS)
SOURCE_LABELS = {TRACKS: "Animals (tracks)", DETECTIONS: "Boxes (detections)"}


#: Name of the survey folder that is used when none was chosen.
DEFAULT_FOLDER_NAME = "bambi_survey"


class SurveyError(RuntimeError):
    """The survey cannot be analysed as configured."""


# ---------------------------------------------------------------------------
# The survey folder
# ---------------------------------------------------------------------------

def default_folder(project_home: str, flight_folders: Sequence[str]) -> str:
    """Where a survey keeps its results when no folder was chosen:
    ``bambi_survey`` next to the QGIS project - or, while the project has
    not been saved anywhere, next to the first flight of the survey.
    ``""`` when there is neither."""
    if project_home:
        return os.path.join(project_home, DEFAULT_FOLDER_NAME)
    for folder in flight_folders:
        if folder:
            parent = os.path.dirname(os.path.normpath(folder))
            return os.path.join(parent, DEFAULT_FOLDER_NAME)
    return ""


def holds_results(folder: str) -> bool:
    """True when *folder* exists and something is in it - a survey has
    written there (results, the survey area)."""
    try:
        return bool(folder) and os.path.isdir(folder) and \
            bool(os.listdir(folder))
    except OSError:
        return False


def folder_to_keep(in_use: str, default: str) -> str:
    """The folder a survey stays in although its default has moved.

    The default depends on where the QGIS project is saved and, before it
    is, on which flight is ticked first - so saving the project for the
    first time moved it. Everything written until then (results, the survey
    area, the transects' distances) stayed where it was while the tools went
    on in the new, empty folder: one survey in two places.

    :param in_use: the default folder used so far
    :param default: the default folder as it would be now
    :return: *in_use* when it holds results and *default* is another
        folder - the survey keeps it; ``""`` when nothing has to be kept
    """
    if not in_use or not holds_results(in_use):
        return ""
    same = default and os.path.normcase(os.path.abspath(default)) == \
        os.path.normcase(os.path.abspath(in_use))
    return "" if same else in_use


@dataclass
class SurveyFlight:
    """One processed flight of the survey."""

    name: str
    target_folder: str
    epsg: Optional[int] = None
    #: DEM metadata JSON (mesh -> world origin) and mesh; the population
    #: estimate places a flight's transects with them.
    dem: str = ""
    dem_path: str = ""
    correction_path: str = ""

    @property
    def key(self) -> str:
        return os.path.normcase(os.path.abspath(self.target_folder))


@dataclass
class Survey:
    folder: str
    modality: str = "t"
    flights: List[SurveyFlight] = field(default_factory=list)
    #: Species names to count; ``None`` = every species.
    species: Optional[List[str]] = None
    per_species: bool = False
    include_manual: bool = True
    area_enabled: bool = False
    #: What every analytic of the survey counts - one choice for all of them
    #: (see :func:`chosen_source`): ``"tracks"``, each tracked animal once,
    #: or ``"detections"``, every box - what a still-photo survey without
    #: tracking has.
    source: str = TRACKS
    #: Count only animals that both cameras saw: tracks of the survey's
    #: camera that the cross-modal matching paired with a track of the
    #: other one. The survey's camera still says where an animal is.
    confirmed: bool = False

    # -- (de)serialisation ------------------------------------------------- #

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Survey":
        data = dict(data or {})
        flights = [f if isinstance(f, SurveyFlight) else SurveyFlight(**{
            k: v for k, v in f.items()
            if k in SurveyFlight.__dataclass_fields__})
            for f in data.pop("flights", []) or []]
        known = {k: v for k, v in data.items()
                 if k in cls.__dataclass_fields__}
        survey = cls(flights=flights, **known)
        if survey.species is not None:
            survey.species = list(survey.species)
        return survey

    @classmethod
    def from_config(cls, config: dict) -> "Survey":
        data = config.get("survey")
        if not data:
            raise SurveyError(
                "No survey is configured. Tick the flights to analyse at the "
                "top of the Survey Analytics tab.")
        return cls.from_dict(data)

    # -- checks ------------------------------------------------------------- #

    def validate(self) -> "Survey":
        if self.modality not in MODALITIES:
            raise SurveyError(f"Unknown camera {self.modality!r}.")
        if self.source not in SOURCES:
            raise SurveyError(
                f"Unknown survey source {self.source!r}; expected "
                f"{' or '.join(SOURCES)}.")
        if not self.flights:
            raise SurveyError(
                "No flight is selected. Tick at least one processed flight "
                "at the top of the Survey Analytics tab.")
        if not (self.folder or "").strip():
            raise SurveyError(
                "Set the survey folder - where the survey results and the "
                "survey area are stored.")
        missing = [f.name for f in self.flights
                   if not os.path.isdir(f.target_folder)]
        if missing:
            names = ", ".join(missing)
            raise SurveyError(
                f"The folder of these flights does not exist: {names}")
        seen = set()
        for flight in self.flights:
            if flight.key in seen:
                raise SurveyError(f"Flight '{flight.name}' is listed twice.")
            seen.add(flight.key)
        _ = self.epsg  # raises on a mixed CRS
        unmatched = self.unmatched_flights() if self.confirmed else []
        if unmatched:
            raise SurveyError(
                "Only animals that both cameras saw are counted, and the "
                "tracks of these flights have not been matched between "
                f"the cameras: {', '.join(unmatched)}. Run 'Match Tracks' "
                "for them, untick them, or count all tracks.")
        return self

    def unmatched_flights(self) -> List[str]:
        """The flights that have tracks of the survey's camera and no
        cross-modal matching run: nobody can say which of their animals
        the other camera saw. A flight without tracks has none to
        confirm, and is not one of these."""
        from . import match_store, store

        tracked = [flight for flight in self.flights
                   if os.path.isfile(store.stage_path(
                       flight.target_folder, store.TRACKS, self.modality))]
        return [flight.name for flight in tracked
                if match_store.active_run(flight.target_folder) is None]

    @property
    def counted_label(self) -> str:
        """``"Thermal"``, or ``"Thermal, confirmed by RGB"``."""
        if not self.confirmed:
            return self.camera_label
        other = CAMERA_LABELS["w" if self.modality == "t" else "t"]
        return f"{self.camera_label}, confirmed by {other}"

    @property
    def other_modality(self) -> str:
        return "w" if self.modality == "t" else "t"

    @property
    def epsg(self) -> Optional[int]:
        """The flights' common target CRS (``None`` when none is known)."""
        codes = {int(f.epsg) for f in self.flights if f.epsg}
        if len(codes) > 1:
            detail = ", ".join(f"{f.name}: EPSG:{f.epsg}" for f in self.flights)
            raise SurveyError(
                "The selected flights use different target CRSs "
                f"({detail}). Pooled results need one CRS - analyse them as "
                "separate surveys or re-process them into the same CRS.")
        return codes.pop() if codes else None

    # -- places ------------------------------------------------------------- #

    @property
    def camera_label(self) -> str:
        return CAMERA_LABELS.get(self.modality, self.modality)

    @property
    def analytics_folder(self) -> str:
        return os.path.join(self.folder, f"analytics_{self.modality}")

    def ensure_analytics_folder(self) -> str:
        os.makedirs(self.analytics_folder, exist_ok=True)
        return self.analytics_folder

    def area(self):
        """The survey area when it is switched on and defined, else ``None``."""
        if not self.area_enabled:
            return None
        from . import survey_area
        return survey_area.load(self.folder, self.epsg)

    @property
    def several(self) -> bool:
        return len(self.flights) > 1

    # -- species ------------------------------------------------------------ #

    def species_ids(self, flight: SurveyFlight,
                    names: Optional[Sequence[str]] = None
                    ) -> Optional[List[int]]:
        """*names* (default: the survey's) as this flight's species ids.

        ``None`` means every species. A flight that has none of the names gets
        an empty list - it has nothing to count, which is a result.
        """
        names = self.species if names is None else names
        if names is None:
            return None
        from .analytics_source import resolve_species_names
        ids, _missing = resolve_species_names(flight.target_folder, names)
        return ids

    def species_names(self) -> List[str]:
        """Every countable species of the selected flights, in schema order
        (first flight first, then names only later flights know)."""
        from .analytics_source import species_options

        ordered: List[str] = []
        for flight in self.flights:
            if not os.path.isdir(flight.target_folder):
                continue
            try:
                options = species_options(flight.target_folder)
            except Exception:  # noqa: BLE001 - an unmigrated flight
                options = []
            for entry in options:
                if entry["name"] not in ordered:
                    ordered.append(entry["name"])
        return ordered

    def strata(self, has_data: Callable[[List[str]], bool]
               ) -> Tuple[List[Tuple[str, Optional[List[str]]]], List[str]]:
        """``(strata, empty)`` - the runs an analytic makes.

        Without "separate result per species" that is one pooled run over the
        selected species. With it, one run per selected species for which
        ``has_data([name])`` is true; the others are returned in *empty*.
        """
        if not self.per_species:
            return [("", self.species)], []
        chosen = self.species_names() if self.species is None else \
            list(self.species)
        strata, empty = [], []
        for name in chosen:
            if has_data([name]):
                strata.append((name, [name]))
            else:
                empty.append(name)
        return strata, empty


def chosen_source(config: dict, key: str = "") -> str:
    """What an analytic counts: ``"tracks"`` or ``"detections"``.

    The survey's one setting (``config["survey"]["source"]``) - unless the
    caller names a source for this run under *key* (``census_source``,
    ``density_source``, ``ds_source``), as a script may. The tools used to
    have a selector each, with different defaults: the census counted
    animals, the heat map and distance sampling boxes, so the same survey
    was three different things depending on the tool.
    """
    named = str((config or {}).get(key) or "").strip().lower() if key else ""
    if named in SOURCES:
        return named
    survey = (config or {}).get("survey") or {}
    source = str(survey.get("source") or "").strip().lower()
    return source if source in SOURCES else TRACKS


def route_suffix(target_folder: str, modality: str,
                 filename: str = "flight_route.geojson") -> Optional[str]:
    """The ``flight_route_{x}`` folder holding *filename* for a flight.

    The route is generated per camera; the survey's camera is preferred and
    the other one accepted, since both describe the same flight path.
    """
    other = "w" if modality == "t" else "t"
    for suffix in (modality, other):
        if os.path.isfile(os.path.join(
                target_folder, f"flight_route_{suffix}", filename)):
            return suffix
    return None


def flight_settings(target_folder: str) -> Dict[str, object]:
    """CRS, DEM and correction of a flight from its stored configuration."""
    from . import flights
    from .crs_utils import parse_epsg_from_text

    stored = flights.load_config(target_folder)
    crs = (stored.get("Input/TargetCrs") or "").strip().upper()
    return {
        "epsg": parse_epsg_from_text(crs) if crs else None,
        "dem": flights.dem_metadata_path(target_folder),
        "dem_path": (stored.get(flights.DEM_PATH_KEY) or "").strip(),
        "correction_path": (stored.get("Input/CorrectionPath") or "").strip(),
    }

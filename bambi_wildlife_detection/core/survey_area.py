# -*- coding: utf-8 -*-
"""Survey plot and exclusion zones that limit the survey analytics.

A drone census is usually reported for a defined area - a hunting district, a
reserve, a forest compartment - not for whatever ground the flight happened
to see. This module holds that area:

* **plot polygons** - the census area. Only animals inside count, and only the
  ground inside is the surveyed area.
* **exclusion zones** - parts of the plot that are removed again, each with an
  optional reason (thick forest cover, a building, a road). Animals there are
  not counted and their ground is not part of the area.

``effective area = union(plots) - union(exclusions)``. A project may define
exclusion zones without a plot; they are then cut out of the whole flight.

The polygons live in ``<target folder>/survey_area/survey_area.gpkg`` (layers
``plot`` and ``exclusion_zones``) in the project's target CRS. It is a normal
feature GeoPackage, so the polygons are drawn and edited with the ordinary
QGIS digitising tools; the analytics read it with ``sqlite3`` and shapely,
without QGIS. Every project folder carries its own file, so pooled
multi-project analytics clip each flight to its own plot.

Unlike the pipeline stage files, this file is user-owned input: QGIS may hold
it open for editing, and this module only ever reads it (apart from creating
an empty one).
"""

import os
import sqlite3
import struct
from typing import Iterable, List, Optional, Sequence, Tuple

from . import gpkg

FOLDER = "survey_area"
FILENAME = "survey_area.gpkg"
PLOT_LAYER = "plot"
EXCLUSION_LAYER = "exclusion_zones"
LAYERS = (PLOT_LAYER, EXCLUSION_LAYER)

#: Envelope byte length per GeoPackage binary envelope indicator (flags 1-3).
_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


class SurveyAreaError(RuntimeError):
    """The survey-area file exists but cannot be used."""


def path(target_folder: str) -> str:
    """Where *target_folder* keeps its survey area."""
    return os.path.join(target_folder, FOLDER, FILENAME)


# ---------------------------------------------------------------------------
# GeoPackage geometry encoding (OGC GeoPackage 1.3.0, clause 2.1.3)
# ---------------------------------------------------------------------------

def decode_geometry(blob: bytes):
    """A shapely geometry from a GeoPackage geometry blob (``None`` if empty)."""
    from shapely import wkb

    if blob is None:
        return None
    blob = bytes(blob)
    if len(blob) < 8 or blob[:2] != b"GP":
        raise SurveyAreaError("Not a GeoPackage geometry blob.")
    flags = blob[3]
    if flags & 0b0010_0000:
        raise SurveyAreaError(
            "Extended GeoPackage geometries are not supported.")
    if flags & 0b0001_0000:
        return None  # the empty-geometry flag
    envelope = (flags >> 1) & 0b111
    if envelope not in _ENVELOPE_BYTES:
        raise SurveyAreaError(f"Invalid GeoPackage envelope code {envelope}.")
    start = 8 + _ENVELOPE_BYTES[envelope]
    geometry = wkb.loads(blob[start:])
    return None if geometry.is_empty else geometry


def encode_geometry(geometry, srs_id: int) -> bytes:
    """A GeoPackage geometry blob (little endian, XY envelope)."""
    from shapely import wkb

    if geometry is None or geometry.is_empty:
        raise SurveyAreaError("Cannot store an empty survey-area polygon.")
    flags = 0b0000_0011  # XY envelope, little endian
    min_x, min_y, max_x, max_y = geometry.bounds
    header = b"GP" + bytes([0, flags]) + struct.pack("<i", srs_id)
    envelope = struct.pack("<4d", min_x, max_x, min_y, max_y)
    return header + envelope + wkb.dumps(geometry, byte_order=1)


# ---------------------------------------------------------------------------
# Creating the (empty) file
# ---------------------------------------------------------------------------

def _crs_wkt(epsg: int) -> str:
    try:
        from pyproj import CRS
        return CRS.from_epsg(int(epsg)).to_wkt(version="WKT1_GDAL")
    except Exception:  # pragma: no cover - pyproj ships with the engine
        # GDAL resolves the CRS from organization/organization_coordsys_id
        # when the definition is unusable.
        return "undefined"


def create(target_folder: str, epsg: int,
           plots: Iterable[Tuple[object, str]] = (),
           exclusions: Iterable[Tuple[object, str, str]] = ()) -> str:
    """Create ``survey_area.gpkg`` with both layers, optionally pre-filled.

    Existing layers are kept as they are - only a missing file or a missing
    layer is created, so this is safe to call before every "Draw" action.

    :param plots: ``(geometry, name)`` pairs to insert into a new plot layer.
    :param exclusions: ``(geometry, name, reason)`` for a new exclusion layer.
    :return: the file path.
    """
    file_path = path(target_folder)
    epsg = int(epsg)
    conn = gpkg.connect(file_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO gpkg_spatial_ref_sys "
            "(srs_name, srs_id, organization, organization_coordsys_id, "
            " definition, description) VALUES (?, ?, 'EPSG', ?, ?, '')",
            (f"EPSG:{epsg}", epsg, epsg, _crs_wkt(epsg)))
        known = {row["name"] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        contents = (
            (PLOT_LAYER, "Survey plot (census area)", list(
                (geometry, name, "") for geometry, name in plots)),
            (EXCLUSION_LAYER, "Exclusion zones inside the survey plot",
             list(exclusions)),
        )
        for table, description, rows in contents:
            if table in known:
                continue
            # Table names are module constants, never user input.
            conn.execute(
                f"CREATE TABLE {table} ("
                "fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, "
                "geom MULTIPOLYGON, name TEXT, reason TEXT)")
            conn.execute(
                "INSERT INTO gpkg_contents (table_name, data_type, identifier,"
                " description, srs_id) VALUES (?, 'features', ?, ?, ?)",
                (table, table, description, epsg))
            conn.execute(
                "INSERT INTO gpkg_geometry_columns (table_name, column_name, "
                "geometry_type_name, srs_id, z, m) "
                "VALUES (?, 'geom', 'MULTIPOLYGON', ?, 0, 0)", (table, epsg))
            for geometry, name, reason in rows:
                conn.execute(
                    f"INSERT INTO {table} (geom, name, reason) "  # nosec B608
                    "VALUES (?, ?, ?)",
                    (encode_geometry(_as_multipolygon(geometry), epsg),
                     name, reason))
        conn.commit()
    finally:
        conn.close()
    return file_path


def _as_multipolygon(geometry):
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(geometry, Polygon):
        return MultiPolygon([geometry])
    return geometry


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _connect_read(file_path: str) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(f"file:{file_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        # A read-only open of a WAL database whose -shm is missing fails;
        # a normal open recreates it.
        conn = sqlite3.connect(file_path)
    conn.row_factory = sqlite3.Row
    return conn


def _layer_epsg(conn: sqlite3.Connection, table: str) -> Optional[int]:
    row = conn.execute(
        "SELECT s.organization, s.organization_coordsys_id, s.srs_id "
        "FROM gpkg_geometry_columns g "
        "JOIN gpkg_spatial_ref_sys s ON s.srs_id = g.srs_id "
        "WHERE g.table_name = ?", (table,)).fetchone()
    if row is None:
        return None
    if str(row["organization"]).upper() == "EPSG":
        return int(row["organization_coordsys_id"])
    return int(row["srs_id"]) if int(row["srs_id"]) > 0 else None


def _read_layer(conn: sqlite3.Connection, table: str) -> Tuple[list, Optional[int]]:
    """``([(geometry, name, reason)], epsg)``; empty when the layer is absent."""
    column = conn.execute(
        "SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?",
        (table,)).fetchone()
    if column is None:
        return [], None
    geom_col = column["column_name"]
    columns = {row["name"] for row in conn.execute(
        "SELECT name FROM pragma_table_info(?)", (table,))}
    name_sql = "name" if "name" in columns else "NULL"
    reason_sql = "reason" if "reason" in columns else "NULL"
    if not geom_col.replace("_", "").isalnum():
        raise SurveyAreaError(f"Unexpected geometry column {geom_col!r}.")
    # Identifiers come from the GeoPackage's own catalogue and are validated
    # above; the table name is one of this module's constants.
    query = (f'SELECT "{geom_col}" AS geom, {name_sql} AS name, '
             f"{reason_sql} AS reason FROM {table}")  # nosec B608
    features = []
    for row in conn.execute(query):
        geometry = decode_geometry(row["geom"])
        if geometry is None:
            continue
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry.geom_type not in ("Polygon", "MultiPolygon") or \
                geometry.is_empty:
            continue
        features.append((geometry, row["name"] or "", row["reason"] or ""))
    return features, _layer_epsg(conn, table)


def _reproject(features: list, source_epsg: int, target_epsg: int) -> list:
    try:
        from pyproj import Transformer
        from shapely.ops import transform
    except ImportError as exc:  # pragma: no cover
        raise SurveyAreaError(
            f"The survey area is in EPSG:{source_epsg} but the project uses "
            f"EPSG:{target_epsg}, and pyproj is not available to convert it."
        ) from exc
    transformer = Transformer.from_crs(
        int(source_epsg), int(target_epsg), always_xy=True)
    return [(transform(transformer.transform, geometry), name, reason)
            for geometry, name, reason in features]


def load(target_folder: str, epsg: Optional[int] = None) -> Optional["SurveyArea"]:
    """The survey area of *target_folder*, or ``None`` when none is defined.

    "Defined" means at least one plot or exclusion polygon; an empty file (say,
    right after "Draw plot" was pressed and cancelled) counts as no area.

    :param epsg: the CRS the analytics work in. Polygons stored in another CRS
        are reprojected into it.
    """
    file_path = path(target_folder)
    if not os.path.isfile(file_path):
        return None
    if not gpkg.is_geopackage(file_path):
        raise SurveyAreaError(f"{file_path} is not a GeoPackage.")
    conn = _connect_read(file_path)
    try:
        plots, plot_epsg = _read_layer(conn, PLOT_LAYER)
        exclusions, exclusion_epsg = _read_layer(conn, EXCLUSION_LAYER)
    except sqlite3.DatabaseError as exc:
        raise SurveyAreaError(f"Cannot read {file_path}: {exc}") from exc
    finally:
        conn.close()

    if epsg:
        if plot_epsg and int(plot_epsg) != int(epsg):
            plots = _reproject(plots, plot_epsg, epsg)
        if exclusion_epsg and int(exclusion_epsg) != int(epsg):
            exclusions = _reproject(exclusions, exclusion_epsg, epsg)

    if not plots and not exclusions:
        return None
    return SurveyArea(
        plots=[(geometry, name) for geometry, name, _reason in plots],
        exclusions=exclusions,
        epsg=int(epsg or plot_epsg or exclusion_epsg or 0) or None,
        source=file_path)


# ---------------------------------------------------------------------------
# The area itself
# ---------------------------------------------------------------------------

class SurveyArea:
    """Plot polygons minus exclusion zones, in one metric CRS."""

    def __init__(self, plots: Sequence[Tuple[object, str]] = (),
                 exclusions: Sequence[Tuple[object, str, str]] = (),
                 epsg: Optional[int] = None, source: str = ""):
        from shapely.ops import unary_union

        self.plots = list(plots)
        self.exclusions = list(exclusions)
        self.epsg = epsg
        self.source = source
        self._plot = (unary_union([g for g, _ in self.plots])
                      if self.plots else None)
        self._excluded = (unary_union([g for g, _n, _r in self.exclusions])
                          if self.exclusions else None)
        if self._plot is not None and self._excluded is not None:
            self._effective = self._plot.difference(self._excluded)
        else:
            self._effective = self._plot
        self._prepared = None

    # -- geometry ---------------------------------------------------------- #

    @property
    def has_plot(self) -> bool:
        return self._plot is not None

    @property
    def geometry(self):
        """The effective area, or ``None`` when only exclusions are defined
        (then everything outside them counts)."""
        return self._effective

    @property
    def excluded_geometry(self):
        return self._excluded

    def contains_xy(self, xs, ys):
        """Boolean array: which points count. A point on the boundary counts."""
        import numpy as np
        import shapely

        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        inside = np.ones(xs.shape, dtype=bool)
        if self._effective is not None:
            shapely.prepare(self._effective)
            inside &= shapely.intersects_xy(self._effective, xs, ys)
        elif self._excluded is not None:
            shapely.prepare(self._excluded)
            # The interior only: an animal on an exclusion's edge still counts.
            inside &= ~shapely.contains_xy(self._excluded, xs, ys)
        return inside

    def contains(self, x: float, y: float) -> bool:
        return bool(self.contains_xy([x], [y])[0])

    def filter_points(self, points: Sequence[Tuple[float, float]]
                      ) -> Tuple[list, int]:
        """``(points inside, number dropped)``."""
        if not points:
            return [], 0
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        keep = self.contains_xy(xs, ys)
        kept = [p for p, k in zip(points, keep) if k]
        return kept, len(points) - len(kept)

    def clip(self, geometry):
        """*geometry* restricted to the effective area (``None`` if nothing
        is left)."""
        if geometry is None or geometry.is_empty:
            return None
        if self._effective is not None:
            clipped = geometry.intersection(self._effective)
        elif self._excluded is not None:
            clipped = geometry.difference(self._excluded)
        else:
            clipped = geometry
        return None if clipped.is_empty else clipped

    def clipped_length(self, coordinates: Sequence[Sequence[float]]) -> float:
        """Length of the polyline *coordinates* that runs inside the area."""
        from shapely.geometry import LineString

        if len(coordinates) < 2:
            return 0.0
        line = LineString([(float(c[0]), float(c[1])) for c in coordinates])
        clipped = self.clip(line)
        return 0.0 if clipped is None else float(clipped.length)

    # -- areas -------------------------------------------------------------- #

    @property
    def plot_m2(self) -> Optional[float]:
        return None if self._plot is None else float(self._plot.area)

    @property
    def excluded_m2(self) -> float:
        """Excluded ground that lies inside the plot (all of it without one)."""
        if self._excluded is None:
            return 0.0
        if self._plot is None:
            return float(self._excluded.area)
        return float(self._plot.intersection(self._excluded).area)

    @property
    def adjusted_m2(self) -> Optional[float]:
        """Plot minus exclusions; ``None`` without a plot."""
        return None if self._effective is None else float(self._effective.area)

    def describe(self) -> dict:
        """What the area was, for the result documents (provenance)."""
        def ha(value):
            return None if value is None else round(value / 10000.0, 4)

        return {
            "source": self.source,
            "crs": f"EPSG:{self.epsg}" if self.epsg else None,
            "plots": [name or f"plot {i + 1}"
                      for i, (_g, name) in enumerate(self.plots)],
            "exclusions": [
                {"name": name or f"zone {i + 1}", "reason": reason or ""}
                for i, (_g, name, reason) in enumerate(self.exclusions)],
            "plot_ha": ha(self.plot_m2),
            "excluded_ha": ha(self.excluded_m2),
            "adjusted_ha": ha(self.adjusted_m2),
        }

    def summary(self) -> str:
        """One line for the log and the dock status label."""
        parts = []
        if self.plot_m2 is not None:
            parts.append(f"{len(self.plots)} plot polygon(s), "
                         f"{self.plot_m2 / 10000.0:.2f} ha")
        else:
            parts.append("no plot (whole flight)")
        if self.exclusions:
            parts.append(f"{len(self.exclusions)} exclusion zone(s), "
                         f"-{self.excluded_m2 / 10000.0:.2f} ha")
        if self.adjusted_m2 is not None and self.exclusions:
            parts.append(f"adjusted {self.adjusted_m2 / 10000.0:.2f} ha")
        return "; ".join(parts)


def union_adjusted_m2(areas: Iterable[Optional["SurveyArea"]]) -> Optional[float]:
    """Adjusted area of several projects' plots, shared ground counted once.

    Repeat flights over the same plot must not multiply its area, while
    flights over neighbouring plots add theirs - the union does both. ``None``
    when any project has no plot (its area is then unknown).
    """
    from shapely.ops import unary_union

    geometries: List[object] = []
    for area in areas:
        if area is None or area.geometry is None:
            return None
        geometries.append(area.geometry)
    if not geometries:
        return None
    return float(unary_union(geometries).area)

# -*- coding: utf-8 -*-
"""Where the plugin's layers go in the QGIS layer tree (9.0.0).

Layers are grouped per flight, and inside a flight per camera ("Thermal",
"RGB"); survey-wide products live in the survey group. Within a group the
plugin's layers keep one fixed order - vector overlays on top, rasters at
the bottom - instead of "whatever was added last is on top":

    survey area and exclusion zones, flight route, FoV, density heatmap,
    coverage map, transect areas, track inventory, matched pairs, tracks,
    detections, distance sampling (perpendicular distances), segmentation
    masks, reconstruction rasters, orthomosaic, ALFS, per-frame GeoTIFFs

A new layer or group is inserted before the first sibling that ranks after
it. Siblings the plugin does not know (the user's own layers) have no rank
and are left where they are, so a manual re-ordering is respected as far
as it can be: the plugin never moves an existing node.

Everything here works on names only, so it is testable without QGIS.
"""

from typing import List, Optional, Sequence

#: Camera subgroup names inside a flight group, in display order.
MODALITY_GROUPS = {"t": "Thermal", "w": "RGB"}

#: (category, name fragments) in display order, top to bottom. The first
#: category with a fragment in the name wins, so more specific names come
#: first where fragments overlap ("SAM3 Segmentation Orthomosaic").
CATEGORIES = (
    ("survey_area", ("Survey Plot", "Survey Area", "Exclusion")),
    ("flight_route", ("Flight Route",)),
    ("fov", ("FoV",)),
    ("density", ("Density",)),
    ("coverage", ("Coverage Map",)),
    ("transects", ("Transect Areas",)),
    ("track_inventory", ("Track Inventory",)),
    ("habitat", ("Habitat Tracks",)),
    ("matched_pairs", ("Matched Pairs",)),
    ("tracks", ("Wildlife Tracks",)),
    ("detections", ("Frame Detections", "Detections")),
    ("distance", ("Perpendicular",)),
    ("segmentation", ("Segmentation",)),
    ("reconstruction", ("Reconstruction",)),
    ("orthomosaic", ("Orthomosaic",)),
    ("alfs", ("ALFS",)),
    ("geotiffs", ("GeoTIFF",)),
)

_FIRST_MODALITY_RANK = 1000

#: The environment rasters (satellite indices, land cover, terrain wetness)
#: are base layers covering the whole flight: they go below everything else
#: in a flight group, the camera subgroups included, or they would hide it.
BASE_GROUPS = ("BAMBI Environment",)
_BASE_RANK = 2000


def category(name: str) -> Optional[str]:
    """The category of a plugin layer or group name, or None."""
    name = name or ""
    for key, fragments in CATEGORIES:
        if any(f.lower() in name.lower() for f in fragments):
            return key
    return None


def modality(name: str) -> Optional[str]:
    """``"t"`` / ``"w"`` for a name ending in "(Thermal)" / "(RGB)"."""
    lowered = (name or "").lower()
    if "(thermal)" in lowered:
        return "t"
    if "(rgb)" in lowered:
        return "w"
    return None


def rank(name: str) -> Optional[int]:
    """Sort key of a node name among its siblings (lower = further up), or
    None for names the plugin does not order. Camera subgroups come after
    a flight's camera-independent layers, Thermal before RGB."""
    for i, group in enumerate(MODALITY_GROUPS.values()):
        if name == group:
            return _FIRST_MODALITY_RANK + i
    for i, group in enumerate(BASE_GROUPS):
        if name == group:
            return _BASE_RANK + i
    key = category(name)
    if key is None:
        return None
    return [k for k, _ in CATEGORIES].index(key)


def insert_index(siblings: Sequence[str], name: str,
                 at_top: bool = True) -> int:
    """Index at which a new node called *name* goes among *siblings*.

    Before the first sibling ranking after it; else below the last ranked
    sibling. A name without a rank goes to the top (or the bottom with
    ``at_top=False``), as the plugin did before.
    """
    new = rank(name)
    if new is None:
        return 0 if at_top else len(siblings)
    last_ranked = -1
    for i, sibling in enumerate(siblings):
        r = rank(sibling)
        if r is None:
            continue
        if r > new:
            return i
        last_ranked = i
    return last_ranked + 1 if last_ranked >= 0 else 0


def ordered(names: Sequence[str]) -> List[str]:
    """*names* in display order (unranked names keep their place at the
    end) - what a group looks like when everything was added by the plugin."""
    out: List[str] = []
    for name in names:
        out.insert(insert_index(out, name, at_top=False), name)
    return out

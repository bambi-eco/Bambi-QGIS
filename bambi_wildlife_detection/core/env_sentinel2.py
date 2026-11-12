# -*- coding: utf-8 -*-
"""Vegetation, moisture, water and snow indices from Sentinel-2.

Sentinel-2 is the only free satellite whose pixels (10-20 m) are small
enough to say something about the patches inside one drone flight, and it
passes every two to five days. This module finds the scene nearest to the
flight that is clear *over the flight* and turns it into four indices on the
flight's 10 m grid (:mod:`core.environment`):

========  =====================================  ==========================
``ndvi``  (NIR - red) / (NIR + red)              greenness, forage, leaf-on
``ndmi``  (NIR narrow - SWIR) / (NIR narrow + …) vegetation water content
``ndwi``  (green - NIR) / (green + NIR)          open water
``ndsi``  (green - SWIR) / (green + SWIR)        snow
========  =====================================  ==========================

The scenes come from a STAC catalogue of cloud-optimised GeoTIFFs. The
default is Element 84's Earth Search on AWS, which needs no account;
:class:`Catalogue` is the seam for adding others later, including ones that
want credentials.

**Clear over the flight, not over the tile.** A tile is 100 km wide and its
cloud percentage says little about three kilometres of it, so every
candidate's scene classification is read over the flight's extent and the
share of usable cells decides. Cloud, cloud shadow, cirrus and defective
cells never reach an index - they are written as no-data.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

from . import env_remote
from .environment import (EnvironmentFailure, Grid, check_cancelled,
                          grid_bounds_wgs84, iso)

DEFAULT_WINDOW_DAYS = 10
DEFAULT_MIN_CLEAR = 0.80

#: Scene-classification (SCL) classes that an index may be computed on:
#: dark area, vegetation, bare ground, water, unclassified and snow. The
#: rest - no data, saturated, cloud shadow, cloud, cirrus - is masked.
SCL_USABLE = (2, 4, 5, 6, 7, 11)
SCL_NAMES = {
    0: "no data", 1: "saturated or defective", 2: "dark area",
    3: "cloud shadow", 4: "vegetation", 5: "bare ground", 6: "water",
    7: "unclassified", 8: "cloud (medium probability)",
    9: "cloud (high probability)", 10: "thin cirrus", 11: "snow or ice",
}

#: The spectral bands the indices need, by their STAC asset role.
BANDS = ("green", "red", "nir", "nir08", "swir16")


@dataclass(frozen=True)
class Catalogue:
    """A STAC catalogue of Sentinel-2 L2A scenes as cloud-optimised GeoTIFFs.

    To add a service, describe it here: its search endpoint, its collection
    and what it calls the assets. One that needs credentials would carry
    them into ``search`` and the raster reads; nothing else has to change.
    """

    key: str
    title: str
    search_url: str
    collection: str
    #: band role (see :data:`BANDS`, plus ``scl``) -> asset key
    assets: Dict[str, str] = field(default_factory=dict)
    attribution: str = ""
    requires_credentials: bool = False


EARTH_SEARCH = Catalogue(
    key="earth-search",
    title="Earth Search (Element 84, AWS open data)",
    search_url="https://earth-search.aws.element84.com/v1/search",
    collection="sentinel-2-l2a",
    assets={"green": "green", "red": "red", "nir": "nir", "nir08": "nir08",
            "swir16": "swir16", "scl": "scl"},
    attribution="Contains modified Copernicus Sentinel data, served by "
                "Element 84 Earth Search.",
)

CATALOGUES: Dict[str, Catalogue] = {EARTH_SEARCH.key: EARTH_SEARCH}


@dataclass
class Scene:
    """One candidate scene."""

    scene_id: str
    time: datetime
    cloud_cover: Optional[float]
    #: band role -> ``(href, scale, offset)``
    assets: Dict[str, Tuple[str, float, float]]
    platform: str = ""
    tile: str = ""

    def offset_days(self, flight_time: datetime) -> float:
        """Days from the flight to the scene; negative = before the flight."""
        return (self.time - flight_time).total_seconds() / 86400.0


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

def _scene_from_item(item: dict, catalogue: Catalogue) -> Optional[Scene]:
    """A STAC item as a :class:`Scene`, or ``None`` when it lacks a band."""
    from .environment import parse_time

    properties = item.get("properties") or {}
    moment = parse_time(properties.get("datetime"))
    if moment is None:
        return None
    # Since processing baseline 04.00 the L2A values carry an offset of
    # -1000. Earth Search removes it from the pixels of the scenes it marks
    # ``boa_offset_applied`` while still listing it in the band metadata, so
    # the flag decides - applying the listed offset again would shift every
    # reflectance by 0.1.
    offset_applied = bool(properties.get("earthsearch:boa_offset_applied"))
    assets: Dict[str, Tuple[str, float, float]] = {}
    for role, key in catalogue.assets.items():
        asset = (item.get("assets") or {}).get(key)
        if not asset or not asset.get("href"):
            return None
        scale, offset = 1.0, 0.0
        if role != "scl":
            band = (asset.get("raster:bands") or [{}])[0] or {}
            scale = float(band.get("scale") or 0.0001)
            offset = 0.0 if offset_applied else float(band.get("offset") or 0.0)
        assets[role] = (str(asset["href"]), scale, offset)
    cloud = properties.get("eo:cloud_cover")
    return Scene(
        scene_id=str(item.get("id") or ""),
        time=moment,
        cloud_cover=None if cloud is None else float(cloud),
        assets=assets,
        platform=str(properties.get("platform") or ""),
        tile=str(properties.get("grid:code") or ""),
    )


def search_scenes(bounds_wgs84, start: datetime, end: datetime,
                  catalogue: Catalogue = EARTH_SEARCH,
                  fetch_json: Callable = env_remote.fetch_json,
                  max_pages: int = 5) -> List[Scene]:
    """Every scene of *catalogue* touching the bounds between two moments."""
    body: Optional[dict] = {
        "collections": [catalogue.collection],
        "bbox": [round(float(v), 6) for v in bounds_wgs84],
        "datetime": f"{iso(start)}/{iso(end)}",
        "limit": 100,
    }
    url = catalogue.search_url
    scenes: List[Scene] = []
    for _page in range(max_pages):
        document = fetch_json(url, body=body, what=catalogue.title)
        for item in document.get("features") or []:
            scene = _scene_from_item(item, catalogue)
            if scene is not None:
                scenes.append(scene)
        following = next((link for link in document.get("links") or []
                          if link.get("rel") == "next"), None)
        if not following or not following.get("href"):
            break
        url = following["href"]
        if following.get("body"):
            merged = dict(body) if following.get("merge") else {}
            merged.update(following["body"])
            body = merged
    return scenes


def rank(scenes: List[Scene], flight_time: datetime) -> List[Scene]:
    """Nearest in time first; of two equally near, the less cloudy tile."""
    return sorted(scenes, key=lambda scene: (
        round(abs(scene.offset_days(flight_time)), 2),
        scene.cloud_cover if scene.cloud_cover is not None else 100.0))


# --------------------------------------------------------------------------- #
# Indices
# --------------------------------------------------------------------------- #

def usable_mask(scl):
    """Boolean array: the cells of a scene classification an index may use."""
    import numpy as np

    return np.isin(np.asarray(scl), SCL_USABLE)


def clear_fraction(scl) -> float:
    """Share of the cells that are usable (0 for an empty array)."""
    import numpy as np

    mask = usable_mask(scl)
    return float(np.count_nonzero(mask)) / mask.size if mask.size else 0.0


def scl_composition(scl) -> Dict[str, float]:
    """``class name -> share`` of a scene classification, largest first."""
    import numpy as np

    values, counts = np.unique(np.asarray(scl), return_counts=True)
    total = float(counts.sum()) or 1.0
    shares = {SCL_NAMES.get(int(v), f"class {int(v)}"): round(c / total, 4)
              for v, c in zip(values, counts)}
    return dict(sorted(shares.items(), key=lambda kv: -kv[1]))


def reflectance(raw, scale: float, offset: float):
    """Digital numbers as surface reflectance; no-data (0) becomes NaN."""
    import numpy as np

    raw = np.asarray(raw)
    values = raw.astype(np.float32) * np.float32(scale) + np.float32(offset)
    values[raw == 0] = np.nan
    # A handful of dark cells come out slightly negative after atmospheric
    # correction; a ratio of a negative and a positive number is noise.
    return np.clip(values, 0.0, None)


def normalised_difference(a, b):
    """``(a - b) / (a + b)``, NaN where either is missing or both are zero."""
    import numpy as np

    with np.errstate(divide="ignore", invalid="ignore"):
        total = a + b
        result = np.where(total > 0, (a - b) / total, np.nan)
    return result.astype(np.float32)


def compute_indices(bands: Dict[str, "object"], usable) -> Dict[str, "object"]:
    """The four indices from reflectance *bands*, masked to *usable* cells."""
    import numpy as np

    indices = {
        "ndvi": normalised_difference(bands["nir"], bands["red"]),
        "ndmi": normalised_difference(bands["nir08"], bands["swir16"]),
        "ndwi": normalised_difference(bands["green"], bands["nir"]),
        "ndsi": normalised_difference(bands["green"], bands["swir16"]),
    }
    return {key: np.where(usable, values, np.nan).astype(np.float32)
            for key, values in indices.items()}


# --------------------------------------------------------------------------- #
# The source
# --------------------------------------------------------------------------- #

def fetch(grid: Grid, flight_time: datetime,
          window_days: float = DEFAULT_WINDOW_DAYS,
          min_clear: float = DEFAULT_MIN_CLEAR,
          catalogue: Catalogue = EARTH_SEARCH,
          search: Callable = search_scenes,
          read: Callable = env_remote.read_to_grid,
          log: Optional[Callable[[str], None]] = None,
          cancelled: Optional[Callable[[], bool]] = None,
          progress: Optional[Callable[[int], None]] = None,
          ) -> Tuple[Dict[str, "object"], dict]:
    """``(indices, metadata)`` of the clear scene nearest to the flight.

    Candidates are tried nearest first, before or after the flight alike; the
    first whose usable share over *grid* reaches *min_clear* is taken.

    :raises EnvironmentFailure: no scene in the window is clear enough
    """
    def say(message: str) -> None:
        if log:
            log(message)

    def check() -> None:
        check_cancelled(cancelled)

    window = timedelta(days=float(window_days))
    scenes = rank(search(grid_bounds_wgs84(grid), flight_time - window,
                         flight_time + window, catalogue), flight_time)
    if not scenes:
        raise EnvironmentFailure(
            f"{catalogue.title} lists no Sentinel-2 scene over the flight "
            f"within {window_days:g} days of {iso(flight_time)}.")
    say(f"{len(scenes)} Sentinel-2 scene(s) within {window_days:g} days of "
        f"the flight; checking them nearest first.")
    if progress:
        progress(10)

    considered = []
    chosen: Optional[Scene] = None
    chosen_scl = None
    for index, scene in enumerate(scenes):
        check()
        scl, _nodata = read(scene.assets["scl"][0], grid, "nearest")
        share = clear_fraction(scl)
        offset = scene.offset_days(flight_time)
        considered.append({
            "scene": scene.scene_id, "time": iso(scene.time),
            "offset_days": round(offset, 2),
            "clear_fraction": round(share, 4),
            "tile_cloud_cover": scene.cloud_cover})
        say(f"  {scene.scene_id} ({offset:+.1f} days): "
            f"{share * 100:.0f}% clear over the flight")
        if progress:
            progress(10 + int(40 * (index + 1) / len(scenes)))
        if share >= min_clear:
            chosen, chosen_scl = scene, scl
            break

    if chosen is None:
        best = max(considered, key=lambda entry: entry["clear_fraction"])
        raise EnvironmentFailure(
            f"No Sentinel-2 scene within {window_days:g} days of the flight "
            f"is at least {min_clear * 100:.0f}% clear over it. The best is "
            f"{best['scene']} ({best['offset_days']:+.1f} days) with "
            f"{best['clear_fraction'] * 100:.0f}%. Widen the window or "
            "lower the required clear share on the Environment tab.")

    bands = {}
    for index, role in enumerate(BANDS):
        check()
        href, scale, offset = chosen.assets[role]
        raw, _nodata = read(href, grid, "bilinear")
        bands[role] = reflectance(raw, scale, offset)
        if progress:
            progress(50 + int(45 * (index + 1) / len(BANDS)))

    usable = usable_mask(chosen_scl)
    indices = compute_indices(bands, usable)
    offset_days = chosen.offset_days(flight_time)
    metadata = {
        "catalogue": catalogue.key,
        "attribution": catalogue.attribution,
        "scene": chosen.scene_id,
        "platform": chosen.platform,
        "tile": chosen.tile,
        "scene_time": iso(chosen.time),
        "flight_time": iso(flight_time),
        "offset_days": round(offset_days, 2),
        "clear_fraction": round(clear_fraction(chosen_scl), 4),
        "tile_cloud_cover": chosen.cloud_cover,
        "scene_classification": scl_composition(chosen_scl),
        "window_days": float(window_days),
        "min_clear": float(min_clear),
        "considered": considered,
    }
    say(f"Using {chosen.scene_id}: taken {abs(offset_days):.1f} day(s) "
        f"{'before' if offset_days < 0 else 'after'} the flight, "
        f"{metadata['clear_fraction'] * 100:.0f}% clear over it.")
    return indices, metadata


def means(indices: Dict[str, "object"]) -> Dict[str, Optional[float]]:
    """Mean of every index over its valid cells (``None`` when it has none)."""
    import numpy as np

    result = {}
    for key, values in indices.items():
        valid = np.asarray(values)[np.isfinite(values)]
        result[key] = round(float(valid.mean()), 4) if valid.size else None
    return result

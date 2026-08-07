# -*- coding: utf-8 -*-
"""One-directional exports from the 6.0 store (EXCHANGE_FORMAT_PLAN.md §8.1).

Every exporter reads the store and writes throwaway files. None of them is an
input format, so nothing here has to survive a round trip — the store is the
source of truth and these are views onto it.

Two families:

* **training and tracking** — COCO, YOLO, MOT, TRex ``.npz``. These describe
  boxes, so ``not-an-animal`` is dropped by default.
* **recording and publishing** — GeoJSON, Camtrap DP, Darwin Core Archive.
  These describe a survey, so a labelled false positive is information and is
  kept (Darwin Core excepted: it publishes taxa, and a rejected detection is
  not one).
"""

from .biodiversity import export_camtrap_dp, export_darwin_core
from .common import ExportError
from .detections import export_coco, export_mot, export_yolo
from .geo import (
    export_geojson, export_segmentation_geojson, export_trex_npz,
)

#: ``key -> (label, callable, output_is_folder)``, for the export UI.
EXPORTERS = {
    "coco": ("COCO (detections)", export_coco, False),
    "yolo": ("YOLO (training labels)", export_yolo, True),
    "mot": ("MOT (tracking)", export_mot, True),
    "trex": ("TRex tracklets (.npz)", export_trex_npz, True),
    "geojson": ("GeoJSON (animals)", export_geojson, False),
    "geojson_segmentation": ("GeoJSON (segmentations)",
                             export_segmentation_geojson, False),
    "camtrap": ("Camtrap DP (survey package)", export_camtrap_dp, True),
    "dwca": ("Darwin Core Archive (GBIF)", export_darwin_core, True),
}

#: Default file name offered for the single-file formats.
DEFAULT_FILENAME = {"coco": "detections_coco.json",
                    "geojson": "animals.geojson",
                    "geojson_segmentation": "segmentations.geojson"}

#: Formats that publish latitude/longitude and therefore need the project CRS.
NEEDS_CRS = frozenset({"geojson", "geojson_segmentation",
                       "camtrap", "dwca"})

#: Formats describing boxes for training or tracking, which drop
#: ``not-an-animal`` by default. The rest are survey records and keep it (§8.1).
TRAINING_FORMATS = frozenset({"coco", "yolo", "mot", "trex"})

#: Formats needing the frame size to normalise or declare image dimensions.
NEEDS_FRAME_SIZE = frozenset({"coco", "yolo"})

#: Formats that reference the frames, and can therefore carry them. The frames
#: are the heaviest thing a project owns, so copying them is always a choice —
#: but YOLO and MOT are folder layouts that do not resolve without them, and a
#: Camtrap DP package is only self-contained with its media. GeoJSON and the
#: TRex ``.npz`` reference no image at all and are absent for that reason.
SUPPORTS_IMAGES = frozenset({"coco", "yolo", "mot", "camtrap"})


def run_export(key: str, target_folder: str, modality: str, output: str,
               epsg=None, include_not_an_animal=None, image_size=None,
               include_images=None, log_fn=None):
    """Run one exporter, passing only the arguments it accepts.

    The formats differ in what they can carry, so the caller should not have to
    know which takes an ``epsg``, which needs the frame size, or which drops
    false positives by default — those are the per-format decisions of §8.1,
    made here once.

    *image_size* is optional even for the formats that need it: they fall back
    to probing an extracted frame, and say so clearly if there is none.

    *include_images* is ignored by formats that reference no image.
    """
    if key not in EXPORTERS:
        raise ExportError(f"Unknown export format '{key}'.")
    _label, function, _is_folder = EXPORTERS[key]

    kwargs = {"log_fn": log_fn}
    if key in NEEDS_CRS:
        kwargs["epsg"] = epsg
    if key in NEEDS_FRAME_SIZE and image_size is not None:
        kwargs["image_size"] = image_size
    if key in SUPPORTS_IMAGES and include_images is not None:
        kwargs["include_images"] = include_images
    if key != "dwca":
        # Darwin Core always excludes false positives: a rejected detection is
        # by definition not an occurrence of anything.
        if include_not_an_animal is None:
            include_not_an_animal = key not in TRAINING_FORMATS
        kwargs["include_not_an_animal"] = include_not_an_animal

    return function(target_folder, modality, output, **kwargs)


__all__ = [
    "DEFAULT_FILENAME", "EXPORTERS", "ExportError", "NEEDS_CRS",
    "NEEDS_FRAME_SIZE", "SUPPORTS_IMAGES", "TRAINING_FORMATS", "run_export",
    "export_camtrap_dp", "export_coco", "export_darwin_core",
    "export_geojson", "export_mot", "export_segmentation_geojson",
    "export_trex_npz", "export_yolo",
]

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
from .geo import export_geojson, export_trex_npz

#: ``key -> (label, callable, output_is_folder)``, for the export UI.
EXPORTERS = {
    "coco": ("COCO (detections)", export_coco, False),
    "yolo": ("YOLO (training labels)", export_yolo, True),
    "mot": ("MOT (tracking)", export_mot, True),
    "trex": ("TRex tracklets (.npz)", export_trex_npz, True),
    "geojson": ("GeoJSON (geo-referenced)", export_geojson, False),
    "camtrap": ("Camtrap DP (survey package)", export_camtrap_dp, True),
    "dwca": ("Darwin Core Archive (GBIF)", export_darwin_core, True),
}

__all__ = [
    "EXPORTERS", "ExportError",
    "export_camtrap_dp", "export_coco", "export_darwin_core",
    "export_geojson", "export_mot", "export_trex_npz", "export_yolo",
]

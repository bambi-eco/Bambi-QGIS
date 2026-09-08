# -*- coding: utf-8 -*-
"""Detection of already-completed pipeline steps in a target folder.

Moved from ``bambi_dock_widget._check_existing_outputs`` (which delegates
here and maps the result onto its status labels). The camera selection of
each step is passed in as a suffix dict so this stays widget-free.
"""

import os
from typing import Dict, List, Sequence

# Steps whose status is derived from output folders (reset to "Not started"
# before re-checking).
FOLDER_STATUS_STEPS = (
    "extract_thermal_frames", "extract_rgb_frames", "detection",
    "georeference", "tracking", "calculate_fov", "flight_route",
    "alfs", "export_geotiffs", "orthomosaic", "sam3_segmentation",
    "sam3_georeference", "trex_import", "track_matching", "track_inventory",
)
PERPENDICULAR_STEPS = ("perpendicular", "track_perpendicular")

# Classification steps are store-backed and recorded per modality, and their
# status is read from the store's stage states rather than from folders:
# ``complete``, ``stale`` (an upstream step re-ran since) or ``pending``.
CLASSIFICATION_STATUS_STEPS = (
    "embeddings", "classify_occlusion", "classify_species", "classify_sex",
    "life_stage",
)
#: Which status label each classification task reports on, and which stage
#: carries its staleness.
CLASSIFICATION_TASKS = {
    "occlusion": ("classify_occlusion", "classification"),
    "species": ("classify_species", "classification"),
    "sex": ("classify_sex", "classification"),
    "life_stage": ("life_stage", "life_stage"),
}


def cross_modal_states(target_folder: str) -> Dict[str, str]:
    """``status_key -> "complete" | "stale" | "pending"`` for the steps
    recorded against both cameras at once (cross-modal track matching).

    The file scan can only say the matches exist; whether they still refer
    to the current tracking runs is what the stage state adds.
    """
    from . import stages

    if not target_folder or not os.path.isdir(target_folder):
        return {}
    recorded = stages.states(target_folder, stages.CROSS_MODAL)
    result: Dict[str, str] = {}
    for stage in stages.STAGE_SHARED_STORE_KIND:
        row = recorded.get(stage)
        if row is None or row.get("state") not in (stages.COMPLETE, stages.STALE):
            result[stage] = ("complete" if stages.has_output(
                target_folder, stage, stages.CROSS_MODAL) else "pending")
        else:
            result[stage] = ("stale" if row["state"] == stages.STALE
                             else "complete")
    return result


def _combine(states: List[str]) -> str:
    """One state for a step that spans several modalities.

    Pending anywhere means the step has not been run as configured; stale
    anywhere means part of its answer is out of date.
    """
    if not states or "pending" in states:
        return "pending"
    return "stale" if "stale" in states else "complete"


def classification_states(target_folder: str,
                          embeddings_targets: Sequence[str],
                          task_targets: Dict[str, Sequence[str]]
                          ) -> Dict[str, str]:
    """``status_key -> "complete" | "stale" | "pending"`` for the
    classification steps, as configured.

    :param target_folder: the pipeline output root folder
    :param embeddings_targets: modalities the embeddings step is set to
        cover (``"t"``, ``"w"`` or both for *matched*)
    :param task_targets: per task (``occlusion``, ``species``, ``sex``,
        ``life_stage``), the modalities its answers are recorded on
    """
    from . import classification_store, stages

    result: Dict[str, str] = {}
    if not target_folder or not os.path.isdir(target_folder):
        return result

    recorded = {m: stages.states(target_folder, m) for m in ("t", "w")}

    def _stage_state(stage: str, modality: str) -> str:
        row = recorded.get(modality, {}).get(stage)
        if row is None or row.get("state") not in (stages.COMPLETE, stages.STALE):
            return "pending"
        return "stale" if row["state"] == stages.STALE else "complete"

    result["embeddings"] = _combine(
        [_stage_state("embeddings", m) for m in embeddings_targets])

    predicted = {m: set(classification_store.predicted_tasks(target_folder, m))
                 for m in ("t", "w")}
    for task, (status_key, stage) in CLASSIFICATION_TASKS.items():
        per_modality = []
        for modality in task_targets.get(task, ()):
            if task not in predicted.get(modality, set()):
                per_modality.append("pending")
            else:
                state = _stage_state(stage, modality)
                # The predictions exist, so the stage ran even if its row
                # was never written (older projects).
                per_modality.append("complete" if state == "pending" else state)
        result[status_key] = _combine(per_modality)
    return result

# (subfolder_base, status_step_key, additional_check_file, camera_key)
_FOLDER_STATUS_MAPPING = [
    ("flight_route", "flight_route", None, "flight_route"),
    ("detections", "detection", None, "detection"),
    ("georeferenced", "georeference", None, "detection"),
    ("tracks", "tracking", None, "tracking"),
    ("fov", "calculate_fov", None, "fov"),
    ("alfs", "alfs", None, "alfs"),
    ("geotiffs", "export_geotiffs", None, "geotiff"),
    ("orthomosaic", "orthomosaic", None, "ortho"),
    ("segmentation", "sam3_segmentation", "segmentation_pixel.json", "sam3"),
    ("segmentation", "sam3_georeference", "segmentation_georef.json", "sam3"),
    ("tracks_pixel", "trex_import", None, "tracking"),
]


def _store_path(target_folder: str, status_key: str,
                modality: str) -> str:
    """Path of the 6.0 store file backing *status_key*, or ``""``."""
    from . import stages as stage_graph
    from . import store

    kind = stage_graph.STAGE_STORE_KIND.get(status_key)
    if not kind or modality not in store.MODALITIES:
        return ""
    return store.stage_path(target_folder, kind, modality)


def check_existing_outputs(target_folder: str,
                           cameras: Dict[str, str]) -> List[str]:
    """Return the status keys of pipeline steps with existing outputs.

    :param target_folder: the pipeline output root folder
    :param cameras: camera suffix (``"_t"`` / ``"_w"``) per step group;
        expected keys: ``extract``, ``flight_route``, ``detection``,
        ``tracking``, ``fov``, ``alfs``, ``geotiff``, ``ortho``, ``sam3``
    :return: completed step keys, in check order (a key appears once)
    """
    completed: List[str] = []
    if not target_folder or not os.path.isdir(target_folder):
        return completed

    # Frame extraction for the selected camera
    suffix = cameras.get("extract", "_t")
    frames_path = os.path.join(target_folder, "frames" + suffix)
    poses_path = os.path.join(target_folder, f"poses{suffix}.json")
    if os.path.isdir(frames_path) and os.listdir(frames_path) and os.path.isfile(poses_path):
        completed.append(
            "extract_thermal_frames" if suffix == "_t" else "extract_rgb_frames")

    # A stage counts as done when *either* its 6.0 store file or its legacy
    # folder is present, so a migrated project and a 5.x one both read
    # correctly, and so does a 6.x project with legacy text output switched off.
    from . import stages as stage_graph

    for subfolder_base, status_key, check_file, camera_key in _FOLDER_STATUS_MAPPING:
        modality = cameras.get(camera_key, "_t").lstrip("_")
        if status_key in stage_graph.STAGE_STORE_KIND:
            store_path = _store_path(target_folder, status_key, modality)
            if store_path and os.path.isfile(store_path):
                completed.append(status_key)
                continue

        subfolder_path = os.path.join(
            target_folder, subfolder_base + cameras.get(camera_key, "_t"))

        if not os.path.isdir(subfolder_path):
            continue

        if check_file:
            check_path_subfolder = os.path.join(subfolder_path, check_file)
            check_path_target = os.path.join(target_folder, check_file)
            if not os.path.isfile(check_path_subfolder) and not os.path.isfile(check_path_target):
                continue

        if os.listdir(subfolder_path):
            completed.append(status_key)

    # Cross-modal matching belongs to both cameras at once, so its store sits
    # beside project.gpkg rather than under a modality, and no camera combo
    # has a say in whether it counts.
    from . import store
    if os.path.isfile(store.matches_path(target_folder)):
        completed.append("track_matching")

    # The track inventory is a report per camera in that camera's analytics
    # folder.
    inventory_csv = os.path.join(
        target_folder, "analytics" + cameras.get("inventory", "_t"),
        "track_inventory.csv")
    if os.path.isfile(inventory_csv):
        completed.append("track_inventory")

    # Perpendicular results live in the selected flight-route camera's folder
    # and are suffixed with the camera of the detections/tracks they were
    # computed for.
    route_folder = os.path.join(
        target_folder, "flight_route" + cameras.get("flight_route", "_t"))
    for status_key, filename in (
            ("perpendicular",
             f"perpendicular{cameras.get('detection', '_t')}.json"),
            ("track_perpendicular",
             f"perpendicular_tracks{cameras.get('tracking', '_t')}.json")):
        if os.path.isfile(os.path.join(route_folder, filename)):
            completed.append(status_key)

    return completed

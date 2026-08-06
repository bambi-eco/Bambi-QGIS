# -*- coding: utf-8 -*-
"""Stage state, the dependency cascade, and resetting a stage.

Implements EXCHANGE_FORMAT_PLAN.md §7. Three jobs:

* **record** what each stage produced, when, and from which inputs;
* **invalidate** every transitive dependent when a stage is re-run or reset —
  replacing the hand-maintained cleanup in ``LabelStore.replace_detections``,
  which knew about ``tracks_{m}`` and ``tracks_pixel_{m}`` and nothing else
  (§1.3);
* **reconcile** the recorded state with what is actually on disk, because
  deleting a stage file in Explorer has to keep working.

Two rules run through all of it:

* **The files win.** The ``stages`` table is bookkeeping, not truth. If it
  disagrees with the folder, the folder is right.
* **Stale is flagged, not deleted.** Invalidation marks dependents; only an
  explicit reset removes anything, so a destructive act stays a deliberate one.
"""

import os
import shutil
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

from . import store

#: Frame extraction, the root of the graph. Not a store stage — its output is
#: the ``frames_{m}`` folder and ``poses_{m}.json``.
FRAMES = "frames"

#: What each stage needs before it can run. Keys match the status keys
#: ``output_inventory`` already uses, so saved configs and step ids are
#: unaffected (§10.1).
STAGE_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    "flight_route": (FRAMES,),
    "detection": (FRAMES,),
    "trex_import": (FRAMES,),
    "georeference": ("detection",),
    "tracking": ("georeference",),
    "calculate_fov": (FRAMES,),
    "alfs": (FRAMES,),
    "export_geotiffs": (FRAMES,),
    "orthomosaic": ("export_geotiffs",),
    "sam3_segmentation": (FRAMES,),
    "sam3_georeference": ("sam3_segmentation",),
    # Classification. Everything here is keyed on track ids, so re-running
    # tracking invalidates all of it — which is exactly what the cascade does.
    "track_matching": ("tracking",),
    "embeddings": ("tracking",),
    "classification": ("embeddings",),
    # Life stage is read from box area, not from features, so it needs no
    # embeddings — only tracks and the geo-referencing that makes the area
    # metric.
    "life_stage": ("tracking", "georeference"),
    "perpendicular": ("georeference", "flight_route"),
    "track_perpendicular": ("tracking", "flight_route"),
    "population": ("tracking",),
    "density": ("tracking",),
    "coverage": ("calculate_fov",),
}

#: Stages whose output is a store file, and which one.
STAGE_STORE_KIND: Dict[str, str] = {
    "detection": store.DETECTIONS,
    "georeference": store.GEOREFERENCED,
    "tracking": store.TRACKS,
    "calculate_fov": store.FOV,
    "sam3_segmentation": store.SEGMENTATION,
    # ``classification`` and ``life_stage`` write into the same file, so only
    # the stage that creates it claims it here: letting either reset delete the
    # file would take the other's results with it. Their own outputs are
    # cleared row-wise by core.classification_store.
    "embeddings": store.CLASSIFICATION,
}

#: Stages whose output is a modality-independent store, and which one. These
#: are recorded against :data:`CROSS_MODAL` rather than a single modality.
STAGE_SHARED_STORE_KIND: Dict[str, str] = {
    "track_matching": store.MATCHES,
}

#: The modality a cross-modal stage is recorded under. A match belongs to both
#: sides at once, and recording it twice would let one be reset while the other
#: still claimed to be complete.
CROSS_MODAL = "tw"

#: Legacy output folders per stage, still written during 6.x. A reset has to
#: remove these too or the stage would still look complete (§7).
STAGE_LEGACY_FOLDERS: Dict[str, Tuple[str, ...]] = {
    "detection": ("detections",),
    "georeference": ("georeferenced",),
    "tracking": ("tracks", "tracks_pixel"),
    "calculate_fov": ("fov",),
    "flight_route": ("flight_route",),
    "alfs": ("alfs",),
    "export_geotiffs": ("geotiffs",),
    "orthomosaic": ("orthomosaic",),
    "sam3_segmentation": ("segmentation",),
    "trex_import": ("tracks_pixel",),
}

#: Current (non-legacy) output folders per stage. The embedding vectors are
#: files beside the frames rather than rows in the GeoPackage, so a reset has
#: to remove them too — they are the expensive part, and leaving them behind
#: would make a "reset" look like it had done nothing.
STAGE_OUTPUT_FOLDERS: Dict[str, Tuple[str, ...]] = {
    "embeddings": ("embeddings",),
}

PENDING = "pending"
RUNNING = "running"
COMPLETE = "complete"
FAILED = "failed"
STALE = "stale"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def dependents(stage: str) -> List[str]:
    """Every stage that transitively depends on *stage*.

    This is the cascade: re-running or resetting *stage* makes all of these
    stale, which is what the 5.x code did by hand for two folders and not at
    all for the rest.
    """
    found: Set[str] = set()
    frontier = [stage]
    while frontier:
        current = frontier.pop()
        for candidate, needs in STAGE_DEPENDENCIES.items():
            if current in needs and candidate not in found:
                found.add(candidate)
                frontier.append(candidate)
    return sorted(found)


def dependencies(stage: str) -> Tuple[str, ...]:
    """What *stage* needs before it can run."""
    return STAGE_DEPENDENCIES.get(stage, ())


def known_stages() -> List[str]:
    return sorted(STAGE_DEPENDENCIES)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _project(target_folder: str):
    return store.open_store(store.project_path(target_folder), store.PROJECT)


def set_state(target_folder: str, stage: str, modality: str, state: str,
              row_count: Optional[int] = None,
              fingerprint: str = "") -> None:
    """Record *stage*'s state, without touching its dependents."""
    conn = _project(target_folder)
    try:
        with store.transaction(conn):
            now = _utc_now()
            existing = conn.execute(
                "SELECT started_at FROM stages WHERE stage = ? AND modality = ?",
                (stage, modality)).fetchone()
            started = existing["started_at"] if existing else None
            conn.execute(
                "INSERT OR REPLACE INTO stages (stage, modality, state, "
                "row_count, started_at, finished_at, generation, "
                "input_fingerprint) VALUES (?, ?, ?, ?, ?, ?, "
                "(SELECT IFNULL(MAX(generation), 0) + 1 FROM stages "
                " WHERE stage = ? AND modality = ?), ?)",
                (stage, modality, state, row_count,
                 now if state == RUNNING else started,
                 now if state in (COMPLETE, FAILED) else None,
                 stage, modality, fingerprint))
    finally:
        conn.close()


def mark_running(target_folder: str, stage: str, modality: str) -> None:
    set_state(target_folder, stage, modality, RUNNING)


def mark_complete(target_folder: str, stage: str, modality: str,
                  row_count: Optional[int] = None,
                  fingerprint: str = "") -> List[str]:
    """Record success and mark every dependent stale. Returns what went stale."""
    set_state(target_folder, stage, modality, COMPLETE, row_count, fingerprint)
    return mark_dependents_stale(target_folder, stage, modality)


def mark_failed(target_folder: str, stage: str, modality: str) -> None:
    set_state(target_folder, stage, modality, FAILED)


def mark_dependents_stale(target_folder: str, stage: str,
                          modality: str) -> List[str]:
    """Flag every transitive dependent of *stage* as stale.

    Nothing is deleted: stale results stay on disk until the user resets them,
    so losing work is always a deliberate click.
    """
    affected = []
    conn = _project(target_folder)
    try:
        with store.transaction(conn):
            for dependent in dependents(stage):
                row = conn.execute(
                    "SELECT state FROM stages WHERE stage = ? AND modality = ?",
                    (dependent, modality)).fetchone()
                if row is None or row["state"] not in (COMPLETE, STALE):
                    continue
                conn.execute(
                    "UPDATE stages SET state = ? WHERE stage = ? AND "
                    "modality = ?", (STALE, dependent, modality))
                if row["state"] == COMPLETE:
                    affected.append(dependent)
    finally:
        conn.close()
    return affected


def states(target_folder: str, modality: str = "") -> Dict[str, dict]:
    """Recorded stage state, keyed by stage (or ``(stage, modality)``)."""
    if not os.path.isfile(store.project_path(target_folder)):
        return {}
    conn = _project(target_folder)
    try:
        if modality:
            rows = conn.execute(
                "SELECT * FROM stages WHERE modality = ?", (modality,))
            return {row["stage"]: dict(row) for row in rows}
        rows = conn.execute("SELECT * FROM stages")
        return {(row["stage"], row["modality"]): dict(row) for row in rows}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Reconciliation — the files win
# ---------------------------------------------------------------------------

def stage_outputs(target_folder: str, stage: str,
                  modality: str) -> List[str]:
    """Paths this stage owns: its store file plus any output folders."""
    paths = []
    kind = STAGE_STORE_KIND.get(stage)
    if kind:
        paths.append(store.stage_path(target_folder, kind, modality))
    shared = STAGE_SHARED_STORE_KIND.get(stage)
    if shared == store.MATCHES:
        paths.append(store.matches_path(target_folder))
    for base in STAGE_LEGACY_FOLDERS.get(stage, ()):
        paths.append(os.path.join(target_folder, f"{base}_{modality}"))
    for base in STAGE_OUTPUT_FOLDERS.get(stage, ()):
        paths.append(os.path.join(target_folder, f"{base}_{modality}"))
    return paths


def has_output(target_folder: str, stage: str, modality: str) -> bool:
    """True when anything this stage produces exists on disk.

    Replaces ``output_inventory``'s "folder exists and is non-empty" check with
    the same question asked of the store file as well.
    """
    for path in stage_outputs(target_folder, stage, modality):
        if os.path.isfile(path):
            return True
        if os.path.isdir(path) and os.listdir(path):
            return True
    return False


def reconcile(target_folder: str, modality: str) -> List[str]:
    """Bring the recorded state in line with the folder. Returns what changed.

    Someone deleting ``bambi_t/tracks.gpkg`` in Explorer must still get the
    right answer, so anything recorded complete whose output has gone is reset
    to pending, and its dependents go stale.
    """
    if not os.path.isfile(store.project_path(target_folder)):
        return []

    changed = []
    for stage, row in states(target_folder, modality).items():
        if row["state"] not in (COMPLETE, STALE):
            continue
        if has_output(target_folder, stage, modality):
            continue
        set_state(target_folder, stage, modality, PENDING)
        mark_dependents_stale(target_folder, stage, modality)
        changed.append(stage)
    return changed


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class StageLockedError(RuntimeError):
    """A stage file could not be removed because something holds it open.

    On Windows a layer loaded in the QGIS Browser locks the file. The plugin
    itself never file-backs a layer (§11), so this means the user opened it —
    which is worth saying plainly rather than surfacing a raw ``PermissionError``.
    """


def reset(target_folder: str, stage: str, modality: str,
          cascade: bool = False, log_fn=None) -> Dict[str, List[str]]:
    """Delete *stage*'s outputs and mark it pending.

    With *cascade* the dependents' outputs go too; otherwise they are only
    flagged stale. Returns ``{"removed": [paths], "stale": [stages]}``.
    """
    to_reset = [stage] + (dependents(stage) if cascade else [])
    removed: List[str] = []

    for current in to_reset:
        for path in stage_outputs(target_folder, current, modality):
            if not os.path.exists(path):
                continue
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except OSError as exc:
                raise StageLockedError(
                    f"{os.path.basename(path)} could not be removed: {exc}. "
                    "If you opened it in the QGIS Browser, close that layer "
                    "and try again.") from exc
            removed.append(path)

    stale: List[str] = []
    if os.path.isfile(store.project_path(target_folder)):
        for current in to_reset:
            set_state(target_folder, current, modality, PENDING)
        stale = mark_dependents_stale(target_folder, stage, modality)

    if log_fn:
        stale_note = (f", {len(stale)} dependent stage(s) marked stale"
                      if stale else "")
        log_fn(f"Reset '{stage}' ({modality}): "
               f"removed {len(removed)} output(s){stale_note}")
    return {"removed": removed, "stale": stale}


def completed_stages(target_folder: str, modality: str) -> List[str]:
    """Stages with output on disk, reconciled against the recorded state."""
    return sorted(stage for stage in known_stages()
                  if has_output(target_folder, stage, modality))


def stale_stages(target_folder: str, modality: str) -> List[str]:
    """Stages recorded as stale — output present but out of date."""
    return sorted(stage for stage, row in states(target_folder, modality).items()
                  if row["state"] == STALE)


def describe(target_folder: str, modality: str) -> List[dict]:
    """Per-stage summary for the UI: state, counts and what depends on it."""
    recorded = states(target_folder, modality)
    summary = []
    for stage in known_stages():
        row = recorded.get(stage, {})
        on_disk = has_output(target_folder, stage, modality)
        state = row.get("state")
        if state in (COMPLETE, STALE) and not on_disk:
            state = PENDING
        elif state is None:
            state = COMPLETE if on_disk else PENDING
        summary.append({
            "stage": stage,
            "state": state,
            "on_disk": on_disk,
            "row_count": row.get("row_count"),
            "finished_at": row.get("finished_at"),
            "dependents": dependents(stage),
        })
    return summary


def fingerprint(config: dict, keys: Iterable[str],
                upstream: Iterable[str] = ()) -> str:
    """Hash the config values a stage actually depends on, plus its inputs.

    Answers "was this really recomputed?" directly, which folder mtimes never
    could (§7).
    """
    import hashlib
    import json

    payload = {
        "config": {key: config.get(key) for key in sorted(keys)},
        "upstream": sorted(upstream),
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

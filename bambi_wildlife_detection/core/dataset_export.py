# -*- coding: utf-8 -*-
"""A survey exported as a dataset: several flights, each in one split.

The exporters (:mod:`core.exporters`) write one flight. A training set is
made of many - and of a decision the exporters know nothing about: which
flights are trained on, which are validated on and which are tested on. A
flight goes into a split whole. Its frames are seconds apart and show the
same animals from almost the same place; frames of one flight on both
sides of a split measure how well a model remembers, not how well it
generalises.

What is written::

    <output>/
    ├── splits.json            {"train": [flight, ...], "val": [...],
    │                           "test": [...]}
    ├── dataset.json           format, camera, classes, what each flight gave
    ├── train/<flight>/...     the flight as its exporter writes it
    ├── val/<flight>/...
    └── test/<flight>/...

and, for the formats a training reads as one set, what joins the flights:
species ids are a flight's own, so the classes are named alike across all
of them (by species name), and

* COCO: ``<split>/annotations.json`` - every flight of the split in one
  file, the images as ``<flight>/images/<name>``;
* YOLO: ``data.yaml`` at the top, with the image folders of every flight
  per split.

No QGIS here.
"""

import json
import os
import random
import re
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from . import exporters

TRAIN, VAL, TEST = "train", "val", "test"
SPLITS = (TRAIN, VAL, TEST)
#: A flight that is in the survey and in no split is not exported.
LEFT_OUT = ""
SPLIT_LABELS = {TRAIN: "Train", VAL: "Validation", TEST: "Test",
                LEFT_OUT: "Leave out"}
#: Per cent of the flights for train, validation and test.
DEFAULT_SHARES = (70, 15, 15)
SPLITS_FILE = "splits.json"
DATASET_FILE = "dataset.json"
COCO_FILE = "annotations.json"


class DatasetExportError(RuntimeError):
    """The dataset cannot be exported as asked."""


# ---------------------------------------------------------------------------
# Which flight goes where
# ---------------------------------------------------------------------------

def counts_for(total: int, shares: Sequence[float]) -> List[int]:
    """How many of *total* flights each split gets for *shares* (any
    scale): the nearest whole numbers that add up to *total*. A split with
    a share gets at least one flight where there are enough flights for
    every such split - training first, where there are not."""
    shares = [max(0.0, float(share)) for share in shares]
    if total <= 0:
        return [0 for _ in shares]
    if sum(shares) <= 0:
        raise DatasetExportError(
            "At least one split needs a share of the flights.")
    exact = [total * share / sum(shares) for share in shares]
    counts = [int(value) for value in exact]
    by_remainder = sorted(range(len(shares)),
                          key=lambda i: (counts[i] - exact[i], i))
    for index in by_remainder[:total - sum(counts)]:
        counts[index] += 1
    wanted = [index for index, share in enumerate(shares) if share > 0]
    if total >= len(wanted):
        for index in wanted:
            while counts[index] == 0:
                richest = max(range(len(counts)), key=lambda i: counts[i])
                counts[richest] -= 1
                counts[index] += 1
    return counts


def assign(flights: Iterable[str], shares: Sequence[float] = DEFAULT_SHARES,
           seed: int = 0) -> Dict[str, str]:
    """Deal *flights* into train / validation / test by *shares*, at
    random but the same way every time for one *seed* - whatever order the
    flights come in.

    :return: ``{flight: split}``
    """
    flights = sorted(set(flights))
    # Not for secrecy: a repeatable shuffle.
    random.Random(int(seed)).shuffle(flights)  # nosec B311
    dealt: Dict[str, str] = {}
    position = 0
    for split, count in zip(SPLITS, counts_for(len(flights), shares)):
        for flight in flights[position:position + count]:
            dealt[flight] = split
        position += count
    return dealt


def by_split(assignment: Dict[str, str]) -> Dict[str, List[str]]:
    """``{"train": [...], "val": [...], "test": [...]}`` of an assignment,
    each list sorted; what is left out is in none."""
    return {split: sorted(name for name, where in assignment.items()
                          if where == split) for split in SPLITS}


def describe(assignment: Dict[str, str]) -> str:
    """``"5 train, 1 validation, 1 test, 2 left out"``."""
    lists = by_split(assignment)
    parts = [f"{len(lists[TRAIN])} train", f"{len(lists[VAL])} validation",
             f"{len(lists[TEST])} test"]
    left = sum(1 for where in assignment.values() if where not in SPLITS)
    if left:
        parts.append(f"{left} left out")
    return ", ".join(parts)


def folder_names(names: Sequence[str]) -> Dict[str, str]:
    """A folder name for each flight: its name with what a file system
    does not take replaced, and a number where two would be alike."""
    taken: Dict[str, str] = {}
    used = set()
    for name in names:
        base = re.sub(r"[^\w.-]+", "_", str(name).strip()).strip("._")
        base = base or "flight"
        candidate, number = base, 2
        while candidate.lower() in used:
            candidate = f"{base}_{number}"
            number += 1
        used.add(candidate.lower())
        taken[name] = candidate
    return taken


def check_output(output: str, assignment: Dict[str, str]) -> None:
    """Refuse an output folder in which a flight would end up in two
    splits: one that holds an earlier export with a flight in another
    split than now, or split folders nobody can account for.

    :raises DatasetExportError: naming the flights
    """
    if not os.path.isdir(output):
        return
    earlier_path = os.path.join(output, SPLITS_FILE)
    folders = [os.path.join(output, split) for split in SPLITS]
    present = [os.path.basename(folder) for folder in folders
               if os.path.isdir(folder) if os.listdir(folder)]
    if not os.path.isfile(earlier_path):
        if present:
            raise DatasetExportError(
                f"{output} already holds {', '.join(present)} folder(s) "
                "that are not an export of this tool. Choose an empty "
                "folder, so that no flight ends up in two splits.")
        return
    try:
        with open(earlier_path, encoding="utf-8") as handle:
            earlier = json.load(handle)
    except (OSError, ValueError) as exc:
        raise DatasetExportError(
            f"{earlier_path} cannot be read ({exc}). Choose an empty "
            "folder.") from exc
    moved = []
    for split in SPLITS:
        for name in earlier.get(split) or []:
            now = assignment.get(name, LEFT_OUT)
            if now != split:
                moved.append(
                    f"{name} ({split} -> "
                    f"{SPLIT_LABELS.get(now, now).lower()})")
    if moved:
        raise DatasetExportError(
            f"{output} holds an earlier export in which these flights "
            f"were in another split: {', '.join(moved)}. Exporting over it "
            "would leave them in both. Choose an empty folder, or remove "
            "the earlier export first.")


# ---------------------------------------------------------------------------
# Joining the flights
# ---------------------------------------------------------------------------

def _read_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, document) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)


def _class_index(classes: List[str], name: str) -> int:
    if name not in classes:
        classes.append(name)
    return classes.index(name)


def join_coco(output: str, done: Sequence[dict], classes: List[str]) -> None:
    """Name the classes of every flight's COCO file alike, and write one
    ``annotations.json`` per split with all its flights.

    Image and annotation ids are a flight's own; in the joined file they
    are numbered through, and an image is ``<flight>/images/<name>`` with
    its ``flight`` beside it - a track id means something only together
    with that.
    """
    for flight in done:
        document = _read_json(flight["path"])
        local = {category["id"]: _class_index(classes, category["name"])
                 for category in document.get("categories", [])}
        for annotation in document.get("annotations", []):
            annotation["category_id"] = local[annotation["category_id"]]
        flight["document"] = document
    categories = [{"id": index, "name": name}
                  for index, name in enumerate(classes)]
    for split in SPLITS:
        images, annotations = [], []
        for flight in done:
            document = flight.pop("document") if flight["split"] == split \
                else None
            if document is None:
                continue
            document["categories"] = categories
            _write_json(flight["path"], document)
            image_ids = {}
            for image in document.get("images", []):
                image_ids[image["id"]] = len(images)
                images.append(dict(
                    image, id=len(images), flight=flight["name"],
                    file_name=f"{flight['folder']}/images/"
                              f"{image['file_name']}"))
            for annotation in document.get("annotations", []):
                annotations.append(dict(
                    annotation, id=len(annotations),
                    image_id=image_ids[annotation["image_id"]]))
            flight["boxes"] = len(document.get("annotations", []))
        if not images and not os.path.isdir(os.path.join(output, split)):
            continue
        os.makedirs(os.path.join(output, split), exist_ok=True)
        _write_json(os.path.join(output, split, COCO_FILE), {
            "info": {"description": f"BAMBI detections ({split})",
                     "split": split},
            "images": images, "annotations": annotations,
            "categories": categories})


def _yolo_names(path: str) -> Dict[int, str]:
    """``{index: name}`` of a ``data.yaml`` as the YOLO exporter writes
    it."""
    names: Dict[int, str] = {}
    listing = False
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("names:"):
                listing = True
                continue
            if not listing:
                continue
            match = re.match(r"\s+(\d+):\s?(.*)$", line.rstrip("\n"))
            if match is None:
                break
            names[int(match.group(1))] = match.group(2)
    return names


def _write_yolo_yaml(path: str, root: str, sets: Dict[str, List[str]],
                     classes: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"path: {root}\n")
        for split in SPLITS:
            folders = sets.get(split) or []
            if not folders:
                continue
            handle.write(f"{split}:\n")
            for folder in folders:
                handle.write(f"  - {folder}\n")
        handle.write(f"nc: {len(classes)}\n")
        handle.write("names:\n")
        for index, name in enumerate(classes):
            handle.write(f"  {index}: {name}\n")


def join_yolo(output: str, done: Sequence[dict], classes: List[str]) -> None:
    """Number the classes of every flight's labels alike, and write the
    ``data.yaml`` that names the image folders of every flight per
    split."""
    for flight in done:
        local = {index: _class_index(classes, name) for index, name in
                 _yolo_names(os.path.join(flight["path"], "data.yaml")).items()}
        labels = os.path.join(flight["path"], "labels")
        boxes = 0
        for name in sorted(os.listdir(labels)) if os.path.isdir(labels) \
                else []:
            if not name.endswith(".txt"):
                continue
            path = os.path.join(labels, name)
            with open(path, encoding="utf-8") as handle:
                lines = [line.split() for line in handle if line.strip()]
            with open(path, "w", encoding="utf-8") as handle:
                for cells in lines:
                    cells[0] = str(local[int(cells[0])])
                    handle.write(" ".join(cells) + "\n")
            boxes += len(lines)
        flight["boxes"] = boxes
    sets: Dict[str, List[str]] = {split: [] for split in SPLITS}
    for flight in done:
        sets[flight["split"]].append(
            f"{flight['split']}/{flight['folder']}/images")
        # the flight's own file, with the classes as they now are: on its
        # own it is still what the exporter makes of one flight
        _write_yolo_yaml(os.path.join(flight["path"], "data.yaml"),
                         flight["path"], {TRAIN: ["images"], VAL: ["images"]},
                         classes)
    _write_yolo_yaml(os.path.join(output, "data.yaml"), output, sets, classes)


_JOINERS = {"coco": join_coco, "yolo": join_yolo}


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------

def export(key: str, flights: Sequence[dict], modality: str, output: str,
           include_images: Optional[bool] = None,
           include_not_an_animal: Optional[bool] = None,
           progress: Optional[Callable[[int], None]] = None,
           cancelled: Optional[Callable[[], bool]] = None,
           log: Optional[Callable[[str], None]] = None,
           run_export=None, confirmed: bool = False) -> dict:
    """Export the flights of a survey as one dataset.

    :param key: the format, a key of ``core.exporters.EXPORTERS``
    :param flights: ``{"name", "target_folder", "split", "epsg"}`` each;
        a flight whose split is none of :data:`SPLITS` is left out
    :param output: the dataset's folder
    :param run_export: stands in for ``core.exporters.run_export`` (tests)
    :param confirmed: only what both cameras saw - the boxes of tracks
        the cross-modal matching paired with a track of the other camera
    :return: the content of ``dataset.json``
    :raises DatasetExportError: when nothing is to export, the format is
        unknown, the output folder holds another export, or no flight
        could be exported
    """
    log = log or (lambda message: None)
    run_export = run_export or exporters.run_export
    if key not in exporters.EXPORTERS:
        raise DatasetExportError(f"Unknown export format '{key}'.")
    label, _function, is_folder = exporters.EXPORTERS[key]
    if not (output or "").strip():
        raise DatasetExportError("Choose the folder to export into.")

    names = [flight["name"] for flight in flights]
    if len(set(names)) != len(names):
        raise DatasetExportError(
            "Two flights of the survey have the same name; a dataset "
            "tells its flights apart by name.")
    assignment = {flight["name"]: (flight.get("split") or LEFT_OUT)
                  for flight in flights}
    chosen = [flight for flight in flights
              if assignment[flight["name"]] in SPLITS]
    if not chosen:
        raise DatasetExportError(
            "No flight is in a split. Put the flights into train, "
            "validation or test first.")
    check_output(output, assignment)
    folders = folder_names(names)

    done, failed = [], []
    for index, flight in enumerate(chosen):
        if cancelled and cancelled():
            raise DatasetExportError("The export was cancelled.")
        split = assignment[flight["name"]]
        folder = folders[flight["name"]]
        path = os.path.join(output, split, folder)
        os.makedirs(path, exist_ok=True)
        if not is_folder:
            path = os.path.join(path, exporters.DEFAULT_FILENAME.get(
                key, "export.json"))
        log(f"{flight['name']} -> {split}/{folder}")
        try:
            run_export(
                key, flight["target_folder"], modality, path,
                epsg=flight.get("epsg"),
                include_not_an_animal=include_not_an_animal,
                include_images=include_images,
                log_fn=lambda message: log(f"  {message}"),
                **({"confirmed": True} if confirmed else {}))
        except exporters.ExportError as exc:
            failed.append({"name": flight["name"], "split": split,
                           "error": str(exc)})
            log(f"  {flight['name']} was not exported: {exc}")
            if os.path.isdir(os.path.join(output, split, folder)) and \
                    not os.listdir(os.path.join(output, split, folder)):
                os.rmdir(os.path.join(output, split, folder))
        else:
            done.append({"name": flight["name"], "split": split,
                         "folder": folder, "path": path})
        if progress:
            progress(int(90 * (index + 1) / len(chosen)))
    if not done:
        raise DatasetExportError(
            "No flight could be exported: " + "; ".join(
                f"{entry['name']}: {entry['error']}" for entry in failed))

    classes: List[str] = []
    if key in _JOINERS:
        _JOINERS[key](output, done, classes)

    exported = {entry["name"]: entry["split"] for entry in done}
    _write_json(os.path.join(output, SPLITS_FILE), by_split(exported))
    summary = {
        "format": key, "format_label": label,
        "camera": {"t": "thermal", "w": "rgb"}.get(modality, modality),
        "confirmed_by_other_camera": bool(confirmed),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "splits": by_split(exported),
        "classes": classes,
        "flights": [{k: v for k, v in entry.items() if k != "path"}
                    for entry in done],
        "not_exported": failed,
        "left_out": sorted(name for name, where in assignment.items()
                           if where not in SPLITS),
    }
    _write_json(os.path.join(output, DATASET_FILE), summary)
    if progress:
        progress(100)
    log(f"Dataset written to {output}: {describe(exported)}")
    return summary

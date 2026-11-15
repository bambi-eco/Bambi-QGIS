# -*- coding: utf-8 -*-
"""COCO, YOLO and MOT exporters.

All three are one-directional: they read the store and write throwaway files.
None of them is an input format, so nothing has to survive a round trip.

The interesting difference between them is how much of the 6.0 model they can
carry. COCO tolerates extra keys on an annotation, so custom fields map onto it
directly and it is the reference detection export. YOLO has room for a class id
and nothing else. MOT has a fixed column layout, so anything extra goes to a
sidecar keyed by ``(frame, track_id)``.
"""

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

from . import common


def _cell(value) -> str:
    """A value as a CSV cell: nothing for "not known", lower-case booleans."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def export_coco(target_folder: str, modality: str, output_path: str,
                include_not_an_animal: bool = False,
                image_size: Optional[Tuple[int, int]] = None,
                include_images: bool = True, log_fn=None) -> str:
    """Write a COCO detection JSON, and optionally the images beside it.

    The reference export: COCO permits extra keys on an annotation, so every
    custom field travels with its detection instead of being dropped the way
    ``labels.csv`` dropped them.

    ``file_name`` is a bare name, which readers resolve against an image root
    they are told about separately. *include_images* copies the frames into an
    ``images`` folder next to the JSON so that root is the export itself.
    """
    vocabulary = common.load_vocabulary(target_folder)
    rows, _animals = common.load_export_rows(
        target_folder, modality, vocabulary, include_not_an_animal,
        include_geo=False)
    width, height = common.require_frame_size(
        target_folder, modality, image_size)
    frames = {f["frame"]: f for f in common.load_frames(target_folder, modality)}
    rows = common.without_duplicated(rows, frames)

    mapping, names = common.class_map(
        (row["species_id"] for row in rows), vocabulary["species"])

    used_frames = sorted({row["frame"] for row in rows})
    images = [{
        "id": frame,
        "file_name": frames.get(frame, {}).get(
            "imagefile", f"frame_{frame:06d}.jpg"),
        "width": width,
        "height": height,
    } for frame in used_frames]

    annotations = []
    for row in rows:
        w = row["x2"] - row["x1"]
        h = row["y2"] - row["y1"]
        annotation = {
            "id": row["detection_id"],
            "image_id": row["frame"],
            "category_id": mapping[row["species_id"]],
            "bbox": [row["x1"], row["y1"], w, h],
            "area": w * h,
            "iscrowd": 0,
            "score": row["confidence"],
        }
        if row.get("track_id") is not None:
            annotation["track_id"] = row["track_id"]
        # the box's own fields and what is known of the animal: sex, age,
        # whether it was approved by hand
        if row["properties"]:
            annotation["attributes"] = row["properties"]
        annotations.append(annotation)

    document = {
        "info": {"description": "BAMBI detections",
                 "modality": modality},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index, "name": name}
                       for index, name in enumerate(names)],
    }

    folder = os.path.dirname(os.path.abspath(output_path))
    common.ensure_folder(folder)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2)

    copied = None
    if include_images:
        copied = common.copy_frames(
            target_folder, modality, [image["file_name"] for image in images],
            os.path.join(folder, "images"))

    if log_fn:
        log_fn(f"COCO: {len(annotations)} annotation(s) over "
               f"{len(images)} image(s) → {output_path}")
        for line in common.describe_copy(copied) if copied else []:
            log_fn(f"COCO: {line}")
    return output_path


def export_yolo(target_folder: str, modality: str, output_folder: str,
                include_not_an_animal: bool = False,
                image_size: Optional[Tuple[int, int]] = None,
                include_images: bool = True, log_fn=None) -> str:
    """Write a YOLO dataset: ``images/``, ``labels/`` and ``data.yaml``.

    One ``.txt`` per frame that has detections, named after the frame image, in
    normalised ``cls cx cy w h``. Custom fields have nowhere to go in this
    format and are deliberately not smuggled into it - use COCO for those.

    With *include_images* the frames are copied in, because a YOLO dataset is a
    folder layout rather than a manifest: there is no field naming the images,
    so the labels are only found through ``images``/``labels`` siblings. Only
    frames carrying a detection are copied. Turning it off leaves the labels
    alone for callers who already have the images arranged - and saves
    duplicating what is usually the heaviest part of a project.
    """
    vocabulary = common.load_vocabulary(target_folder)
    rows, _animals = common.load_export_rows(
        target_folder, modality, vocabulary, include_not_an_animal,
        include_geo=False)
    width, height = common.require_frame_size(
        target_folder, modality, image_size)
    frames = {f["frame"]: f for f in common.load_frames(target_folder, modality)}
    rows = common.without_duplicated(rows, frames)

    mapping, names = common.class_map(
        (row["species_id"] for row in rows), vocabulary["species"])

    # ``images`` and ``labels`` must be siblings with matching stems: YOLO
    # finds a label by swapping the last "images" segment of the image path for
    # "labels". Nothing declares that mapping, so the layout is the contract.
    images_folder = common.ensure_folder(os.path.join(output_folder, "images"))
    labels_folder = common.ensure_folder(os.path.join(output_folder, "labels"))
    by_frame: Dict[int, List[dict]] = {}
    for row in rows:
        by_frame.setdefault(row["frame"], []).append(row)

    used = [frames.get(frame, {}).get("imagefile", f"frame_{frame:06d}.jpg")
            for frame in sorted(by_frame)]

    for frame, frame_rows in sorted(by_frame.items()):
        imagefile = frames.get(frame, {}).get(
            "imagefile", f"frame_{frame:06d}.jpg")
        stem = os.path.splitext(imagefile)[0]

        with open(os.path.join(labels_folder, f"{stem}.txt"),
                  "w", encoding="utf-8") as fh:
            for row in frame_rows:
                cx = ((row["x1"] + row["x2"]) / 2.0) / width
                cy = ((row["y1"] + row["y2"]) / 2.0) / height
                w = (row["x2"] - row["x1"]) / width
                h = (row["y2"] - row["y1"]) / height
                fh.write(f"{mapping[row['species_id']]} "
                         f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    yaml_path = os.path.join(output_folder, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(f"path: {output_folder}\n")
        fh.write("train: images\n")
        # Everything exported has labels, so there is no held-out set to point
        # at. Naming the training images keeps the file valid for a training
        # run; split it properly before believing any validation number.
        fh.write("val: images\n")
        fh.write(f"nc: {len(names)}\n")
        fh.write("names:\n")
        for index, name in enumerate(names):
            fh.write(f"  {index}: {name}\n")

    copied = None
    if include_images:
        copied = common.copy_frames(target_folder, modality, used,
                                    images_folder)

    if log_fn:
        log_fn(f"YOLO: {len(rows)} box(es) over {len(by_frame)} frame(s) "
               f"→ {output_folder}")
        if copied is not None:
            for line in common.describe_copy(copied):
                log_fn(f"YOLO: {line}")
        else:
            log_fn("YOLO: images not included - point 'train' at the frames "
                   "yourself, or the labels will not be found.")
    return output_folder


#: MOTChallenge class numbers: the object the benchmark is about, and
#: something a tracker should neither find nor be punished for finding.
MOT_TARGET_CLASS = 1
MOT_DISTRACTOR_CLASS = 8


def export_mot(target_folder: str, modality: str, output_folder: str,
               include_not_an_animal: bool = False,
               include_images: bool = True, log_fn=None) -> str:
    """Write the flight as a MOTChallenge sequence.

    ::

        <output>/
        ├── seqinfo.ini        name, frame rate, length, image size
        ├── gt/gt.txt          the tracks
        ├── img1/000001.jpg …  the frames (with *include_images*)
        └── attributes.csv     what the columns cannot hold

    ``gt.txt`` has the nine ground-truth columns
    ``frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility``:
    frames count from 1, ``conf`` is the "evaluate this row" flag and
    ``class`` is 1 for every animal - the class the benchmark tools
    evaluate. A box kept although it is ``not-an-animal`` is written the way
    the format marks such things: flag 0, class 8 (distractor).

    Species, sex, age, the detector's confidence and every custom field are
    in ``attributes.csv``, keyed by the same ``frame`` and ``id``.

    Only boxes that belong to a track are written - MOT is a tracking
    format, and a box without a track has no id to give.

    *include_images* copies **every** frame of the flight into ``img1``
    under the number MOT knows it by. A sequence is its frames, with or
    without an animal on them: a tracker run on ``img1`` numbers its output
    by image, and that number has to be the one in ``gt.txt``.

    Up to 9.0 this wrote ``gt.txt`` beside the images with the detector's
    confidence as the flag, a class counted from 0 (the benchmark tools
    refuse 0) and a tenth column; no ``seqinfo.ini``; and only the frames
    with a box, under their own names.
    """
    vocabulary = common.load_vocabulary(target_folder)
    rows, _animals = common.load_export_rows(
        target_folder, modality, vocabulary, include_not_an_animal,
        include_geo=False)
    tracked = sorted((row for row in rows if row.get("track_id") is not None),
                     key=lambda r: (r["frame"], r["track_id"]))
    frames = common.load_frames(target_folder, modality)

    common.ensure_folder(output_folder)
    gt_folder = common.ensure_folder(os.path.join(output_folder, "gt"))
    gt_path = os.path.join(gt_folder, "gt.txt")
    with open(gt_path, "w", encoding="utf-8") as fh:
        for row in tracked:
            false_positive = row["species_id"] == common.NOT_AN_ANIMAL
            fh.write(f"{row['frame'] + 1},{row['track_id']},"
                     f"{row['x1']:.2f},{row['y1']:.2f},"
                     f"{row['x2'] - row['x1']:.2f},"
                     f"{row['y2'] - row['y1']:.2f},"
                     f"{0 if false_positive else 1},"
                     f"{MOT_DISTRACTOR_CLASS if false_positive else MOT_TARGET_CLASS},"
                     "1\n")

    # Everything MOT's fixed columns cannot carry, keyed back to (frame, id).
    sidecar_path = os.path.join(output_folder, "attributes.csv")
    extra_names = sorted({key for row in tracked for key in row["properties"]})
    with open(sidecar_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["frame", "id", "species", "confidence"] + extra_names)
        for row in tracked:
            cells = [_cell(row["properties"].get(name))
                     for name in extra_names]
            writer.writerow(
                [row["frame"] + 1, row["track_id"],
                 vocabulary["species"].get(row["species_id"], ""),
                 f"{row['confidence']:.4f}", *cells])

    length = max(len(frames),
                 max((row["frame"] for row in tracked), default=-1) + 1)
    extension = _frame_extension(frames)
    size = common.frame_size(target_folder, modality)
    _write_seqinfo(output_folder, length, _frame_rate(frames), size,
                   extension)

    copied = None
    if include_images and tracked:
        copied = _copy_sequence(target_folder, modality, frames,
                                os.path.join(output_folder, "img1"))

    if log_fn:
        skipped = len(rows) - len(tracked)
        message = f"MOT: {len(tracked)} row(s) → {gt_path}"
        if skipped:
            message += f" ({skipped} untracked detection(s) omitted)"
        if not tracked and rows:
            message += common.no_tracks_hint(target_folder, modality)
        log_fn(message)
        if size is None:
            log_fn("MOT: the frame size is not known (no extracted frame "
                   "could be read), so seqinfo.ini has no imWidth/imHeight.")
        for line in common.describe_copy(copied) if copied else []:
            log_fn(f"MOT: {line}")
    return output_folder


def _frame_extension(frames: List[dict]) -> str:
    """The extension most of the flight's frames have (``.jpg`` when there
    is nothing to go by) - MOT names one for the whole sequence."""
    counts: Dict[str, int] = {}
    for frame in frames:
        extension = os.path.splitext(frame.get("imagefile") or "")[1]
        if extension:
            counts[extension] = counts.get(extension, 0) + 1
    return max(counts, key=lambda e: (counts[e], e)) if counts else ".jpg"


def _frame_rate(frames: List[dict], default: int = 30) -> int:
    """Frames per second, from the capture times: the median step between
    consecutive frames. *default* when the times are missing or useless."""
    moments = [common.capture_time(frame.get("epoch")) for frame in frames]
    steps = sorted(
        (later - earlier).total_seconds()
        for earlier, later in zip(moments, moments[1:])
        if earlier is not None and later is not None)
    steps = [step for step in steps if step > 0]
    if not steps:
        return default
    return max(1, int(round(1.0 / steps[len(steps) // 2])))


def _write_seqinfo(output_folder: str, length: int, frame_rate: int,
                   size: Optional[Tuple[int, int]], extension: str) -> str:
    name = os.path.basename(os.path.normpath(output_folder)) or "sequence"
    lines = ["[Sequence]", f"name={name}", "imDir=img1",
             f"frameRate={frame_rate}", f"seqLength={length}"]
    if size is not None:
        lines += [f"imWidth={size[0]}", f"imHeight={size[1]}"]
    lines.append(f"imExt={extension}")
    path = os.path.join(output_folder, "seqinfo.ini")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _copy_sequence(target_folder: str, modality: str, frames: List[dict],
                   destination: str) -> dict:
    """Copy every frame of the flight into *destination* as
    ``000001.<ext>``, ``000002.<ext>``, ... - frame *n* of the flight is
    image *n + 1* of the sequence. Returns what :func:`common.copy_frames`
    returns."""
    import shutil

    source_folder = os.path.join(target_folder, f"frames_{modality}")
    common.ensure_folder(destination)
    copied, missing = 0, []
    for frame in frames:
        name = frame.get("imagefile") or ""
        source = os.path.join(source_folder, name)
        if not name or not os.path.isfile(source):
            missing.append(name or f"frame {frame['frame']}")
            continue
        extension = os.path.splitext(name)[1]
        shutil.copy2(source, os.path.join(
            destination, f"{frame['frame'] + 1:06d}{extension}"))
        copied += 1
    return {"copied": copied, "missing": missing, "source": source_folder,
            "destination": destination}

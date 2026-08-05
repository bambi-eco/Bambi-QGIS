# -*- coding: utf-8 -*-
"""COCO, YOLO and MOT exporters (EXCHANGE_FORMAT_PLAN.md §8.1).

All three are one-directional: they read the store and write throwaway files.
None of them is an input format, so nothing has to survive a round trip.

The interesting difference between them is how much of the 6.0 model they can
carry. COCO tolerates extra keys on an annotation, so custom fields map onto it
directly and it is the reference detection export. YOLO has room for a class id
and nothing else. MOT has a fixed column layout, so anything extra goes to a
sidecar keyed by ``(frame, track_id)``.
"""

import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

from . import common


def export_coco(target_folder: str, modality: str, output_path: str,
                include_not_an_animal: bool = False,
                image_size: Optional[Tuple[int, int]] = None,
                log_fn=None) -> str:
    """Write a COCO detection JSON.

    The reference export: COCO permits extra keys on an annotation, so every
    custom field travels with its detection instead of being dropped the way
    ``labels.csv`` dropped them (§1.2b).
    """
    vocabulary = common.load_vocabulary(target_folder)
    rows = common.load_detections(
        target_folder, modality, include_not_an_animal, include_geo=False)
    width, height = common.require_frame_size(
        target_folder, modality, image_size)
    frames = {f["frame"]: f for f in common.load_frames(target_folder, modality)}

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
        attributes = common.resolve_attributes(
            row["attributes"], vocabulary["enum_labels"])
        if attributes:
            annotation["attributes"] = attributes
        annotations.append(annotation)

    document = {
        "info": {"description": "BAMBI detections",
                 "modality": modality},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index, "name": name}
                       for index, name in enumerate(names)],
    }

    common.ensure_folder(os.path.dirname(os.path.abspath(output_path)))
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2)

    if log_fn:
        log_fn(f"COCO: {len(annotations)} annotation(s) over "
               f"{len(images)} image(s) → {output_path}")
    return output_path


def export_yolo(target_folder: str, modality: str, output_folder: str,
                include_not_an_animal: bool = False,
                image_size: Optional[Tuple[int, int]] = None,
                copy_images: bool = True, log_fn=None) -> str:
    """Write a YOLO dataset: ``images/``, ``labels/`` and ``data.yaml``.

    One ``.txt`` per frame that has detections, named after the frame image, in
    normalised ``cls cx cy w h``. Custom fields have nowhere to go in this
    format and are deliberately not smuggled into it — use COCO for those.

    The frames themselves are copied in, because a YOLO dataset is a folder
    layout rather than a manifest: there is no field naming the images, so the
    labels are only found through ``images``/``labels`` siblings. Only frames
    carrying a detection are copied. *copy_images* turns that off for callers
    that already have the images arranged.
    """
    vocabulary = common.load_vocabulary(target_folder)
    rows = common.load_detections(
        target_folder, modality, include_not_an_animal, include_geo=False)
    width, height = common.require_frame_size(
        target_folder, modality, image_size)
    frames = {f["frame"]: f for f in common.load_frames(target_folder, modality)}

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

    source_folder = os.path.join(target_folder, f"frames_{modality}")
    copied, missing = 0, []

    for frame, frame_rows in sorted(by_frame.items()):
        imagefile = frames.get(frame, {}).get(
            "imagefile", f"frame_{frame:06d}.jpg")
        stem = os.path.splitext(imagefile)[0]

        if copy_images:
            source = os.path.join(source_folder, imagefile)
            if os.path.isfile(source):
                shutil.copy2(source, os.path.join(images_folder, imagefile))
                copied += 1
            else:
                missing.append(imagefile)

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

    if log_fn:
        log_fn(f"YOLO: {len(rows)} box(es) over {len(by_frame)} frame(s) "
               f"→ {output_folder}")
        if copy_images:
            log_fn(f"YOLO: copied {copied} image(s) into {images_folder}")
        else:
            log_fn("YOLO: images not copied — point 'train' at the frames "
                   "yourself, or the labels will not be found.")
        if missing:
            shown = ", ".join(missing[:3])
            more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
            log_fn(f"YOLO: warning — {len(missing)} labelled frame(s) had no "
                   f"image in {source_folder}: {shown}{more}. Those labels "
                   "will be ignored by YOLO.")
    return output_folder


def export_mot(target_folder: str, modality: str, output_folder: str,
               include_not_an_animal: bool = False, log_fn=None) -> str:
    """Write MOT ``gt.txt`` plus a sidecar for anything the format cannot hold.

    Columns are ``frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z`` with
    1-based frame numbers, as MOT expects. Only detections belonging to a track
    are written — MOT is a tracking format, and a detection with no track has no
    id to give.
    """
    vocabulary = common.load_vocabulary(target_folder)
    rows = common.load_detections(
        target_folder, modality, include_not_an_animal, include_geo=False)
    tracked = [row for row in rows if row.get("track_id") is not None]

    mapping, names = common.class_map(
        (row["species_id"] for row in tracked), vocabulary["species"])

    common.ensure_folder(output_folder)
    gt_path = os.path.join(output_folder, "gt.txt")
    with open(gt_path, "w", encoding="utf-8") as fh:
        for row in sorted(tracked, key=lambda r: (r["frame"], r["track_id"])):
            fh.write(f"{row['frame'] + 1},{row['track_id']},"
                     f"{row['x1']:.2f},{row['y1']:.2f},"
                     f"{row['x2'] - row['x1']:.2f},"
                     f"{row['y2'] - row['y1']:.2f},"
                     f"{row['confidence']:.4f},"
                     f"{mapping[row['species_id']]},1,-1\n")

    # Everything MOT's fixed columns cannot carry, keyed back to (frame, id).
    sidecar_path = os.path.join(output_folder, "attributes.csv")
    extra_names = sorted({
        key
        for row in tracked
        for key in common.resolve_attributes(
            row["attributes"], vocabulary["enum_labels"])
    })
    with open(sidecar_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("frame,id,species," + ",".join(extra_names) + "\n")
        for row in sorted(tracked, key=lambda r: (r["frame"], r["track_id"])):
            attributes = common.resolve_attributes(
                row["attributes"], vocabulary["enum_labels"])
            values = ",".join(
                str(attributes.get(name, "")) for name in extra_names)
            species = vocabulary["species"].get(row["species_id"], "")
            fh.write(
                f"{row['frame'] + 1},{row['track_id']},{species},{values}\n")

    classes_path = os.path.join(output_folder, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as fh:
        for index, name in enumerate(names):
            fh.write(f"{index} {name}\n")

    if log_fn:
        skipped = len(rows) - len(tracked)
        message = f"MOT: {len(tracked)} row(s) → {gt_path}"
        if skipped:
            message += f" ({skipped} untracked detection(s) omitted)"
        if not tracked and rows:
            message += common.no_tracks_hint(target_folder, modality)
        log_fn(message)
    return output_folder

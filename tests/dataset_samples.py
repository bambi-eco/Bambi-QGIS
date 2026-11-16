# -*- coding: utf-8 -*-
"""A small flight of the BAMBI dataset, made up: what Zenodo holds of it -
the recordings, and the labelled flight with its poses, labels and
correction - shaped as the published files are, and a server that hands
both out. For the tests of core/bambi_dataset.py and of what is built on
it."""
import datetime
import io
import json
import os
import zipfile

from bambi_wildlife_detection.core import bambi_dataset as bd

from .fakes import FakeResponse, RangeSession
from .test_flight_archive import ARCHIVE

FLIGHT = 7
#: When the first frame was taken.
START = datetime.datetime(
    2024, 11, 19, 10, 19, 36, 314500,
    tzinfo=datetime.timezone(datetime.timedelta(hours=1)))
FRAME_MS = 36
#: The published picture's field of view, and that of the plugin's own
#: frames of the same flight: the same view, cut closer.
PUBLISHED_FOVY = 48.8879
OWN_FOVY = 43.7173
EPSG = 32633

GT = (
    # frame, track, left, top, width, height, confidence, class,
    # visibility, species, sex, age, propagated
    "0,4668,570,223,49,76,1.00,23,0.50,Capreolus capreolus (Roe deer),2,2,0\n"
    "4,4668,566,231,49,76,1.00,23,1.00,Capreolus capreolus (Roe deer),2,2,0\n"
    "10,4668,560,243,49,76,1.00,23,1.00,Capreolus capreolus (Roe deer),2,2,0\n"
    "2,4669,100,900,40,40,1.00,7,1.00,Sus scrofa (Wild boar),1,1,0\n"
    "8,4669,130,930,40,40,1.00,7,1.00,Sus scrofa (Wild boar),1,1,0\n")

CORRECTION = {
    "rotation": {"x": 0.0, "y": 0.0, "z": 0.033},
    "translation": {"x": 0.0, "y": 0.0, "z": 46.65},
    # as the dataset writes one that holds for the whole flight
    "fine_corrections": [{
        "start_frame": 0, "end_frame": 11,
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.033},
        "translation": {"x": 0.0, "y": 0.0, "z": 46.65}}],
}

METADATA = {
    "flight_key": str(FLIGHT), "split": "val", "frame_count": 5,
    "flight_info": {"start_time": "2024-11-19T10:02:32",
                    "drone_name": "Matric 30 Thermal"},
    "species_present": {
        "Q122069": {"species_name": "Capreolus capreolus (Roe deer)",
                    "num_labels": 1, "num_label_states": 3},
        "Q58697": {"species_name": "Sus scrofa (Wild boar)",
                   "num_labels": 1, "num_label_states": 2}},
}


def moment(frame: int) -> str:
    """When a frame of the published video was taken."""
    return (START + datetime.timedelta(
        milliseconds=FRAME_MS * frame)).isoformat()


def published_poses(frames: int = 12) -> dict:
    """The poses file of the labelled flight: a camera going south-east
    over Styria, a frame every 36 ms."""
    images = [{
        "imagefile": f"{frame}-{frame}-{frame}.jpg",
        "location": [1200.0 + frame, -670.0 - 2.0 * frame, 373.8],
        "rotation": [0.0, 0, 185.7], "fovy": [PUBLISHED_FOVY],
        "lat": 47.0050 - frame * 2e-5, "lng": 15.9770 + frame * 1e-5,
        "timestamp": moment(frame),
    } for frame in range(frames)]
    return {"images": images,
            "origin": {"latitude": 47.0113, "longitude": 15.9613,
                       "altitude": 0},
            "drone": "DJI_M30T", "camera": "Thermal", "samplingRate": 0,
            "mask": "mask_T.png"}


def own_poses(published: dict, before: int = 1, after: int = 2,
              dropped=()) -> dict:
    """What an extraction of the plugin's makes of the same flight: the
    published frames - but for those in *dropped* - with *before* frames
    ahead of them and *after* behind, numbered its own way."""
    frames = list(range(-before, len(published["images"]) + after))
    frames = [frame for frame in frames if frame not in dropped]
    images = []
    for number, frame in enumerate(frames):
        images.append({
            "imagefile": f"{number}-{frame + before}-{frame + before}.jpg",
            "location": [700.0 + frame, 500.0 - 2.0 * frame, 90.0],
            "rotation": [0.0, 0, 185.7], "fovy": [OWN_FOVY],
            "lat": 47.0050 - frame * 2e-5, "lng": 15.9770 + frame * 1e-5,
            "timestamp": moment(frame), "grid_convergence": 0.7148})
    return {"images": images,
            "origin": {"latitude": 47.0006, "longitude": 15.9673,
                       "altitude": 283.9},
            "headings": {"north": "map", "epsg": EPSG,
                         "grid_convergence": 0.7148}}


def labelled_archive(flight: int = FLIGHT, gt: str = GT,
                     correction=CORRECTION, poses=None,
                     video=300_000, leave_out=()) -> bytes:
    """The labelled flight as Zenodo holds it: the small files, and a
    video that is most of the archive."""
    poses = published_poses() if poses is None else poses
    members = {
        f"{flight}_matched_processed.mp4": (
            video if isinstance(video, bytes) else os.urandom(video)),
        f"{flight}_mask_t.png": b"thermal mask",
        f"{flight}_mask_w.png": b"rgb mask",
        f"{flight}_matched_poses.json": json.dumps(poses).encode(),
        f"{flight}_gt.txt": gt.encode(),
        f"{flight}_correction.json": json.dumps(correction).encode(),
        f"{flight}_metadata.json": json.dumps(METADATA).encode(),
        f"{flight}_track_mapping.json": b'{"4668": 23, "4669": 7}',
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            if any(part in name for part in leave_out):
                continue
            kind = (zipfile.ZIP_STORED if name.endswith(".mp4")
                    else zipfile.ZIP_DEFLATED)
            bundle.writestr(zipfile.ZipInfo(name), data, kind)
    return buffer.getvalue()


def entry(flight: int = FLIGHT, raw: bytes = ARCHIVE, labelled=None,
          **more) -> bd.Flight:
    """The flight as the dataset's repository lists it - with what Zenodo
    says of the stand-in archives already asked (their link, size and
    checksum), so that no test needs Zenodo's answer."""
    import hashlib

    def archive(record, data):
        if data is None:
            return None
        name = f"flight_{flight}.zip"
        return bd.Archive(
            record, name, len(data),
            hashlib.md5(data).hexdigest(),  # nosec B324
            f"https://zenodo.org/api/records/{record}/files/{name}/content")

    values = dict(id=flight, raw=archive(111, raw),
                  labelled=archive(222, labelled), split="val")
    values.update(more)
    return bd.Flight(**values)


class Zenodo:
    """A session that serves a flight's recordings as a stream and its
    labelled archive by ranges - as the one session of a download does."""

    def __init__(self, flight: bd.Flight, raw: bytes = ARCHIVE,
                 labelled: bytes = b""):
        self.flight = flight
        self.raw = raw
        self.labelled = labelled
        self.ranges = RangeSession(
            {flight.labelled.url: labelled} if flight.labelled else {})
        self.streamed = []
        self.headers = {}
        #: An exception for the recordings to fail with.
        self.raw_fails = None

    def get(self, url, headers=None, **more):
        if headers and "Range" in headers and not more.get("stream"):
            return self.ranges.get(url, headers=headers, **more)
        self.streamed.append(url)
        if self.raw_fails is not None:
            raise self.raw_fails
        whole = self.raw
        if self.flight.labelled and url == self.flight.labelled.url:
            whole = self.labelled
        size = max(1, len(whole) // 3)
        chunks = [whole[i:i + size] for i in range(0, len(whole), size)]
        return FakeResponse(chunks=chunks, content_length=len(whole),
                            headers={"etag": '"v1"'})


def dem_metadata(folder: str, name: str = "gem15_dem") -> str:
    """An elevation model in *folder*, as the worldwide download leaves
    one: the mesh, and the metadata that says where its origin is."""
    from pyproj import Transformer

    origin = {"latitude": 47.0006, "longitude": 15.9673, "altitude": 283.9}
    east, north = Transformer.from_crs(4326, EPSG, always_xy=True).transform(
        origin["longitude"], origin["latitude"])
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{name}.glb"), "wb") as handle:
        handle.write(b"glTF")
    path = os.path.join(folder, f"{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"crs": f"EPSG:{EPSG}",
                   "origin": [east, north, origin["altitude"]],
                   "origin_wgs84": origin}, handle)
    return path


def put_published(target: str, flight: int = FLIGHT, gt: str = GT,
                  correction=CORRECTION, poses=None,
                  kinds=(bd.POSES, bd.LABELS, bd.CORRECTION)) -> dict:
    """The published files of a flight in its target folder, as
    ``fetch_published`` leaves them; returns the poses."""
    poses = published_poses() if poses is None else poses
    texts = {bd.POSES: json.dumps(poses), bd.LABELS: gt,
             bd.CORRECTION: json.dumps(correction)}
    os.makedirs(bd.published_folder(target), exist_ok=True)
    for kind in kinds:
        with open(bd.published_path(target, flight, kind), "w",
                  encoding="utf-8") as handle:
            handle.write(texts[kind])
    return poses


def put_extraction(target: str, poses: dict, size=(640, 512)) -> None:
    """An extraction of the plugin's in a target folder: its poses, and a
    first frame to measure."""
    from PIL import Image

    os.makedirs(os.path.join(target, "frames_t"), exist_ok=True)
    with open(os.path.join(target, "poses_t.json"), "w",
              encoding="utf-8") as handle:
        json.dump(poses, handle)
    Image.new("L", size).save(os.path.join(
        target, "frames_t", poses["images"][0]["imagefile"]))


def processed_video(folder: str, frames: int = 12, size: int = 32) -> bytes:
    """A processed video as the dataset has them, small: thermal on the
    left (dark, brighter with every frame), RGB on the right (light)."""
    import cv2
    import numpy as np

    path = os.path.join(folder, "made.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                             (2 * size, size))
    for frame in range(frames):
        picture = np.zeros((size, 2 * size, 3), np.uint8)
        picture[:, :size] = 10 + 8 * frame
        picture[:, size:] = 220
        writer.write(picture)
    writer.release()
    with open(path, "rb") as handle:
        data = handle.read()
    os.remove(path)
    return data

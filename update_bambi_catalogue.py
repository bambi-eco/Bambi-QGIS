#!/usr/bin/env python3
"""Rebuild the list of BAMBI dataset flights that ships with the plugin.

The plugin offers the flights of the BAMBI dataset by their number
("Download Example Flights..." on the Input tab). Which flights there are,
where each one lies on Zenodo, how large it is and what was labelled on it
is read from ``bambi_wildlife_detection/core/bambi_flights.json`` - so that
the dialog opens at once, without asking Zenodo for 55 records first. This
script writes that file. Run it when the dataset gets new parts or new
versions:

    python3 update_bambi_catalogue.py --cache ~/.cache/bambi_catalogue

What it does:

1. asks Zenodo for the records of the raw dataset ("The BAMBI Raw Dataset")
   and of the labelled dataset ("The BAMBI Dataset: Multimodal Nadir
   UAV-Recordings of Forest Wildlife"), newest versions only, and notes for
   every ``flight_<n>.zip`` its record, its size and its MD5;
2. reads each labelled flight's ``<n>_metadata.json`` out of its archive -
   a few kilobytes by range requests, not the archive - for the day it was
   flown, the drone, and how many animals of which species were labelled.
   What was read is kept in the cache directory: a later run reads only
   the flights it does not have yet;
3. looks into the archives of recordings that are too small to hold a
   flight's videos, and notes the flights that were published without
   them (``"videos": false``);
4. writes the catalogue, one flight per line.

Only what the dialog shows is taken from a flight's metadata.

Needs ``requests``. Zenodo is asked slowly (``--pause``): a first run reads
some 390 archives' tables of contents and takes about a quarter of an hour.
"""

import argparse
import datetime
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from bambi_wildlife_detection.core import remote_zip  # noqa: E402

ZENODO = "https://zenodo.org"
OUTPUT = os.path.join(ROOT, "bambi_wildlife_detection", "core",
                      "bambi_flights.json")
#: The two datasets: what their records' titles begin with.
DATASETS = (
    ("raw", "The BAMBI Raw Dataset: Unprocessed Nadir UAV-Recordings of "
            "Forest Wildlife (Part"),
    ("labelled", "The BAMBI Dataset: Multimodal Nadir UAV-Recordings of "
                 "Forest Wildlife (Part"),
)
#: Zenodo's search returns no more than this per page to an anonymous
#: client.
PAGE = 25
FLIGHT_ARCHIVE = re.compile(r"^flight_(\d+)\.zip$")
#: An archive of recordings smaller than this is looked into: the smallest
#: flight with its videos is 122 MB.
SMALL_ARCHIVE = 50_000_000
#: Names the dataset's metadata misspells.
DRONES = {"Matric 30 Thermal": "Matrice 30 Thermal"}


def say(message: str) -> None:
    print(message, flush=True)


def search(session, title: str, pause: float) -> list:
    """Every record whose title begins with *title*."""
    # the words of the title, as a phrase: the search takes no prefixes
    phrase = title.split(" (Part")[0].replace('"', " ")
    found, page = {}, 1
    while True:
        response = session.get(f"{ZENODO}/api/records", params={
            "q": f'title:"{phrase}"', "size": PAGE, "page": page,
            "sort": "bestmatch", "all_versions": "false"}, timeout=60)
        response.raise_for_status()
        hits = response.json().get("hits", {}).get("hits", [])
        for hit in hits:
            if str(hit.get("metadata", {}).get("title", "")).startswith(title):
                found[int(hit["id"])] = hit
        if len(hits) < PAGE:
            break
        page += 1
        time.sleep(pause)
    return [found[record] for record in sorted(found)]


def archives(records: list) -> dict:
    """``{flight number: (record, size, md5)}`` of the flights in
    *records*. A flight that is in two records stays with the newer."""
    flights = {}
    for record in records:
        for entry in record.get("files") or []:
            named = FLIGHT_ARCHIVE.match(str(entry.get("key", "")))
            if not named:
                continue
            checksum = str(entry.get("checksum", ""))
            if not checksum.startswith("md5:"):
                raise SystemExit(
                    f"record {record['id']}: {entry['key']} has no MD5 "
                    f"({checksum!r})")
            flights[int(named.group(1))] = (
                int(record["id"]), int(entry["size"]), checksum[4:])
    return flights


def authors(record: dict) -> str:
    """``Praschl, C., Schedl, D. C., et al.`` of a record's creators."""
    short = []
    for creator in (record.get("metadata", {}).get("creators") or [])[:2]:
        family, _comma, given = str(creator.get("name", "")).partition(",")
        initials = " ".join(f"{part[0]}." for part in given.split() if part)
        short.append(f"{family.strip()}, {initials}".strip(", "))
    if not short:
        return ""
    more = len(record["metadata"]["creators"]) > len(short)
    return ", ".join(short) + (", et al." if more else "")


def flight_metadata(session, flight: int, archive: tuple, cache: str,
                    pause: float):
    """A labelled flight's metadata document, from the cache or from its
    archive on Zenodo; ``None`` when the archive has none."""
    path = os.path.join(cache, f"{flight}.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    record, size, _md5 = archive
    name = f"{flight}_metadata.json"
    url = f"{ZENODO}/api/records/{record}/files/flight_{flight}.zip/content"
    found = remote_zip.read_members(session, url, [name], size=size)
    time.sleep(pause)
    if name not in found:
        return None
    document = json.loads(found[name])
    os.makedirs(cache, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(found[name])
    return document


def has_videos(session, flight: int, archive: tuple, cache: str,
               pause: float) -> bool:
    """Whether a flight's archive of recordings holds a video - asked of
    the archive's table of contents, and only of archives that are too
    small to be sure."""
    record, size, _md5 = archive
    if size >= SMALL_ARCHIVE:
        return True
    path = os.path.join(cache, f"{flight}.members.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            names = json.load(handle)
    else:
        url = (f"{ZENODO}/api/records/{record}/files/flight_{flight}.zip/"
               "content")
        names = [name for name, _size in
                 remote_zip.list_members(session, url, size=size)]
        time.sleep(pause)
        os.makedirs(cache, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(names, handle)
    return any(str(name).lower().endswith(".mp4") for name in names)


def described(document) -> dict:
    """What the catalogue says of a flight, out of its metadata."""
    if not isinstance(document, dict):
        return {}
    info = document.get("flight_info") or {}
    species = []
    for entry in (document.get("species_present") or {}).values():
        name = str(entry.get("species_name") or "").strip()
        if name:
            species.append((name, int(entry.get("num_labels") or 0),
                            int(entry.get("num_label_states") or 0)))
    species.sort(key=lambda item: (-item[1], item[0]))
    drone = str(info.get("drone_name") or "").strip()
    return {
        "date": str(info.get("start_time") or "")[:10],
        "species": [[name, animals] for name, animals, _boxes in species],
        "animals": sum(animals for _name, animals, _boxes in species),
        "boxes": sum(boxes for _name, _animals, boxes in species),
        "drone": DRONES.get(drone, drone),
        "split": str(document.get("split") or ""),
    }


def build(session, cache: str, pause: float) -> dict:
    found = {}
    for kind, title in DATASETS:
        found[kind] = search(session, title, pause)
        say(f"{kind}: {len(found[kind])} record(s) on Zenodo")
        if not found[kind]:
            raise SystemExit(
                f"Zenodo knows no record called '{title}...': nothing was "
                "written.")
    raw, labelled = archives(found["raw"]), archives(found["labelled"])
    say(f"{len(raw)} flight(s) with recordings, {len(labelled)} labelled")

    records = {}
    for kind, _title in DATASETS:
        for record in found[kind]:
            metadata = record["metadata"]
            records[str(record["id"])] = {
                "kind": kind, "title": metadata["title"],
                "year": int(str(metadata.get("publication_date"))[:4])}

    flights = []
    for done, flight in enumerate(sorted(set(raw) | set(labelled)), 1):
        entry = {"id": flight}
        if flight in raw:
            entry["raw"] = list(raw[flight])
            if not has_videos(session, flight, raw[flight], cache, pause):
                entry["videos"] = False
        if flight in labelled:
            entry["labelled"] = list(labelled[flight])
            entry.update(described(flight_metadata(
                session, flight, labelled[flight], cache, pause)))
        flights.append(entry)
        if done % 50 == 0:
            say(f"  {done} flight(s) described")
    return {
        "about": ("The flights of the BAMBI dataset on Zenodo. Written by "
                  "update_bambi_catalogue.py - do not edit by hand."),
        "generated": datetime.date.today().isoformat(),
        "licence": "CC BY 4.0",
        "authors": authors(found["raw"][0]),
        "records": records,
        "flights": flights,
    }


def write(catalogue: dict, path: str) -> None:
    """The catalogue as JSON, a record and a flight per line - so that a
    new version of it can be read as a difference."""
    lines = ["{"]
    for key in ("about", "generated", "licence", "authors"):
        lines.append(f" {json.dumps(key)}: "
                     f"{json.dumps(catalogue[key], ensure_ascii=False)},")
    lines.append(' "records": {')
    records = sorted(catalogue["records"].items(), key=lambda item: int(item[0]))
    for index, (record, value) in enumerate(records):
        comma = "," if index < len(records) - 1 else ""
        lines.append(f"  {json.dumps(record)}: "
                     f"{json.dumps(value, ensure_ascii=False)}{comma}")
    lines.append(" },")
    lines.append(' "flights": [')
    for index, flight in enumerate(catalogue["flights"]):
        comma = "," if index < len(catalogue["flights"]) - 1 else ""
        lines.append(f"  {json.dumps(flight, ensure_ascii=False)}{comma}")
    lines.append(" ]")
    lines.append("}")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(temporary, "r", encoding="utf-8") as handle:
        json.load(handle)                   # it is what it should be
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the list of BAMBI dataset flights that ships "
                    "with the plugin.")
    parser.add_argument(
        "--cache", default=os.path.join(ROOT, ".bambi_catalogue_cache"),
        help="where the flights' metadata are kept between runs "
             "(default: %(default)s)")
    parser.add_argument("--output", default=OUTPUT,
                        help="the catalogue to write (default: %(default)s)")
    parser.add_argument("--pause", type=float, default=1.5,
                        help="seconds between two questions to Zenodo "
                             "(default: %(default)s)")
    options = parser.parse_args()

    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "bambi-qgis-plugin"
    catalogue = build(session, options.cache, options.pause)
    write(catalogue, options.output)
    flights = catalogue["flights"]
    size = sum(flight["raw"][1] for flight in flights if "raw" in flight)
    without = [flight["id"] for flight in flights
               if flight.get("videos") is False]
    say(f"{options.output}: {len(flights)} flight(s), "
        f"{sum(1 for flight in flights if 'labelled' in flight)} of them "
        f"labelled, {size / 1e12:.2f} TB of recordings; published without "
        f"their videos: {without or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

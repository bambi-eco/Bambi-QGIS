# -*- coding: utf-8 -*-
"""The script that writes the list of BAMBI flights shipped with the
plugin (update_bambi_catalogue.py, at the top of the repository) - with
Zenodo's answers made up."""
import importlib.util
import json
import os

import pytest

from bambi_wildlife_detection.core import bambi_dataset

from . import dataset_samples as samples
from .fakes import RangeSession

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "update_bambi_catalogue",
        os.path.join(ROOT, "update_bambi_catalogue.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(number, title, files, creators=("Praschl, Christoph",
                                           "Schedl, David C.",
                                           "Maschek, Anna")):
    return {"id": number, "metadata": {
        "title": title, "publication_date": "2026-03-16",
        "creators": [{"name": name} for name in creators]},
        "files": [{"key": key, "size": size, "checksum": checksum}
                  for key, size, checksum in files]}


RAW = "The BAMBI Raw Dataset: Unprocessed Nadir UAV-Recordings of Forest " \
      "Wildlife (Part 27)"
LABELLED = "The BAMBI Dataset: Multimodal Nadir UAV-Recordings of Forest " \
           "Wildlife (Part 7)"


class Search:
    """Zenodo's search, and the labelled archives by ranges."""

    class Response:
        def __init__(self, hits):
            self._hits = hits

        def raise_for_status(self):
            pass

        def json(self):
            return {"hits": {"hits": self._hits}}

    def __init__(self, records, archives):
        self.records = records
        self.ranges = RangeSession(archives)
        self.asked = []
        self.headers = {}

    def get(self, url, params=None, headers=None, **more):
        if params is None:
            return self.ranges.get(url, headers=headers, **more)
        self.asked.append(dict(params))
        phrase = params["q"][len('title:"'):-1]
        hits = [entry for entry in self.records
                if phrase in entry["metadata"]["title"]]
        size, page = params["size"], params["page"]
        return self.Response(hits[(page - 1) * size:page * size])


def test_a_flights_archive_is_noted_with_its_checksum(script):
    records = [record(19047570, RAW, [
        ("flight_306.zip", 121842908, "md5:" + "ad" * 16),
        ("flight_7.zip", 500, "md5:" + "07" * 16),
        ("README.md", 100, "md5:" + "00" * 16)])]

    assert script.archives(records) == {
        306: (19047570, 121842908, "ad" * 16), 7: (19047570, 500, "07" * 16)}


def test_an_archive_without_a_checksum_stops_everything(script):
    records = [record(1, RAW, [("flight_306.zip", 10, "sha1:abc")])]

    with pytest.raises(SystemExit, match="flight_306.zip has no MD5"):
        script.archives(records)


def test_who_made_the_dataset(script):
    assert script.authors(record(1, RAW, [])) == \
        "Praschl, C., Schedl, D. C., et al."
    assert script.authors(record(1, RAW, [], creators=("Doe, Jane",))) == \
        "Doe, J."
    assert script.authors(record(1, RAW, [], creators=())) == ""


def test_only_what_the_list_shows_is_taken_from_a_flights_metadata(script):
    document = dict(samples.METADATA)
    document["flight_info"] = dict(
        document["flight_info"], sharepoint_link="https://example.org/x",
        weather=1, drone_instance_id=5)

    described = script.described(document)

    assert described == {
        "date": "2024-11-19",
        "species": [["Capreolus capreolus (Roe deer)", 1],
                    ["Sus scrofa (Wild boar)", 1]],
        "animals": 2, "boxes": 5,
        "drone": "Matrice 30 Thermal",        # as it is spelled
        "split": "val"}
    assert "example.org" not in json.dumps(described)


@pytest.mark.parametrize("document", [None, [], "text"])
def test_metadata_that_is_no_document(script, document):
    assert script.described(document) == {}


def test_a_flight_whose_metadata_says_nothing(script):
    assert script.described({}) == {
        "date": "", "species": [], "animals": 0, "boxes": 0, "drone": "",
        "split": ""}


def test_the_species_with_most_animals_comes_first(script):
    document = {"species_present": {
        "a": {"species_name": "Unknown", "num_labels": 2,
              "num_label_states": 8},
        "b": {"species_name": "Sus scrofa (Wild boar)", "num_labels": 29,
              "num_label_states": 367}}}

    assert script.described(document)["species"] == [
        ["Sus scrofa (Wild boar)", 29], ["Unknown", 2]]


def test_the_catalogue_from_zenodo_to_the_file(script, tmp_path):
    data = samples.labelled_archive()
    url = "https://zenodo.org/api/records/21881459/files/flight_7.zip/content"
    raw = {7: _zip(["air_data.csv", "DJI_0001_T.MP4", "DJI_0001_T.SRT"]),
           12: _zip(["air_data.csv", "DJI_0001_T_point0.SRT"])}
    session = Search(
        [record(19047570, RAW, [
            ("flight_7.zip", len(raw[7]), "md5:" + "07" * 16),
            ("flight_12.zip", len(raw[12]), "md5:" + "12" * 16)]),
         record(21881459, LABELLED,
                [("flight_7.zip", len(data), "md5:" + "aa" * 16)]),
         record(5, "The BAMBI Dataset: Multimodal Nadir UAV-Recordings of "
                   "Forest Wildlife (Sample)", [])],
        {url: data, **{
            "https://zenodo.org/api/records/19047570/files/"
            f"flight_{number}.zip/content": archive
            for number, archive in raw.items()}})
    cache, output = str(tmp_path / "cache"), str(tmp_path / "flights.json")

    catalogue = script.build(session, cache, pause=0.0)
    script.write(catalogue, output)

    # what the plugin reads
    flights = bambi_dataset.flights(output)
    assert [flight.id for flight in flights] == [7, 12]
    seven, twelve = flights
    assert seven.raw == bambi_dataset.Archive(
        19047570, "flight_7.zip", len(raw[7]), "07" * 16)
    # the one that was published without its videos is told
    assert (seven.videos, twelve.videos) == (True, False)
    assert seven.labelled.record == 21881459
    assert (seven.date, seven.animals, seven.boxes, seven.drone) == (
        "2024-11-19", 2, 5, "Matrice 30 Thermal")
    assert twelve.labelled is None and twelve.species == ()
    assert bambi_dataset.citation(seven.raw, output).startswith(
        "Praschl, C., Schedl, D. C., et al. (2026): The BAMBI Raw Dataset")
    # the sample record holds no flight archive: it is not a part
    assert set(bambi_dataset.catalogue(output)["records"]) == {19047570,
                                                               21881459}
    # a flight's metadata is kept, and not asked for twice
    assert sorted(os.listdir(cache)) == ["12.members.json", "7.json",
                                         "7.members.json"]
    asked = len(session.ranges.asked)
    script.build(session, cache, pause=0.0)
    assert len(session.ranges.asked) == asked


def _zip(names):
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name in names:
            bundle.writestr(name, b"x" * 50)
    return buffer.getvalue()


def test_a_small_archive_is_looked_into_for_its_videos(script, tmp_path):
    whole = _zip(["air_data.csv", "DJI_0001_T.MP4", "DJI_0001_T.SRT"])
    logs_only = _zip(["air_data.csv", "DJI_0001_T_point0.SRT"])
    base = "https://zenodo.org/api/records/5/files/flight_{}.zip/content"
    session = RangeSession({base.format(1): whole, base.format(2): logs_only})
    cache = str(tmp_path / "cache")

    assert script.has_videos(session, 1, (5, len(whole), "00"), cache, 0.0)
    assert not script.has_videos(session, 2, (5, len(logs_only), "00"),
                                 cache, 0.0)
    # asked once: the answer is kept
    asked = len(session.asked)
    assert not script.has_videos(session, 2, (5, len(logs_only), "00"),
                                 cache, 0.0)
    assert len(session.asked) == asked


def test_an_archive_large_enough_for_a_flight_is_not_asked(script, tmp_path):
    session = RangeSession({})

    assert script.has_videos(session, 306, (5, 121842908, "00"),
                             str(tmp_path), 0.0)
    assert session.asked == []


def test_a_flight_and_a_record_per_line(script, tmp_path):
    catalogue = {
        "about": "x", "generated": "2026-11-16", "licence": "CC BY 4.0",
        "authors": "Doe, J.",
        "records": {"2": {"kind": "raw", "title": "B", "year": 2026},
                    "1": {"kind": "raw", "title": "A", "year": 2026}},
        "flights": [{"id": 1, "raw": [1, 10, "00"]},
                    {"id": 2, "raw": [2, 20, "11"]}]}
    path = str(tmp_path / "flights.json")

    script.write(catalogue, path)

    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    assert json.loads("\n".join(lines)) == catalogue
    assert '  {"id": 1, "raw": [1, 10, "00"]},' in lines
    assert sum(1 for line in lines if '"kind"' in line) == 2
    assert os.listdir(tmp_path) == ["flights.json"]


def test_zenodo_without_the_dataset_writes_nothing(script, tmp_path):
    session = Search([], {})

    with pytest.raises(SystemExit, match="nothing was written"):
        script.build(session, str(tmp_path / "cache"), pause=0.0)

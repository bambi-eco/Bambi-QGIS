# -*- coding: utf-8 -*-
"""Flights of the BAMBI dataset (core/bambi_dataset.py): which there are,
how their recordings and their published files are fetched, and how the
published poses, correction and labels are fitted to a flight as the plugin
has it - with made-up archives standing in for Zenodo, and no network.

What happens to them inside a flight's folder is in
tests/test_bambi_dataset_flight.py."""
import hashlib
import json
import math
import os

import pytest

from bambi_wildlife_detection.core import bambi_dataset as bd
from bambi_wildlife_detection.core import checked_download, remote_zip

from . import dataset_samples as samples
from .fakes import FakeResponse, FakeSession, RangeSession
from .test_flight_archive import ARCHIVE, MEMBERS, archive_of


def quiet_fetch(**more):
    return dict({"sleep": lambda seconds: None}, **more)


def serving(data=ARCHIVE, pieces=3):
    size = max(1, len(data) // pieces)
    chunks = [data[i:i + size] for i in range(0, len(data), size)]
    return FakeSession([FakeResponse(chunks=chunks, content_length=len(data),
                                     headers={"etag": '"v1"'})])


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

CATALOGUE = {
    "generated": "2026-11-16", "authors": "Praschl, C., Schedl, D. C., et al.",
    "records": {
        "111": {"kind": "raw", "year": 2026,
                "title": "The BAMBI Raw Dataset (Part 1)"},
        "222": {"kind": "labelled", "year": 2026,
                "title": "The BAMBI Dataset (Part 1)"}},
    "flights": [
        {"id": 12, "raw": [111, 500, "ab" * 16]},
        {"id": 7, "raw": [111, 1200, "cd" * 16],
         "labelled": [222, 300, "ef" * 16], "date": "2024-11-19",
         "species": [["Sus scrofa (Wild boar)", 29], ["Unknown", 2]],
         "animals": 31, "boxes": 375, "drone": "Matrice 30 Thermal",
         "split": "train"}],
}


@pytest.fixture
def catalogue(tmp_path):
    path = tmp_path / "flights.json"
    path.write_text(json.dumps(CATALOGUE), encoding="utf-8")
    return str(path)


class TestCatalogue:
    def test_the_flights_by_their_number(self, catalogue):
        flights = bd.flights(catalogue)

        assert [flight.id for flight in flights] == [7, 12]
        seven = flights[0]
        assert seven.name == "BAMBI flight 7"
        assert seven.folder == "flight_7"
        assert (seven.date, seven.drone, seven.split) == (
            "2024-11-19", "Matrice 30 Thermal", "train")
        assert seven.species == (("Sus scrofa (Wild boar)", 29),
                                 ("Unknown", 2))
        assert (seven.animals, seven.boxes) == (31, 375)
        assert seven.species_text == "wild boar, unknown"

    def test_where_a_flight_lies(self, catalogue):
        seven = bd.flight(7, catalogue)

        assert seven.raw == bd.Archive(111, "flight_7.zip", 1200, "cd" * 16)
        assert seven.raw.url == ("https://zenodo.org/api/records/111/files/"
                                 "flight_7.zip/content")
        assert seven.raw.doi == "10.5281/zenodo.111"
        assert seven.labelled.record == 222
        assert seven.labelled.url.startswith(
            "https://zenodo.org/api/records/222/files/flight_7.zip")

    def test_a_checksum_written_as_a_number(self, tmp_path):
        """As the shipped list has them (update_bambi_catalogue.py says
        why) - noughts at the front included."""
        document = dict(CATALOGUE, flights=[
            {"id": 7, "raw": [111, 1200, int("00ab" + "cd" * 14, 16)],
             "labelled": [222, 300, 0xadce6a7336ff27414b02fa6b995c98a4]}])
        path = tmp_path / "flights.json"
        path.write_text(json.dumps(document), encoding="utf-8")

        seven = bd.flight(7, str(path))

        assert seven.raw.md5 == "00ab" + "cd" * 14
        assert seven.labelled.md5 == "adce6a7336ff27414b02fa6b995c98a4"

    def test_a_flight_that_was_not_labelled(self, catalogue):
        twelve = bd.flight(12, catalogue)

        assert twelve.labelled is None
        assert (twelve.date, twelve.species, twelve.animals) == ("", (), 0)

    def test_a_flight_published_without_its_videos(self, tmp_path):
        document = dict(CATALOGUE, flights=[
            {"id": 397, "raw": [111, 233594, "ab" * 16], "videos": False},
            {"id": 7, "raw": [111, 1200, "cd" * 16]}])
        path = tmp_path / "flights.json"
        path.write_text(json.dumps(document), encoding="utf-8")

        assert bd.flight(397, str(path)).videos is False
        assert bd.flight(7, str(path)).videos is True

    def test_a_flight_the_dataset_does_not_have(self, catalogue):
        with pytest.raises(bd.DatasetError, match="has no flight 99"):
            bd.flight(99, catalogue)
        with pytest.raises(bd.DatasetError, match="has no flight seven"):
            bd.flight("seven", catalogue)

    @pytest.mark.parametrize("text", ["", "not json", "{}", '{"flights": 3}',
                                      '{"flights": [{"raw": [1, 2, "x"]}]}'])
    def test_a_catalogue_that_cannot_be_read(self, tmp_path, text):
        path = tmp_path / "broken.json"
        path.write_text(text, encoding="utf-8")

        with pytest.raises(bd.DatasetError, match="cannot be read"):
            bd.flights(str(path))
        with pytest.raises(bd.DatasetError, match="cannot be read"):
            bd.flights(str(tmp_path / "none.json"))

    def test_how_a_flight_is_cited(self, catalogue):
        seven = bd.flight(7, catalogue)

        assert bd.citation(seven.raw, catalogue) == (
            "Praschl, C., Schedl, D. C., et al. (2026): The BAMBI Raw "
            "Dataset (Part 1). Zenodo. https://doi.org/10.5281/zenodo.111 "
            "(CC BY 4.0)")
        assert "The BAMBI Dataset (Part 1)" in bd.citation(seven.labelled,
                                                           catalogue)

    @pytest.mark.parametrize("name, common, scientific", [
        ("Capreolus capreolus (Roe deer)", "roe deer", "Capreolus capreolus"),
        ("Dama dama (Fallow Deer)", "fallow deer", "Dama dama"),
        ("Sus scrofa x Sus domesticus (Hybrid Pig)", "hybrid pig",
         "Sus scrofa x Sus domesticus"),
        ("Unknown", "unknown", ""), ("No-animal", "no-animal", ""),
        ("", "", ""), (None, "", ""),
    ])
    def test_the_two_names_of_a_species(self, name, common, scientific):
        assert bd.common_name(name) == common
        assert bd.scientific_name(name) == scientific


class TestTheShippedCatalogue:
    """The list that ships with the plugin, as update_bambi_catalogue.py
    wrote it."""

    def test_every_flight_of_the_dataset(self):
        flights = bd.flights()

        assert len(flights) == 400
        assert [flight.id for flight in flights] == sorted(
            {flight.id for flight in flights})
        assert all(flight.raw is not None for flight in flights)
        assert sum(1 for flight in flights if flight.labelled) == 386
        for flight in flights:
            for archive in (flight.raw, flight.labelled):
                if archive is None:
                    continue
                assert archive.size > 0
                assert len(archive.md5) == 32 and int(archive.md5, 16) >= 0
                assert archive.filename == f"flight_{flight.id}.zip"

    def test_every_archive_is_in_a_record_that_is_named(self):
        known = bd.catalogue()

        for flight in known["flights"].values():
            assert known["records"][flight.raw.record]["kind"] == "raw"
            if flight.labelled is not None:
                assert known["records"][flight.labelled.record]["kind"] == \
                    "labelled"
        for record in known["records"].values():
            assert record["title"].startswith("The BAMBI")
            assert record["year"] >= 2026
        assert known["authors"].startswith("Praschl, C., Schedl, D. C.")

    def test_the_example_is_the_flight_this_release_was_tested_with(self):
        example = bd.flight(bd.EXAMPLE)

        assert example.id == 306
        assert example.raw.size == 121842908
        assert example.raw.url == ("https://zenodo.org/api/records/19047570/"
                                   "files/flight_306.zip/content")
        assert len(bd.PINNED_SHA256[306]) == 64
        assert example.species == (("Capreolus capreolus (Roe deer)", 2),)
        assert example.date == "2024-11-19"
        assert "Part 27" in bd.citation(example.raw)

    def test_seven_flights_were_published_without_their_videos(self):
        without = [flight.id for flight in bd.flights() if not flight.videos]

        assert without == [397, 398, 399, 400, 401, 402, 403]
        # ... and no flight with videos is as small as they are
        assert min(flight.raw.size for flight in bd.flights()
                   if flight.videos) == bd.flight(bd.EXAMPLE).raw.size
        assert max(bd.flight(number).raw.size for number in without) < 2e6

    def test_it_holds_what_the_list_shows_and_nothing_else(self):
        with open(bd.CATALOGUE_FILE, encoding="utf-8") as handle:
            text = handle.read()
        document = json.loads(text)

        allowed = {"id", "raw", "labelled", "date", "species", "animals",
                   "boxes", "drone", "split", "videos"}
        assert all(set(flight) <= allowed for flight in document["flights"])
        # a flight's metadata on Zenodo says more than a list needs
        assert "sharepoint" not in text.lower() and "http" not in text
        assert text.isascii()

    def test_it_holds_nothing_a_scan_for_secrets_stops_at(self):
        """786 checksums in quotes are 786 "potential secrets" to the scan
        of run_checks.sh; they are written as the numbers they are."""
        import re

        with open(bd.CATALOGUE_FILE, encoding="utf-8") as handle:
            text = handle.read()

        assert not re.search(r'"[0-9a-fA-F]{20,}"', text)
        assert bd.flight(306).raw.md5 == "adce6a7336ff27414b02fa6b995c98a4"

    def test_the_species_are_ones_the_plugin_can_name(self):
        names = {name for flight in bd.flights()
                 for name, _count in flight.species}

        assert "Capreolus capreolus (Roe deer)" in names
        for name in names:
            assert bd.common_name(name)
            assert bd.scientific_name(name) or \
                bd.common_name(name) in bd.NO_SPECIES


# ---------------------------------------------------------------------------
# The recordings
# ---------------------------------------------------------------------------

class TestArchive:
    def _written(self, tmp_path, data):
        path = str(tmp_path / "flight_7.zip")
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def test_the_published_file_has_no_problem(self, tmp_path):
        assert bd.archive_problem(self._written(tmp_path, ARCHIVE),
                                  samples.entry().raw) is None

    def test_a_file_of_another_length(self, tmp_path):
        problem = bd.archive_problem(self._written(tmp_path, ARCHIVE[:-10]),
                                     samples.entry().raw)

        assert problem == (f"it is {len(ARCHIVE) - 10} bytes long, not "
                           f"{len(ARCHIVE)}")

    def test_a_file_with_other_contents(self, tmp_path):
        changed = bytearray(ARCHIVE)
        changed[100] ^= 0xFF

        problem = bd.archive_problem(self._written(tmp_path, bytes(changed)),
                                     samples.entry().raw)

        assert problem.startswith("its contents are not the published ones "
                                  "(MD5 ")

    def test_a_file_that_is_not_there(self, tmp_path):
        assert bd.archive_problem(str(tmp_path / "none.zip"),
                                  samples.entry().raw).startswith(
            "it cannot be read")

    def test_an_archive_this_release_was_tested_with_is_checked_twice(
            self, tmp_path):
        path = self._written(tmp_path, ARCHIVE)
        raw = samples.entry().raw

        assert bd.archive_problem(
            path, raw, sha256=hashlib.sha256(ARCHIVE).hexdigest()) is None
        assert bd.archive_problem(path, raw, sha256="00" * 32).startswith(
            "its contents are not the ones this version of the plugin was "
            "tested with (SHA-256 ")

    def test_checking_can_be_stopped(self, tmp_path):
        with pytest.raises(checked_download.Cancelled):
            bd.archive_problem(self._written(tmp_path, ARCHIVE),
                               samples.entry().raw, cancelled=lambda: True)


class TestRoom:
    def test_the_archive_and_what_is_unpacked_from_it(self, tmp_path):
        raw = bd.Archive(1, "flight_7.zip", 1_000_000_000, "00" * 16)

        assert bd.space_problem(str(tmp_path), raw, free=2_000_000_000) is None
        assert bd.space_problem(str(tmp_path), raw, free=1_999_999_999) == (
            f"2.0 GB are needed in {tmp_path} - the archive, and what is "
            "unpacked from it - and 2.0 GB are free")

    def test_what_arrived_already_needs_no_room(self, tmp_path):
        raw = bd.Archive(1, "flight_7.zip", 1_000_000_000, "00" * 16)

        assert bd.space_problem(str(tmp_path), raw, there=600_000_000,
                                free=1_400_000_000) is None

    def test_the_disk_of_a_folder_that_is_not_there_yet(self, tmp_path):
        small = bd.Archive(1, "flight_7.zip", 10, "00" * 16)
        huge = bd.Archive(1, "flight_7.zip", 10 ** 18, "00" * 16)
        folder = str(tmp_path / "not" / "there" / "yet")

        assert bd.space_problem(folder, small) is None
        assert "are free" in bd.space_problem(folder, huge)


class TestRecordings:
    def test_fetched_unpacked_and_ready(self, tmp_path):
        folder, log, seen = str(tmp_path / "flight_7"), [], []

        files = bd.fetch_recordings(
            samples.entry(), folder, serving(), progress=seen.append,
            log=log.append, **quiet_fetch())

        assert sorted(os.listdir(folder)) == sorted(MEMBERS)   # no archive
        assert os.path.basename(files["airdata"]) == "air_data.csv"
        assert [os.path.basename(path) for path in
                files["thermal_videos"]] == ["DJI_20241119101935_0003_T.MP4"]
        assert log[0] == "Downloading flight_7.zip (0 MB) from Zenodo..."
        assert "Unpacking..." in log
        assert seen == sorted(seen) and seen[-1] == 100

    def test_it_is_asked_for_where_the_catalogue_says(self, tmp_path):
        session = serving()

        bd.fetch_recordings(samples.entry(), str(tmp_path / "f"), session,
                            **quiet_fetch())

        assert session.requested_urls == [
            "https://zenodo.org/api/records/111/files/flight_7.zip/content"]

    def test_the_archive_can_be_kept(self, tmp_path):
        folder = str(tmp_path / "flight_7")

        bd.fetch_recordings(samples.entry(), folder, serving(),
                            keep_archive=True, **quiet_fetch())

        assert "flight_7.zip" in os.listdir(folder)

    def test_a_folder_that_holds_the_flight_is_taken_as_it_is(self, tmp_path):
        folder = tmp_path / "flight_7"
        folder.mkdir()
        for name, data in MEMBERS.items():
            (folder / name).write_bytes(data)
        session, log, seen = FakeSession([]), [], []

        files = bd.fetch_recordings(samples.entry(), str(folder), session,
                                    progress=seen.append, log=log.append)

        assert session.requested_urls == []
        assert log == [f"Flight 7 is already in {folder}"]
        assert files["rgb_calibration"].endswith("W_calib.json")
        assert seen == [100]

    def test_an_archive_that_is_there_is_not_fetched_again(self, tmp_path):
        folder = tmp_path / "flight_7"
        folder.mkdir()
        (folder / "flight_7.zip").write_bytes(ARCHIVE)
        session, log = FakeSession([]), []

        bd.fetch_recordings(samples.entry(), str(folder), session,
                            log=log.append)

        assert session.requested_urls == []
        assert log[0] == "flight_7.zip is already there"
        assert "air_data.csv" in os.listdir(folder)

    def test_one_that_is_not_the_published_file_is(self, tmp_path):
        folder = tmp_path / "flight_7"
        folder.mkdir()
        (folder / "flight_7.zip").write_bytes(b"half of it")
        log = []

        bd.fetch_recordings(samples.entry(), str(folder), serving(),
                            log=log.append, **quiet_fetch())

        assert log[0].startswith(
            "The flight_7.zip that is there cannot be used - it is 10 bytes "
            "long")
        assert "air_data.csv" in os.listdir(folder)

    def test_what_arrives_and_is_not_the_file_is_not_kept(self, tmp_path):
        page = b"<html>503 Service Unavailable</html>".ljust(len(ARCHIVE))

        with pytest.raises(bd.DatasetError, match="not the file expected"):
            bd.fetch_recordings(samples.entry(), str(tmp_path / "f"),
                                serving(page), **quiet_fetch())

        assert os.listdir(tmp_path / "f") == []

    def test_the_tested_archive_has_to_be_the_tested_one(self, tmp_path,
                                                         monkeypatch):
        monkeypatch.setitem(bd.PINNED_SHA256, samples.FLIGHT, "00" * 32)

        with pytest.raises(bd.DatasetError, match="not the file expected"):
            bd.fetch_recordings(samples.entry(), str(tmp_path / "f"),
                                serving(), **quiet_fetch())

        assert os.listdir(tmp_path / "f") == []

    def test_a_flight_that_is_no_longer_published(self, tmp_path):
        session = FakeSession([FakeResponse(status_code=404)])

        with pytest.raises(bd.DatasetError, match="Flight 7 is not where it "
                                                  "was published"):
            bd.fetch_recordings(samples.entry(), str(tmp_path / "f"),
                                session, **quiet_fetch())

    def test_a_connection_that_never_holds(self, tmp_path):
        session = FakeSession([ConnectionError("reset")] * 2)

        with pytest.raises(bd.DatasetError, match="could not be downloaded"):
            bd.fetch_recordings(samples.entry(), str(tmp_path / "f"),
                                session, **quiet_fetch(attempts=2))

    def test_cancelling_keeps_what_arrived(self, tmp_path):
        calls = []

        def cancelled():
            calls.append(1)
            return len(calls) > 3

        with pytest.raises(checked_download.Cancelled):
            bd.fetch_recordings(samples.entry(), str(tmp_path / "f"),
                                serving(pieces=4), cancelled=cancelled,
                                **quiet_fetch())

        left = sorted(os.listdir(tmp_path / "f"))
        assert "flight_7.zip" not in left
        assert "flight_7.zip.part" in left

    def test_cancelled_between_fetching_and_unpacking(self, tmp_path):
        folder = tmp_path / "flight_7"
        folder.mkdir()
        (folder / "flight_7.zip").write_bytes(ARCHIVE)
        calls = []

        def cancelled():                   # after the archive was checked
            calls.append(1)
            return len(calls) > 1

        with pytest.raises(checked_download.Cancelled):
            bd.fetch_recordings(samples.entry(), str(folder),
                                FakeSession([]), cancelled=cancelled)

        assert os.listdir(folder) == ["flight_7.zip"]

    def test_an_archive_that_holds_no_flight(self, tmp_path):
        data = archive_of({"readme.txt": b"nothing to see"})

        with pytest.raises(bd.DatasetError, match="is not a flight"):
            bd.fetch_recordings(samples.entry(raw=data), str(tmp_path / "f"),
                                serving(data), **quiet_fetch())

    def test_an_archive_that_would_write_outside_its_folder(self, tmp_path):
        import io
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr(zipfile.ZipInfo("../outside.txt"), b"gotcha")
        data = buffer.getvalue()

        with pytest.raises(bd.DatasetError, match="outside the folder"):
            bd.fetch_recordings(samples.entry(raw=data),
                                str(tmp_path / "in" / "f"), serving(data),
                                **quiet_fetch())

        assert not os.path.exists(tmp_path / "in" / "outside.txt")

    def test_a_flight_without_recordings(self, tmp_path):
        with pytest.raises(bd.DatasetError, match="has no recordings of "
                                                  "flight 7"):
            bd.fetch_recordings(samples.entry(raw=None), str(tmp_path / "f"),
                                FakeSession([]))

    def test_a_flight_published_without_its_videos_is_not_begun(
            self, tmp_path):
        session = FakeSession([])

        with pytest.raises(bd.DatasetError, match="Flight 7 was published "
                                                  "without its videos"):
            bd.fetch_recordings(samples.entry(videos=False),
                                str(tmp_path / "f"), session)

        assert session.requested_urls == []
        assert not os.path.exists(tmp_path / "f")

    def test_a_flight_that_does_not_fit_is_not_begun(self, tmp_path,
                                                     monkeypatch):
        monkeypatch.setattr(bd, "space_problem",
                            lambda folder, archive, there=0: "no room")
        session = FakeSession([])

        with pytest.raises(bd.DatasetError,
                           match="Flight 7 does not fit: no room."):
            bd.fetch_recordings(samples.entry(), str(tmp_path / "f"),
                                session)

        assert session.requested_urls == []

    def test_a_download_that_broke_off_counts_towards_the_room(
            self, tmp_path, monkeypatch):
        folder = tmp_path / "f"
        folder.mkdir()
        (folder / "flight_7.zip.part").write_bytes(ARCHIVE[:1000])
        seen = []
        monkeypatch.setattr(
            bd, "space_problem",
            lambda folder, archive, there=0: seen.append(there))

        bd.fetch_recordings(samples.entry(), str(folder), serving(),
                            **quiet_fetch())

        assert seen == [1000]


# ---------------------------------------------------------------------------
# What is published beside the labelled video
# ---------------------------------------------------------------------------

class TestPublishedFiles:
    @pytest.fixture
    def zenodo(self):
        data = samples.labelled_archive()
        flight = samples.entry(labelled=data)
        return flight, RangeSession({flight.labelled.url: data}), data

    def test_the_small_files_land_in_the_flights_folder(self, zenodo,
                                                        tmp_path):
        flight, session, _data = zenodo
        target = str(tmp_path / "qgis")

        there = bd.fetch_published(flight, target, session)

        folder = os.path.join(target, "bambi_dataset")
        assert sorted(os.listdir(folder)) == [
            "7_correction.json", "7_gt.txt", "7_matched_poses.json",
            "7_metadata.json", "7_track_mapping.json"]
        assert there[bd.POSES] == os.path.join(folder, "7_matched_poses.json")
        with open(there[bd.LABELS], encoding="utf-8") as handle:
            assert handle.read() == samples.GT
        with open(there[bd.CORRECTION], encoding="utf-8") as handle:
            assert json.load(handle) == samples.CORRECTION

    def test_the_video_stays_where_it_is(self, tmp_path):
        data = samples.labelled_archive(video=3_000_000)
        flight = samples.entry(labelled=data)
        session = RangeSession({flight.labelled.url: data})

        bd.fetch_published(flight, str(tmp_path / "qgis"), session,
                           kinds=[bd.POSES, bd.LABELS, bd.CORRECTION])

        assert len(data) > 3_000_000
        assert session.transferred < len(data) / 4

    def test_only_what_is_asked_for(self, zenodo, tmp_path):
        flight, session, _data = zenodo

        there = bd.fetch_published(flight, str(tmp_path / "qgis"), session,
                                   kinds=[bd.CORRECTION])

        assert list(there) == [bd.CORRECTION]
        assert os.listdir(tmp_path / "qgis" / "bambi_dataset") == [
            "7_correction.json"]

    def test_what_is_there_is_not_fetched_again(self, zenodo, tmp_path):
        flight, session, _data = zenodo
        target = str(tmp_path / "qgis")
        bd.fetch_published(flight, target, session)
        asked = len(session.asked)
        with open(bd.published_path(target, 7, bd.LABELS), "w") as handle:
            handle.write("mine now\n")

        there = bd.fetch_published(flight, target, session)

        assert len(session.asked) == asked
        with open(there[bd.LABELS]) as handle:
            assert handle.read() == "mine now\n"

    def test_a_file_the_archive_does_not_have(self, tmp_path):
        data = samples.labelled_archive(leave_out=("_gt.txt",))
        flight = samples.entry(labelled=data)
        session = RangeSession({flight.labelled.url: data})

        there = bd.fetch_published(flight, str(tmp_path / "qgis"), session)

        assert bd.LABELS not in there and bd.POSES in there

    def test_a_flight_that_was_not_labelled(self, tmp_path):
        with pytest.raises(bd.DatasetError, match="has no flight 7: neither "
                                                  "poses nor labels"):
            bd.fetch_published(samples.entry(), str(tmp_path / "qgis"),
                               RangeSession({}))

    def test_a_labelled_flight_that_is_no_longer_published(self, tmp_path):
        flight = samples.entry(labelled=samples.labelled_archive())

        with pytest.raises(bd.DatasetError, match="The labelled flight 7 is "
                                                  "not where it was"):
            bd.fetch_published(flight, str(tmp_path / "qgis"),
                               RangeSession({}))

    def test_an_archive_that_cannot_be_read(self, tmp_path):
        flight = samples.entry(labelled=samples.labelled_archive())
        session = RangeSession({flight.labelled.url: b"no archive" * 5000})

        with pytest.raises(bd.DatasetError, match="The published files of "
                                                  "flight 7 could not be "
                                                  "read"):
            bd.fetch_published(flight, str(tmp_path / "qgis"), session)

    def test_it_can_be_stopped(self, zenodo, tmp_path):
        flight, session, _data = zenodo

        with pytest.raises(remote_zip.Cancelled):
            bd.fetch_published(flight, str(tmp_path / "qgis"), session,
                               cancelled=lambda: True)


class TestState:
    def test_a_flight_that_is_not_of_the_dataset(self, tmp_path):
        assert bd.read_state(str(tmp_path)) == {}

    def test_what_was_taken_is_noted_in_the_flights_folder(self, tmp_path):
        target = str(tmp_path / "qgis")

        bd.write_state(target, flight=7)
        state = bd.write_state(target, labels={"wanted": True})

        assert state == {"flight": 7, "labels": {"wanted": True}}
        assert bd.read_state(target) == state
        assert os.listdir(os.path.join(target, "bambi_dataset")) == [
            "dataset.json"]

    @pytest.mark.parametrize("text", ["", "{not json", "[1, 2]"])
    def test_a_note_that_cannot_be_read_is_none(self, tmp_path, text):
        folder = tmp_path / "bambi_dataset"
        folder.mkdir()
        (folder / "dataset.json").write_text(text)

        assert bd.read_state(str(tmp_path)) == {}


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------

class TestPoses:
    ORIGIN = {"latitude": 47.0006, "longitude": 15.9673, "altitude": 283.9}

    def test_a_frames_place_is_relative_to_the_elevation_model(self):
        pyproj = pytest.importorskip("pyproj")
        published = samples.published_poses()

        poses = bd.flight_poses(published, self.ORIGIN, samples.EPSG)

        to_map = pyproj.Transformer.from_crs(4326, samples.EPSG,
                                             always_xy=True)
        east0, north0 = to_map.transform(15.9673, 47.0006)
        for image, theirs in zip(poses["images"], published["images"]):
            east, north = to_map.transform(theirs["lng"], theirs["lat"])
            assert image["location"][0] == pytest.approx(east - east0,
                                                         abs=1e-6)
            assert image["location"][1] == pytest.approx(north - north0,
                                                         abs=1e-6)
            # above the model's origin: the published height is above sea
            # level, its own origin being at nought
            assert image["location"][2] == pytest.approx(373.8 - 283.9)

    def test_the_height_of_the_datasets_origin_counts(self):
        pytest.importorskip("pyproj")
        published = samples.published_poses(2)
        published["origin"]["altitude"] = 100.0

        poses = bd.flight_poses(published, self.ORIGIN, samples.EPSG)

        assert poses["images"][0]["location"][2] == pytest.approx(
            373.8 + 100.0 - 283.9)

    def test_everything_else_is_as_published(self):
        pytest.importorskip("pyproj")
        published = samples.published_poses(3)

        poses = bd.flight_poses(published, self.ORIGIN, samples.EPSG,
                                {"dataset": "BAMBI", "flight": 7})

        for image, theirs in zip(poses["images"], published["images"]):
            for key in ("imagefile", "rotation", "fovy", "lat", "lng",
                        "timestamp"):
                assert image[key] == theirs[key]
        assert poses["origin"] == self.ORIGIN
        assert (poses["drone"], poses["mask"]) == ("DJI_M30T", "mask_T.png")
        assert poses["source"] == {"dataset": "BAMBI", "flight": 7,
                                   "frames": "published"}
        # what was handed in is not touched
        assert published["images"][0]["location"] == [1200.0, -670.0, 373.8]

    def test_published_poses_are_told_from_an_extractions(self):
        pytest.importorskip("pyproj")
        published = samples.published_poses(2)

        assert bd.are_published(
            bd.flight_poses(published, self.ORIGIN, samples.EPSG))
        assert not bd.are_published(samples.own_poses(published))
        assert not bd.are_published({}) and not bd.are_published(None)
        assert not bd.are_published({"source": "somewhere"})

    def test_they_are_written_as_the_flights_own(self, tmp_path):
        path = bd.write_poses(str(tmp_path / "qgis"), {"images": [1]})

        assert path == str(tmp_path / "qgis" / "poses_t.json")
        with open(path) as handle:
            assert json.load(handle) == {"images": [1]}
        assert os.listdir(tmp_path / "qgis") == ["poses_t.json"]

    @pytest.mark.parametrize("document", [
        "not json", "{}", '{"images": []}', '{"images": [{"lat": 1}]}',
        '{"images": [{"lat": "north", "lng": 1, "location": [0, 0, 0]}]}',
        '{"images": [{"lat": 1, "lng": 1, "location": [0]}]}'])
    def test_a_file_that_is_no_poses_file(self, tmp_path, document):
        path = tmp_path / "7_matched_poses.json"
        path.write_text(document)

        with pytest.raises(bd.DatasetError, match="7_matched_poses.json is "
                                                  "not a poses file"):
            bd.read_poses(str(path))

    def test_a_poses_file(self, tmp_path):
        path = tmp_path / "7_matched_poses.json"
        path.write_text(json.dumps(samples.published_poses(4)))

        assert len(bd.read_poses(str(path))["images"]) == 4


# ---------------------------------------------------------------------------
# The correction
# ---------------------------------------------------------------------------

class TestCorrection:
    def test_the_grid_convergence_leaves_the_rotation(self):
        correction = bd.flight_correction(samples.CORRECTION, 0.7148)

        assert correction["rotation"] == {
            "x": 0.0, "y": 0.0,
            "z": pytest.approx(0.033 - math.radians(0.7148))}
        assert correction["translation"] == {"x": 0.0, "y": 0.0, "z": 46.65}

    def test_without_one_it_is_as_published(self):
        correction = bd.flight_correction(samples.CORRECTION)

        assert correction["rotation"]["z"] == 0.033

    def test_a_range_that_says_what_holds_anyway_is_left_out(self):
        assert bd.flight_correction(samples.CORRECTION, 0.7)[
            "additional"] == []

    def test_a_range_of_its_own_is_kept_in_the_plugins_words(self):
        published = dict(samples.CORRECTION, fine_corrections=[
            {"start_frame": 1889, "end_frame": 3631,
             "rotation": {"x": 0.0, "y": 0.0, "z": -0.0094},
             "translation": {"x": 0.0, "y": 0.0, "z": -1.0}}])

        correction = bd.flight_correction(published, 1.0)

        assert correction["additional"] == [{
            "start": 1889, "end": 3631,
            "translation": {"x": 0.0, "y": 0.0, "z": -1.0},
            "rotation": {"x": 0.0, "y": 0.0,
                         "z": pytest.approx(-0.0094 - math.radians(1.0))}}]

    def test_a_range_without_values_of_its_own_takes_the_flights(self):
        published = dict(samples.CORRECTION, fine_corrections=[
            {"start_frame": 3, "end_frame": 5,
             "translation": {"z": 2.0}}])

        (part,) = bd.flight_correction(published)["additional"]

        assert part["rotation"] == {"x": 0.0, "y": 0.0, "z": 0.033}
        assert part["translation"] == {"x": 0.0, "y": 0.0, "z": 2.0}

    def test_ranges_are_numbered_as_the_flights_own_frames(self):
        published = dict(samples.CORRECTION, fine_corrections=[
            {"start_frame": 2, "end_frame": 5, "translation": {"z": 2.0}},
            {"start_frame": 6, "end_frame": 9, "translation": {"z": 3.0}},
            {"start_frame": 20, "end_frame": 30, "translation": {"z": 4.0}}])
        # the flight has a frame ahead of the published ones, and lacks 4-7
        mapping = {0: 1, 1: 2, 2: 3, 3: 4, 8: 5, 9: 6}

        ranges = bd.flight_correction(published, 0.0, mapping)["additional"]

        assert [(part["start"], part["end"], part["translation"]["z"])
                for part in ranges] == [(3, 4, 2.0), (5, 6, 3.0)]

    def test_nothing_published_is_no_correction(self):
        assert bd.flight_correction({}) == {
            "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}, "additional": []}

    def test_it_is_written_as_the_input_tab_reads_it(self, tmp_path):
        correction = bd.flight_correction(samples.CORRECTION, 0.7148)

        path = bd.write_correction(str(tmp_path / "into"), correction)

        assert path == str(tmp_path / "into" / "correction.json")
        with open(path) as handle:
            assert json.load(handle) == correction
        assert set(correction) == {"translation", "rotation", "additional"}

    @pytest.mark.parametrize("document", [
        "not json", "[1]", '{"rotation": 3}', '{"translation": {"x": "a"}}'])
    def test_a_file_that_is_no_correction(self, tmp_path, document):
        path = tmp_path / "7_correction.json"
        path.write_text(document)

        with pytest.raises(bd.DatasetError, match="7_correction.json is not "
                                                  "a correction file"):
            bd.read_correction(str(path))


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

class TestLabelFile:
    def test_a_row_is_an_animal_on_a_key_frame(self):
        labels = bd.parse_labels(samples.GT)

        assert len(labels) == 5
        assert labels[0] == bd.Label(
            frame=0, track=4668, x1=570.0, y1=223.0, x2=619.0, y2=299.0,
            species="Capreolus capreolus (Roe deer)", sex="male",
            age="adult", occluded=True)
        assert labels[3].sex == "female" and labels[3].age == "juvenile"
        assert not labels[1].occluded

    def test_one_is_female_and_two_is_male(self):
        """As the dataset's notes on its second version say ("recorded as
        gender=1 (Female)") - and not as the ``contains_male`` /
        ``contains_female`` flags of a flight's metadata would have it.
        Two rows of flight 104: a hind the review of that version called
        female, and a stag it called male."""
        hind, stag = bd.parse_labels(
            "32,1821,1005,425,19,48,1.00,58,0.50,"
            "Cervus elaphus (Red deer),1,1,0\n"
            "1535,1825,996,375,28,40,1.00,12,0.50,"
            "Cervus elaphus (Red deer),2,2,0\n")

        assert (hind.sex, hind.age) == ("female", "juvenile")
        assert (stag.sex, stag.age) == ("male", "adult")

    def test_sex_and_age_that_are_not_known(self):
        (label,) = bd.parse_labels("0,1,10,10,5,5,1.0,3,1.0,Unknown,0,0,0\n")

        assert (label.species, label.sex, label.age) == (
            "Unknown", "unknown", "unknown")
        (label,) = bd.parse_labels("0,1,10,10,5,5,1.0,3,1.0,Fox,x,9\n")
        assert (label.sex, label.age) == ("unknown", "unknown")

    def test_a_short_row_says_what_it_can(self):
        (label,) = bd.parse_labels("3,1,10,20,5,6\n")

        assert (label.frame, label.x2, label.y2) == (3, 15.0, 26.0)
        assert (label.species, label.sex, label.occluded) == (
            "", "unknown", False)

    def test_rows_that_cannot_be_read_are_left_out(self):
        text = ("# frame, track, ...\n"
                "\n"
                "frame,track,left,top,width,height\n"
                "1,2,3\n"
                "5,1,10,10,0,5,1.0,3,1.0,Fox,0,0,0\n"       # no width
                "-1,1,10,10,5,5,1.0,3,1.0,Fox,0,0,0\n"      # no frame
                "6,1,10,10,5,5,1.0,3,1.0,Fox,0,0,0\n")

        assert [label.frame for label in bd.parse_labels(text)] == [6]

    def test_a_file_that_cannot_be_read(self, tmp_path):
        with pytest.raises(bd.DatasetError, match="7_gt.txt cannot be read"):
            bd.read_labels(str(tmp_path / "7_gt.txt"))

        path = tmp_path / "7_gt.txt"
        path.write_text(samples.GT)
        assert len(bd.read_labels(str(path))) == 5


class TestFrames:
    def test_a_frame_is_found_by_the_moment_it_was_taken(self):
        published = samples.published_poses(6)
        own = samples.own_poses(published, before=2, after=1)

        assert bd.frame_mapping(published, own) == {
            0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7}

    def test_a_frame_the_flight_does_not_have_is_not_there(self):
        published = samples.published_poses(6)
        own = samples.own_poses(published, before=0, after=0, dropped=(2, 3))

        assert bd.frame_mapping(published, own) == {0: 0, 1: 1, 4: 2, 5: 3}

    def test_a_frame_of_another_moment_is_another_frame(self):
        published = samples.published_poses(3)
        own = samples.own_poses(published, before=0, after=0)
        # 10 ms off: nearer than the next frame, and not the same one
        own["images"][1]["timestamp"] = (
            samples.START + __import__("datetime").timedelta(
                milliseconds=samples.FRAME_MS + 10)).isoformat()

        assert bd.frame_mapping(published, own) == {0: 0, 2: 2}
        assert bd.frame_mapping(published, own, tolerance_s=0.02) == {
            0: 0, 1: 1, 2: 2}

    def test_the_published_poses_are_their_own_frames(self):
        pytest.importorskip("pyproj")
        published = samples.published_poses(4)
        own = bd.flight_poses(published, TestPoses.ORIGIN, samples.EPSG)
        for image in own["images"]:
            image.pop("timestamp")           # even where nothing says when

        assert bd.frame_mapping(published, own) == {0: 0, 1: 1, 2: 2, 3: 3}

    def test_frames_that_do_not_say_when(self):
        published = samples.published_poses(3)
        own = samples.own_poses(published)
        for image in own["images"]:
            image["timestamp"] = ""

        assert bd.frame_mapping(published, own) == {}
        assert bd.frame_mapping({"images": [{}]}, samples.own_poses(
            published)) == {}

    def test_the_same_frames_are_told_from_others(self):
        published = samples.published_poses(4)
        own = samples.own_poses(published)

        assert bd.frames_fingerprint(own) == bd.frames_fingerprint(
            json.loads(json.dumps(own)))
        assert bd.frames_fingerprint(own) != bd.frames_fingerprint(published)
        assert bd.frames_fingerprint(own) != bd.frames_fingerprint(
            samples.own_poses(published, after=3))
        assert bd.frames_fingerprint({}) == bd.frames_fingerprint(None)


class TestPixels:
    def test_the_middle_stays_the_middle(self):
        published = samples.published_poses(1)
        own = samples.own_poses(published)

        transform = bd.pixel_transform(published, own, (640, 512))

        assert transform(512.0, 512.0) == (320.0, 256.0)

    def test_a_pixel_moves_by_the_ratio_of_the_focal_lengths(self):
        published = samples.published_poses(1)
        own = samples.own_poses(published)
        theirs = 512.0 / math.tan(math.radians(samples.PUBLISHED_FOVY) / 2)
        mine = 256.0 / math.tan(math.radians(samples.OWN_FOVY) / 2)

        x, y = bd.pixel_transform(published, own, (640, 512))(612.0, 312.0)

        assert x == pytest.approx(320.0 + 100.0 * mine / theirs)
        assert y == pytest.approx(256.0 - 200.0 * mine / theirs)

    def test_the_same_picture_is_left_as_it_is(self):
        published = samples.published_poses(1)

        transform = bd.pixel_transform(published, published, (1024, 1024))

        assert transform(17.0, 900.5) == pytest.approx((17.0, 900.5))

    def test_the_real_flight(self):
        """Flight 306: the plugin's frames have the published picture's
        size and a narrower view - a pixel moves outwards by 1.1331."""
        published = {"images": [{"fovy": [48.887902511473634]}]}
        own = {"images": [{"fovy": [43.71726995855834]}]}

        x, y = bd.pixel_transform(published, own, (1024, 1024))(612.0, 512.0)

        assert (x - 512.0) / 100.0 == pytest.approx(1.1331, abs=1e-4)
        assert y == 512.0

    @pytest.mark.parametrize("fovy", [None, [], "wide", 0, [200.0]])
    def test_a_view_that_is_not_known(self, fovy):
        published = samples.published_poses(1)
        own = {"images": [{"fovy": fovy}]}

        with pytest.raises(bd.DatasetError, match="field of view"):
            bd.pixel_transform(published, own, (640, 512))


class TestLabelTracks:
    def tracks(self, labels=None, mapping=None, size=(1024, 1024),
               transform=lambda x, y: (x, y)):
        labels = bd.parse_labels(samples.GT) if labels is None else labels
        if mapping is None:
            mapping = {frame: frame for frame in range(12)}
        return bd.label_tracks(labels, mapping, transform, size)

    def test_an_animal_is_a_track_of_key_frames(self):
        deer, boar = self.tracks()

        assert (deer["track"], deer["species"], deer["sex"], deer["age"]) == (
            4668, "Capreolus capreolus (Roe deer)", "male", "adult")
        assert sorted(deer["keyframes"]) == [0, 4, 10]
        assert deer["keyframes"][0] == {
            "box": (570.0, 223.0, 619.0, 299.0), "occluded": True}
        assert deer["keyframes"][4]["occluded"] is False
        assert (boar["track"], boar["species"], boar["sex"], boar["age"]) == (
            4669, "Sus scrofa (Wild boar)", "female", "juvenile")
        assert sorted(boar["keyframes"]) == [2, 8]

    def test_key_frames_go_where_the_flights_frames_are(self):
        mapping = {0: 3, 4: 7, 10: 13, 2: 5}          # frame 8 is missing

        deer, boar = self.tracks(mapping=mapping)

        assert sorted(deer["keyframes"]) == [3, 7, 13]
        assert sorted(boar["keyframes"]) == [5]

    def test_boxes_are_scaled_to_the_flights_picture(self):
        deer, _boar = self.tracks(
            transform=lambda x, y: (x / 2.0, y / 2.0), size=(512, 512))

        assert deer["keyframes"][0]["box"] == (285.0, 111.5, 309.5, 149.5)

    def test_a_box_over_the_edge_is_cut_there(self):
        labels = bd.parse_labels("0,1,1000,-10,60,40,1,1,1,Fox,0,0,0\n")

        (track,) = self.tracks(labels)

        assert track["keyframes"][0]["box"] == (1000.0, 0.0, 1024.0, 30.0)

    def test_a_box_outside_the_picture_is_left_out(self):
        labels = bd.parse_labels(
            "0,1,1030,10,60,40,1,1,1,Fox,0,0,0\n"       # right of it
            "1,1,10,-50,60,40,1,1,1,Fox,0,0,0\n"        # above it
            "2,1,1023.5,10,60,40,1,1,1,Fox,0,0,0\n"     # a sliver of it
            "3,1,10,10,60,40,1,1,1,Fox,0,0,0\n")

        (track,) = self.tracks(labels)

        assert sorted(track["keyframes"]) == [3]

    def test_an_animal_none_of_whose_frames_the_flight_has(self):
        assert self.tracks(mapping={}) == []

    def test_a_track_is_what_most_of_its_key_frames_say(self):
        labels = bd.parse_labels(
            "0,1,10,10,5,5,1,1,1,Cervus elaphus (Red deer),1,2,0\n"
            "1,1,10,10,5,5,1,1,1,Cervus elaphus (Red deer),1,2,0\n"
            "2,1,10,10,5,5,1,1,1,Dama dama (Fallow Deer),2,1,0\n")

        (track,) = self.tracks(labels)

        assert (track["species"], track["sex"], track["age"]) == (
            "Cervus elaphus (Red deer)", "female", "adult")

    def test_an_animal_recognised_once_is_not_unknown(self):
        labels = bd.parse_labels(
            "0,1,10,10,5,5,1,1,1,Unknown,0,0,0\n"
            "1,1,10,10,5,5,1,1,1,Unknown,0,0,0\n"
            "2,1,10,10,5,5,1,1,1,Cervus elaphus (Red deer),1,2,0\n")

        (track,) = self.tracks(labels)

        assert (track["species"], track["sex"], track["age"]) == (
            "Cervus elaphus (Red deer)", "female", "adult")

    def test_an_animal_nobody_recognised(self):
        labels = bd.parse_labels("0,1,10,10,5,5,1,1,1,Unknown,0,0,0\n"
                                 "1,1,10,10,5,5\n")

        (track,) = self.tracks(labels)

        assert (track["species"], track["sex"], track["age"]) == (
            "Unknown", "unknown", "unknown")

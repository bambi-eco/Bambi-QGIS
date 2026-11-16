# -*- coding: utf-8 -*-
"""The public example flight (core/demo_data.py): its archive is fetched
whole and as published, unpacked where it belongs, and what is in the folder
is what the Input tab is filled with - with a small archive standing in for
the real one, and no network."""
import hashlib
import io
import os
import zipfile

import pytest

from bambi_wildlife_detection.core import checked_download, demo_data

from .fakes import FakeResponse, FakeSession

AIRDATA = (
    "time(millisecond),latitude,longitude,isVideo\n"
    "0,47.0100,15.9900,0\n"                    # on the way there
    "100,47.0021,15.9695,1\n"
    "200,47.0084,15.9785,1\n"
    "300,0,0,1\n"                              # no fix
    "400,47.0050,15.9750,1\n")
MEMBERS = {
    "air_data.csv": AIRDATA.encode(),
    "DJI_20241119101935_0003_T.MP4": b"thermal video" * 400,
    "DJI_20241119101935_0003_T.SRT": b"1\n00:00:00,000 --> 00:00:00,033\n",
    "DJI_20241119101935_0003_W.MP4": b"rgb video" * 900,
    "DJI_20241119101935_0003_W.SRT": b"1\n00:00:00,000 --> 00:00:00,033\n",
    "T_calib.json": b'{"mtx": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}',
    "W_calib.json": b'{"mtx": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]}',
}


def archive_of(members=MEMBERS, compress=zipfile.ZIP_STORED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compress) as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


ARCHIVE = archive_of()


def demo_for(data=ARCHIVE, **changed):
    """The pin of a stand-in archive."""
    values = dict(demo_data.DEMO._asdict(), filename="flight_test.zip",
                  url="https://example.org/flight_test.zip", size=len(data),
                  sha256=hashlib.sha256(data).hexdigest())
    values.update(changed)
    return demo_data.DemoFlight(**values)


def serving(data=ARCHIVE, pieces=3):
    size = max(1, len(data) // pieces)
    chunks = [data[i:i + size] for i in range(0, len(data), size)]
    return FakeSession([FakeResponse(chunks=chunks, content_length=len(data),
                                     headers={"etag": '"v1"'})])


def quiet_fetch(**more):
    return dict({"sleep": lambda seconds: None}, **more)


# ---------------------------------------------------------------------------
# The flight that is published
# ---------------------------------------------------------------------------

class TestThePin:
    def test_it_is_named_by_what_it_is(self):
        demo = demo_data.DEMO

        assert demo.filename == "flight_306.zip"
        assert demo.url == ("https://zenodo.org/api/records/19047570/files/"
                            "flight_306.zip/content")
        assert demo.size == 121842908
        assert len(demo.sha256) == 64 and int(demo.sha256, 16)
        assert demo.url.startswith("https://")

    def test_it_says_whose_it_is(self):
        demo = demo_data.DEMO

        assert demo.licence == "CC BY 4.0"
        assert demo.doi in demo.citation and "BAMBI Raw Dataset" in \
            demo.citation
        assert demo.name == "BAMBI demo flight 306"
        assert "Styria" in demo.description


# ---------------------------------------------------------------------------
# The archive
# ---------------------------------------------------------------------------

class TestArchive:
    def _written(self, tmp_path, data):
        path = str(tmp_path / "flight_test.zip")
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def test_the_published_file_has_no_problem(self, tmp_path):
        assert demo_data.archive_problem(
            self._written(tmp_path, ARCHIVE), demo_for()) is None

    def test_a_file_of_another_length(self, tmp_path):
        problem = demo_data.archive_problem(
            self._written(tmp_path, ARCHIVE[:-10]), demo_for())

        assert problem == (f"it is {len(ARCHIVE) - 10} bytes long, not "
                           f"{len(ARCHIVE)}")

    def test_a_file_with_other_contents(self, tmp_path):
        changed = bytearray(ARCHIVE)
        changed[100] ^= 0xFF

        problem = demo_data.archive_problem(
            self._written(tmp_path, bytes(changed)), demo_for())

        assert problem.startswith("its contents are not the published ones")

    def test_a_file_that_is_not_there(self, tmp_path):
        assert demo_data.archive_problem(
            str(tmp_path / "none.zip"), demo_for()).startswith(
            "it cannot be read")


class TestDownload:
    def test_it_arrives_whole_and_is_the_published_file(self, tmp_path):
        session, log, seen = serving(), [], []

        path = demo_data.download(str(tmp_path / "demo"), session, demo_for(),
                                  progress=seen.append, log=log.append,
                                  **quiet_fetch())

        assert path == str(tmp_path / "demo" / "flight_test.zip")
        with open(path, "rb") as handle:
            assert handle.read() == ARCHIVE
        assert os.listdir(tmp_path / "demo") == ["flight_test.zip"]
        assert log[0] == ("Downloading flight_test.zip (0 MB) from "
                          "Zenodo...")
        assert seen[-1] == 100

    def test_an_archive_that_is_there_is_not_fetched_again(self, tmp_path):
        folder = tmp_path / "demo"
        folder.mkdir()
        (folder / "flight_test.zip").write_bytes(ARCHIVE)
        session, log = FakeSession([]), []

        demo_data.download(str(folder), session, demo_for(), log=log.append)

        assert session.requested_urls == []
        assert log == ["flight_test.zip is already there"]

    def test_one_that_is_not_the_published_file_is(self, tmp_path):
        folder = tmp_path / "demo"
        folder.mkdir()
        (folder / "flight_test.zip").write_bytes(b"half of it")
        session, log = serving(), []

        path = demo_data.download(str(folder), session, demo_for(),
                                  log=log.append, **quiet_fetch())

        assert log[0].startswith(
            "The flight_test.zip that is there cannot be used - it is 10 "
            "bytes long")
        with open(path, "rb") as handle:
            assert handle.read() == ARCHIVE

    def test_what_arrives_and_is_not_the_file_is_not_kept(self, tmp_path):
        page = b"<html>503 Service Unavailable</html>".ljust(len(ARCHIVE))
        session = serving(page)

        with pytest.raises(demo_data.DemoError,
                           match="not the file expected"):
            demo_data.download(str(tmp_path / "demo"), session, demo_for(),
                               **quiet_fetch())

        assert os.listdir(tmp_path / "demo") == []

    def test_a_flight_that_is_no_longer_published(self, tmp_path):
        session = FakeSession([FakeResponse(status_code=404)])

        with pytest.raises(demo_data.DemoError,
                           match="not where it was published"):
            demo_data.download(str(tmp_path / "demo"), session, demo_for(),
                               **quiet_fetch())

    def test_a_connection_that_never_holds(self, tmp_path):
        session = FakeSession([ConnectionError("reset")] * 2)

        with pytest.raises(demo_data.DemoError, match="could not be "
                                                      "downloaded"):
            demo_data.download(str(tmp_path / "demo"), session, demo_for(),
                               **quiet_fetch(attempts=2))

    def test_cancelling_keeps_what_arrived(self, tmp_path):
        session = serving(pieces=4)
        calls = []

        def cancelled():
            calls.append(1)
            return len(calls) > 3

        with pytest.raises(checked_download.Cancelled):
            demo_data.download(str(tmp_path / "demo"), session, demo_for(),
                               cancelled=cancelled, **quiet_fetch())

        left = sorted(os.listdir(tmp_path / "demo"))
        assert "flight_test.zip" not in left
        assert "flight_test.zip.part" in left


class TestExtract:
    def _archive(self, tmp_path, members=MEMBERS, **more):
        path = str(tmp_path / "flight_test.zip")
        with open(path, "wb") as handle:
            handle.write(archive_of(members, **more))
        return path

    def test_the_files_land_in_the_folder(self, tmp_path):
        log = []

        written = demo_data.extract(self._archive(tmp_path),
                                    str(tmp_path / "flight"), log.append)

        assert sorted(os.listdir(tmp_path / "flight")) == sorted(MEMBERS)
        assert len(written) == 7
        assert (tmp_path / "flight" / "air_data.csv").read_text() == AIRDATA
        assert log == [f"Unpacked 7 file(s) into {tmp_path / 'flight'}"]

    @pytest.mark.parametrize("name", [
        "../outside.txt", "nested/../../outside.txt", "/tmp/outside.txt"])
    def test_a_file_that_would_land_outside_is_refused(self, tmp_path, name):
        archive = str(tmp_path / "bad.zip")
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("air_data.csv", AIRDATA)
            bundle.writestr(zipfile.ZipInfo(name), b"gotcha")

        with pytest.raises(demo_data.DemoError, match="outside the folder"):
            demo_data.extract(archive, str(tmp_path / "flight"))

        # nothing of it was unpacked, inside or out
        assert not os.path.exists(tmp_path / "flight" / "air_data.csv")
        assert not os.path.exists(tmp_path / "outside.txt")
        assert not os.path.exists("/tmp/outside.txt")

    def test_an_archive_that_swells_is_refused(self, tmp_path):
        bomb = {"air_data.csv": b"0" * 4_000_000}

        with pytest.raises(demo_data.DemoError, match="it is not unpacked"):
            demo_data.extract(
                self._archive(tmp_path, bomb, compress=zipfile.ZIP_DEFLATED),
                str(tmp_path / "flight"))

        assert not os.path.exists(tmp_path / "flight")

    def test_what_is_no_archive(self, tmp_path):
        path = str(tmp_path / "flight_test.zip")
        with open(path, "wb") as handle:
            handle.write(b"this is not a zip archive")

        with pytest.raises(demo_data.DemoError, match="not a ZIP archive"):
            demo_data.extract(path, str(tmp_path / "flight"))

    def test_folders_in_an_archive_are_followed(self, tmp_path):
        nested = {"flight/air_data.csv": AIRDATA.encode()}

        demo_data.extract(self._archive(tmp_path, nested),
                          str(tmp_path / "out"))

        assert os.path.isfile(tmp_path / "out" / "flight" / "air_data.csv")


# ---------------------------------------------------------------------------
# The flight in the folder
# ---------------------------------------------------------------------------

@pytest.fixture
def flight(tmp_path):
    folder = tmp_path / "flight"
    folder.mkdir()
    for name, data in MEMBERS.items():
        (folder / name).write_bytes(data)
    return str(folder)


class TestFlightFiles:
    def test_what_the_input_tab_is_filled_with(self, flight):
        files = demo_data.flight_files(flight)

        def names(key):
            value = files[key]
            return ([os.path.basename(v) for v in value]
                    if isinstance(value, list) else os.path.basename(value))

        assert names("thermal_videos") == ["DJI_20241119101935_0003_T.MP4"]
        assert names("thermal_srts") == ["DJI_20241119101935_0003_T.SRT"]
        assert names("rgb_videos") == ["DJI_20241119101935_0003_W.MP4"]
        assert names("rgb_srts") == ["DJI_20241119101935_0003_W.SRT"]
        assert names("airdata") == "air_data.csv"
        assert names("thermal_calibration") == "T_calib.json"
        assert names("rgb_calibration") == "W_calib.json"
        assert demo_data.is_complete(files)

    def test_several_recordings_come_in_the_order_they_were_made(self,
                                                                 tmp_path):
        for name in ("DJI_0002_T.MP4", "DJI_0001_T.MP4", "dji_0003_t.mp4",
                     "DJI_0001_W.MP4", "DJI_0001_V.MP4", "notes.mp4"):
            (tmp_path / name).write_bytes(b"x")

        files = demo_data.flight_files(str(tmp_path))

        assert [os.path.basename(p) for p in files["thermal_videos"]] == [
            "DJI_0001_T.MP4", "DJI_0002_T.MP4", "dji_0003_t.mp4"]
        assert [os.path.basename(p) for p in files["rgb_videos"]] == [
            "DJI_0001_V.MP4", "DJI_0001_W.MP4"]

    def test_a_folder_without_a_flight(self, tmp_path):
        files = demo_data.flight_files(str(tmp_path))

        assert files["thermal_videos"] == [] and files["airdata"] == ""
        assert not demo_data.is_complete(files)
        assert not demo_data.is_complete(
            demo_data.flight_files(str(tmp_path / "nowhere")))

    def test_a_video_without_its_srt_is_no_flight_yet(self, flight):
        os.remove(os.path.join(flight, "DJI_20241119101935_0003_T.SRT"))

        assert not demo_data.is_complete(demo_data.flight_files(flight))

    def test_where_the_drone_was_while_it_recorded(self, flight):
        bounds = demo_data.recording_bounds(
            os.path.join(flight, "air_data.csv"))

        # the way there (47.01, 15.99) and the row without a fix are out
        assert bounds == (15.9695, 47.0021, 15.9785, 47.0084)

    def test_a_log_that_does_not_say_when_it_recorded(self, tmp_path):
        path = tmp_path / "log.csv"
        path.write_text("latitude,longitude\n47.0,15.9\n47.1,16.0\n")

        assert demo_data.recording_bounds(str(path)) == (15.9, 47.0, 16.0,
                                                         47.1)

    @pytest.mark.parametrize("text", [
        "", "time,height\n1,2\n", "latitude,longitude\nnorth,east\n",
        "latitude,longitude,isVideo\n0,0,1\n"])
    def test_a_log_without_a_position(self, tmp_path, text):
        path = tmp_path / "log.csv"
        path.write_text(text)

        assert demo_data.recording_bounds(str(path)) is None
        assert demo_data.recording_bounds(str(tmp_path / "none.csv")) is None


# ---------------------------------------------------------------------------
# All of it
# ---------------------------------------------------------------------------

class TestPrepare:
    def test_fetched_unpacked_and_ready(self, tmp_path):
        folder, log, seen = str(tmp_path / "demo"), [], []

        files = demo_data.prepare(folder, serving(), demo_for(),
                                  progress=seen.append, log=log.append,
                                  **quiet_fetch())

        assert demo_data.is_complete(files)
        assert sorted(os.listdir(folder)) == sorted(MEMBERS)   # no archive
        assert "Unpacking..." in log
        assert seen == sorted(seen) and seen[-1] == 100

    def test_the_archive_can_be_kept(self, tmp_path):
        folder = str(tmp_path / "demo")

        demo_data.prepare(folder, serving(), demo_for(), keep_archive=True,
                          **quiet_fetch())

        assert "flight_test.zip" in os.listdir(folder)

    def test_a_folder_that_holds_the_flight_is_taken_as_it_is(self, flight):
        session, log = FakeSession([]), []

        files = demo_data.prepare(flight, session, demo_for(),
                                  log=log.append)

        assert session.requested_urls == []
        assert log == [f"The demo flight is already in {flight}"]
        assert demo_data.is_complete(files)

    def test_an_archive_that_holds_no_flight(self, tmp_path):
        data = archive_of({"readme.txt": b"nothing to see"})

        with pytest.raises(demo_data.DemoError, match="is not a flight"):
            demo_data.prepare(str(tmp_path / "demo"), serving(data),
                              demo_for(data), **quiet_fetch())

    def test_cancelled_between_fetching_and_unpacking(self, tmp_path):
        folder = tmp_path / "demo"
        folder.mkdir()
        (folder / "flight_test.zip").write_bytes(ARCHIVE)

        with pytest.raises(checked_download.Cancelled):
            demo_data.prepare(str(folder), FakeSession([]), demo_for(),
                              cancelled=lambda: True)

        assert os.listdir(folder) == ["flight_test.zip"]

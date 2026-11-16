# -*- coding: utf-8 -*-
"""A flight's recordings out of an archive and in a folder
(core/flight_archive.py): the archive is unpacked where it belongs and
nowhere else, and what is in the folder is what the Input tab is filled
with - with a small archive standing in for a real one."""
import io
import os
import zipfile

import pytest

from bambi_wildlife_detection.core import flight_archive

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


class TestExtract:
    def _archive(self, tmp_path, members=MEMBERS, **more):
        path = str(tmp_path / "flight_test.zip")
        with open(path, "wb") as handle:
            handle.write(archive_of(members, **more))
        return path

    def test_the_files_land_in_the_folder(self, tmp_path):
        log = []

        written = flight_archive.extract(
            self._archive(tmp_path), str(tmp_path / "flight"), log.append)

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

        with pytest.raises(flight_archive.ArchiveError, match="outside the folder"):
            flight_archive.extract(archive, str(tmp_path / "flight"))

        # nothing of it was unpacked, inside or out
        assert not os.path.exists(tmp_path / "flight" / "air_data.csv")
        assert not os.path.exists(tmp_path / "outside.txt")
        assert not os.path.exists("/tmp/outside.txt")

    def test_an_archive_that_swells_is_refused(self, tmp_path):
        bomb = {"air_data.csv": b"0" * 4_000_000}

        with pytest.raises(flight_archive.ArchiveError, match="it is not unpacked"):
            flight_archive.extract(
                self._archive(tmp_path, bomb, compress=zipfile.ZIP_DEFLATED),
                str(tmp_path / "flight"))

        assert not os.path.exists(tmp_path / "flight")

    def test_what_is_no_archive(self, tmp_path):
        path = str(tmp_path / "flight_test.zip")
        with open(path, "wb") as handle:
            handle.write(b"this is not a zip archive")

        with pytest.raises(flight_archive.ArchiveError, match="not a ZIP archive"):
            flight_archive.extract(path, str(tmp_path / "flight"))

    def test_folders_in_an_archive_are_followed(self, tmp_path):
        nested = {"flight/air_data.csv": AIRDATA.encode()}

        flight_archive.extract(self._archive(tmp_path, nested),
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
        files = flight_archive.flight_files(flight)

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
        assert flight_archive.is_complete(files)

    def test_several_recordings_come_in_the_order_they_were_made(self,
                                                                 tmp_path):
        for name in ("DJI_0002_T.MP4", "DJI_0001_T.MP4", "dji_0003_t.mp4",
                     "DJI_0001_W.MP4", "DJI_0001_V.MP4", "notes.mp4"):
            (tmp_path / name).write_bytes(b"x")

        files = flight_archive.flight_files(str(tmp_path))

        assert [os.path.basename(p) for p in files["thermal_videos"]] == [
            "DJI_0001_T.MP4", "DJI_0002_T.MP4", "dji_0003_t.mp4"]
        assert [os.path.basename(p) for p in files["rgb_videos"]] == [
            "DJI_0001_V.MP4", "DJI_0001_W.MP4"]

    def test_a_folder_without_a_flight(self, tmp_path):
        files = flight_archive.flight_files(str(tmp_path))

        assert files["thermal_videos"] == [] and files["airdata"] == ""
        assert not flight_archive.is_complete(files)
        assert not flight_archive.is_complete(
            flight_archive.flight_files(str(tmp_path / "nowhere")))

    def test_a_video_without_its_srt_is_no_flight_yet(self, flight):
        os.remove(os.path.join(flight, "DJI_20241119101935_0003_T.SRT"))

        assert not flight_archive.is_complete(flight_archive.flight_files(flight))

    def test_where_the_drone_was_while_it_recorded(self, flight):
        bounds = flight_archive.recording_bounds(
            os.path.join(flight, "air_data.csv"))

        # the way there (47.01, 15.99) and the row without a fix are out
        assert bounds == (15.9695, 47.0021, 15.9785, 47.0084)

    def test_a_log_that_does_not_say_when_it_recorded(self, tmp_path):
        path = tmp_path / "log.csv"
        path.write_text("latitude,longitude\n47.0,15.9\n47.1,16.0\n")

        assert flight_archive.recording_bounds(str(path)) == (
            15.9, 47.0, 16.0, 47.1)

    @pytest.mark.parametrize("text", [
        "", "time,height\n1,2\n", "latitude,longitude\nnorth,east\n",
        "latitude,longitude,isVideo\n0,0,1\n"])
    def test_a_log_without_a_position(self, tmp_path, text):
        path = tmp_path / "log.csv"
        path.write_text(text)

        assert flight_archive.recording_bounds(str(path)) is None
        assert flight_archive.recording_bounds(str(tmp_path / "none.csv")) is None

# -*- coding: utf-8 -*-
"""A few files out of a large archive on a server (core/remote_zip.py): the
table of contents and the files asked for are fetched, the video beside
them is not."""
import io
import os
import zipfile

import pytest

from bambi_wildlife_detection.core import remote_zip
from bambi_wildlife_detection.core.remote_zip import RangeFile

from .fakes import RangeSession

URL = "https://zenodo.org/api/records/1/files/flight_7.zip/content"
VIDEO = os.urandom(3_000_000)                 # does not compress
POSES = b'{"images": [' + b'{"lat": 47.1, "lng": 14.2}, ' * 4000 + b"{}]}"
LABELS = b"0,1,10,20,30,40,1.00,3,1.00,Roe deer,2,2,0\n" * 40


def archive(members=None, method=zipfile.ZIP_DEFLATED):
    """A flight's archive: a video and the small files beside it."""
    members = members if members is not None else [
        ("7_correction.json", b'{"translation": {"z": 1.5}}'),
        ("7_gt.txt", LABELS),
        ("7_matched_poses.json", POSES),
        ("7_matched_processed.mp4", VIDEO),
        ("7_metadata.json", b'{"flight_key": "7"}'),
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", method) as bundle:
        for name, data in members:
            bundle.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture
def server():
    return RangeSession({URL: archive()})


# ---------------------------------------------------------------------------
# A file on a server, read a range at a time
# ---------------------------------------------------------------------------

class TestRangeFile:
    DATA = bytes(range(256)) * 4000           # 1,024,000 bytes

    def _file(self, **more):
        session = RangeSession({URL: self.DATA}, **{
            key: more.pop(key) for key in ("busy", "retry_after", "whole")
            if key in more})
        return session, lambda **kw: RangeFile(session, URL, **dict(more,
                                                                    **kw))

    def test_how_long_it_is_is_asked_once(self):
        session, make = self._file()

        remote = make()

        assert remote.size == len(self.DATA)
        assert session.asked == [(URL, 0, 0)]

    def test_a_length_that_is_known_is_not_asked_for(self):
        session, make = self._file()

        remote = make(size=len(self.DATA))

        assert remote.size == len(self.DATA) and session.asked == []

    def test_it_reads_like_a_file(self):
        _session, make = self._file()
        remote = make(size=len(self.DATA))

        assert remote.read(4) == self.DATA[:4] and remote.tell() == 4
        assert remote.seek(1000) == 1000
        assert remote.read(10) == self.DATA[1000:1010]
        assert remote.seek(-5, os.SEEK_END) == len(self.DATA) - 5
        assert remote.read() == self.DATA[-5:]
        assert remote.read(10) == b""                  # at the end
        assert remote.seek(-3, os.SEEK_CUR) == len(self.DATA) - 3
        assert remote.read(100) == self.DATA[-3:]
        assert (remote.readable(), remote.seekable(), remote.writable()) == (
            True, True, False)
        buffer = bytearray(6)
        remote.seek(250)
        assert remote.readinto(buffer) == 6
        assert bytes(buffer) == self.DATA[250:256]

    def test_small_steps_are_served_from_what_was_fetched(self):
        """A ZIP reader takes a table of contents apart field by field."""
        session, make = self._file()
        remote = make(size=len(self.DATA))

        for _ in range(500):
            remote.read(46)
            remote.seek(remote.tell() + 20)

        assert len(session.asked) == 1
        assert session.asked[0] == (URL, 0, remote_zip.CHUNK - 1)

    def test_near_the_end_the_chunk_that_ends_there_is_fetched(self):
        session, make = self._file()
        remote = make(size=len(self.DATA))

        remote.seek(-22, os.SEEK_END)
        assert remote.read(22) == self.DATA[-22:]
        # ...and what a reader looks for just before it is there already
        remote.seek(-70000, os.SEEK_END)
        assert remote.read(100) == self.DATA[-70000:-69900]

        assert session.asked == [(URL, len(self.DATA) - remote_zip.CHUNK,
                                  len(self.DATA) - 1)]

    def test_a_read_larger_than_a_chunk_is_one_request(self):
        session, make = self._file()
        remote = make(size=len(self.DATA))

        assert remote.read(700_000) == self.DATA[:700_000]

        assert session.asked == [(URL, 0, 699_999)]

    def test_what_is_asked_for_ahead_is_clipped_to_the_file(self):
        session, make = self._file()
        remote = make(size=len(self.DATA))

        remote.fetch(len(self.DATA) - 10, 5000)
        remote.fetch(len(self.DATA) + 10, 5000)        # beyond: nothing
        remote.fetch(0, 0)

        assert session.asked == [(URL, len(self.DATA) - 10,
                                  len(self.DATA) - 1)]

    def test_requests_and_bytes_are_counted(self):
        _session, make = self._file()
        remote = make()                                # asks for the length
        remote.read(10)

        assert remote.requests == 2
        assert remote.transferred == 1 + remote_zip.CHUNK

    def test_only_https(self):
        with pytest.raises(remote_zip.RemoteArchiveError, match="non-HTTPS"):
            RangeFile(RangeSession({}), "http://example.org/a.zip", size=10)

    def test_a_file_that_is_not_there(self):
        with pytest.raises(remote_zip.NotThere):
            RangeFile(RangeSession({}), URL)

    def test_a_server_without_range_requests_is_not_sent_the_whole_file(self):
        _session, make = self._file(whole=True)

        with pytest.raises(remote_zip.RemoteArchiveError,
                           match="does not answer range requests"):
            make()

    def test_a_connection_that_breaks_is_noticed(self):
        session, make = self._file()
        remote = make(size=len(self.DATA))
        session.short = True

        with pytest.raises(remote_zip.RemoteArchiveError,
                           match="the connection broke"):
            remote.read(10)

    def test_another_part_than_asked_for_is_refused(self):
        session, make = self._file()
        remote = make(size=len(self.DATA))
        session.shifted = True

        with pytest.raises(remote_zip.RemoteArchiveError,
                           match="another part of the file"):
            remote.read(10)

    def test_no_connection(self):
        session, make = self._file()
        remote = make(size=len(self.DATA))
        session.fails = OSError("name resolution failed")

        with pytest.raises(remote_zip.RemoteArchiveError,
                           match="could not be reached"):
            remote.read(10)

    def test_it_can_be_cancelled(self):
        _session, make = self._file()
        remote = make(size=len(self.DATA), cancelled=lambda: True)

        with pytest.raises(remote_zip.Cancelled):
            remote.read(10)

    def test_a_busy_server_is_asked_again_after_a_while(self):
        waited = []
        session, make = self._file(busy=2)
        remote = make(size=len(self.DATA), sleep=waited.append)

        assert remote.read(4) == self.DATA[:4]

        assert len(session.asked) == 3
        assert sum(waited) == pytest.approx(2.0 + 4.0)   # 2 s, then twice it

    def test_it_waits_as_long_as_the_server_says(self):
        waited = []
        session, make = self._file(busy=1, retry_after=7)
        remote = make(size=len(self.DATA), sleep=waited.append)

        remote.read(4)

        assert sum(waited) == pytest.approx(7.0)
        assert max(waited) <= 1.0                  # a second at a time

    def test_a_server_that_stays_busy_is_given_up_on(self):
        waited = []
        session, make = self._file(busy=99)
        remote = make(size=len(self.DATA), sleep=waited.append)

        with pytest.raises(remote_zip.RemoteArchiveError,
                           match="The server is busy"):
            remote.read(4)

        assert len(session.asked) == remote_zip.RETRIES + 1

    def test_waiting_can_be_cancelled(self):
        state = {"waited": 0.0}
        session, make = self._file(busy=5, retry_after=30)

        def sleep(seconds):
            state["waited"] += seconds

        remote = make(size=len(self.DATA), sleep=sleep,
                      cancelled=lambda: state["waited"] >= 3)

        with pytest.raises(remote_zip.Cancelled):
            remote.read(4)

        assert state["waited"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Files out of an archive
# ---------------------------------------------------------------------------

class TestArchive:
    def test_what_is_in_it(self, server):
        members = remote_zip.list_members(server, URL)

        assert members == [
            ("7_correction.json", 27), ("7_gt.txt", len(LABELS)),
            ("7_matched_poses.json", len(POSES)),
            ("7_matched_processed.mp4", len(VIDEO)),
            ("7_metadata.json", 19)]
        # the length, and the table of contents at the end
        assert len(server.asked) == 2
        assert server.transferred < 300_000

    def test_the_small_files_without_the_video(self, server):
        log = []

        found = remote_zip.read_members(
            server, URL, ["7_matched_poses.json", "7_gt.txt"],
            log=log.append)

        assert found == {"7_matched_poses.json": POSES, "7_gt.txt": LABELS}
        size = len(server.files[URL])
        assert server.transferred < size / 5
        assert log == [
            f"Read 2 file(s) out of flight_7.zip - "
            f"{server.transferred / 1e6:.1f} MB of {size / 1e6:.0f} MB "
            "transferred"]

    def test_a_file_behind_the_video_is_one_request(self, server):
        """The metadata is the last file of the archive: its header and
        contents are fetched together, not the megabytes before it."""
        size = len(server.files[URL])

        found = remote_zip.read_members(server, URL, ["7_metadata.json"],
                                        size=size)

        assert found == {"7_metadata.json": b'{"flight_key": "7"}'}
        assert len(server.asked) == 1                 # it lay in the tail

    def test_a_file_before_the_video_is_one_request_more(self, server):
        size = len(server.files[URL])

        remote_zip.read_members(server, URL, ["7_gt.txt"], size=size)

        assert len(server.asked) == 2
        assert server.asked[1][1] < 1000              # near the beginning

    def test_a_name_the_archive_does_not_have_is_not_in_the_result(
            self, server):
        found = remote_zip.read_members(server, URL,
                                        ["7_gt.txt", "7_rgb_gt.txt"])

        assert sorted(found) == ["7_gt.txt"]

    def test_files_are_found_by_their_base_name(self):
        """An archive made on another system may hold a folder."""
        data = archive([("flight_7/7_gt.txt", LABELS),
                        ("flight_7\\7_metadata.json", b"{}"),
                        ("flight_7/", b"")])
        server = RangeSession({URL: data})

        found = remote_zip.read_members(server, URL,
                                        ["7_gt.txt", "7_metadata.json"])

        assert found == {"7_gt.txt": LABELS, "7_metadata.json": b"{}"}

    def test_the_first_of_two_files_of_one_name_is_taken(self):
        data = archive([("7_gt.txt", b"first\n"), ("copy/7_gt.txt", b"2\n")])

        found = remote_zip.read_members(RangeSession({URL: data}), URL,
                                        ["7_gt.txt"])

        assert found == {"7_gt.txt": b"first\n"}

    def test_stored_files_are_read_as_well(self):
        data = archive(method=zipfile.ZIP_STORED)

        found = remote_zip.read_members(RangeSession({URL: data}), URL,
                                        ["7_gt.txt"])

        assert found == {"7_gt.txt": LABELS}

    def test_a_file_larger_than_such_a_file_can_be_is_refused(self, server):
        with pytest.raises(remote_zip.RemoteArchiveError) as raised:
            remote_zip.read_members(server, URL, ["7_matched_processed.mp4"],
                                    max_bytes=1 << 20)

        assert str(raised.value) == (
            "7_matched_processed.mp4 is 3 MB in the archive: larger than "
            "such a file can be.")
        assert server.transferred < 300_000           # it was not fetched

    def test_what_arrives_is_checked_against_the_archive(self, server):
        with zipfile.ZipFile(io.BytesIO(server.files[URL])) as bundle:
            info = bundle.getinfo("7_gt.txt")
        server.spoil(URL, info.header_offset + 30 + len("7_gt.txt") + 5)

        with pytest.raises(remote_zip.RemoteArchiveError,
                           match="7_gt.txt did not arrive as the archive "
                                 "records it"):
            remote_zip.read_members(server, URL, ["7_gt.txt"])

    def test_something_that_is_no_archive(self):
        server = RangeSession({URL: b"<html>Service unavailable</html>" * 9})

        with pytest.raises(remote_zip.RemoteArchiveError) as raised:
            remote_zip.read_members(server, URL, ["7_gt.txt"])

        assert str(raised.value).startswith(
            "flight_7.zip is not a ZIP archive that can be read")

    def test_an_archive_that_is_not_there(self):
        with pytest.raises(remote_zip.NotThere):
            remote_zip.list_members(RangeSession({}), URL)

    def test_reading_can_be_cancelled(self, server):
        calls = []

        def cancelled():
            calls.append(1)
            return len(calls) > 2

        with pytest.raises(remote_zip.Cancelled):
            remote_zip.read_members(server, URL, ["7_gt.txt"],
                                    cancelled=cancelled)

    def test_an_archive_is_named_by_its_file_not_by_the_end_of_its_address(
            self):
        assert remote_zip._archive_name(URL) == "flight_7.zip"
        assert remote_zip._archive_name(
            "https://example.org/data/a.zip?download=1") == "a.zip"
        assert remote_zip._archive_name("https://example.org/x") == "x"

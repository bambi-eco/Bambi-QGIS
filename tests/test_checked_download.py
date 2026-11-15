# -*- coding: utf-8 -*-
"""Large downloads arrive whole, or not at all.

An elevation tile was written straight to its place in the cache while it
came in, and whatever was there afterwards counted as the tile: a file
shorter than announced, an error page sent with "200 OK", the first third of
a tile left by a QGIS that was closed. And a download that broke at 90 %
began again at zero.
"""
import os

import numpy as np
import pytest

from bambi_wildlife_detection.core import checked_download as cd

from .fakes import FakeResponse, FakeSession, tiny_tiff

URL = "https://example.org/tiles/N1E2.tif"
TILE = tiny_tiff(40000)
ETAG = {"etag": '"abc123"'}


def _cut(data, *sizes):
    """*data* in pieces of *sizes* bytes, and the rest."""
    pieces, start = [], 0
    for size in sizes:
        pieces.append(data[start:start + size])
        start += size
    pieces.append(data[start:])
    return [piece for piece in pieces if piece]


def _whole(headers=None, chunks=None):
    return FakeResponse(chunks=chunks or _cut(TILE, 15000, 15000),
                        content_length=len(TILE), headers=dict(headers or ETAG))


def _rest(start, status=206, headers=None):
    """The answer to a request for the file from byte *start* on."""
    rest = TILE[start:]
    head = {"content-range": f"bytes {start}-{len(TILE) - 1}/{len(TILE)}"}
    head.update(headers or ETAG)
    return FakeResponse(status_code=status, chunks=[rest],
                        content_length=len(rest), headers=head)


@pytest.fixture
def target(tmp_path):
    return str(tmp_path / "cache" / "N1E2.tif")


def _fetch(session, target, **kwargs):
    kwargs.setdefault("sleep", lambda seconds: None)
    return cd.fetch(session, URL, target, **kwargs)


def _files(target):
    return sorted(os.listdir(os.path.dirname(target)))


# ---------------------------------------------------------------------------
# Is a GeoTIFF whole?
# ---------------------------------------------------------------------------

class TestTiffProblem:
    def _written(self, tmp_path, data):
        path = str(tmp_path / "file.tif")
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def test_a_whole_file_has_none(self, tmp_path):
        assert cd.tiff_problem(self._written(tmp_path, TILE)) is None

    @pytest.mark.parametrize("missing", [1, 50, 200, 20000, 39000])
    def test_a_file_that_ends_early(self, tmp_path, missing):
        problem = cd.tiff_problem(self._written(tmp_path, TILE[:-missing]))
        assert "ends after" in problem or "end is missing" in problem

    def test_what_is_not_a_tiff(self, tmp_path):
        page = b"<html><body><h1>502 Bad Gateway</h1></body></html>"
        assert cd.tiff_problem(self._written(tmp_path, page)) \
            == "it is not a TIFF file"
        assert cd.tiff_problem(self._written(tmp_path, b"")) \
            == "it is not a TIFF file"
        assert cd.tiff_problem(self._written(tmp_path, b"II")) \
            == "it is not a TIFF file"

    def test_a_file_that_is_not_there(self, tmp_path):
        assert "cannot be read" in cd.tiff_problem(str(tmp_path / "none.tif"))

    @pytest.mark.parametrize("options", [
        {},
        {"tiled": True, "blockxsize": 128, "blockysize": 128},
        {"tiled": True, "blockxsize": 256, "blockysize": 256,
         "compress": "deflate"},
        {"BIGTIFF": "YES", "tiled": True, "blockxsize": 128, "blockysize": 128},
        {"ENDIANNESS": "BIG"},
    ], ids=["strips", "tiles", "deflate with overviews", "BigTIFF",
            "big-endian"])
    def test_the_geotiffs_gdal_writes(self, tmp_path, options):
        """Whole: nothing to say. Cut anywhere - a thousandth before the
        end, or after the first kilobyte - it is found."""
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        path = str(tmp_path / "dem.tif")
        heights = np.random.default_rng(0).normal(
            500, 50, (600, 700)).astype("float32")
        with rasterio.open(
                path, "w", driver="GTiff", height=600, width=700, count=1,
                dtype="float32", crs="EPSG:3035",
                transform=from_origin(4400000, 2700000, 1, 1),
                **options) as dataset:
            dataset.write(heights, 1)
        if options.get("compress"):
            with rasterio.open(path, "r+") as dataset:
                dataset.build_overviews([2, 4])
        assert cd.tiff_problem(path) is None

        with open(path, "rb") as handle:
            whole = handle.read()
        for share in (0.999, 0.9, 0.5, 0.1, 0.0005):
            cut = self._written(tmp_path, whole[:int(len(whole) * share)])
            assert cd.tiff_problem(cut), f"cut at {share:.2%} not found"


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

class TestFetch:
    def test_a_file_arrives_and_is_put_in_place(self, target):
        progress = []
        session = FakeSession([_whole()])

        assert _fetch(session, target, progress=progress.append) == target
        with open(target, "rb") as handle:
            assert handle.read() == TILE
        assert _files(target) == ["N1E2.tif"]         # no part, no note
        assert progress[-1] == 100 and progress == sorted(progress)
        assert session.requested_headers == [{"Accept-Encoding": "identity"}]

    def test_nothing_has_the_files_name_while_it_comes_in(self, target):
        seen = []

        def cancelled():
            seen.append(_files(target))
            return False

        _fetch(FakeSession([_whole()]), target, cancelled=cancelled)
        assert ["N1E2.tif.part", "N1E2.tif.part.json"] in seen
        assert not any("N1E2.tif" in names for names in seen)

    def test_a_file_shorter_than_announced_is_not_accepted(self, target):
        """The connection ended early and nothing raised - what urllib3
        lets through unless told otherwise. It used to be the tile."""
        short = FakeResponse(chunks=[TILE[:25000]], content_length=len(TILE),
                             headers=dict(ETAG))
        said = []
        session = FakeSession([short, _rest(25000)])

        _fetch(session, target, log=said.append)

        with open(target, "rb") as handle:
            assert handle.read() == TILE
        assert any("the connection ended after 25000 of" in line
                   for line in said)

    def test_it_continues_where_the_connection_broke(self, target):
        broken = _whole(chunks=[TILE[:15000], ConnectionError("reset by peer")])
        said = []
        session = FakeSession([broken, _rest(15000)])

        _fetch(session, target, log=said.append)

        with open(target, "rb") as handle:
            assert handle.read() == TILE
        assert session.requested_headers[1] == {
            "Accept-Encoding": "identity", "Range": "bytes=15000-",
            "If-Range": '"abc123"'}
        assert "  Continuing at 0.0 MB" in said
        assert any("Interrupted (reset by peer); trying again" in line
                   for line in said)

    def test_a_date_does_for_knowing_it_is_the_same_file(self, target):
        dated = {"last-modified": "Fri, 15 Sep 2023 10:00:00 GMT"}
        broken = _whole(headers=dated, chunks=[TILE[:100], OSError("gone")])
        session = FakeSession([broken, _rest(100, headers=dated)])

        _fetch(session, target)
        assert session.requested_headers[1]["If-Range"] == dated["last-modified"]

    def test_a_server_that_cannot_continue_sends_it_all_again(self, target):
        """200 instead of 206: the file changed, or ranges are not served.
        What was there is not kept in front of it."""
        broken = _whole(chunks=[TILE[:15000], ConnectionError("reset")])
        session = FakeSession([broken, _whole()])

        _fetch(session, target)
        with open(target, "rb") as handle:
            assert handle.read() == TILE

    def test_a_part_of_unknown_origin_is_not_continued(self, target):
        """No ETag, no date: nothing says the rest belongs to it."""
        nameless = {"x-served-by": "cache"}
        broken = _whole(headers=nameless,
                        chunks=[TILE[:15000], ConnectionError("reset")])
        session = FakeSession([broken, _whole(headers=nameless)])

        _fetch(session, target)

        assert "Range" not in session.requested_headers[1]
        with open(target, "rb") as handle:
            assert handle.read() == TILE

    def test_a_part_left_by_an_earlier_run_is_continued(self, target):
        """QGIS was closed at 60 %."""
        session = FakeSession([
            _whole(chunks=[TILE[:24000], ConnectionError("reset")])])
        with pytest.raises(cd.DownloadError, match="continues from there"):
            _fetch(session, target, attempts=1)
        assert _files(target) == ["N1E2.tif.part", "N1E2.tif.part.json"]
        assert os.path.getsize(target + ".part") == 24000

        later = FakeSession([_rest(24000)])
        _fetch(later, target)

        assert later.requested_headers[0]["Range"] == "bytes=24000-"
        with open(target, "rb") as handle:
            assert handle.read() == TILE
        assert _files(target) == ["N1E2.tif"]

    def test_a_part_of_another_address_is_not_continued(self, target):
        session = FakeSession([
            _whole(chunks=[TILE[:24000], ConnectionError("reset")])])
        with pytest.raises(cd.DownloadError):
            _fetch(session, target, attempts=1)

        elsewhere = FakeSession([_whole()])
        cd.fetch(elsewhere, URL.replace("tiles", "older"), target)
        assert "Range" not in elsewhere.requested_headers[0]

    def test_a_part_that_was_whole_already(self, target):
        """All of it had arrived when the connection broke: the server has
        nothing to add (416)."""
        os.makedirs(os.path.dirname(target))
        with open(target + ".part", "wb") as handle:
            handle.write(TILE)
        with open(target + ".part.json", "w") as handle:
            handle.write('{"url": "%s", "etag": "\\"abc123\\"", "size": %d}'
                         % (URL, len(TILE)))

        _fetch(FakeSession([FakeResponse(status_code=416)]), target)
        with open(target, "rb") as handle:
            assert handle.read() == TILE

    def test_cancelling_keeps_what_arrived(self, target):
        calls = {"n": 0}

        def cancelled():
            calls["n"] += 1
            return calls["n"] > 3

        with pytest.raises(cd.Cancelled):
            _fetch(FakeSession([_whole(chunks=_cut(TILE, 9000, 9000, 9000))]),
                   target, cancelled=cancelled)

        assert _files(target) == ["N1E2.tif.part", "N1E2.tif.part.json"]
        assert os.path.getsize(target + ".part") == 18000

    def test_an_error_page_is_not_the_file(self, target):
        """Sent with "200 OK" by a proxy; it was saved as the tile."""
        page = b"<html><body>Service temporarily unavailable</body></html>"
        with pytest.raises(cd.NotTheFile, match="it is not a TIFF file"):
            _fetch(FakeSession([FakeResponse(chunks=[page])]), target)
        assert _files(target) == []

    def test_a_file_that_is_not_there(self, target):
        with pytest.raises(cd.NotThere):
            _fetch(FakeSession([FakeResponse(status_code=404)]), target)
        assert _files(target) == []

    def test_a_server_error_is_tried_again_and_then_reported(self, target):
        waited, said = [], []
        session = FakeSession([FakeResponse(status_code=503)] * 3)

        with pytest.raises(cd.DownloadError,
                           match="could not be downloaded: HTTP 503"):
            _fetch(session, target, attempts=3, sleep=waited.append,
                   log=said.append)

        assert len(session.requested_urls) == 3
        assert waited == [2.0, 4.0]                 # a pause before each retry
        assert said[-1] == "  Interrupted (HTTP 503)"

    def test_more_than_announced_is_not_the_file(self, target):
        long = FakeResponse(chunks=[TILE + b"extra"], content_length=len(TILE))
        with pytest.raises(cd.DownloadError, match="were announced"):
            _fetch(FakeSession([long]), target)
        assert _files(target) == []

    def test_a_packed_stream_is_not_measured_by_its_packed_length(self, target):
        packed = FakeResponse(chunks=[TILE], content_length=1234,
                              headers={"content-encoding": "gzip"})
        _fetch(FakeSession([packed]), target)
        assert os.path.getsize(target) == len(TILE)

    def test_without_a_check_anything_whole_is_taken(self, target):
        _fetch(FakeSession([FakeResponse(chunks=[b"not a tiff at all"])]),
               target, check=None)
        with open(target, "rb") as handle:
            assert handle.read() == b"not a tiff at all"

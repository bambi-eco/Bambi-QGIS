# -*- coding: utf-8 -*-
"""Embedding vectors on disk.

The naming rule is the whole contract: the store records only *whether* a
detection was embedded, so if the writer and the reader ever disagreed about
which file a vector lives in, the vectors would be silently unreachable.
"""
import os

import numpy as np
import pytest

from bambi_wildlife_detection.core import embedding_files as ef


# ---------------------------------------------------------------------------
# Naming — one function, used in both directions
# ---------------------------------------------------------------------------

class TestNaming:

    def test_mirrors_the_frame_image(self):
        assert ef.frame_file_name(123, "frame_000123.jpg") == "frame_000123.npz"

    def test_photo_mode_keeps_the_original_name(self):
        """Photo flights carry the camera's own file names, not frame_%06d."""
        assert ef.frame_file_name(0, "DJI_0042.JPG") == "DJI_0042.npz"

    def test_any_extension_is_replaced(self):
        for name in ("f.jpg", "f.JPG", "f.png", "f.tif", "f.jpeg"):
            assert ef.frame_file_name(1, name) == "f.npz"

    def test_a_frame_without_an_image_falls_back_to_its_number(self):
        # The same fallback the rest of the pipeline uses, so both agree on
        # what an unnamed frame is called.
        assert ef.frame_file_name(123) == "frame_000123.npz"
        assert ef.frame_file_name(7, "") == "frame_000007.npz"
        assert ef.frame_file_name(7, "   ") == "frame_000007.npz"

    def test_a_path_is_reduced_to_its_basename(self):
        assert ef.frame_file_name(1, "frames_t/sub/f.jpg") == "f.npz"

    def test_array_keys_round_trip(self):
        assert ef.array_key(4711) == "det_4711"
        assert ef.detection_of_key("det_4711") == 4711

    def test_foreign_keys_are_ignored(self):
        for key in ("", "meta", "det_", "det_x", "xdet_1", "det_1x"):
            assert ef.detection_of_key(key) is None

    def test_the_recorded_folder_is_relative(self):
        # So moving a flight folder does not strand its vectors.
        folder = ef.relative_run_folder("t", "non_geo")
        assert folder == "embeddings_t/non_geo"
        assert not os.path.isabs(folder)

    def test_the_run_folder_separates_projections(self):
        first = ef.run_folder("/flight", "t", "non_geo")
        second = ef.run_folder("/flight", "t", "geo_2k")
        assert first != second
        assert first.endswith(os.path.join("embeddings_t", "non_geo"))


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

@pytest.fixture
def folder(tmp_path):
    return str(tmp_path)


def _vector(seed, dim=8):
    return np.arange(seed, seed + dim, dtype=np.float32)


class TestRoundTrip:

    def test_a_frame_of_vectors_round_trips(self, folder):
        path = os.path.join(folder, "frame_000001.npz")
        written = ef.write_frame(path, {1: _vector(0), 2: _vector(10)})

        assert written == 2
        archive = ef.read_frame(path)
        assert set(archive) == {1, 2}
        assert np.array_equal(archive[1], _vector(0))

    def test_vectors_are_stored_as_float32(self, folder):
        """Float64 would double the size of the expensive part for nothing."""
        path = os.path.join(folder, "f.npz")
        ef.write_frame(path, {1: np.arange(4, dtype=np.float64)})
        assert ef.read_frame(path)[1].dtype == np.float32

    def test_the_folder_is_created(self, folder):
        path = os.path.join(folder, "embeddings_t", "non_geo", "f.npz")
        ef.write_frame(path, {1: _vector(0)})
        assert os.path.isfile(path)

    def test_rewriting_a_frame_replaces_it(self, folder):
        path = os.path.join(folder, "f.npz")
        ef.write_frame(path, {1: _vector(0), 2: _vector(10)})
        ef.write_frame(path, {3: _vector(20)})
        assert set(ef.read_frame(path)) == {3}

    def test_an_absent_file_reads_as_empty(self, folder):
        assert ef.read_frame(os.path.join(folder, "nope.npz")) == {}

    def test_a_truncated_archive_reads_as_empty(self, folder):
        """An interrupted write must mean 'needs recomputing', not a crash."""
        path = os.path.join(folder, "f.npz")
        ef.write_frame(path, {1: _vector(0)})
        with open(path, "r+b") as handle:
            handle.truncate(20)
        assert ef.read_frame(path) == {}

    def test_a_file_that_is_not_an_archive_reads_as_empty(self, folder):
        path = os.path.join(folder, "f.npz")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("not an npz")
        assert ef.read_frame(path) == {}


# ---------------------------------------------------------------------------
# Bulk reads
# ---------------------------------------------------------------------------

class TestReadVectors:

    def _populate(self, folder):
        for frame, image, ids in ((1, "a.jpg", (10, 11)), (2, "b.jpg", (12,))):
            ef.write_frame(
                ef.frame_path(folder, "t", "non_geo", frame, image),
                {i: _vector(i) for i in ids})

    def test_reads_across_frames(self, folder):
        self._populate(folder)
        vectors = ef.read_vectors(folder, "t", "non_geo", [
            {"detection_id": 10, "frame": 1, "imagefile": "a.jpg"},
            {"detection_id": 12, "frame": 2, "imagefile": "b.jpg"},
        ])
        assert set(vectors) == {10, 12}
        assert np.array_equal(vectors[12], _vector(12))

    def test_opens_each_frame_once(self, folder, monkeypatch):
        """One np.load per frame, not per detection — that is why the vectors
        are grouped by frame in the first place."""
        self._populate(folder)
        opened = []
        original = ef.read_frame
        monkeypatch.setattr(
            ef, "read_frame",
            lambda path: (opened.append(path), original(path))[1])

        ef.read_vectors(folder, "t", "non_geo", [
            {"detection_id": 10, "frame": 1, "imagefile": "a.jpg"},
            {"detection_id": 11, "frame": 1, "imagefile": "a.jpg"},
            {"detection_id": 12, "frame": 2, "imagefile": "b.jpg"},
        ])
        assert len(opened) == 2

    def test_a_detection_with_no_vector_is_simply_absent(self, folder):
        self._populate(folder)
        vectors = ef.read_vectors(folder, "t", "non_geo", [
            {"detection_id": 10, "frame": 1, "imagefile": "a.jpg"},
            {"detection_id": 99, "frame": 1, "imagefile": "a.jpg"},
        ])
        assert set(vectors) == {10}

    def test_a_missing_frame_file_is_not_fatal(self, folder):
        self._populate(folder)
        vectors = ef.read_vectors(folder, "t", "non_geo", [
            {"detection_id": 10, "frame": 1, "imagefile": "a.jpg"},
            {"detection_id": 50, "frame": 9, "imagefile": "gone.jpg"},
        ])
        assert set(vectors) == {10}

    def test_present_ids_reconciles_the_store_against_the_files(self, folder):
        """A .npz deleted by hand must be noticed, not believed."""
        self._populate(folder)
        os.remove(ef.frame_path(folder, "t", "non_geo", 2, "b.jpg"))

        wanted = [
            {"detection_id": 10, "frame": 1, "imagefile": "a.jpg"},
            {"detection_id": 11, "frame": 1, "imagefile": "a.jpg"},
            {"detection_id": 12, "frame": 2, "imagefile": "b.jpg"},
        ]
        assert ef.present_ids(folder, "t", "non_geo", wanted) == [10, 11]

    def test_projections_do_not_read_each_others_vectors(self, folder):
        self._populate(folder)
        wanted = [{"detection_id": 10, "frame": 1, "imagefile": "a.jpg"}]
        assert ef.read_vectors(folder, "t", "geo_2k", wanted) == {}

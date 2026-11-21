# -*- coding: utf-8 -*-
"""A custom embedding model (core.custom_models, kind "embedder"): a class
of the user's own in place of DINOv3 in step C1."""
import os

import numpy as np
import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import (
    classification, classification_store, custom_models as cm,
    embedding_files)

from .test_embeddings_step import _config, flight  # noqa: F401 - fixture

EMBEDDER = '''
BAMBI_API = 1
import numpy as np

SEEN = []


class Mean:
    BAMBI_KIND = "embedder"

    def setup(self, context):
        self.camera = context.camera
        return dict(context.options.get("info") or {})

    def run(self, crops, items):
        SEEN.append((len(crops), [item.get("detection_id") for item in items],
                     [crop.shape for crop in crops]))
        return np.array([[crop.mean(), crop.std(), len(crops), 1.0]
                         for crop in crops], dtype=np.float32)
'''


def write(tmp_path, source=EMBEDDER, name="mean_embedder.py"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return str(path)


def opened(path, **options):
    context = cm.Context(kind=cm.EMBEDDER, camera=cm.THERMAL)
    return cm.open_implementation(
        {"path": path, "class": "", "options": options, "source": cm.SOURCE},
        cm.EMBEDDER, context)


def crops(count, size=32):
    return [np.full((size, size, 3), 10 * (index + 1), np.uint8)
            for index in range(count)]


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------

class TestInterface:
    def test_an_embedder_is_a_kind_of_its_own(self, tmp_path):
        path = write(tmp_path)

        assert cm.EMBEDDER in cm.KINDS
        assert cm.find_classes(path, cm.EMBEDDER) == ["Mean"]
        # it says what it is, so it is not offered as a detector
        assert cm.find_classes(path, cm.DETECTOR) == []

    def test_one_vector_per_crop(self, tmp_path):
        impl = opened(write(tmp_path))

        vectors = impl.embed(crops(3))

        assert vectors.shape == (3, 4)
        assert vectors.dtype == np.float32
        assert vectors[:, 0].tolist() == [10.0, 20.0, 30.0]
        assert impl.dim == 4

    def test_it_may_say_its_width_and_its_batch(self, tmp_path):
        impl = opened(write(tmp_path), info={"dim": 4, "batch_size": 8})

        assert (impl.dim, impl.batch_size) == (4, 8)

    def test_a_declared_width_that_is_not_kept_is_refused(self, tmp_path):
        impl = opened(write(tmp_path), info={"dim": 7})

        with pytest.raises(cm.CustomModelError, match="4-d vectors, not"):
            impl.embed(crops(1))

    @pytest.mark.parametrize("info", [{"dim": "wide"}, {"batch_size": -1}])
    def test_a_width_that_is_no_number_is_refused(self, tmp_path, info):
        with pytest.raises(cm.CustomModelError, match="dim|batch_size"):
            opened(write(tmp_path), info=info)

    @pytest.mark.parametrize("returned, complaint", [
        ("np.zeros((len(crops) + 1, 4))", "vector.s. for"),
        ("np.zeros((len(crops), 2, 2))", "one vector per crop - not one of shape"),
        ("np.full((len(crops), 4), np.nan)", "NaN or infinite"),
        ("[[1.0, 2.0], [1.0]][:len(crops)]", "one vector per crop"),
        ("np.zeros((len(crops), 0))", "empty vectors"),
    ])
    def test_what_does_not_look_like_vectors_is_refused(
            self, tmp_path, returned, complaint):
        source = EMBEDDER.replace(
            "return np.array([[crop.mean(), crop.std(), len(crops), 1.0]\n"
            "                         for crop in crops], dtype=np.float32)",
            f"return {returned}")
        impl = opened(write(tmp_path, source))

        with pytest.raises(cm.CustomModelError, match=complaint):
            impl.embed(crops(2))

    def test_a_torch_tensor_is_taken_too(self):
        class Tensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.ones((2, 3), np.float64)

        assert cm.normalise_embeddings(Tensor(), 2).dtype == np.float32

    def test_who_made_a_run(self):
        assert cm.is_custom_backbone("custom:mean.py:Mean@12345678")
        assert not cm.is_custom_backbone(
            "facebook/dinov3-vith16plus-pretrain-lvd1689m")
        assert not cm.is_custom_backbone(None)


class TestAsABackbone:
    def test_it_finds_out_its_width_with_a_black_crop(self, tmp_path):
        embedder = cm.Embedder(opened(write(tmp_path)), crop_size=48)
        assert embedder.dim == 0

        embedder.load()

        assert embedder.dim == 4
        assert embedder.model_id.startswith("custom:mean_embedder.py:Mean@")

    def test_a_declared_width_needs_no_trial(self, tmp_path):
        source = EMBEDDER.replace("SEEN.append", "raise RuntimeError('no'); "
                                  "SEEN.append")
        embedder = cm.Embedder(opened(write(tmp_path, source),
                                      info={"dim": 4}))

        embedder.load()                 # run() would have raised

        assert embedder.dim == 4


# ---------------------------------------------------------------------------
# Step C1
# ---------------------------------------------------------------------------

def run_with(root, path, **options):
    logs = []
    config = _config(root, custom_implementations={"embedder": {
        "path": path, "class": "", "options": options,
        "source": cm.SOURCE}})
    BambiProcessor().run_embeddings(config, log_fn=logs.append)
    return logs


def vectors_of(root):
    return embedding_files.read_vectors(root, "t", "non_geo", [
        {"detection_id": 1, "frame": 0, "imagefile": "frame_000000.jpg"},
        {"detection_id": 2, "frame": 0, "imagefile": "frame_000000.jpg"},
        {"detection_id": 3, "frame": 1, "imagefile": "frame_000001.jpg"}])


class TestStep:
    def test_the_custom_model_makes_the_vectors(self, flight, tmp_path,  # noqa: F811
                                                monkeypatch):
        def never(*_args, **_kwargs):
            raise AssertionError("DINOv3 must not be loaded")

        monkeypatch.setattr(classification, "Backbone", never)

        logs = run_with(flight, write(tmp_path))

        vectors = vectors_of(flight)
        assert sorted(vectors) == [1, 2, 3]
        assert all(vector.shape == (4,) for vector in vectors.values())
        # the frames are a flat grey: every crop's mean is the grey
        assert vectors[1][0] == pytest.approx(100.0, abs=1.0)
        assert any("custom:mean_embedder.py:Mean@" in line for line in logs)

    def test_the_run_is_recorded_under_the_file(self, flight, tmp_path):  # noqa: F811
        path = write(tmp_path)
        run_with(flight, path)

        run = classification_store.active_embedding_run(flight, "t")

        assert run["backbone"] == cm.label(path, "Mean")
        assert run["dim"] == 4
        assert cm.is_custom_backbone(run["backbone"])

    def test_a_changed_file_starts_a_new_run(self, flight, tmp_path):  # noqa: F811
        path = write(tmp_path)
        run_with(flight, path)
        first = classification_store.active_embedding_run(flight, "t")

        write(tmp_path, EMBEDDER + "\n# changed\n")
        run_with(flight, path)

        second = classification_store.active_embedding_run(flight, "t")
        assert second["run_id"] != first["run_id"]
        assert second["backbone"] != first["backbone"]

    def test_a_second_run_embeds_nothing(self, flight, tmp_path):  # noqa: F811
        path = write(tmp_path)
        run_with(flight, path)

        logs = run_with(flight, path)

        assert any("already embedded" in line for line in logs)

    def test_it_is_told_what_each_crop_is(self, flight, tmp_path):  # noqa: F811
        import importlib.util

        path = write(tmp_path)
        run_with(flight, path, info={"dim": 4})

        seen = _seen(path)
        assert sorted(ids for _n, batch, _s in seen for ids in batch) == \
            [1, 2, 3]
        assert all(shape == (32, 32, 3)
                   for _n, _i, shapes in seen for shape in shapes)
        assert importlib.util.find_spec("numpy") is not None

    def test_it_takes_as_many_crops_at_once_as_it_asked_for(
            self, flight, tmp_path):  # noqa: F811
        path = write(tmp_path)
        run_with(flight, path, info={"dim": 4, "batch_size": 1})

        assert [count for count, _i, _s in _seen(path)] == [1, 1, 1]

    def test_dinov3_is_used_without_one(self, flight):  # noqa: F811
        BambiProcessor().run_embeddings(_config(flight))

        run = classification_store.active_embedding_run(flight, "t")
        assert run["backbone"] == "fake/backbone"


def _seen(path):
    """What the embedder's ``run`` was handed, from the loaded module."""
    import sys

    for module in list(sys.modules.values()):
        if getattr(module, "__file__", None) == path and hasattr(
                module, "SEEN"):
            return module.SEEN
    raise AssertionError("the embedder module is not loaded")


# ---------------------------------------------------------------------------
# The classifiers that read the vectors
# ---------------------------------------------------------------------------

class TestClassifiers:
    def test_a_published_classifier_cannot_read_them(self):
        config = {"classification_embedded_by":
                  "custom:mean_embedder.py:Mean@12345678"}

        with pytest.raises(ValueError, match="reads DINOv3 vectors"):
            BambiProcessor()._open_head(
                config, "species", {"model": "default"}, "thermal",
                "non_geo")

    def test_a_classifier_of_ones_own_is_given_their_width(
            self, tmp_path, monkeypatch):
        opened_with = {}

        def head(path, feature_dim=0, log_fn=None):
            opened_with["dim"] = feature_dim
            return "head"

        monkeypatch.setattr(classification, "Head", head)
        weights = tmp_path / "head.pt"
        weights.write_bytes(b"x")
        config = {"classification_embedded_by": "custom:m.py:M@1",
                  "classification_backbone_dim": 4}

        BambiProcessor()._open_head(
            config, "species", {"model": "custom", "path": str(weights)},
            "thermal", "non_geo")

        assert opened_with["dim"] == 4


# ---------------------------------------------------------------------------
# The sample
# ---------------------------------------------------------------------------

class TestSample:
    PATH = os.path.join(cm.samples_folder(), "embedder_colour_histogram.py")

    def test_it_is_shipped(self):
        assert "embedder_colour_histogram.py" in cm.sample_files()
        assert cm.find_classes(self.PATH, cm.EMBEDDER) == [
            "ColourHistogramEmbedder"]

    def test_it_embeds(self):
        impl = opened(self.PATH, bins=4)

        vectors = impl.embed(crops(2))

        assert vectors.shape == (2, 12)
        # every channel's histogram adds up to all the pixels
        assert vectors.reshape(2, 3, 4).sum(axis=2) == pytest.approx(1.0)

# -*- coding: utf-8 -*-
"""Local SAM3 segmentation through transformers.

The point of the module is that the segmentation step cannot tell which
backend ran: the local model has to hand back exactly the structure the
Roboflow branch writes. So the tests drive it with fakes standing in for
``transformers`` and check the file the step writes, not the model.
"""
import contextlib
import json
import os
import sys
import types

import numpy as np
import pytest

from bambi_wildlife_detection.core import sam3_local


# ---------------------------------------------------------------------------
# Mask -> polygon
# ---------------------------------------------------------------------------

def _square_mask(size=32, lo=8, hi=24):
    mask = np.zeros((size, size), dtype=bool)
    mask[lo:hi, lo:hi] = True
    return mask


def test_a_square_mask_becomes_one_ring():
    pytest.importorskip("cv2")
    rings = sam3_local.mask_to_polygons(_square_mask())
    assert len(rings) == 1
    xs = [x for x, _ in rings[0]]
    ys = [y for _, y in rings[0]]
    # CHAIN_APPROX_SIMPLE keeps the corners; the outline spans the square.
    assert min(xs) == 8 and max(xs) == 23
    assert min(ys) == 8 and max(ys) == 23
    assert all(isinstance(v, float) for point in rings[0] for v in point)


def test_two_regions_become_two_rings():
    pytest.importorskip("cv2")
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:10, 2:10] = 1
    mask[20:30, 20:35] = 1
    assert len(sam3_local.mask_to_polygons(mask)) == 2


def test_an_empty_mask_has_no_polygon():
    assert sam3_local.mask_to_polygons(np.zeros((8, 8), dtype=bool)) == []


def test_a_leading_channel_axis_is_tolerated():
    pytest.importorskip("cv2")
    assert len(sam3_local.mask_to_polygons(_square_mask()[None])) == 1


# ---------------------------------------------------------------------------
# Post-processed result -> Roboflow structure
# ---------------------------------------------------------------------------

def test_predictions_are_normalised_to_the_roboflow_shape():
    pytest.importorskip("cv2")
    masks = np.stack([_square_mask(), np.zeros((32, 32), dtype=bool)])
    processed = {
        "masks": masks,
        "scores": np.array([0.91, 0.42]),
        "boxes": np.array([[8, 8, 24, 24], [0, 0, 0, 0]], dtype=float),
    }
    predictions = sam3_local.normalise_predictions(processed)

    # The empty mask yields nothing, as a Roboflow response would not have
    # listed it.
    assert len(predictions) == 1
    assert predictions[0]["confidence"] == pytest.approx(0.91)
    assert predictions[0]["box"] == [8.0, 8.0, 24.0, 24.0]
    assert isinstance(predictions[0]["polygons"][0][0], list)


def test_missing_masks_yield_no_predictions():
    assert sam3_local.normalise_predictions({}) == []


def test_tensors_are_accepted():
    """Anything with detach/cpu/numpy, so real torch output works."""
    pytest.importorskip("cv2")

    class Tensor:
        def __init__(self, array):
            self.array = array

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    processed = {"masks": Tensor(_square_mask()[None]),
                 "scores": Tensor(np.array([0.7]))}
    predictions = sam3_local.normalise_predictions(processed)
    assert len(predictions) == 1 and "box" not in predictions[0]


# ---------------------------------------------------------------------------
# transformers without SAM3
# ---------------------------------------------------------------------------

def test_an_old_transformers_is_named_with_the_needed_version(monkeypatch):
    fake = types.ModuleType("transformers")
    fake.__version__ = "4.57.6"
    monkeypatch.setitem(sys.modules, "transformers", fake)
    with pytest.raises(sam3_local.Sam3LocalError, match="4.57.6.*5.0.0"):
        sam3_local._sam3_classes()


# ---------------------------------------------------------------------------
# The model wrapper, with a fake transformers
# ---------------------------------------------------------------------------

class _FakeInputs(dict):
    def to(self, _device):
        return self


class _FakeProcessor:
    """Records the prompts asked; returns one square mask per prompt."""
    asked = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.kwargs = kwargs
        cls.model_id = model_id
        return cls()

    def __call__(self, images, text, return_tensors):
        self.asked.append(text)
        return _FakeInputs(original_sizes=np.array([[32, 32]]), text=text)

    def post_process_instance_segmentation(self, outputs, threshold,
                                           mask_threshold, target_sizes):
        return [{"masks": _square_mask()[None],
                 "scores": np.array([0.8]),
                 "boxes": np.array([[8, 8, 24, 24]], dtype=float)}]


class _FakeModel:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        return {"seen": inputs}


@pytest.fixture
def fake_transformers(monkeypatch):
    _FakeProcessor.asked = []
    monkeypatch.setattr(sam3_local, "_sam3_classes",
                        lambda: (_FakeModel, _FakeProcessor))
    torch = types.ModuleType("torch")
    torch.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        "bambi_wildlife_detection.core.classification.resolve_device",
        lambda preference="auto": "cpu")
    return _FakeProcessor


def test_the_model_is_asked_once_per_prompt(fake_transformers, tmp_path):
    pytest.importorskip("cv2")
    logs = []
    model = sam3_local.LocalSam3(models_dir=str(tmp_path), token="hf_x",
                                 log_fn=logs.append)
    results = model.segment(object(), ["deer", "wild boar"], confidence=0.3)

    assert fake_transformers.asked == ["deer", "wild boar"]
    assert [r["prompt"] for r in results] == ["deer", "wild boar"]
    assert results[0]["predictions"][0]["confidence"] == pytest.approx(0.8)
    assert "polygons" in results[0]["predictions"][0]
    assert any("ready" in line for line in logs)


def test_the_token_and_shared_cache_reach_from_pretrained(fake_transformers,
                                                          tmp_path):
    model = sam3_local.LocalSam3(model_id="", models_dir=str(tmp_path),
                                 token="hf_secret")
    model.load()
    assert fake_transformers.model_id == sam3_local.DEFAULT_SAM3_REPO
    assert fake_transformers.kwargs["token"] == "hf_secret"
    # Same cache the DINOv3 backbone uses, so one download serves everything.
    assert fake_transformers.kwargs["cache_dir"] == os.path.join(
        str(tmp_path), "hf_cache")


def test_a_pinned_revision_is_split_off(fake_transformers, tmp_path):
    model = sam3_local.LocalSam3(model_id="me/sam3-finetune@abc123")
    model.load()
    assert fake_transformers.model_id == "me/sam3-finetune"
    assert fake_transformers.kwargs["revision"] == "abc123"


def test_a_gated_refusal_is_explained(monkeypatch, tmp_path):
    class Refusing:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            raise RuntimeError("401 Client Error: gated repo")

    monkeypatch.setattr(sam3_local, "_sam3_classes",
                        lambda: (Refusing, Refusing))
    monkeypatch.setattr(
        "bambi_wildlife_detection.core.classification.resolve_device",
        lambda preference="auto": "cpu")
    with pytest.raises(sam3_local.Sam3LocalError, match="gated"):
        sam3_local.LocalSam3(models_dir=str(tmp_path)).load()


# ---------------------------------------------------------------------------
# The processing step
# ---------------------------------------------------------------------------

def _project(tmp_path, frames=2):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "frames_t"))
    images = []
    for index in range(frames):
        name = f"frame_{index:06d}.jpg"
        with open(os.path.join(root, "frames_t", name), "wb") as fh:
            fh.write(b"not really a jpeg")
        images.append({"imagefile": name, "timestamp": ""})
    with open(os.path.join(root, "poses_t.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": images}, fh)
    return root


class _FakeLocalSam3:
    """Stands in for the model inside the processing step."""
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.loaded = False
        self.paths = []
        _FakeLocalSam3.instances.append(self)

    def load(self):
        self.loaded = True

    def segment_path(self, image_path, prompts, confidence):
        self.paths.append(os.path.basename(image_path))
        return [{"prompt": prompt,
                 "predictions": [{"confidence": 0.9,
                                  "polygons": [[[1.0, 1.0], [5.0, 1.0],
                                                [5.0, 5.0]]]}]}
                for prompt in prompts]


def test_the_step_runs_locally_without_a_roboflow_key(monkeypatch, tmp_path):
    from bambi_wildlife_detection.bambi_processing import BambiProcessor

    _FakeLocalSam3.instances = []
    monkeypatch.setattr(sam3_local, "LocalSam3", _FakeLocalSam3)
    root = _project(tmp_path)
    logs = []

    BambiProcessor().run_sam3_segmentation({
        "target_folder": root,
        "sam3_local": True,
        "sam3_api_key": "",
        "hf_token": "hf_token",
        "sam3_model": "",
        "classification_device": "cpu",
        "sam3_prompts": ["deer", "boar"],
        "sam3_confidence": 0.4,
    }, log_fn=logs.append)

    model = _FakeLocalSam3.instances[0]
    assert model.loaded
    assert model.kwargs["token"] == "hf_token"
    assert model.kwargs["device"] == "cpu"
    assert model.paths == ["frame_000000.jpg", "frame_000001.jpg"]

    with open(os.path.join(root, "segmentation_t",
                           "segmentation_pixel.json"), encoding="utf-8") as fh:
        results = json.load(fh)
    assert [r["frame_idx"] for r in results] == [0, 1]
    assert [p["prompt"] for p in results[0]["prompts"]] == ["deer", "boar"]
    assert results[0]["prompts"][0]["predictions"][0]["polygons"]
    assert any("local transformers model" in line for line in logs)


def test_the_roboflow_backend_still_insists_on_its_key(tmp_path):
    from bambi_wildlife_detection.bambi_processing import BambiProcessor

    with pytest.raises(ValueError, match="Roboflow API key"):
        BambiProcessor().run_sam3_segmentation({
            "target_folder": _project(tmp_path),
            "sam3_local": False,
            "sam3_api_key": "",
            "sam3_prompts": ["deer"],
        })


def test_a_model_that_fails_to_load_stops_before_the_frames(monkeypatch,
                                                            tmp_path):
    """One clear error up front, not one warning per frame."""
    from bambi_wildlife_detection.bambi_processing import BambiProcessor

    class Broken(_FakeLocalSam3):
        def load(self):
            raise sam3_local.Sam3LocalError("facebook/sam3 is gated")

    monkeypatch.setattr(sam3_local, "LocalSam3", Broken)
    with pytest.raises(sam3_local.Sam3LocalError, match="gated"):
        BambiProcessor().run_sam3_segmentation({
            "target_folder": _project(tmp_path),
            "sam3_local": True,
            "sam3_prompts": ["deer"],
        })

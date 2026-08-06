# -*- coding: utf-8 -*-
"""The DINOv3 backbone wrapper, against fake torch/transformers.

Neither library is installed in the unit tier — and both are heavy enough that
requiring them would make this suite unrunnable on a laptop — so the contract
is exercised through stand-ins. What is being tested is our plumbing: that the
CLS vector is taken from the right place, that inference runs under
``no_grad``, and that the cache is redirected without touching the environment
of the whole QGIS process.
"""
import contextlib
import os
import sys
import types

import numpy as np
import pytest

from bambi_wildlife_detection.core import classification as cl


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeTensor:
    def __init__(self, array, dtype="float32"):
        self.array = np.asarray(array)
        self.dtype = dtype

    def to(self, _device):
        return self

    def half(self):
        return FakeTensor(self.array, dtype="float16")

    def detach(self):
        return self

    def float(self):
        return FakeTensor(self.array, dtype="float32")

    def numpy(self):
        return self.array

    def __getitem__(self, item):
        return FakeTensor(self.array[item], self.dtype)


class FakeModel:
    def __init__(self, hidden_size=1280, pooler=True):
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.calls = []
        self.moved_to = None
        self.halved = False
        self.evalled = False
        self._pooler = pooler
        self._hidden = hidden_size

    def to(self, device):
        self.moved_to = device
        return self

    def half(self):
        self.halved = True
        return self

    def eval(self):
        self.evalled = True
        return self

    def __call__(self, **inputs):
        self.calls.append(inputs)
        count = len(inputs["pixel_values"].array)
        cls = FakeTensor(np.ones((count, self._hidden), dtype=np.float32))
        sequence = FakeTensor(
            np.full((count, 5, self._hidden), 7.0, dtype=np.float32))
        return types.SimpleNamespace(
            pooler_output=cls if self._pooler else None,
            last_hidden_state=sequence)


def _install_fakes(monkeypatch, model=None, cuda=False, record=None):
    torch = types.ModuleType("torch")
    torch.float32 = "float32"

    @contextlib.contextmanager
    def no_grad():
        if record is not None:
            record.append("no_grad")
        yield

    torch.no_grad = no_grad
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    monkeypatch.setitem(sys.modules, "torch", torch)

    the_model = model if model is not None else FakeModel()

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            if record is not None:
                record.append(("model", model_id, kwargs))
            return the_model

    class AutoImageProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            if record is not None:
                record.append(("processor", model_id, kwargs))

            def process(images, return_tensors=None):
                return {"pixel_values": FakeTensor(np.zeros((len(images), 3)))}
            return process

    transformers = types.ModuleType("transformers")
    transformers.AutoModel = AutoModel
    transformers.AutoImageProcessor = AutoImageProcessor
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return the_model


@pytest.fixture
def crops():
    return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(3)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoading:

    def test_the_cache_goes_to_the_shared_models_folder(self, monkeypatch,
                                                        tmp_path):
        record = []
        _install_fakes(monkeypatch, record=record)
        backbone = cl.Backbone(models_dir=str(tmp_path), device="cpu")
        backbone.load()

        for kind, _model_id, kwargs in record:
            if kind in ("model", "processor"):
                assert kwargs["cache_dir"].endswith("hf_cache")

    def test_hf_home_is_never_touched(self, monkeypatch, tmp_path):
        """Exporting HF_HOME would relocate the cache for everything else in
        the QGIS process that uses Hugging Face."""
        monkeypatch.delenv("HF_HOME", raising=False)
        _install_fakes(monkeypatch)
        cl.Backbone(models_dir=str(tmp_path), device="cpu").load()
        assert "HF_HOME" not in os.environ

    def test_the_token_is_passed_through(self, monkeypatch, tmp_path):
        record = []
        _install_fakes(monkeypatch, record=record)
        cl.Backbone(models_dir=str(tmp_path), token="hf_x",
                    device="cpu").load()
        assert all(kwargs["token"] == "hf_x"
                   for kind, _m, kwargs in record if kind == "model")

    def test_an_empty_token_becomes_none(self, monkeypatch, tmp_path):
        record = []
        _install_fakes(monkeypatch, record=record)
        cl.Backbone(models_dir=str(tmp_path), token="", device="cpu").load()
        assert all(kwargs["token"] is None
                   for kind, _m, kwargs in record if kind == "model")

    def test_loading_twice_loads_once(self, monkeypatch, tmp_path):
        record = []
        _install_fakes(monkeypatch, record=record)
        backbone = cl.Backbone(models_dir=str(tmp_path), device="cpu")
        backbone.load()
        backbone.load()
        # ~840 M parameters; paying for it twice would dominate the run.
        assert sum(1 for kind, *_ in record if kind == "model") == 1

    def test_the_model_is_put_in_eval_mode(self, monkeypatch, tmp_path):
        model = _install_fakes(monkeypatch)
        cl.Backbone(models_dir=str(tmp_path), device="cpu").load()
        assert model.evalled and model.moved_to == "cpu"

    def test_half_precision_only_on_the_gpu(self, monkeypatch, tmp_path):
        model = _install_fakes(monkeypatch, cuda=True)
        backbone = cl.Backbone(models_dir=str(tmp_path), device="auto",
                               fp16=True)
        backbone.load()
        assert backbone.device == "cuda" and model.halved

        cpu_model = _install_fakes(monkeypatch, model=FakeModel(), cuda=False)
        cl.Backbone(models_dir=str(tmp_path), device="cpu", fp16=True).load()
        assert not cpu_model.halved

    def test_requesting_cuda_without_it_is_an_error_not_a_silent_cpu_run(
            self, monkeypatch):
        _install_fakes(monkeypatch, cuda=False)
        with pytest.raises(cl.BackboneError) as caught:
            cl.resolve_device("cuda")
        assert "GPU support" in str(caught.value)

    def test_a_load_failure_is_wrapped_with_advice(self, monkeypatch,
                                                   tmp_path):
        _install_fakes(monkeypatch)
        transformers = sys.modules["transformers"]

        def _boom(model_id, **kwargs):
            raise RuntimeError("401 Client Error: gated repo")

        transformers.AutoImageProcessor.from_pretrained = staticmethod(_boom)
        with pytest.raises(cl.BackboneError) as caught:
            cl.Backbone(models_dir=str(tmp_path), device="cpu").load()
        assert "Request access" in str(caught.value)

    def test_a_pinned_revision_is_passed_through(self, monkeypatch, tmp_path):
        """An unpinned model resolves to whatever `main` holds today, which
        would change the features under a user's saved results."""
        record = []
        _install_fakes(monkeypatch, record=record)
        cl.Backbone(model_id="facebook/x@abc123", models_dir=str(tmp_path),
                    device="cpu").load()

        for kind, model_id, kwargs in record:
            assert model_id == "facebook/x"
            assert kwargs["revision"] == "abc123"

    def test_an_unpinned_model_sends_no_revision(self, monkeypatch, tmp_path):
        record = []
        _install_fakes(monkeypatch, record=record)
        cl.Backbone(model_id="facebook/x", models_dir=str(tmp_path),
                    device="cpu").load()
        assert all("revision" not in kwargs for _k, _m, kwargs in record)

    @pytest.mark.parametrize("value,repo,revision", [
        ("facebook/x", "facebook/x", ""),
        ("facebook/x@abc", "facebook/x", "abc"),
        ("  facebook/x@abc  ", "facebook/x", "abc"),
        ("facebook/x@refs/pr/1", "facebook/x", "refs/pr/1"),
        ("", "", ""),
    ])
    def test_revision_splitting(self, value, repo, revision):
        assert cl.split_revision(value) == (repo, revision)

    def test_missing_transformers_is_reported_usefully(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(cl.BackboneError) as caught:
            cl.Backbone(device="cpu").load()
        assert "Dependency Manager" in str(caught.value)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class TestEmbed:

    def test_returns_one_float32_vector_per_crop(self, monkeypatch, crops,
                                                 tmp_path):
        _install_fakes(monkeypatch)
        vectors = cl.Backbone(models_dir=str(tmp_path),
                              device="cpu").embed(crops)
        assert vectors.shape == (3, 1280)
        assert vectors.dtype == np.float32

    def test_inference_runs_under_no_grad(self, monkeypatch, crops, tmp_path):
        """Without it the autograd graph grows across the whole flight and the
        run dies on memory rather than on time."""
        record = []
        _install_fakes(monkeypatch, record=record)
        cl.Backbone(models_dir=str(tmp_path), device="cpu").embed(crops)
        assert "no_grad" in record

    def test_the_cls_token_comes_from_the_pooler(self, monkeypatch, crops,
                                                 tmp_path):
        _install_fakes(monkeypatch, model=FakeModel(pooler=True))
        vectors = cl.Backbone(models_dir=str(tmp_path),
                              device="cpu").embed(crops)
        assert (vectors == 1.0).all()

    def test_without_a_pooler_the_first_sequence_position_is_used(
            self, monkeypatch, crops, tmp_path):
        """Not every checkpoint exposes a pooler; the CLS token is position 0
        either way, and that is what the published heads were trained on."""
        _install_fakes(monkeypatch, model=FakeModel(pooler=False))
        vectors = cl.Backbone(models_dir=str(tmp_path),
                              device="cpu").embed(crops)
        assert vectors.shape == (3, 1280)
        assert (vectors == 7.0).all()

    def test_an_empty_batch_returns_an_empty_array(self, monkeypatch,
                                                   tmp_path):
        _install_fakes(monkeypatch)
        vectors = cl.Backbone(models_dir=str(tmp_path), device="cpu").embed([])
        assert vectors.shape == (0, 1280)

    def test_the_width_follows_the_loaded_model(self, monkeypatch, crops,
                                                tmp_path):
        _install_fakes(monkeypatch, model=FakeModel(hidden_size=768))
        backbone = cl.Backbone(models_dir=str(tmp_path), device="cpu")
        vectors = backbone.embed(crops)
        assert backbone.dim == 768
        assert vectors.shape == (3, 768)

# -*- coding: utf-8 -*-
"""The DINOv3 backbone wrapper, against fake torch/transformers.

Neither library is installed in the unit tier - and both are heavy enough that
requiring them would make this suite unrunnable on a laptop - so the contract
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


class TestTransformersImportMessage:
    """The loader must say what actually broke: transformers imports lazily and
    a failure deep in its tail (2026-09-07: a user-site pyarrow whose DLL could
    not load inside QGIS) was reported as "transformers is not installed"."""

    def _message(self, exc):
        from bambi_wildlife_detection.core import classification
        return classification._transformers_import_message(exc)

    def test_missing_transformers_says_not_installed(self):
        exc = ModuleNotFoundError("No module named 'transformers'",
                                  name="transformers")
        assert "not installed" in self._message(exc)

    def test_broken_dependency_names_the_culprit(self):
        root = ImportError("DLL load failed while importing lib: The specified "
                           "procedure could not be found.", name="pyarrow.lib")
        try:
            try:
                raise root
            except ImportError as inner:
                raise ImportError("could not import module") from inner
        except ImportError as outer:
            message = self._message(outer)
        assert "not installed" not in message
        assert "pyarrow.lib" in message
        assert "procedure could not be found" in message
        assert "shadows" in message
        assert "Dependency Manager" in message

    def test_other_import_errors_keep_their_text(self):
        exc = ImportError("numpy.core.multiarray failed to import")
        message = self._message(exc)
        assert "could not be imported" in message
        assert "multiarray" in message
        assert "shadows" not in message


class TestLoadProgressLogging:
    """Between "Loading ..." and "Backbone ready" a first run downloads
    gigabytes in silence (2026-09-07). The load now says whether it is
    downloading or loading, and how far the download is."""

    def test_repo_folder_follows_the_hub_naming(self, tmp_path):
        folder = cl.repo_cache_folder(str(tmp_path), "facebook/dinov3-vith16plus")
        assert os.path.basename(folder) == "models--facebook--dinov3-vith16plus"
        assert cl.repo_cache_folder("", "facebook/x") == ""

    def test_cached_bytes_counts_partial_downloads(self, tmp_path):
        folder = tmp_path / "models--a--b"
        (folder / "blobs").mkdir(parents=True)
        (folder / "blobs" / "abc").write_bytes(b"x" * 100)
        (folder / "blobs" / "def.incomplete").write_bytes(b"y" * 50)
        assert cl.cached_bytes(str(folder)) == 150
        # On Windows the finished file lives under snapshots/ instead.
        (folder / "snapshots" / "rev").mkdir(parents=True)
        (folder / "snapshots" / "rev" / "model.safetensors").write_bytes(b"z" * 30)
        assert cl.cached_bytes(str(folder)) == 180
        assert cl.cached_bytes(str(tmp_path / "missing")) == 0
        assert cl.cached_bytes("") == 0

    def test_reporter_logs_growth_with_a_percentage(self, tmp_path):
        folder = tmp_path / "models--a--b" / "blobs"
        folder.mkdir(parents=True)
        logs = []
        reporter = cl.DownloadReporter(str(tmp_path), "a/b", 4 * 1024 * 1024,
                                       log_fn=logs.append, interval=0.01)
        reporter.report()                              # nothing on disk yet
        assert logs == []
        (folder / "w.incomplete").write_bytes(b"0" * (1024 * 1024))
        reporter.report()
        reporter.report()                              # unchanged: no repeat
        assert logs == ["Downloading weights: 1 MB of 4 MB (25%)"]
        (folder / "w.incomplete").write_bytes(b"0" * (4 * 1024 * 1024))
        reporter.report()
        assert logs[-1] == "Downloading weights: 4 MB of 4 MB (100%)"

    def test_reporter_is_silent_when_the_cache_is_warm(self, tmp_path):
        logs = []
        with cl.DownloadReporter(str(tmp_path), "a/b", None, log_fn=logs.append,
                                 active=False, interval=0.01) as reporter:
            assert reporter._thread is None
        assert logs == []

    def test_reporter_thread_samples_while_loading(self, tmp_path):
        import time as _time
        folder = tmp_path / "models--a--b" / "blobs"
        folder.mkdir(parents=True)
        logs = []
        with cl.DownloadReporter(str(tmp_path), "a/b", None, log_fn=logs.append,
                                 interval=0.02):
            (folder / "w.incomplete").write_bytes(b"0" * (2 * 1024 * 1024))
            deadline = _time.time() + 2.0
            while not logs and _time.time() < deadline:
                _time.sleep(0.02)
        assert logs and logs[0].startswith("Downloading weights: 2 MB")

    def test_describe_cache_without_the_hub_says_downloading(self, monkeypatch,
                                                              tmp_path):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        logs = []
        state = cl.describe_cache("a/b", str(tmp_path), "", None,
                                  log_fn=logs.append)
        assert state.complete is False
        assert len(logs) == 1 and "not in the local cache" in logs[0]
        assert "several GB" in logs[0]

    def test_load_logs_the_phases(self, monkeypatch, tmp_path):
        logs = []
        _install_fakes(monkeypatch)
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        cl.Backbone(models_dir=str(tmp_path), device="cpu",
                    log_fn=logs.append).load()
        text = "\n".join(logs)
        assert "not in the local cache" in text
        assert "Image processor ready" in text
        assert "Weights loaded in" in text
        assert "Backbone ready" in text and "on cpu" in text


# ---------------------------------------------------------------------------
# Precision: bf16 over fp16, and no NaN leaves the backbone
# ---------------------------------------------------------------------------

class PrecisionModel(FakeModel):
    """A model whose output the test controls per call, and which records
    every dtype it is cast to."""

    def __init__(self, outputs, hidden_size=4):
        super().__init__(hidden_size=hidden_size)
        self._outputs = list(outputs)
        self.casts = []
        self.floated = 0
        self.raise_on_float = False

    def to(self, target):
        self.casts.append(target)
        if isinstance(target, str):
            self.moved_to = target
        return self

    def float(self):
        # Out of memory only on the GPU; the CPU copy always fits.
        if self.raise_on_float and self.moved_to != "cpu":
            raise RuntimeError("CUDA out of memory")
        self.floated += 1
        return self

    def __call__(self, **inputs):
        rows = self._outputs.pop(0)
        return types.SimpleNamespace(pooler_output=FakeTensor(np.asarray(rows)))


def _with_bf16(monkeypatch, supported):
    import sys as _sys
    torch = _sys.modules["torch"]
    torch.cuda.is_bf16_supported = lambda: supported
    torch.bfloat16 = "bfloat16"
    torch.cuda.empty_cache = lambda: None


class TestPrecision:

    def test_bf16_is_preferred_where_the_gpu_supports_it(self, monkeypatch, tmp_path):
        model = PrecisionModel([np.ones((2, 4))])
        _install_fakes(monkeypatch, model=model, cuda=True)
        _with_bf16(monkeypatch, True)
        backbone = cl.Backbone(models_dir=str(tmp_path), device="cuda", fp16=True)
        backbone.load()
        assert "bfloat16" in model.casts and not model.halved
        assert backbone.precision == "bf16"

    def test_fp16_when_bf16_is_not_available(self, monkeypatch, tmp_path):
        model = PrecisionModel([np.ones((2, 4))])
        _install_fakes(monkeypatch, model=model, cuda=True)
        _with_bf16(monkeypatch, False)
        backbone = cl.Backbone(models_dir=str(tmp_path), device="cuda", fp16=True)
        backbone.load()
        assert model.halved and backbone.precision == "fp16"

    def test_a_non_finite_batch_is_redone_in_fp32(self, monkeypatch, tmp_path, crops):
        nan_rows = np.full((3, 4), np.nan)
        nan_rows[0] = 1.0
        model = PrecisionModel([nan_rows, np.ones((3, 4))])
        _install_fakes(monkeypatch, model=model, cuda=True)
        _with_bf16(monkeypatch, False)
        logs = []
        backbone = cl.Backbone(models_dir=str(tmp_path), device="cuda",
                               fp16=True, log_fn=logs.append)
        out = backbone.embed(crops)
        assert np.isfinite(out).all() and out.shape == (3, 4)
        assert model.floated == 1 and backbone.precision == "fp32"
        assert any("2 of 3 feature vector(s) came out non-finite in fp16" in line
                   and "fp32 on cuda" in line for line in logs)
        # Later batches stay in fp32: no second demotion, no NaN.
        model._outputs.append(np.ones((3, 4)))
        backbone.embed(crops)
        assert model.floated == 1

    def test_fp32_moves_to_the_cpu_when_the_gpu_is_full(self, monkeypatch, tmp_path, crops):
        model = PrecisionModel([np.full((3, 4), np.nan), np.ones((3, 4))])
        model.raise_on_float = True
        _install_fakes(monkeypatch, model=model, cuda=True)
        _with_bf16(monkeypatch, True)
        logs = []
        backbone = cl.Backbone(models_dir=str(tmp_path), device="cuda",
                               fp16=True, log_fn=logs.append)
        backbone.embed(crops)
        assert backbone.device == "cpu" and model.moved_to == "cpu"
        assert any("do not fit on the GPU" in line for line in logs)

    def test_nan_in_fp32_is_an_error_not_a_file(self, monkeypatch, tmp_path, crops):
        model = PrecisionModel([np.full((3, 4), np.nan)])
        _install_fakes(monkeypatch, model=model, cuda=False)
        backbone = cl.Backbone(models_dir=str(tmp_path), device="cpu", fp16=False)
        with pytest.raises(cl.BackboneError, match="non-finite even in fp32"):
            backbone.embed(crops)

# -*- coding: utf-8 -*-
"""The default models are the published ones, by commit, size and SHA-256
(core.model_pins).

They used to be fetched from the current state of their repository, and any
file over a megabyte counted as a good download: a model could change under
a survey unnoticed, and a download cut off half-way was kept and used.
"""
import hashlib
import os
import re

import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import hf_access, model_pins
from bambi_wildlife_detection.core.model_pins import Pin
from tests.fakes import FakeResponse

PUBLISHED = b"published weights " * 4000
OTHER = b"some other version" * 4000


def _pin(data=PUBLISHED, revision="0123abc"):
    return Pin(revision, len(data), hashlib.sha256(data).hexdigest())


class Web:
    """``requests.get`` that serves *data* and remembers what was asked."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.urls = []

    def __call__(self, url, **_kwargs):
        self.urls.append(url)
        answer = self.answers.pop(0) if len(self.answers) > 1 \
            else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return FakeResponse(chunks=[answer[:1000], answer[1000:]])


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

def test_every_pin_is_a_commit_a_size_and_a_checksum():
    assert model_pins.REVISIONS
    for repo, revision in model_pins.REVISIONS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", revision), repo
    for repo, files in model_pins._FILES.items():
        assert repo in model_pins.REVISIONS, repo
        for path, (size, sha256) in files.items():
            assert size > 100000, (repo, path)
            assert re.fullmatch(r"[0-9a-f]{64}", sha256), (repo, path)
    assert re.fullmatch(r"[0-9a-f]{64}", model_pins.OSNET_X0_25_MSMT17.sha256)


def test_every_default_model_the_plugin_downloads_is_pinned():
    from bambi_wildlife_detection.core import sam3_local

    for spec in BambiProcessor.DEFAULT_MODELS.values():
        assert model_pins.pin_of(spec["repo"], spec["remote_path"]), spec
    for task, repo in hf_access.DEFAULT_HEAD_REPOS.items():
        if not repo:
            continue
        for projection in ("non_geo", "geo_1k", "geo_2k"):
            for modality in ("thermal", "rgb", "matched"):
                remote = hf_access.head_repo_path(task, projection, modality)
                assert model_pins.pin_of(repo, remote), (repo, remote)
    assert model_pins.revision_of(hf_access.DEFAULT_BACKBONE)
    assert model_pins.revision_of(sam3_local.DEFAULT_SAM3_REPO)


def test_a_pinned_file_is_fetched_at_its_commit():
    repo = "cpraschl/bambi-thermal-detection"
    url = model_pins.resolve_url(repo, "thermal_animal_detector.pt")
    assert url == (f"https://huggingface.co/{repo}/resolve/"
                   f"{model_pins.REVISIONS[repo]}/thermal_animal_detector.pt")
    assert "/resolve/main/" not in url


def test_a_repository_of_ones_own_is_not_pinned():
    assert model_pins.revision_of("me/my-detector") == ""
    assert model_pins.pin_of("me/my-detector", "best.pt") is None
    assert model_pins.resolve_url("me/my-detector", "best.pt").endswith(
        "/me/my-detector/resolve/main/best.pt")
    # ... nor a file added to one of ours after this release
    assert model_pins.pin_of("cpraschl/bambi-models", "new/best.pt") is None


# ---------------------------------------------------------------------------
# Checking a file
# ---------------------------------------------------------------------------

def test_the_published_file_passes(tmp_path):
    path = tmp_path / "m.pt"
    path.write_bytes(PUBLISHED)
    assert model_pins.difference(str(path), _pin()) is None
    model_pins.verify(str(path), _pin())


@pytest.mark.parametrize("data, why", [
    (PUBLISHED[:5000], "bytes where"),                 # cut off
    (PUBLISHED[:-1] + b"X", "SHA-256"),                # same size, altered
    (OTHER, "SHA-256"),
])
def test_anything_else_does_not(tmp_path, data, why):
    path = tmp_path / "m.pt"
    path.write_bytes(data)
    assert why in model_pins.difference(str(path), _pin())
    with pytest.raises(model_pins.ModelIntegrityError,
                       match="not the published file"):
        model_pins.verify(str(path), _pin())


def test_a_missing_file_is_said_to_be_missing(tmp_path):
    assert "missing" in model_pins.difference(str(tmp_path / "x"), _pin())


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def test_a_download_is_checked_before_it_is_put_in_place(tmp_path):
    target = tmp_path / "models" / "m.pt"
    web = Web(PUBLISHED)
    assert model_pins.download(
        "https://x.example/m.pt", str(target), _pin(), get=web) == str(target)
    assert target.read_bytes() == PUBLISHED
    assert os.listdir(target.parent) == ["m.pt"]


@pytest.mark.parametrize("data", [OTHER, PUBLISHED[:9000], b"<html>404</html>"])
def test_what_is_not_the_published_file_is_not_kept(tmp_path, data):
    target = tmp_path / "m.pt"
    with pytest.raises(model_pins.ModelIntegrityError):
        model_pins.download("https://x.example/m.pt", str(target), _pin(),
                            get=Web(data))
    assert os.listdir(tmp_path) == []


def test_an_interrupted_transfer_leaves_nothing(tmp_path):
    def get(url, **_kwargs):
        response = FakeResponse(chunks=[PUBLISHED[:1000]])

        def chunks(chunk_size=None):
            yield PUBLISHED[:1000]
            raise ConnectionError("reset by peer")

        response.iter_content = chunks
        return response

    with pytest.raises(ConnectionError):
        model_pins.download("https://x.example/m.pt", str(tmp_path / "m.pt"),
                            _pin(), get=get)
    assert os.listdir(tmp_path) == []


def test_only_https_is_downloaded(tmp_path):
    for url in ("http://x.example/m.pt", "ftp://x.example/m.pt", "file:///m"):
        with pytest.raises(ValueError, match="only https"):
            model_pins.download(url, str(tmp_path / "m.pt"), get=Web(b""))


def test_without_a_pin_the_size_is_all_there_is(tmp_path):
    target = tmp_path / "m.pt"
    model_pins.download("https://x.example/m.pt", str(target), None,
                        min_size=1000, get=Web(OTHER))
    assert target.read_bytes() == OTHER
    with pytest.raises(model_pins.ModelIntegrityError, match="at least"):
        model_pins.download("https://x.example/m.pt", str(tmp_path / "n.pt"),
                            None, min_size=1000, get=Web(b"tiny"))


# ---------------------------------------------------------------------------
# The file on this computer
# ---------------------------------------------------------------------------

def test_the_published_file_is_used_without_a_download(tmp_path):
    target = tmp_path / "m.pt"
    target.write_bytes(PUBLISHED)
    web = Web(OTHER)
    assert model_pins.ensure(str(target), "https://x.example/m.pt", _pin(),
                             get=web) == str(target)
    assert web.urls == []


def test_another_version_is_replaced_and_kept_aside(tmp_path):
    target = tmp_path / "m.pt"
    target.write_bytes(OTHER)
    log = []
    model_pins.ensure(str(target), "https://x.example/m.pt", _pin(),
                      what="detection model", get=Web(PUBLISHED),
                      log_fn=log.append)
    assert target.read_bytes() == PUBLISHED
    assert (tmp_path / "m.pt.previous").read_bytes() == OTHER
    assert any("not the published version" in line for line in log)


def test_without_a_network_the_file_that_was_there_is_used(tmp_path):
    target = tmp_path / "m.pt"
    target.write_bytes(OTHER)
    log = []
    assert model_pins.ensure(
        str(target), "https://x.example/m.pt", _pin(), min_size=1000,
        get=Web(ConnectionError("no route")), log_fn=log.append
    ) == str(target)
    assert target.read_bytes() == OTHER
    assert sorted(os.listdir(tmp_path)) == ["m.pt"]
    assert any("not the version this release was tested with" in line
               for line in log)


def test_a_broken_transfer_is_tried_once_more(tmp_path):
    target = tmp_path / "m.pt"
    web = Web(ConnectionError("reset"), PUBLISHED)
    log = []
    model_pins.ensure(str(target), "https://x.example/m.pt", _pin(),
                      get=web, log_fn=log.append)
    assert target.read_bytes() == PUBLISHED and len(web.urls) == 2
    assert any("trying once more" in line for line in log)


def test_a_wrong_file_is_not_asked_for_twice(tmp_path):
    web = Web(OTHER)
    with pytest.raises(model_pins.ModelIntegrityError):
        model_pins.ensure(str(tmp_path / "m.pt"), "https://x.example/m.pt",
                          _pin(), get=web)
    assert len(web.urls) == 1


def test_a_missing_model_without_a_network_is_an_error(tmp_path):
    with pytest.raises(ConnectionError):
        model_pins.ensure(str(tmp_path / "m.pt"), "https://x.example/m.pt",
                          _pin(), get=Web(ConnectionError("no route")))
    assert os.listdir(tmp_path) == []


# ---------------------------------------------------------------------------
# The detection model
# ---------------------------------------------------------------------------

@pytest.fixture
def detector(tmp_path, monkeypatch):
    """A processor whose thermal model is pinned to :data:`PUBLISHED`."""
    import requests

    spec = BambiProcessor.DEFAULT_MODELS["T"]
    monkeypatch.setattr(model_pins, "_FILES", {spec["repo"]: {
        spec["remote_path"]: (len(PUBLISHED),
                              hashlib.sha256(PUBLISHED).hexdigest())}})
    monkeypatch.setattr(BambiProcessor, "_get_default_model_dir",
                        staticmethod(lambda: str(tmp_path / "models")))
    web = Web(PUBLISHED)
    monkeypatch.setattr(requests, "get", web)
    return BambiProcessor(), web, tmp_path / "models" / spec["filename"]


def test_the_detection_model_comes_from_its_pinned_commit(detector):
    processor, web, path = detector
    assert processor.download_default_model(camera="T") == str(path)
    assert path.read_bytes() == PUBLISHED
    repo = BambiProcessor.DEFAULT_MODELS["T"]["repo"]
    assert web.urls == [
        f"https://huggingface.co/{repo}/resolve/{model_pins.REVISIONS[repo]}"
        "/thermal_animal_detector.pt"]


def test_a_half_downloaded_detection_model_is_not_used(detector):
    """Anything over a megabyte used to pass as the model."""
    processor, web, path = detector
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\0" * 1500000)
    log = []
    assert processor.download_default_model(log.append, "T") == str(path)
    assert path.read_bytes() == PUBLISHED
    assert len(web.urls) == 1


def test_a_failed_download_names_the_pinned_file(detector, monkeypatch):
    import requests

    processor, _web, _path = detector
    monkeypatch.setattr(requests, "get", Web(ConnectionError("no route")))
    with pytest.raises(RuntimeError) as caught:
        processor.download_default_model(camera="T")
    repo = BambiProcessor.DEFAULT_MODELS["T"]["repo"]
    assert f"/resolve/{model_pins.REVISIONS[repo]}/" in str(caught.value)


# ---------------------------------------------------------------------------
# The classifiers
# ---------------------------------------------------------------------------

def _download(data, calls):
    def download(repo, task, projection, modality, destination, token,
                 log_fn=None):
        calls.append((repo, task, projection, modality))
        if isinstance(data, Exception):
            raise data
        with open(destination, "wb") as handle:
            handle.write(data)
        return destination
    return download


@pytest.fixture
def head(tmp_path, monkeypatch):
    repo = hf_access.DEFAULT_HEAD_REPOS["species"]
    remote = hf_access.head_repo_path("species", "non_geo", "rgb")
    monkeypatch.setattr(model_pins, "_FILES", {repo: {remote: (
        len(PUBLISHED), hashlib.sha256(PUBLISHED).hexdigest())}})
    return repo, str(tmp_path / "species_rgb.pt")


def test_a_classifier_that_is_the_published_one_is_used(head):
    repo, path = head
    with open(path, "wb") as handle:
        handle.write(PUBLISHED)
    calls = []
    assert hf_access.ensure_head(repo, "species", "non_geo", "rgb", path,
                                 download=_download(OTHER, calls)) == path
    assert calls == []


def test_another_version_of_a_classifier_is_replaced(head):
    """The stored predictions name the model by its file name - the same
    for every version of it."""
    repo, path = head
    with open(path, "wb") as handle:
        handle.write(OTHER)
    calls, log = [], []
    hf_access.ensure_head(repo, "species", "non_geo", "rgb", path,
                          log_fn=log.append,
                          download=_download(PUBLISHED, calls))
    assert open(path, "rb").read() == PUBLISHED
    assert open(path + ".previous", "rb").read() == OTHER
    assert calls == [(repo, "species", "non_geo", "rgb")]
    assert any("not the published version" in line for line in log)


def test_offline_the_classifier_that_was_there_is_used(head):
    repo, path = head
    with open(path, "wb") as handle:
        handle.write(OTHER)
    log = []
    hf_access.ensure_head(repo, "species", "non_geo", "rgb", path,
                          log_fn=log.append,
                          download=_download(RuntimeError("offline"), []))
    assert open(path, "rb").read() == OTHER
    assert not os.path.exists(path + ".previous")
    assert any("could not be downloaded" in line for line in log)


def test_a_missing_classifier_is_downloaded_or_the_error_raised(head):
    repo, path = head
    hf_access.ensure_head(repo, "species", "non_geo", "rgb", path,
                          download=_download(PUBLISHED, []))
    assert os.path.isfile(path)
    os.remove(path)
    with pytest.raises(RuntimeError, match="offline"):
        hf_access.ensure_head(repo, "species", "non_geo", "rgb", path,
                              download=_download(RuntimeError("offline"), []))


def test_a_classifier_of_an_unpinned_repository_is_left_alone(tmp_path):
    path = str(tmp_path / "x.pt")
    with open(path, "wb") as handle:
        handle.write(OTHER)
    calls = []
    assert hf_access.ensure_head("me/heads", "species", "non_geo", "rgb",
                                 path, download=_download(b"", calls)) == path
    assert calls == []


# ---------------------------------------------------------------------------
# The models the hub library fetches
# ---------------------------------------------------------------------------

def test_the_default_backbone_and_sam3_are_pinned():
    from bambi_wildlife_detection.core import classification, sam3_local

    for repo in (hf_access.DEFAULT_BACKBONE, sam3_local.DEFAULT_SAM3_REPO):
        assert classification.resolve_revision(repo) == (
            repo, model_pins.REVISIONS[repo])


def test_a_named_revision_wins_and_main_follows_the_latest():
    from bambi_wildlife_detection.core import classification

    repo = hf_access.DEFAULT_BACKBONE
    assert classification.resolve_revision(f"{repo}@abc123") == (repo, "abc123")
    assert classification.resolve_revision(f"{repo}@main") == (repo, "main")


def test_a_model_of_ones_own_is_theirs_to_pin():
    from bambi_wildlife_detection.core import classification

    assert classification.resolve_revision("me/my-backbone") == (
        "me/my-backbone", "")

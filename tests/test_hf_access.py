# -*- coding: utf-8 -*-
"""Hugging Face token resolution, access checking and the model layout.

The gated backbone is the most likely thing to go wrong in the classification
feature, so the states it can be in are pinned here rather than discovered in
the field.
"""
import os
import sys
import types

import pytest

from bambi_wildlife_detection.core import hf_access


# ---------------------------------------------------------------------------
# Fake huggingface_hub
# ---------------------------------------------------------------------------

class _FakeGated(Exception):
    pass


class _FakeMissing(Exception):
    pass


def _install_fake_hub(monkeypatch, *, model_info=None, token=None,
                      errors_module=True):
    """Install a fake ``huggingface_hub`` package into ``sys.modules``.

    *errors_module* selects which of the two error-import paths the module
    under test has to take, so both are exercised.
    """
    hub = types.ModuleType("huggingface_hub")

    class HfApi:
        def model_info(self, repo_id, token=None):
            if model_info is None:
                return {"id": repo_id}
            return model_info(repo_id, token)

    hub.HfApi = HfApi
    hub.get_token = lambda: token

    errors = types.ModuleType(
        "huggingface_hub.errors" if errors_module else "huggingface_hub.utils")
    errors.GatedRepoError = _FakeGated
    errors.RepositoryNotFoundError = _FakeMissing

    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    if errors_module:
        monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
        # Force the fallback path to fail so the canonical one must be used.
        monkeypatch.setitem(sys.modules, "huggingface_hub.utils", None)
    else:
        monkeypatch.setitem(sys.modules, "huggingface_hub.utils", errors)
        monkeypatch.setitem(sys.modules, "huggingface_hub.errors", None)
    return hub


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No ambient Hugging Face configuration leaks into these tests."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Token precedence
# ---------------------------------------------------------------------------

class TestResolveToken:

    def test_stored_token_wins(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "from-env")
        _install_fake_hub(monkeypatch, token="from-file")
        assert hf_access.resolve_token("  from-settings  ") == (
            "from-settings", "settings")

    def test_environment_used_when_nothing_stored(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "from-env")
        _install_fake_hub(monkeypatch, token="from-file")
        assert hf_access.resolve_token("") == ("from-env", "environment")

    def test_legacy_environment_variable_also_read(self, monkeypatch):
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "legacy")
        assert hf_access.resolve_token("") == ("legacy", "environment")

    def test_falls_back_to_cli_token(self, monkeypatch):
        _install_fake_hub(monkeypatch, token="from-file")
        assert hf_access.resolve_token("") == ("from-file", "huggingface-cli")

    def test_no_token_anywhere_is_a_state_not_an_error(self, monkeypatch):
        _install_fake_hub(monkeypatch, token=None)
        assert hf_access.resolve_token("") == ("", "")

    def test_survives_huggingface_hub_being_absent(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        assert hf_access.resolve_token("") == ("", "")

    def test_survives_get_token_raising(self, monkeypatch):
        hub = _install_fake_hub(monkeypatch)

        def _boom():
            raise OSError("unreadable token file")

        hub.get_token = _boom
        assert hf_access.resolve_token("") == ("", "")

    def test_whitespace_only_stored_token_is_not_a_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "from-env")
        assert hf_access.resolve_token("   ") == ("from-env", "environment")

    def test_every_source_has_a_description(self):
        for source in hf_access.TOKEN_SOURCES:
            assert hf_access.describe_token_source(source)


# ---------------------------------------------------------------------------
# Access check
# ---------------------------------------------------------------------------

class TestCheckRepoAccess:

    def test_granted(self, monkeypatch):
        _install_fake_hub(monkeypatch)
        result = hf_access.check_repo_access("facebook/x", token="t")
        assert result["status"] == hf_access.ACCESS_GRANTED
        assert result["repo"] == "facebook/x"

    def test_gated_without_token_is_distinguished_from_denial(self, monkeypatch):
        def _gated(repo_id, token):
            raise _FakeGated()

        _install_fake_hub(monkeypatch, model_info=_gated)
        assert hf_access.check_repo_access("facebook/x", token="")["status"] == \
            hf_access.ACCESS_NO_TOKEN
        assert hf_access.check_repo_access("facebook/x", token="t")["status"] == \
            hf_access.ACCESS_GATED

    def test_gated_message_points_at_the_repo_page(self, monkeypatch):
        def _gated(repo_id, token):
            raise _FakeGated()

        _install_fake_hub(monkeypatch, model_info=_gated)
        message = hf_access.check_repo_access("facebook/x", token="")["message"]
        assert "huggingface.co/facebook/x" in message

    def test_missing_repo(self, monkeypatch):
        def _missing(repo_id, token):
            raise _FakeMissing()

        _install_fake_hub(monkeypatch, model_info=_missing)
        result = hf_access.check_repo_access("facebook/nope", token="t")
        assert result["status"] == hf_access.ACCESS_MISSING

    def test_network_failure_is_reported_not_raised(self, monkeypatch):
        def _offline(repo_id, token):
            raise OSError("no route to host")

        _install_fake_hub(monkeypatch, model_info=_offline)
        result = hf_access.check_repo_access("facebook/x", token="t")
        assert result["status"] == hf_access.ACCESS_ERROR
        assert "no route to host" in result["message"]

    def test_reports_when_the_library_is_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        result = hf_access.check_repo_access(token="t")
        assert result["status"] == hf_access.ACCESS_UNAVAILABLE
        assert "Dependency Manager" in result["message"]

    def test_works_with_the_legacy_errors_location(self, monkeypatch):
        def _gated(repo_id, token):
            raise _FakeGated()

        _install_fake_hub(monkeypatch, model_info=_gated, errors_module=False)
        assert hf_access.check_repo_access("facebook/x", token="t")["status"] == \
            hf_access.ACCESS_GATED

    def test_empty_token_is_passed_as_none(self, monkeypatch):
        seen = {}

        def _record(repo_id, token):
            seen["token"] = token
            return {}

        _install_fake_hub(monkeypatch, model_info=_record)
        hf_access.check_repo_access("facebook/x", token="")
        assert seen["token"] is None

    def test_defaults_to_the_backbone(self, monkeypatch):
        seen = {}

        def _record(repo_id, token):
            seen["repo"] = repo_id
            return {}

        _install_fake_hub(monkeypatch, model_info=_record)
        hf_access.check_repo_access(token="t")
        assert seen["repo"] == hf_access.DEFAULT_BACKBONE


# ---------------------------------------------------------------------------
# On-disk layout
# ---------------------------------------------------------------------------

class TestModelLayout:

    def test_head_path_keeps_the_repository_layout(self, tmp_path):
        path = hf_access.head_local_path(
            str(tmp_path), "sex", "non_geo", "rgb")
        assert path == os.path.join(
            str(tmp_path), "classification", "sex", "non_geo", "sex_rgb.pt")

    def test_repo_path_matches_the_published_layout(self):
        assert hf_access.head_repo_path("occlusion", "geo_2k", "matched") == \
            "geo_2k/occlusion_matched.pt"

    def test_backbone_cache_sits_under_the_shared_models_folder(self, tmp_path):
        assert hf_access.backbone_cache_dir(str(tmp_path)) == \
            os.path.join(str(tmp_path), "hf_cache")

    def test_matched_takes_twice_the_backbone_width(self):
        assert hf_access.feature_dim("rgb") == 1280
        assert hf_access.feature_dim("thermal") == 1280
        assert hf_access.feature_dim("matched") == 2560

    def test_feature_dim_follows_a_custom_backbone(self):
        assert hf_access.feature_dim("matched", backbone_dim=768) == 1536

    def test_species_has_no_default_model_yet(self):
        assert not hf_access.has_default_head("species")
        assert hf_access.default_head_repo("species") is None

    def test_released_tasks_have_defaults(self):
        for task in ("occlusion", "sex"):
            assert hf_access.has_default_head(task)
            assert "cpraschl/" in hf_access.default_head_repo(task)

    def test_missing_heads_reports_only_what_is_absent(self, tmp_path):
        present = hf_access.head_local_path(
            str(tmp_path), "sex", "non_geo", "rgb")
        os.makedirs(os.path.dirname(present))
        with open(present, "wb") as handle:
            handle.write(b"weights")

        wanted = [("sex", "non_geo", "rgb"), ("occlusion", "non_geo", "matched")]
        assert hf_access.missing_heads(str(tmp_path), wanted) == [
            ("occlusion", "non_geo", "matched")]

    def test_every_task_is_covered_by_the_default_table(self):
        assert set(hf_access.DEFAULT_HEAD_REPOS) == set(hf_access.TASKS)

    def test_occlusion_runs_before_the_heads_it_gates(self):
        # The order of TASKS is load-bearing: occlusion selects the frames the
        # other two vote over, and sex reuses exactly the species frames.
        assert hf_access.TASKS.index("occlusion") == 0
        assert hf_access.TASKS.index("species") < hf_access.TASKS.index("sex")

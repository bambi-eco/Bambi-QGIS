# -*- coding: utf-8 -*-
"""Hugging Face access and the on-disk layout of the classification models.

The DINOv3 backbone the classification heads consume
(:data:`DEFAULT_BACKBONE`) is a **gated** repository: the user has to request
access on huggingface.co themselves and then supply a token. That is the single
most likely thing to go wrong in the whole feature, so it gets a first-class
check (:func:`check_repo_access`) rather than surfacing as a stack trace three
steps into a long run.

Two boundaries are deliberate here:

* **The token is passed in, never read from QSettings by this module.** A token
  is a user credential, so it lives in QSettings rather than in the project file
  a user shares — but ``QSettings`` is a GUI-layer concern and is not part of the
  headless test stub, so the dock widget reads it and hands the value to
  :func:`resolve_token`.
* **The models root is passed in.** It comes from
  ``QgsApplication.qgisSettingsDirPath()``, and ``qgis.core`` may not be imported
  from ``core`` even lazily (see ``core/__init__``), so the caller resolves it —
  ``BambiProcessor._get_default_model_dir()`` already does exactly this for the
  detection weights.

``huggingface_hub`` is an optional dependency and is imported lazily throughout,
so a project that never classifies anything does not need it installed.
"""

import os
from typing import Dict, List, Optional, Tuple

#: The backbone every published head was trained against. Gated on Hugging Face.
#: The trailing "lvd1689m" makes this read as high-entropy to a secret scanner;
#: it is a public model name, not a credential.
DEFAULT_BACKBONE = "facebook/dinov3-vith16plus-pretrain-lvd1689m"  # pragma: allowlist secret

#: CLS-token width of :data:`DEFAULT_BACKBONE`. A ``matched`` head takes twice
#: this, being the concatenation of the two modalities.
BACKBONE_DIM = 1280

#: Projection variants, matching the sub-folders of the head repositories. The
#: crops a head sees have to come from the imagery it was trained on, so this
#: selects both the repo sub-folder and the source of the crops.
PROJECTIONS = ("non_geo", "geo_1k", "geo_2k")

#: Input configurations a head can be trained for.
MODALITIES = ("rgb", "thermal", "matched")

#: The classification tasks, in the order they must run: occlusion decides
#: which frames the others are allowed to see (see the plan, §5.2a), species
#: fixes what the animal is, and the two demographic heads are chosen by that
#: species.
TASKS = ("occlusion", "species", "sex", "life_stage")

#: Tasks whose model is chosen per species, because the cue is
#: species-specific: antlers mark a male red deer, and nothing about that
#: transfers to another animal. A species with no model is left uncalled
#: rather than guessed at.
PER_SPECIES_TASKS = ("sex", "life_stage")

#: Default head repository per task. ``species`` and ``life_stage`` are not
#: released yet — the code paths are complete, so they start working the day
#: the repos appear, and until then a custom model is the only option.
DEFAULT_HEAD_REPOS: Dict[str, Optional[str]] = {
    "occlusion": "cpraschl/bambi-occlusion-classifiers",
    "sex": "cpraschl/bambi-red-deer-sex-classifiers",
    "species": None,
    "life_stage": None,
}

#: Human labels for the tasks, since ``life_stage`` does not capitalise well.
TASK_LABELS = {
    "occlusion": "Occlusion",
    "species": "Species",
    "sex": "Sex",
    "life_stage": "Life stage",
}

#: Where the token came from, for the UI to report.
TOKEN_SOURCES = ("settings", "environment", "huggingface-cli", "")


class GatedRepoError(RuntimeError):
    """Access to a gated repository was refused, or no token was supplied."""


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

def resolve_token(stored: str = "") -> Tuple[str, str]:
    """Return ``(token, source)``, preferring the most explicit setting.

    The order is what a user would expect: what they typed into the plugin
    wins, then the environment, then whatever ``hf auth login`` left behind —
    so someone already logged in on the command line has nothing to configure.

    *stored* is the value the GUI read out of QSettings; this module never
    touches QSettings itself. An empty return means no token is available
    anywhere, which is a reportable state rather than an error.
    """
    stored = (stored or "").strip()
    if stored:
        return stored, "settings"

    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value, "environment"

    # Last resort: the token file written by ``hf auth login``. Reached only
    # when the environment held nothing, so the source label stays accurate
    # even though ``get_token`` would consult the environment as well.
    try:
        from huggingface_hub import get_token
    except ImportError:
        return "", ""
    try:
        value = (get_token() or "").strip()
    except Exception:
        return "", ""
    return (value, "huggingface-cli") if value else ("", "")


def describe_token_source(source: str) -> str:
    """A human sentence for where the token came from."""
    return {
        "settings": "the token stored in the plugin settings",
        "environment": "the HF_TOKEN environment variable",
        "huggingface-cli": "the token from 'hf auth login'",
    }.get(source, "no token")


# ---------------------------------------------------------------------------
# Access check
# ---------------------------------------------------------------------------

#: Outcomes of :func:`check_repo_access`.
ACCESS_GRANTED = "granted"
ACCESS_GATED = "gated"
ACCESS_NO_TOKEN = "no_token"  # nosec B105 — a status name, not a credential
ACCESS_MISSING = "not_found"
ACCESS_UNAVAILABLE = "unavailable"   # huggingface_hub not installed
ACCESS_ERROR = "error"               # offline, proxy, anything else


def check_repo_access(
    repo_id: str = DEFAULT_BACKBONE,
    # An empty token means "none supplied" — a state this function exists to
    # report, not a hardcoded credential.
    token: str = "",  # nosec B107
) -> Dict[str, str]:
    """Ask Hugging Face whether *token* may read *repo_id*.

    Returns ``{"status", "message", "repo"}``. Never raises: every failure is a
    reportable status, because this exists precisely to turn an exception three
    steps into a long run into an answer before it starts.
    """
    result = {"repo": repo_id, "status": ACCESS_ERROR, "message": ""}

    try:
        from huggingface_hub import HfApi
        try:
            # Canonical since huggingface_hub 0.25; ``.utils`` re-exports these
            # for older releases, so try the new home first and fall back.
            from huggingface_hub.errors import (
                GatedRepoError as HubGatedRepoError,
                RepositoryNotFoundError,
            )
        except ImportError:
            from huggingface_hub.utils import (
                GatedRepoError as HubGatedRepoError,
                RepositoryNotFoundError,
            )
    except ImportError:
        result["status"] = ACCESS_UNAVAILABLE
        result["message"] = (
            "huggingface_hub is not installed. Install the Classification "
            "dependencies from the Dependency Manager first.")
        return result

    try:
        HfApi().model_info(repo_id, token=token or None)
    except HubGatedRepoError:
        result["status"] = ACCESS_NO_TOKEN if not token else ACCESS_GATED
        result["message"] = (
            f"{repo_id} is gated and this token has no access. Open "
            f"https://huggingface.co/{repo_id}, accept the conditions, then "
            "paste a token with read permission here."
            if token else
            f"{repo_id} is gated and no token was supplied. Request access at "
            f"https://huggingface.co/{repo_id}, then paste a read token here.")
        return result
    except RepositoryNotFoundError:
        # A private or gated repo can also present as 404 to an unauthorised
        # caller, so this is not necessarily a typo — say both.
        result["status"] = ACCESS_MISSING
        result["message"] = (
            f"{repo_id} was not found. Check the spelling, or — if it is "
            "private or gated — that your token has access to it.")
        return result
    except Exception as exc:
        result["message"] = (
            f"Could not reach Hugging Face: {exc}. Check the network "
            "connection or any proxy settings.")
        return result

    result["status"] = ACCESS_GRANTED
    result["message"] = f"Access to {repo_id} is granted."
    return result


# ---------------------------------------------------------------------------
# On-disk layout
# ---------------------------------------------------------------------------

def classification_dir(models_dir: str) -> str:
    """Folder holding the downloaded classification heads."""
    return os.path.join(models_dir, "classification")


def backbone_cache_dir(models_dir: str) -> str:
    """Hugging Face cache for the backbone.

    Passed to ``from_pretrained(cache_dir=...)`` rather than exported as
    ``HF_HOME``: setting that environment variable would relocate the cache for
    everything else in the QGIS process that uses Hugging Face, which is not
    ours to do.
    """
    return os.path.join(models_dir, "hf_cache")


def head_filename(task: str, modality: str) -> str:
    """The head's file name, e.g. ``sex_rgb.pt`` — the repos' own convention."""
    return f"{task}_{modality}.pt"


def head_repo_path(task: str, projection: str, modality: str) -> str:
    """Path of a head *inside* its repository, e.g. ``non_geo/sex_rgb.pt``."""
    return f"{projection}/{head_filename(task, modality)}"


def head_local_path(models_dir: str, task: str, projection: str,
                    modality: str) -> str:
    """Where a downloaded head lives locally.

    The repository's own ``{projection}/{task}_{modality}.pt`` layout is kept
    under a per-task folder, so a head fetched by hand into the obvious place is
    picked up with no configuration.
    """
    return os.path.join(classification_dir(models_dir), task, projection,
                        head_filename(task, modality))


def default_head_repo(task: str) -> Optional[str]:
    """The published repository for *task*, or ``None`` if there is not one."""
    return DEFAULT_HEAD_REPOS.get(task)


def has_default_head(task: str) -> bool:
    """True when *task* has a published default model to offer."""
    return bool(DEFAULT_HEAD_REPOS.get(task))


def feature_dim(modality: str, backbone_dim: int = BACKBONE_DIM) -> int:
    """Input width of a head: ``matched`` concatenates both modalities."""
    return backbone_dim * 2 if modality == "matched" else backbone_dim


def missing_heads(models_dir: str, wanted: List[Tuple[str, str, str]]
                  ) -> List[Tuple[str, str, str]]:
    """Which of *wanted* ``(task, projection, modality)`` are not downloaded."""
    return [spec for spec in wanted
            if not os.path.isfile(head_local_path(models_dir, *spec))]

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
  a user shares - but ``QSettings`` is a GUI-layer concern and is not part of the
  headless test stub, so the dock widget reads it and hands the value to
  :func:`resolve_token`.
* **The models root is passed in.** It comes from
  ``QgsApplication.qgisSettingsDirPath()``, and ``qgis.core`` may not be imported
  from ``core`` even lazily (see ``core/__init__``), so the caller resolves it -
  ``BambiProcessor._get_default_model_dir()`` already does exactly this for the
  detection weights.

``huggingface_hub`` is an optional dependency and is imported lazily throughout,
so a project that never classifies anything does not need it installed.
"""

import contextlib
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

#: Default head repository per task. ``life_stage`` has no published head -
#: the size estimate is its default - and the code path is complete, so it
#: starts working the day a repo appears; until then a custom model is the
#: only option. The species heads call ``red_deer`` / ``roe_deer`` /
#: ``wild_boar``; anything else in a survey is forced into one of the three,
#: which is why the class mapping is worth a look on a project with other
#: animals in it.
DEFAULT_HEAD_REPOS: Dict[str, Optional[str]] = {
    "occlusion": "cpraschl/bambi-occlusion-classifiers",
    "species": "cpraschl/bambi-species-classification",
    "sex": "cpraschl/bambi-red-deer-sex-classifiers",
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
    wins, then the environment, then whatever ``hf auth login`` left behind -
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


@contextlib.contextmanager
def token_environment(token: str = ""):  # nosec B107 - "" means none
    """Make *token* visible to every Hugging Face call made inside.

    ``from_pretrained(token=…)`` is not enough: a processor that bundles a
    tokenizer and an image processor fans out into nested loaders, and not
    every one of them is handed the keyword on, so the request for
    ``config.json`` goes out unauthenticated and a gated repository answers
    "please log in" - to a user whose token just passed the access check.
    ``huggingface_hub`` reads ``HF_TOKEN`` from the environment on every
    request, so setting it for the duration of the load reaches all of
    them. Restored afterwards: the token is the user's, not the process's.
    """
    token = (token or "").strip()
    if not token:
        yield
        return
    previous = {name: os.environ.get(name)
                for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
ACCESS_NO_TOKEN = "no_token"  # nosec B105 - a status name, not a credential
ACCESS_MISSING = "not_found"
ACCESS_UNAVAILABLE = "unavailable"   # huggingface_hub not installed
ACCESS_ERROR = "error"               # offline, proxy, anything else


def check_repo_access(
    repo_id: str = DEFAULT_BACKBONE,
    # An empty token means "none supplied" - a state this function exists to
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
        # caller, so this is not necessarily a typo - say both.
        result["status"] = ACCESS_MISSING
        result["message"] = (
            f"{repo_id} was not found. Check the spelling, or - if it is "
            "private or gated - that your token has access to it.")
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
    """The head's file name, e.g. ``sex_rgb.pt`` - the repos' own convention."""
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


#: The one species the published per-species heads were fitted on.
DEFAULT_HEAD_SPECIES = "red deer"


def default_species_source(task: str, species: str) -> str:
    """What decides *task* for *species* when nobody has chosen.

    The published model for the species it was fitted on; for life stage,
    the size-based estimate, which needs no model; otherwise nothing. This
    is what the per-species dialog shows before anything is saved, and it
    is what the run uses for the same case - the two must agree, or the
    dialog promises a default that never runs (2026-09-07: life stage
    reported "nothing to measure" while every species showed "Default
    (size-based)").
    """
    if species == DEFAULT_HEAD_SPECIES and has_default_head(task):
        return "default"
    if task == "life_stage":
        return "size"
    return "off"


def per_species_sources(spec: dict, task: str,
                        species: List[str]) -> Dict[str, dict]:
    """``species -> {"model": ..., ...}`` with the defaults applied.

    A spec whose ``species`` entry was never written takes the defaults for
    every species. Once the dialog has saved, every species is listed
    explicitly, and one that is missing from a saved selection (older
    projects, which dropped "Off") is off.
    """
    saved = (spec or {}).get("species")
    result: Dict[str, dict] = {}
    for name in species:
        entry = (saved or {}).get(name)
        if entry is None:
            source = (default_species_source(task, name) if saved is None
                      else "off")
            entry = {"model": source}
        result[name] = dict(entry)
    return result


def project_species(target_folder: str) -> List[str]:
    """The project's concrete species names, the base classes excluded.

    Falls back to the published species when the project has no vocabulary
    yet, so a default still names something.
    """
    from . import label_store

    vocabulary = label_store.vocabulary(target_folder)
    rows = vocabulary.get("species", []) if vocabulary else []
    names = [row["name"] for row in rows if not row.get("protected")]
    return names or [DEFAULT_HEAD_SPECIES]


def feature_dim(modality: str, backbone_dim: int = BACKBONE_DIM) -> int:
    """Input width of a head: ``matched`` concatenates both modalities."""
    return backbone_dim * 2 if modality == "matched" else backbone_dim


def download_head(repo: str, task: str, projection: str, modality: str,
                  destination: str, token: str = "", log_fn=None) -> str:
    """Fetch a published head into *destination* and return that path.

    Shared by the classification run and the class-mapping dialog: the
    mapping needs the model's class list, the model only used to arrive
    when the classifier ran, and telling a user to run a classifier in order
    to configure it was the wrong way round.
    """
    import shutil

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install the Classification "
            "dependencies from the Dependency Manager.") from exc

    remote = head_repo_path(task, projection, modality)
    if log_fn:
        log_fn(f"Downloading {repo}/{remote} …")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        # nosec B615 - deliberately unpinned, for the same reason as the
        # backbone: the repository is user-overridable, so a hardcoded
        # revision would be wrong for a custom head. Reproducibility comes
        # from the model file itself, which is recorded with every
        # prediction.
        fetched = hf_hub_download(  # nosec B615
            repo_id=repo, filename=remote, token=token or None,
            local_dir=os.path.dirname(os.path.dirname(destination)))
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the {task} classifier "
            f"({repo}/{remote}): {exc}") from exc
    if os.path.abspath(fetched) != os.path.abspath(destination):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(fetched, destination)
    return destination


def missing_heads(models_dir: str, wanted: List[Tuple[str, str, str]]
                  ) -> List[Tuple[str, str, str]]:
    """Which of *wanted* ``(task, projection, modality)`` are not downloaded."""
    return [spec for spec in wanted
            if not os.path.isfile(head_local_path(models_dir, *spec))]

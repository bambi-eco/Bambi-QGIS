# -*- coding: utf-8 -*-
"""The published models the plugin downloads, as this release was tested.

Every default model used to be fetched from the *current* state of its
repository (``.../resolve/main/...``), and a download counted as good when
the file was larger than a megabyte. So a model could change under a survey
without anyone noticing - the same plugin version giving other detections
next month - and a download cut off half-way was kept and used.

A model is now named by **what it is**: the commit of its repository and the
size and SHA-256 of the file. A download is written beside its destination,
checked, and only then moved into place; a cached file that is not the pinned
one is replaced (and kept beside it as ``*.previous``).

Pinning is for the plugin's own defaults. A repository or model file the
user chose is theirs to pin - append ``@<commit>`` to a repository - and
``@main`` on a default follows its latest state deliberately.

To move a pin, read the new values off the repository's "Files" page (or
``https://huggingface.co/api/models/<repo>?blobs=true``) and replace commit,
size and SHA-256 together.

The commits and checksums below are public, and look like keys to a scan
for secrets: each line says so (``pragma: allowlist secret``), so that the
scan of ``run_checks.sh`` stays one that stops at a real one.
"""

import hashlib
import os
from typing import Callable, Dict, NamedTuple, Optional, Tuple

HUGGING_FACE = "https://huggingface.co"


class Pin(NamedTuple):
    """One file as published."""

    #: Commit of the repository the file is taken from; empty for a file
    #: that is not in a versioned repository.
    revision: str
    size: int
    sha256: str


class ModelIntegrityError(RuntimeError):
    """A model file is not the one that was published."""


#: Repository -> the commit this release was tested with. Covers the
#: repositories whose files are listed in :data:`FILES` and those fetched as
#: a whole through the hub library (the DINOv3 backbone, SAM3), which checks
#: every file against the commit itself.
REVISIONS: Dict[str, str] = {
    "cpraschl/bambi-thermal-detection":
        "4e3b73b87fed5cea18472685f31890c9db6f8489",  # pragma: allowlist secret
    "cpraschl/bambi-models": "2230399f9b106f38a51e0d40e1c93e82a22dbf22",  # pragma: allowlist secret
    "cpraschl/bambi-thermal-omni": "796329bd1abcea5cd2503244b904e043e82a7e63",  # pragma: allowlist secret
    "cpraschl/bambi-occlusion-classifiers":
        "e5595ed38dd64a9c8b2b8f725999b85c389a343e",  # pragma: allowlist secret
    "cpraschl/bambi-species-classification":
        "5d46b419e765a20bc5b14217e48ba0c07c8a58c3",  # pragma: allowlist secret
    "cpraschl/bambi-red-deer-sex-classifiers":
        "0fd9204ca7c8c94643a3fac836db8ff75f3cf0e7",  # pragma: allowlist secret
    "facebook/dinov3-vith16plus-pretrain-lvd1689m":  # pragma: allowlist secret
        "c807c9eeea853df70aec4069e6f56b28ddc82acc",  # pragma: allowlist secret
    "facebook/sam3": "3c879f39826c281e95690f02c7821c4de09afae7",  # pragma: allowlist secret
    "facebook/sam3.1": "daa63191845a41281374e725f4c9e51c7a824460",  # pragma: allowlist secret
}

#: ``{repository: {path: (size, sha256)}}`` of the files downloaded one by
#: one, at the commit in :data:`REVISIONS`.
_FILES: Dict[str, Dict[str, Tuple[int, str]]] = {
    "cpraschl/bambi-thermal-detection": {
        "thermal_animal_detector.pt": (
            114466450,
            "c01d6159973ee81ffd8b32f99fa020febe2a5b25d71edb5a166059d1910fbb9d"),  # pragma: allowlist secret
    },
    "cpraschl/bambi-models": {
        "rgb/weights/best.pt": (
            118400997,
            "24215d0210794be5f9327a6a43b209bbae130ed4cc5529708a2f9263be69a02a"),  # pragma: allowlist secret
    },
    "cpraschl/bambi-thermal-omni": {
        "model.pt": (
            10765706,
            "d431d9d9bef78f2219275c0ba34dd311e087a8e3e856d60a908faa83ca5a7e22"),  # pragma: allowlist secret
    },
    "cpraschl/bambi-occlusion-classifiers": {
        "geo_1k/occlusion_matched.pt": (
            5803129,
            "24bf76d6cce82c5c2549a0cc61d9ad3d835458e6d7741144ede1284c3f615cb2"),  # pragma: allowlist secret
        "geo_1k/occlusion_rgb.pt": (
            3170189,
            "f88358ad89d964b5a59867296429967680790505e65ea3e8ed2fb8e660439947"),  # pragma: allowlist secret
        "geo_1k/occlusion_thermal.pt": (
            3170273,
            "7b222f1b8980fe766d60825c14af59c042c5e82a555c0832d7abbf81168e28e5"),  # pragma: allowlist secret
        "geo_2k/occlusion_matched.pt": (
            5803129,
            "d06d3621fe4f29ef25c4e1797d52ca630d0a4ef99db67ea3b7a71ec11c291016"),  # pragma: allowlist secret
        "geo_2k/occlusion_rgb.pt": (
            3170189,
            "09abca496bfa0d2d9a42e667ab0668e4e6107d3078a68652e6b622586edf2c5d"),  # pragma: allowlist secret
        "geo_2k/occlusion_thermal.pt": (
            3170273,
            "e41b3f3593cdf986ca49d6f4cc776b46cd5e4e36c14c1865b2ebffcc346c83c8"),  # pragma: allowlist secret
        "non_geo/occlusion_matched.pt": (
            5803129,
            "9e6e6b91532b1f04f9cf93e62e57230ea8f769b1406016463ca0102965f9418c"),  # pragma: allowlist secret
        "non_geo/occlusion_rgb.pt": (
            3170189,
            "ebfe7acaddfeec307ce89727ab799c30035d1c0652a0a17fb134b11929197e38"),  # pragma: allowlist secret
        "non_geo/occlusion_thermal.pt": (
            3170273,
            "70cadfbb31e1ee4e4486fd480012e3042d833ea81735ddf654eaca8329734178"),  # pragma: allowlist secret
    },
    "cpraschl/bambi-red-deer-sex-classifiers": {
        "geo_1k/sex_matched.pt": (
            5802863,
            "156fa3310a025ed2b7d756aa7a21403d0be8b60caecc1d4626cb23c36d322de6"),  # pragma: allowlist secret
        "geo_1k/sex_rgb.pt": (
            3169551,
            "a82cfcc36fe854b7453e501f31dc8d151c03e4e02255e1842f4f5e94e6ea5c2d"),  # pragma: allowlist secret
        "geo_1k/sex_thermal.pt": (
            3170147,
            "40073d101027a781742e78b759f3dbaf3b8411012026dd671886369a3092779e"),  # pragma: allowlist secret
        "geo_2k/sex_matched.pt": (
            5802863,
            "47833f07f14bc44afe7933844d90261311fb1a4954a570f69dd17d51533dc4ad"),  # pragma: allowlist secret
        "geo_2k/sex_rgb.pt": (
            3169551,
            "b47c875361bdba0c3db86d9b6f03687feda1555cd36bb1b6d18404a102b62aae"),  # pragma: allowlist secret
        "geo_2k/sex_thermal.pt": (
            3170147,
            "2660a2d5374a5647bcd82c8722f0614ad87ae12f7508cfc35e6f96b6f7def1e8"),  # pragma: allowlist secret
        "non_geo/sex_matched.pt": (
            5802863,
            "c5dc559e7f5cb17d3c0fd83b3be181e604d82fd52317c4988fd0f1d7cd35b253"),  # pragma: allowlist secret
        "non_geo/sex_rgb.pt": (
            3169551,
            "12312240ea9353ba33aa9b1b021af0ee2b8a314eedbfb4d4113f7034b13209c3"),  # pragma: allowlist secret
        "non_geo/sex_thermal.pt": (
            3170147,
            "d69022a943609781ee5d6e694c39d517ed3a0a4e9f6581bc5d16d0f166c7d7f0"),  # pragma: allowlist secret
    },
    "cpraschl/bambi-species-classification": {
        "alfs_1k/species_matched.pt": (
            5804347,
            "619f4e29b2946b57c8995a4022291a84a2e51798413064e3b60e14ee0abe4306"),  # pragma: allowlist secret
        "alfs_1k/species_rgb.pt": (
            3172311,
            "2ca04fc167a8b0fd88576a7f725c9cb50254bccb823bca0972ca94483abc28de"),  # pragma: allowlist secret
        "alfs_1k/species_thermal.pt": (
            3172467,
            "ca8eb752022e0d0ec2d74eb2c7631fc18a123d6647147b891a0fe44ea545837e"),  # pragma: allowlist secret
        "alfs_2k/species_matched.pt": (
            5804347,
            "5772056b55620df6e3cb49da354954c1368972f9d438232502a859fa029aab17"),  # pragma: allowlist secret
        "alfs_2k/species_rgb.pt": (
            3172311,
            "9912b2dfe31a17a8ea690c79057da29c40acd7f14fc276524efa42d00d7c9f42"),  # pragma: allowlist secret
        "alfs_2k/species_thermal.pt": (
            3172467,
            "b22840ad2b92668042c00d90e81dfafd4627f61fb58128d32420df59d563f362"),  # pragma: allowlist secret
        "geo_1k/species_matched.pt": (
            5804347,
            "e7b8384cd761e76c24cb1cfbf664aa57916f37a6f81e12e65910fc7cd1e93773"),  # pragma: allowlist secret
        "geo_1k/species_rgb.pt": (
            3172311,
            "1e5d6a60c4e6db6dd31bd70926ec07038360d4341666cc0fa7dcfe6b1fd2fa46"),  # pragma: allowlist secret
        "geo_1k/species_thermal.pt": (
            3172467,
            "febe39d9a753e8e462fea6c83e1771d137412ae900ca2fcfae5fd7eede6b4266"),  # pragma: allowlist secret
        "geo_2k/species_matched.pt": (
            5804347,
            "cb8f442e6eb5649c2b5654049295ff72b6e118117ce0aa51766e8347992f21b5"),  # pragma: allowlist secret
        "geo_2k/species_rgb.pt": (
            3172311,
            "d7d1c7469bb705775e5e1b00276d32df84e36668480086ddabd29febcbdcedfe"),  # pragma: allowlist secret
        "geo_2k/species_thermal.pt": (
            3172467,
            "39baf0ecd7516639f6918d98c62acdfdf0ee38339fa3821ef8922b3befb4212d"),  # pragma: allowlist secret
        "non_geo/species_matched.pt": (
            5804347,
            "b1570c300220383358ef2526bf7543d58cf631b77398a38a0f7f02985ec03a20"),  # pragma: allowlist secret
        "non_geo/species_rgb.pt": (
            3172311,
            "e4161f1080684ba960c476d8069d531deaecc1c8673ef0c482ae4183a06f425d"),  # pragma: allowlist secret
        "non_geo/species_thermal.pt": (
            3172467,
            "0ecd4283e3e81472fb838339487b126de92209b9469c66b20184e4c20fb07476"),  # pragma: allowlist secret
    },
}

#: BoxMOT's ``osnet_x0_25_msmt17.pt``, which the tracker fetches from the
#: Google Drive file BoxMOT itself points to. No repository, so no commit:
#: the checksum is all that says it is still that file.
OSNET_X0_25_MSMT17 = Pin(
    "", 3057863,
    "6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99")  # pragma: allowlist secret


def revision_of(repo: str) -> str:
    """The pinned commit of *repo*, or ``""`` for one that is not pinned."""
    return REVISIONS.get((repo or "").strip(), "")


def pin_of(repo: str, path: str) -> Optional[Pin]:
    """The pin of the file *path* in *repo*; ``None`` when it has none - a
    repository of the user's own, or a file added after this release."""
    entry = _FILES.get((repo or "").strip(), {}).get(path)
    if entry is None:
        return None
    return Pin(REVISIONS[repo.strip()], entry[0], entry[1])


def resolve_url(repo: str, path: str) -> str:
    """Where to download *path* of *repo* from: at its pinned commit, or -
    for a repository that is not pinned - at ``main``."""
    return (f"{HUGGING_FACE}/{repo}/resolve/"
            f"{revision_of(repo) or 'main'}/{path}")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def difference(path: str, pin: Pin) -> Optional[str]:
    """How the file at *path* differs from *pin* - ``None`` when it is the
    pinned file. The size is compared first, so a truncated download is
    recognised without reading it."""
    if not os.path.isfile(path):
        return "the file is missing"
    size = os.path.getsize(path)
    if size != pin.size:
        return f"it has {size} bytes where {pin.size} were published"
    found = file_sha256(path)
    if found != pin.sha256:
        return (f"its SHA-256 is {found[:12]}... where {pin.sha256[:12]}... "
                "was published")
    return None


def verify(path: str, pin: Pin, what: str = "The model") -> None:
    """Raise :class:`ModelIntegrityError` unless *path* is the pinned file."""
    found = difference(path, pin)
    if found:
        raise ModelIntegrityError(
            f"{what} ({os.path.basename(path)}) is not the published file: "
            f"{found}.")


def download(url: str, destination: str, pin: Optional[Pin] = None,
             what: str = "model", min_size: int = 0,
             get: Optional[Callable] = None, log_fn=None) -> str:
    """Fetch *url* into *destination*, complete and checked or not at all.

    The download is written to ``<destination>.part`` and moved into place
    only once it is whole: with a *pin*, when size and SHA-256 are the
    published ones; without, when it has at least *min_size* bytes. A
    transfer that is cut off, or delivers something else - an error page, a
    newer file - leaves nothing behind and raises.

    :param get: ``requests.get``-like; the default is ``requests.get``
    :return: *destination*
    """
    if not url.lower().startswith("https://"):
        raise ValueError(f"Refusing to download a {what} from {url!r}: "
                         "only https is accepted.")
    if get is None:
        import requests
        get = requests.get

    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    partial = f"{destination}.part"
    try:
        response = get(url, stream=True, timeout=300)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0) or 0)
        done, reported = 0, 0
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                handle.write(chunk)
                done += len(chunk)
                if log_fn and total and done - reported >= total / 10:
                    reported = done
                    log_fn(f"Downloading the {what}: "
                           f"{100 * done / total:.0f}% ({done >> 20} MB)")
        if pin is not None:
            verify(partial, pin, f"The downloaded {what}")
        elif os.path.getsize(partial) < max(1, min_size):
            raise ModelIntegrityError(
                f"The downloaded {what} has {os.path.getsize(partial)} "
                f"bytes; at least {min_size} were expected.")
        os.replace(partial, destination)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return destination


def ensure(destination: str, url: str, pin: Optional[Pin] = None,
           what: str = "model", min_size: int = 0,
           get: Optional[Callable] = None, log_fn=None) -> str:
    """The model at *destination*, downloading it where needed.

    A file that is already there is used when it is the pinned one (without
    a pin: when it has *min_size* bytes). One that is not - an older
    version, or what an interrupted download left - is set aside as
    ``<destination>.previous`` and the pinned file fetched; if that fails,
    for want of a network say, the previous file is put back and used, with
    a line in the log saying what it is. A transfer that breaks is tried
    once more; a file that arrives whole but is not the published one is
    not.
    """
    previous = None
    if os.path.isfile(destination):
        if pin is None:
            if os.path.getsize(destination) >= max(1, min_size):
                return destination
            os.remove(destination)
        else:
            found = difference(destination, pin)
            if found is None:
                return destination
            if log_fn:
                log_fn(f"The {what} on this computer is not the published "
                       f"version this release was tested with ({found}). "
                       "Downloading that one; the present file is kept as "
                       f"{os.path.basename(destination)}.previous")
            previous = f"{destination}.previous"
            os.replace(destination, previous)

    try:
        try:
            return download(url, destination, pin, what, min_size, get,
                            log_fn)
        except ModelIntegrityError:
            raise               # what arrived is wrong; again will not help
        except Exception as error:
            # A dropped connection is worth one more go.
            if log_fn:
                log_fn(f"Downloading the {what} failed ({error}); trying "
                       "once more...")
            return download(url, destination, pin, what, min_size, get,
                            log_fn)
    except Exception:
        if previous is not None and os.path.isfile(previous) \
                and os.path.getsize(previous) >= max(1, min_size):
            os.replace(previous, destination)
            if log_fn:
                log_fn(f"The published {what} could not be downloaded; "
                       "the file that was here is used instead. It is not "
                       "the version this release was tested with.")
            return destination
        raise

# -*- coding: utf-8 -*-
"""Which alfspy engine and ray caster this project renders with.

alfspy 3.0 rasterises through one of three interchangeable engines and casts
rays through one of two casters. Both are chosen at *run* time from the
environment - ``$ALFS_ENGINE`` and ``$ALFS_RAYCASTER`` - and both are read
inside ``resolve_engine`` / ``resolve_raycaster`` on every call rather than
once at import, so exporting them before a render is enough and no import
ordering has to be arranged around it.

Which of them will actually *work* is decided at install time: the engines and
casters are pip extras (``AlfsPy[torch,warp]``), so this module is the single
place that names them, and ``core.dependency_ops`` builds its install command
from the same table the dropdowns are filled from. The two cannot drift apart.

This module is deliberately free of Qt and of alfspy itself: the vocabulary has
to be readable to build the UI and the install command before alfspy exists at
all.
"""

import os
from typing import Dict, NamedTuple, Optional, Sequence, Tuple


class Option(NamedTuple):
    """One selectable engine or ray caster."""

    #: The value ``$ALFS_ENGINE`` / ``$ALFS_RAYCASTER`` takes, and the name of
    #: the pip extra that installs it. Deliberately the same string: alfspy
    #: 3.0 renamed its extras to match the selector values precisely so a
    #: selection cannot be spelled two ways.
    name: str
    #: What the dropdown shows.
    label: str
    #: One sentence under it.
    hint: str


#: Render engines, in the order the dropdown lists them.
ENGINES: Tuple[Option, ...] = (
    Option("moderngl", "ModernGL",
           "OpenGL rasteriser. Needs a working GL driver, which a remote or "
           "headless machine often has not."),
    Option("torch", "PyTorch",
           "Renders on the GPU when CUDA is available and needs no GL driver. "
           "Installs PyTorch, which is a large download."),
    Option("vulkan", "Vulkan",
           "Headless GPU rendering through WebGPU - no GL driver and no "
           "virtual display. Needs Python 3.11 or newer."),
)

#: Ray casters, in the order the dropdown lists them.
RAYCASTERS: Tuple[Option, ...] = (
    Option("embree", "Embree (CPU)",
           "The default. Fast enough for the tens of rays a frame's labels "
           "need, and already covered by alfspy's own dependencies."),
    Option("warp", "Warp (GPU)",
           "NVIDIA Warp. Worth it above roughly 10,000 rays per call; below "
           "that its launch overhead only ties Embree."),
)

#: What a project renders with when it has never said otherwise. These match
#: alfspy's own defaults, so an unset environment and an unconfigured project
#: behave identically.
DEFAULT_ENGINE = "moderngl"
DEFAULT_RAYCASTER = "embree"

#: The variables alfspy reads. Named here rather than spelled inline at the
#: two call sites, because a typo in an environment variable fails silently -
#: it just renders with the default and nobody is told.
ENGINE_ENV_VAR = "ALFS_ENGINE"
RAYCASTER_ENV_VAR = "ALFS_RAYCASTER"

#: The device, resolved by alfspy exactly like the two above since 3.0.0 - an
#: explicit ``device=`` argument, then this variable, then the backend's own
#: choice. (Until then it was documented but read nowhere, so exporting it was
#: a silent no-op; the plugin pins a release where it works.)
DEVICE_ENV_VAR = "ALFS_DEVICE"

#: Config keys the selection travels under, from the project to the processor.
ENGINE_KEY = "alfs_engine"
RAYCASTER_KEY = "alfs_raycaster"


def engine_names() -> Tuple[str, ...]:
    """Every selectable engine name."""
    return tuple(option.name for option in ENGINES)


def raycaster_names() -> Tuple[str, ...]:
    """Every selectable ray-caster name."""
    return tuple(option.name for option in RAYCASTERS)


def _resolve(value: object, options: Sequence[Option], default: str) -> str:
    """*value* if it names one of *options*, otherwise *default*.

    Falling back rather than raising is deliberate: this reads a stored project
    setting, and a project written by a newer plugin - or by hand - must still
    open. The selection is a preference, not data, so the honest response to an
    unknown one is to render with the default.
    """
    text = str(value or "").strip().lower()
    return text if text in {option.name for option in options} else default


def resolve_engine(value: object = None) -> str:
    """The engine *value* names, or the default."""
    return _resolve(value, ENGINES, DEFAULT_ENGINE)


def resolve_raycaster(value: object = None) -> str:
    """The ray caster *value* names, or the default."""
    return _resolve(value, RAYCASTERS, DEFAULT_RAYCASTER)


def label_of(name: str, options: Sequence[Option] = ENGINES) -> str:
    """The human label for *name*, or *name* itself if it is not one of ours."""
    for option in options:
        if option.name == name:
            return option.label
    return name


def selection_from_config(config: Optional[dict] = None) -> Tuple[str, str]:
    """``(engine, raycaster)`` for a run, from its configuration.

    Both fall back to the default rather than to whatever is already in the
    environment: a project that names an engine is answering the question, and
    a project that does not should render the same way everywhere.
    """
    config = config or {}
    return (resolve_engine(config.get(ENGINE_KEY)),
            resolve_raycaster(config.get(RAYCASTER_KEY)))


def apply_selection(config: Optional[dict] = None, device: object = None,
                    env: Optional[Dict[str, str]] = None,
                    log_fn=None) -> Tuple[str, str]:
    """Export the project's engine and ray caster into the environment.

    alfspy reads both on every ``make_context`` / ``create_raycaster`` call, so
    this only has to run before the render - not before alfspy is imported.
    Call it at the top of any step that renders.

    *device* is written to ``$ALFS_DEVICE`` when given (``"cuda"``, ``"cpu"``);
    an empty value clears it, so a project that stops asking for CUDA does not
    keep rendering on it because of a variable set earlier in the session.

    :param config: the run configuration; ``None`` applies the defaults
    :param device: the torch/Vulkan device, or ``""`` to leave it to alfspy
    :param env: the mapping to write into, defaulting to ``os.environ``
    :return: the ``(engine, raycaster)`` that were exported
    """
    target = os.environ if env is None else env
    engine, raycaster = selection_from_config(config)

    target[ENGINE_ENV_VAR] = engine
    target[RAYCASTER_ENV_VAR] = raycaster

    text = str(device or "").strip()
    if text and text != "auto":
        target[DEVICE_ENV_VAR] = text
    else:
        # Not merely "leave it alone": a stale value from an earlier run in the
        # same QGIS session would otherwise outlive the setting that put it
        # there.
        target.pop(DEVICE_ENV_VAR, None)

    if log_fn:
        log_fn(f"Rendering with the {label_of(engine)} engine and the "
               f"{label_of(raycaster, RAYCASTERS)} ray caster")
    return engine, raycaster


def extras_for(engine: str, raycaster: str) -> Tuple[str, ...]:
    """The pip extras that make *engine* and *raycaster* work.

    alfspy 3.0 names its extras after the selector values, so this is the
    identity - written out anyway because it is the one place where "what you
    picked" becomes "what gets installed", and a future extra that does not
    follow the rule has somewhere to be handled.
    """
    return (resolve_engine(engine), resolve_raycaster(raycaster))

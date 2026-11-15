# -*- coding: utf-8 -*-
"""An error that is survived is still said.

Sixty-odd places in the plugin caught any exception and did nothing with it:
a metadata file that could not be read, a raster written without its CRS, a
correction file that did not parse, a tile that could not be fetched. Each
is right to carry on - and was wrong to leave no trace: the result was a
route in the wrong place or an uncorrected flight with nothing anywhere
that said why.

Inside the ``except`` block, instead of ``pass``::

    except Exception:
        quiet.problem(f"The correction file {path} could not be read; "
                      "no correction is applied")

* :func:`problem` - something the user would want to know about: a warning;
* :func:`detail` - a fallback taken as a matter of course, a clean-up that
  did not work: kept for whoever looks.

Both add the exception being handled, with credentials masked
(``core.redaction``), and write to the Python logger :data:`LOGGER_NAME`.
In QGIS the plugin shows that logger in the *Log Messages* panel, tab
"BAMBI" (``gui_utils.show_notes_in_qgis``). The same message is written a
few times at most: a handler in a loop over a thousand frames does not fill
the log.

Free of QGIS and Qt.
"""

import logging
import sys
import threading
from typing import Dict

from . import redaction

#: The logger these notes are written to.
LOGGER_NAME = "bambi_wildlife_detection"
#: How often the same message is written before it is only counted.
REPEATS = 3

_logger = logging.getLogger(LOGGER_NAME)
_said: Dict[str, int] = {}
_lock = threading.Lock()


def _reason() -> str:
    error = sys.exc_info()[1]
    if error is None:
        return ""
    text = str(error).strip()
    name = type(error).__name__
    return f"{name}: {text}" if text else name


def _note(level: int, what: str) -> None:
    what = str(what).rstrip(". ")
    with _lock:
        count = _said.get(what, 0) + 1
        _said[what] = count
    if count > REPEATS:
        return
    reason = _reason()
    message = f"{what} ({reason})." if reason else f"{what}."
    if count == REPEATS:
        message += " Further occurrences are not listed."
    _logger.log(level, redaction.redact(message))


def problem(what: str) -> None:
    """Note, as a warning, that *what* did not work - with the exception
    being handled - and return: the caller carries on."""
    _note(logging.WARNING, what)


def detail(what: str) -> None:
    """Like :func:`problem`, for what is no cause for concern: a fallback
    taken, something that could not be tidied up."""
    _note(logging.DEBUG, what)


def forget() -> None:
    """Start counting repeated messages afresh (a new run; the tests)."""
    with _lock:
        _said.clear()

# -*- coding: utf-8 -*-
"""Large files that have to arrive whole: resumed, checked, and only then in
place.

An elevation tile of the Austrian survey is a GeoTIFF of 50 km by 50 km at
one metre - hundreds of megabytes to gigabytes. It used to be written
straight to its place in the cache while it came in, and whatever was there
afterwards counted as the tile:

* a connection that ended early left a file that was shorter than announced
  - nothing compared the two;
* an answer that was no tile at all - a proxy's error page, sent with
  "200 OK" - was saved as one;
* QGIS closed during a download left the part that had arrived, under the
  tile's name. "Using cached" was all the next run said about it, and every
  run after it;
* and a download that broke at 90 % started again at zero.

:func:`fetch` writes to ``<file>.part``, continues a part that is there
(when the server says it still serves the same file), tries again when the
connection breaks, and moves the file into place only when it is as long as
announced and :func:`tiff_problem` finds nothing wrong with it - which is
also how a file already in the cache is checked before it is used.

Free of QGIS and Qt; ``requests`` is passed in as the session to use.
"""

import json
import os
import struct
import time
from typing import Callable, List, Optional
from . import quiet

#: TIFF tags that say where the image data is, and how much of it.
_OFFSET_TAGS = {273: 279, 324: 325}     # Strip-/TileOffsets: their ByteCounts
_WANTED_TAGS = frozenset(_OFFSET_TAGS) | frozenset(_OFFSET_TAGS.values())
#: Bytes per value of the TIFF types these tags come in.
_TYPE_SIZE = {3: 2, 4: 4, 16: 8}        # SHORT, LONG, LONG8
_TYPE_CODE = {3: "H", 4: "I", 16: "Q"}
#: A TIFF has at most this many images (pages, overviews, masks) to us.
_MAX_DIRECTORIES = 64


class DownloadError(RuntimeError):
    """A file could not be fetched whole."""


class NotThere(DownloadError):
    """The server has no such file (404)."""


class Cancelled(DownloadError):
    """The download was cancelled; what arrived is kept for the next try."""


class NotTheFile(DownloadError):
    """Something arrived, whole, and it is not what was asked for."""


class _Truncated(Exception):
    pass


# ---------------------------------------------------------------------------
# Is a GeoTIFF whole?
# ---------------------------------------------------------------------------

def _values(handle, order, kind, count, inline, inline_size, file_size):
    """The *count* values of one directory entry."""
    size = _TYPE_SIZE[kind] * count
    if size <= inline_size:
        data = inline[:size]
    else:
        offset = struct.unpack(order + ("Q" if inline_size == 8 else "I"),
                               inline)[0]
        if offset + size > file_size:
            raise _Truncated()
        handle.seek(offset)
        data = handle.read(size)
    return struct.unpack(f"{order}{count}{_TYPE_CODE[kind]}", data)


def tiff_problem(path: str) -> Optional[str]:
    """What is wrong with the TIFF file at *path* - ``None`` when it is
    whole.

    Whole means: it starts as a TIFF does, and everything its directories
    point at - every strip or tile of every image in it - lies within the
    file. A download that stopped early fails that wherever it stopped: a
    TIFF's directory is either at its end, or points to data up to its end.
    The pixel values are not read (that would take as long as using them);
    a file that is whole but damaged in the middle is not found by this.
    """
    try:
        file_size = os.path.getsize(path)
        with open(path, "rb") as handle:
            head = handle.read(16)
            if len(head) < 8 or head[:2] not in (b"II", b"MM"):
                return "it is not a TIFF file"
            order = "<" if head[:2] == b"II" else ">"
            magic = struct.unpack(order + "H", head[2:4])[0]
            if magic == 42:
                big, entry_size, inline_size = False, 12, 4
                directory = struct.unpack(order + "I", head[4:8])[0]
            elif magic == 43 and len(head) == 16:
                big, entry_size, inline_size = True, 20, 8
                directory = struct.unpack(order + "Q", head[8:16])[0]
            else:
                return "it is not a TIFF file"

            seen = set()
            while directory:
                if directory in seen or len(seen) >= _MAX_DIRECTORIES:
                    break
                seen.add(directory)
                count_size = 8 if big else 2
                if directory + count_size > file_size:
                    return (f"it ends after {file_size} bytes, before its "
                            "table of contents")
                handle.seek(directory)
                entries = struct.unpack(
                    order + ("Q" if big else "H"), handle.read(count_size))[0]
                size = entries * entry_size + (8 if big else 4)
                if entries > 65535 or directory + count_size + size > file_size:
                    return (f"it ends after {file_size} bytes, within its "
                            "table of contents")
                table = handle.read(size)
                found = {}
                for index in range(entries):
                    entry = table[index * entry_size:(index + 1) * entry_size]
                    tag, kind = struct.unpack(order + "HH", entry[:4])
                    if tag not in _WANTED_TAGS or kind not in _TYPE_SIZE:
                        continue
                    number = struct.unpack(
                        order + ("Q" if big else "I"),
                        entry[4:4 + inline_size])[0]
                    found[tag] = (kind, number, entry[4 + inline_size:])
                directory = struct.unpack(
                    order + ("Q" if big else "I"), table[-(8 if big else 4):])[0]

                for offsets_tag, counts_tag in _OFFSET_TAGS.items():
                    if offsets_tag not in found or counts_tag not in found:
                        continue
                    offsets = _values(handle, order, *found[offsets_tag],
                                      inline_size, file_size)
                    counts = _values(handle, order, *found[counts_tag],
                                     inline_size, file_size)
                    end = max((offset + length for offset, length
                               in zip(offsets, counts) if length), default=0)
                    if end > file_size:
                        return (f"it has {file_size} bytes where its "
                                f"contents need {end}: the end is missing")
    except _Truncated:
        return (f"it ends after {file_size} bytes, within its table of "
                "contents")
    except (OSError, ValueError, struct.error) as exc:
        return f"it cannot be read as a TIFF file ({exc})"
    return None


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _note_path(part: str) -> str:
    return part + ".json"


def _read_note(part: str) -> dict:
    try:
        with open(_note_path(part), "r", encoding="utf-8") as handle:
            note = json.load(handle)
        return note if isinstance(note, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_note(part: str, note: dict) -> None:
    with open(_note_path(part), "w", encoding="utf-8") as handle:
        json.dump(note, handle)


def _forget(part: str) -> None:
    for path in (part, _note_path(part)):
        try:
            os.remove(path)
        except OSError:
            pass


def _header(response, name: str) -> str:
    headers = getattr(response, "headers", None) or {}
    for key, value in headers.items():
        if str(key).lower() == name:
            return str(value)
    return ""


def _announced_total(response, resumed_from: int) -> int:
    """How long the whole file is, as far as the answer says; 0 if it does
    not."""
    content_range = _header(response, "content-range")      # bytes a-b/total
    if "/" in content_range:
        total = content_range.rsplit("/", 1)[1].strip()
        if total.isdigit():
            return int(total)
    if _header(response, "content-encoding").lower() not in ("", "identity"):
        return 0            # the length is that of the packed stream
    length = _header(response, "content-length")
    if length.isdigit() and int(length) > 0:
        return int(length) + resumed_from
    return 0


def _range_start(response) -> Optional[int]:
    """Where the part an answer carries begins, if it says."""
    content_range = _header(response, "content-range")      # bytes a-b/total
    first = content_range.replace("bytes", "").strip().split("-", 1)[0]
    return int(first) if first.strip().isdigit() else None


def fetch(session, url: str, destination: str,
          check: Optional[Callable[[str], Optional[str]]] = tiff_problem,
          cancelled: Optional[Callable[[], bool]] = None,
          progress: Optional[Callable[[int], None]] = None,
          log: Optional[Callable[[str], None]] = None,
          attempts: int = 4, timeout: float = 600,
          chunk_size: int = 1 << 20,
          sleep: Callable[[float], None] = time.sleep) -> str:
    """Fetch *url* into *destination*: whole and checked, or not at all.

    What arrives goes to ``<destination>.part``. A part that is already
    there - from a download that was cancelled, or broke, or from a QGIS
    that was closed - is continued when the server still serves the file it
    began with (``ETag`` / ``Last-Modified``) and can send the rest
    (``Range``); otherwise it is started again. A connection that breaks is
    tried again from where it broke, up to *attempts* times. The file is
    moved into place when it is as long as the server announced and *check*
    has no objection.

    :param session: something with ``get(url, stream=True, timeout=...,
        headers=...)`` like a ``requests.Session``
    :param check: ``check(path) -> problem or None`` for the finished file
    :param progress: called with 0..100
    :raises NotThere: the server answered 404
    :raises Cancelled: *cancelled* said so; the part is kept
    :raises NotTheFile: it arrived whole and *check* objects to it
    :raises DownloadError: it could not be fetched whole; the part is kept
    """
    say = log or (lambda message: None)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    part = destination + ".part"
    attempts = max(1, int(attempts))
    problems: List[str] = []

    for attempt in range(attempts):
        if cancelled and cancelled():
            raise Cancelled("cancelled")
        if attempt:
            sleep(min(30.0, 2.0 ** attempt))
        note = _read_note(part)
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        validator = note.get("etag") or note.get("last_modified") or ""
        resume = bool(have and validator and note.get("url") == url)
        if have and not resume:
            _forget(part)       # nothing says it is the start of this file
            have = 0
        # as it is on the server: lengths and ranges are those of the file
        headers = {"Accept-Encoding": "identity"}
        if resume:
            headers.update({"Range": f"bytes={have}-", "If-Range": validator})

        response = None
        try:
            response = session.get(url, stream=True, timeout=timeout,
                                   headers=headers)
            status = getattr(response, "status_code", 200)
            if status == 404:
                raise NotThere(f"{url} does not exist")
            if status == 416 and resume:
                # nothing beyond what is here: it is whole, or not this file
                total = int(note.get("size") or have)
            else:
                response.raise_for_status()
                if resume and status == 206 and _range_start(response) == have:
                    say(f"  Continuing at {have / 1048576:.1f} MB")
                elif resume:
                    # the whole file again: it changed, or cannot be resumed
                    say("  The server sends the file from the start again")
                    have = 0
                total = _announced_total(response, have)
                if not have:
                    _write_note(part, {
                        "url": url, "size": total,
                        "etag": _header(response, "etag"),
                        "last_modified": _header(response, "last-modified")})
                if total:
                    say(f"  Size: {total / 1048576:.1f} MB")
                done, shown = have, -1
                with open(part, "ab" if have else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if cancelled and cancelled():
                            raise Cancelled("cancelled")
                        if not chunk:
                            continue
                        handle.write(chunk)
                        done += len(chunk)
                        if progress and total:
                            percent = min(99, int(done * 100 / total))
                            if percent != shown:
                                shown = percent
                                progress(percent)

            size = os.path.getsize(part) if os.path.isfile(part) else 0
            if total and size < total:
                raise ConnectionError(
                    f"the connection ended after {size} of {total} bytes")
            if total and size > total:
                _forget(part)
                raise DownloadError(
                    f"{size} bytes arrived where {total} were announced")
            problem = check(part) if check else None
            if problem:
                _forget(part)
                raise NotTheFile(
                    f"what {url} sent is not the file expected: {problem}")
            os.replace(part, destination)
            _forget(part)
            if progress:
                progress(100)
            return destination
        except DownloadError:       # not there, cancelled, or not the file
            raise
        except Exception as exc:  # noqa: BLE001 - whatever the network does
            problems.append(str(exc) or type(exc).__name__)
            again = "; trying again" if attempt + 1 < attempts else ""
            say(f"  Interrupted ({problems[-1]}){again}")
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # closing what already broke
                    quiet.detail("The connection could not be closed")

    raise DownloadError(
        f"{url} could not be downloaded: {problems[-1] if problems else '?'}. "
        "What arrived is kept; running it again continues from there.")

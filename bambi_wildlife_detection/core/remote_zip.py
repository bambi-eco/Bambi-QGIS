# -*- coding: utf-8 -*-
"""A few files out of a large archive on a web server.

A flight of the BAMBI dataset is one ZIP archive of several hundred
megabytes: a video, and beside it a handful of small files - the poses, the
labels, a correction. Whoever wants the small ones should not have to fetch
the video.

A ZIP archive ends with its table of contents, and says for every file
where in the archive it lies. So a server that answers *range requests* -
"bytes 1000 to 1999 of this file" - can be asked for the table and then for
exactly the files that are wanted: :class:`RangeFile` makes such a file on
a server look like one on disk (read, seek, tell), and Python's own
``zipfile`` does the rest - checking, too, that what arrived has the
checksum the archive records for it.

What an archive names its files is not trusted: :func:`read_members`
hands back the files whose *base name* the caller asked for, as bytes -
nothing is written anywhere - and refuses a file larger than the caller
allows.

Free of QGIS and Qt; ``requests`` is passed in as the session to use.
"""

import io
import os
import re
import time
import zipfile
from typing import Callable, Dict, Iterable, List, Optional, Tuple

#: A request fetches at least this much: the table of contents of an
#: archive, or a small file with its header, in one go.
CHUNK = 256 * 1024
#: What is kept of what was fetched, so that reading back and forth in the
#: table of contents costs nothing.
KEEP_BYTES = 32 * 1024 * 1024
TIMEOUT_S = 60
#: A server that is busy or asks to be left alone for a moment (429, 502,
#: 503, 504) is asked again this many times, after waiting this long at
#: first and twice as long each further time - or as long as it says.
RETRIES = 4
FIRST_WAIT_S = 2.0
LONGEST_WAIT_S = 60.0
_BUSY = (429, 502, 503, 504)

_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)")


class RemoteArchiveError(RuntimeError):
    """An archive on a server could not be read."""


class NotThere(RemoteArchiveError):
    """The server has no such file (404)."""


class Cancelled(RemoteArchiveError):
    """Reading was cancelled."""


class RangeFile(io.RawIOBase):
    """A file on an HTTPS server, read a range at a time.

    Read-only and seekable. What was fetched is kept (up to
    :data:`KEEP_BYTES`), so the small steps a ZIP reader takes through a
    table of contents do not each become a request.

    :param session: something with ``get(url, headers=..., timeout=...)``
        like a ``requests.Session``
    :param size: how long the file is, where that is known - one request
        less
    :param cancelled: ``cancelled() -> bool``, asked before every request
    :param sleep: ``sleep(seconds)``, for waiting on a busy server
    """

    def __init__(self, session, url: str, size: Optional[int] = None,
                 cancelled: Optional[Callable[[], bool]] = None,
                 timeout: float = TIMEOUT_S,
                 sleep: Callable[[float], None] = time.sleep):
        super().__init__()
        if not str(url).lower().startswith("https://"):
            raise RemoteArchiveError(f"Refusing a non-HTTPS address: {url}")
        self._session = session
        self.url = url
        self._cancelled = cancelled
        self._timeout = timeout
        self._sleep = sleep
        self._position = 0
        self._spans: List[Tuple[int, bytes]] = []
        #: How many requests were made, and how many bytes they brought.
        self.requests = 0
        self.transferred = 0
        self._size = int(size) if size is not None else self._ask_size()

    # -- the server ---------------------------------------------------------

    def _wait(self, seconds: float) -> None:
        """Wait, a second at a time, so that a cancel is heard."""
        remaining = max(0.0, min(float(seconds), LONGEST_WAIT_S))
        while remaining > 0:
            if self._cancelled and self._cancelled():
                raise Cancelled("cancelled")
            step = min(1.0, remaining)
            self._sleep(step)
            remaining -= step

    def _ask(self, first: int, last: int):
        """One answer to "bytes *first* to *last*" - asked again while the
        server is busy."""
        wait = FIRST_WAIT_S
        for attempt in range(RETRIES + 1):
            if self._cancelled and self._cancelled():
                raise Cancelled("cancelled")
            try:
                response = self._session.get(
                    self.url, headers={"Range": f"bytes={first}-{last}"},
                    timeout=self._timeout)
            except Exception as exc:  # noqa: BLE001 - whatever it raises
                raise RemoteArchiveError(
                    f"The server could not be reached ({exc}). Check the "
                    "internet connection and try again.") from exc
            self.requests += 1
            status = getattr(response, "status_code", 0)
            if status not in _BUSY or attempt == RETRIES:
                return response
            asked_for = (getattr(response, "headers", None) or {}).get(
                "Retry-After")
            try:
                pause = float(asked_for)
            except (TypeError, ValueError):
                pause = wait
            self._wait(pause)
            wait *= 2.0
        return response

    def _get(self, first: int, last: int):
        """Bytes *first* to *last* (inclusive) of the file: ``(data,
        total length or None)``."""
        response = self._ask(first, last)
        status = getattr(response, "status_code", 0)
        if status == 404:
            raise NotThere(f"There is no such file on the server: {self.url}")
        if status == 200:
            # the whole file instead of the part asked for
            raise RemoteArchiveError(
                "The server does not answer range requests: the archive "
                "would have to be fetched whole.")
        if status in _BUSY:
            raise RemoteArchiveError(
                f"The server is busy (it answered {status}, {RETRIES + 1} "
                "times). Try again in a few minutes.")
        if status != 206:
            raise RemoteArchiveError(
                f"The server answered {status} for a part of {self.url}.")
        data = response.content
        self.transferred += len(data)
        headers = getattr(response, "headers", None) or {}
        found = _CONTENT_RANGE.search(str(headers.get("Content-Range", "")))
        total = None
        if found:
            if int(found.group(1)) != first:
                raise RemoteArchiveError(
                    "The server sent another part of the file than the one "
                    "asked for.")
            if found.group(3) != "*":
                total = int(found.group(3))
        expected = last - first + 1
        if total is not None:
            expected = min(expected, total - first)
        if len(data) != expected:
            raise RemoteArchiveError(
                f"The server sent {len(data)} bytes where {expected} were "
                "asked for: the connection broke, try again.")
        return data, total

    def _ask_size(self) -> int:
        data, total = self._get(0, 0)
        if total is None:
            raise RemoteArchiveError(
                "The server does not say how long the file is.")
        self._spans.append((0, data))
        return total

    # -- what was fetched ---------------------------------------------------

    def _cached(self, start: int, length: int) -> Optional[bytes]:
        for begin, data in self._spans:
            if begin <= start and start + length <= begin + len(data):
                return data[start - begin:start - begin + length]
        return None

    def fetch(self, start: int, length: int) -> None:
        """Have ``[start, start + length)`` in memory after this, by one
        request - for a file whose place and size in the archive are
        known. Clipped to the file."""
        start = max(0, min(int(start), self._size))
        length = max(0, min(int(length), self._size - start))
        if length == 0 or self._cached(start, length) is not None:
            return
        data, _total = self._get(start, start + length - 1)
        self._spans.append((start, data))
        kept = sum(len(span) for _begin, span in self._spans)
        while kept > KEEP_BYTES and len(self._spans) > 1:
            _begin, dropped = self._spans.pop(0)
            kept -= len(dropped)

    # -- a file -------------------------------------------------------------

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    @property
    def size(self) -> int:
        return self._size

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._position
        size = max(0, min(size, self._size - self._position))
        if size == 0:
            return b""
        data = self._cached(self._position, size)
        if data is None:
            # at least a chunk - and near the end of the file the chunk
            # that ends there: a table of contents is read from its end
            start = self._position
            length = max(size, CHUNK)
            if start + length > self._size:
                start = max(0, min(start, self._size - length))
                length = self._size - start
            self.fetch(start, length)
            data = self._cached(self._position, size)
            if data is None:
                raise RemoteArchiveError("A part that was fetched is missing.")
        self._position += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def readall(self) -> bytes:
        return self.read(-1)


def _basename(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def _archive_name(url: str) -> str:
    """What an archive is called in a message: the last part of its
    address that is a file name (Zenodo ends its addresses in
    ``/content``)."""
    from urllib.parse import urlsplit

    parts = [part for part in urlsplit(url).path.split("/") if part]
    for part in reversed(parts):
        if "." in part:
            return part
    return parts[-1] if parts else url


def open_archive(session, url: str, size: Optional[int] = None,
                 cancelled: Optional[Callable[[], bool]] = None
                 ) -> Tuple[zipfile.ZipFile, RangeFile]:
    """The ZIP archive at *url*, opened by its table of contents alone:
    ``(archive, the file it is read through)``.

    :raises RemoteArchiveError: no answer, no range requests, or no ZIP
        archive
    """
    remote = RangeFile(session, url, size=size, cancelled=cancelled)
    # the table of contents is at the end: one request for it
    tail = min(remote.size, CHUNK)
    remote.fetch(remote.size - tail, tail)
    try:
        return zipfile.ZipFile(remote), remote
    except zipfile.BadZipFile as exc:
        raise RemoteArchiveError(
            f"{_archive_name(url)} is not a ZIP archive that can be read "
            f"({exc}).") from exc


def list_members(session, url: str, size: Optional[int] = None,
                 cancelled: Optional[Callable[[], bool]] = None
                 ) -> List[Tuple[str, int]]:
    """``[(name, uncompressed size)]`` of the files in the archive at
    *url*."""
    archive, _remote = open_archive(session, url, size, cancelled)
    with archive:
        return [(info.filename, info.file_size)
                for info in archive.infolist() if not info.is_dir()]


def read_members(session, url: str, wanted: Iterable[str],
                 size: Optional[int] = None, max_bytes: int = 64 << 20,
                 cancelled: Optional[Callable[[], bool]] = None,
                 log: Optional[Callable[[str], None]] = None
                 ) -> Dict[str, bytes]:
    """The files of the archive at *url* whose base name is in *wanted*:
    ``{base name: contents}``. A name the archive does not have is not in
    the result.

    Only the table of contents and these files are transferred, each file
    by one request. Each is checked against the checksum the archive
    records for it.

    :param max_bytes: no file larger than this is read, uncompressed or
        compressed - what an archive says about itself is not trusted
    :raises RemoteArchiveError: the archive cannot be read, a file is
        larger than allowed, or what arrived is not what the archive
        recorded
    """
    names = set(wanted)
    archive, remote = open_archive(session, url, size, cancelled)
    found: Dict[str, bytes] = {}
    with archive:
        for info in archive.infolist():
            base = _basename(info.filename)
            if info.is_dir() or base not in names or base in found:
                continue
            if max(info.file_size, info.compress_size) > max_bytes:
                raise RemoteArchiveError(
                    f"{base} is {info.file_size / 1e6:.0f} MB in the "
                    "archive: larger than such a file can be.")
            # the file's own header (name and extra field, with room to
            # spare) and its contents, in one request
            header = 30 + len(info.filename.encode("utf-8")) + 1024
            remote.fetch(info.header_offset, header + info.compress_size)
            try:
                found[base] = archive.read(info)
            except zipfile.BadZipFile as exc:
                raise RemoteArchiveError(
                    f"{base} did not arrive as the archive records it "
                    f"({exc}). Try again.") from exc
    if log:
        log(f"Read {len(found)} file(s) out of {_archive_name(url)} - "
            f"{remote.transferred / 1e6:.1f} MB of "
            f"{remote.size / 1e6:.0f} MB transferred")
    return found

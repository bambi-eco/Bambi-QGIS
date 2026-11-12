# -*- coding: utf-8 -*-
"""Remote access of the environment sources (core/env_remote.py): every way
a service can fail becomes one error the sub-step can report."""
import numpy as np
import pytest
import requests

from bambi_wildlife_detection.core import env_remote
from bambi_wildlife_detection.core import environment as env


class _Response:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _patch(monkeypatch, response=None, error=None):
    calls = []

    def request(url, **kwargs):
        calls.append((url, kwargs))
        if error is not None:
            raise error
        return response

    monkeypatch.setattr(requests, "get", request)
    monkeypatch.setattr(requests, "post", request)
    return calls


def test_get_and_post_return_the_document(monkeypatch):
    calls = _patch(monkeypatch, _Response(payload={"ok": 1}))
    assert env_remote.fetch_json("https://x/a", params={"q": 1}) == {"ok": 1}
    assert calls[0][1]["params"] == {"q": 1}
    assert calls[0][1]["timeout"] == env_remote.TIMEOUT_S
    assert env_remote.fetch_json("https://x/a", body={"b": 2}) == {"ok": 1}
    assert calls[1][1]["json"] == {"b": 2}


def test_no_connection_is_reported_with_what_was_asked(monkeypatch):
    _patch(monkeypatch, error=requests.ConnectionError("no route"))
    with pytest.raises(env_remote.SourceUnavailable) as error:
        env_remote.fetch_json("https://x/a", what="Earth Search")
    assert "Earth Search" in str(error.value)
    assert "internet connection" in str(error.value)


@pytest.mark.parametrize("payload,text,expected", [
    ({"reason": "Daily API request limit exceeded."}, "",
     "HTTP 429: Daily API request limit exceeded."),
    ({"description": "bad bbox"}, "", "HTTP 429: bad bbox"),
    (None, "<html>gateway</html>", "HTTP 429: <html>gateway</html>"),
    (["unexpected"], "", "HTTP 429."),
])
def test_an_http_error_carries_the_services_reason(monkeypatch, payload,
                                                   text, expected):
    _patch(monkeypatch, _Response(429, payload, text))
    with pytest.raises(env_remote.SourceUnavailable) as error:
        env_remote.fetch_json("https://x/a", what="the service")
    assert expected in str(error.value)


def test_a_body_that_is_not_json_fails(monkeypatch):
    _patch(monkeypatch, _Response(200, None, "<html>"))
    with pytest.raises(env_remote.SourceUnavailable, match="JSON"):
        env_remote.fetch_json("https://x/a")


def test_plain_http_is_refused(monkeypatch):
    calls = _patch(monkeypatch, _Response(payload={}))
    with pytest.raises(env_remote.SourceUnavailable, match="non-HTTPS"):
        env_remote.fetch_json("http://x/a")
    grid = env.make_grid((0.0, 0.0, 30.0, 20.0), 32633, 0.0)
    with pytest.raises(env_remote.SourceUnavailable, match="non-HTTPS"):
        env_remote.read_to_grid("http://x/a.tif", grid)
    assert calls == []


def test_a_raster_is_read_onto_the_grid(tmp_path):
    """A local file stands in for the remote one: same code path, no network."""
    source = env.make_grid((447000.0, 5350500.0, 447100.0, 5350600.0),
                           32633, 0.0)
    values = np.arange(100, dtype=np.uint16).reshape(source.shape) + 1
    path = env.write_raster(str(tmp_path / "band.tif"), values, source,
                            nodata=0)
    inner = env.make_grid((447020.0, 5350520.0, 447050.0, 5350540.0),
                          32633, 0.0)
    read, nodata = env_remote.read_to_grid(path, inner, "nearest")
    assert nodata == 0
    assert read.tolist() == values[6:8, 2:5].tolist()

    outside = env.make_grid((447090.0, 5350590.0, 447120.0, 5350620.0),
                            32633, 0.0)
    read, _nodata = env_remote.read_to_grid(path, outside, "bilinear")
    assert read[0, 0] == 0 and read[2, 0] == values[0, 9]   # 0 = not covered


def test_an_unreadable_raster_fails(tmp_path):
    grid = env.make_grid((0.0, 0.0, 30.0, 20.0), 32633, 0.0)
    with pytest.raises(env_remote.SourceUnavailable, match="Could not read"):
        env_remote.read_to_grid(str(tmp_path / "missing.tif"), grid)

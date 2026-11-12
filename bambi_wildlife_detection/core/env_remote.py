# -*- coding: utf-8 -*-
"""Reading remote data for the environment sources.

Two things live here so the sources themselves stay testable without a
network: fetching a JSON document over HTTPS, and reading the part of a
cloud-optimised GeoTIFF that covers a :class:`core.environment.Grid`.

A cloud-optimised GeoTIFF is read through GDAL's HTTP range requests, so only
the few tiles under a flight are transferred - a Sentinel-2 band is 100 km
wide and a flight a few kilometres.
"""

from typing import Optional

from .environment import EnvironmentFailure, Grid, rasterio_crs

TIMEOUT_S = 60

#: GDAL settings for range-reading public cloud-optimised GeoTIFFs: no
#: directory listing (one request less per file, and buckets refuse it), a few
#: retries on a dropped connection, and no credential lookup for public data.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_TIMEOUT": str(TIMEOUT_S),
    "AWS_NO_SIGN_REQUEST": "YES",
}

USER_AGENT = "bambi-qgis-plugin"


class SourceUnavailable(EnvironmentFailure):
    """A remote service could not be reached or refused the request."""


def _require_https(url: str) -> None:
    if not str(url).lower().startswith("https://"):
        raise SourceUnavailable(f"Refusing a non-HTTPS address: {url}")


def fetch_json(url: str, params: Optional[dict] = None,
               body: Optional[dict] = None, what: str = "the service"):
    """GET (or POST, with *body*) a JSON document.

    Every failure - no connection, a timeout, an HTTP error, a body that is
    not JSON - becomes a :class:`SourceUnavailable` naming *what* was asked,
    because the sub-step is to fail with something the user can act on.
    """
    import requests

    _require_https(url)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        if body is not None:
            response = requests.post(url, json=body, headers=headers,
                                     timeout=TIMEOUT_S)
        else:
            response = requests.get(url, params=params, headers=headers,
                                    timeout=TIMEOUT_S)
    except requests.RequestException as exc:
        raise SourceUnavailable(
            f"Could not reach {what}: {exc}. Check the internet "
            "connection and try again.") from exc
    if response.status_code >= 400:
        detail = ""
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                payload = {}
            for key in ("reason", "description", "message"):
                if payload.get(key):
                    detail = str(payload[key])
                    break
        except ValueError:
            detail = (response.text or "")[:200]
        reason = f": {detail}" if detail else ""
        raise SourceUnavailable(
            f"{what} answered HTTP {response.status_code}{reason}.")
    try:
        return response.json()
    except ValueError as exc:
        raise SourceUnavailable(
            f"{what} did not answer with JSON.") from exc


def read_to_grid(href: str, grid: Grid, resampling: str = "bilinear"):
    """The part of the raster at *href* under *grid*, on *grid*.

    :param resampling: ``"nearest"`` for classes, ``"bilinear"`` for
        measurements
    :return: ``(array, nodata)`` - the raw values in the source's data type
        (cells the source does not cover carry *nodata*, or 0 when the
        source names none)
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.errors import RasterioError
    from rasterio.transform import Affine
    from rasterio.vrt import WarpedVRT

    if str(href).lower().startswith("http"):
        _require_https(href)
    method = (Resampling.nearest if resampling == "nearest"
              else Resampling.bilinear)
    try:
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(href) as src:
                nodata = src.nodata
                with WarpedVRT(
                        src, crs=rasterio_crs(grid.epsg),
                        transform=Affine(*grid.geotransform),
                        width=grid.width, height=grid.height,
                        resampling=method,
                        src_nodata=nodata, nodata=nodata) as vrt:
                    return vrt.read(1), nodata
    except (RasterioError, OSError) as exc:
        raise SourceUnavailable(
            f"Could not read {href}: {exc}. Check the internet connection "
            "and try again.") from exc

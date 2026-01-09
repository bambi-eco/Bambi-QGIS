# -*- coding: utf-8 -*-
"""
Austria DEM Downloader for QGIS Plugin
======================================

Downloads, merges, and processes Digital Elevation Model (DGM) data from the
Austrian Federal Office of Metrology and Surveying (BEV) ATOM service.

This module uses the Austria-wide 1m ALS-DTM dataset which:
- Covers all of Austria
- Has 1m resolution (from Airborne Laser Scanning)
- Is organized in 50km x 50km tiles in EPSG:3035
- Is freely available under CC-BY-4.0 license

Data source: https://data.bev.gv.at
"""

import json
import logging
import os
import re
import struct
import tempfile
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import numpy as np

from qgis.PyQt.QtCore import QObject, pyqtSignal, QThread

# Setup logging
logger = logging.getLogger(__name__)

# CRS definitions using PROJ4 strings to avoid PROJ database version conflicts in QGIS
# Using PROJ4 strings instead of EPSG codes bypasses the proj.db lookup
WGS84_CRS = "EPSG:4326"
WGS84_PROJ4 = "+proj=longlat +datum=WGS84 +no_defs"

# EPSG:3035 - ETRS89-extended / LAEA Europe (BEV tiles use this)
BEV_CRS = "EPSG:3035"
BEV_CRS_PROJ4 = "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs"

# Default output CRS - UTM zone 33N
DEFAULT_OUTPUT_CRS = "EPSG:32633"
DEFAULT_OUTPUT_CRS_PROJ4 = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs"

# Map of common EPSG codes to their PROJ4 strings
EPSG_TO_PROJ4 = {
    "EPSG:4326": WGS84_PROJ4,
    "EPSG:3035": BEV_CRS_PROJ4,
    "EPSG:32632": "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs",
    "EPSG:32633": "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs",
    "EPSG:32634": "+proj=utm +zone=34 +datum=WGS84 +units=m +no_defs",
}

def get_proj4_for_crs(crs_str: str) -> str:
    """Get PROJ4 string for a CRS, with fallback to the original string."""
    return EPSG_TO_PROJ4.get(crs_str, crs_str)

# BEV ATOM service
BEV_ATOM_SERVICE = "https://data.bev.gv.at/geonetwork/srv/atom/describe/service"
BEV_ATOM_UUID = "208cff7a-c8aa-42fe-bf4f-2b8156e37528"

# Multiple URL patterns to try (in order of preference - newest first)
BEV_URL_PATTERNS = [
    ("20230915", "ALS_DTM_CRS3035RES50000mN{north}E{east}.tif"),
    ("20190915", "CRS3035RES50000mN{north}E{east}.tif"),
    ("20210401", "ALS_DTM_CRS3035RES50000mN{north}E{east}.tif"),
]
BEV_DOWNLOAD_BASE = "https://data.bev.gv.at/download/ALS/DTM/"

# Tile parameters (50km x 50km tiles in EPSG:3035)
TILE_SIZE = 50000  # 50km in meters

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "austria_dem"


@dataclass
class BoundingBox:
    """Represents a geographic bounding box."""
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    @classmethod
    def from_points(cls, lat1: float, lon1: float, lat2: float, lon2: float,
                    padding_meters: float = 0) -> 'BoundingBox':
        """Create bounding box from two GPS points with optional padding."""
        min_lat = min(lat1, lat2)
        max_lat = max(lat1, lat2)
        min_lon = min(lon1, lon2)
        max_lon = max(lon1, lon2)

        if padding_meters > 0:
            lat_center = (min_lat + max_lat) / 2
            meters_per_degree_lat = 111320
            meters_per_degree_lon = 111320 * np.cos(np.radians(lat_center))

            lat_padding = padding_meters / meters_per_degree_lat
            lon_padding = padding_meters / meters_per_degree_lon

            min_lat -= lat_padding
            max_lat += lat_padding
            min_lon -= lon_padding
            max_lon += lon_padding

        return cls(min_lat, min_lon, max_lat, max_lon)

    @classmethod
    def from_airdata_csv(cls, csv_path: str, padding_meters: float = 0) -> Optional['BoundingBox']:
        """
        Create bounding box from AirData CSV file by finding min/max GPS positions.
        
        :param csv_path: Path to the AirData CSV file
        :param padding_meters: Padding in meters to add around the bounding box
        :return: BoundingBox or None if no GPS data found
        """
        try:
            lats = []
            lons = []
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Try to find latitude/longitude columns (AirData uses various naming)
                lat_col = None
                lon_col = None
                
                # Check column names
                fieldnames = reader.fieldnames
                if fieldnames is None:
                    return None
                    
                for col in fieldnames:
                    col_lower = col.lower().strip()
                    if lat_col is None and any(x in col_lower for x in ['latitude', 'lat']):
                        lat_col = col
                    if lon_col is None and any(x in col_lower for x in ['longitude', 'lon', 'lng']):
                        lon_col = col
                
                if lat_col is None or lon_col is None:
                    logger.error(f"Could not find latitude/longitude columns in CSV. Columns: {fieldnames}")
                    return None
                
                for row in reader:
                    try:
                        lat = float(row[lat_col])
                        lon = float(row[lon_col])
                        # Skip invalid values
                        if lat != 0 and lon != 0 and -90 <= lat <= 90 and -180 <= lon <= 180:
                            lats.append(lat)
                            lons.append(lon)
                    except (ValueError, KeyError):
                        continue
            
            if not lats or not lons:
                logger.error("No valid GPS coordinates found in CSV")
                return None
            
            min_lat = min(lats)
            max_lat = max(lats)
            min_lon = min(lons)
            max_lon = max(lons)
            
            logger.info(f"Found GPS bounds: lat [{min_lat:.6f}, {max_lat:.6f}], lon [{min_lon:.6f}, {max_lon:.6f}]")
            
            return cls.from_points(min_lat, min_lon, max_lat, max_lon, padding_meters)
            
        except Exception as e:
            logger.error(f"Error reading AirData CSV: {e}")
            return None

    def to_crs(self, target_crs: str) -> Tuple[float, float, float, float]:
        """Convert to projected coordinates (min_x, min_y, max_x, max_y)."""
        try:
            from pyproj import Transformer, CRS as PyprojCRS
        except ImportError:
            raise ImportError("pyproj is required. Install with: pip install pyproj")
        
        # Use PROJ4 strings to avoid PROJ database version conflicts
        src_proj4 = get_proj4_for_crs(WGS84_CRS)
        dst_proj4 = get_proj4_for_crs(target_crs)
        
        transformer = Transformer.from_crs(
            PyprojCRS.from_proj4(src_proj4),
            PyprojCRS.from_proj4(dst_proj4),
            always_xy=True
        )

        # Transform all four corners to handle projection distortion
        corners = [
            (self.min_lon, self.min_lat),
            (self.max_lon, self.min_lat),
            (self.min_lon, self.max_lat),
            (self.max_lon, self.max_lat)
        ]

        x_coords = []
        y_coords = []
        for lon, lat in corners:
            x, y = transformer.transform(lon, lat)
            x_coords.append(x)
            y_coords.append(y)

        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))


class BEVTileCalculator:
    """Calculates which BEV tiles are needed for a given bounding box."""

    AUSTRIA_BOUNDS = {
        'min_x': 4250000,
        'max_x': 4900000,
        'min_y': 2550000,
        'max_y': 2900000
    }

    def __init__(self):
        self.tile_size = TILE_SIZE

    def get_tile_bounds(self, tile_name: str) -> Optional[Tuple[float, float, float, float]]:
        """Parse tile name to get bounds in EPSG:3035."""
        match = re.search(r'N(\d+)E(\d+)', tile_name)
        if not match:
            return None

        north = int(match.group(1))
        east = int(match.group(2))

        min_x = east
        min_y = north
        max_x = east + self.tile_size
        max_y = north + self.tile_size

        return (min_x, min_y, max_x, max_y)

    def get_tile_name(self, north: int, east: int) -> str:
        """Generate tile identifier from grid coordinates."""
        return f"N{north}E{east}"

    def get_required_tiles(self, bbox: BoundingBox) -> List[str]:
        """Get list of tile names that intersect the bounding box."""
        min_x, min_y, max_x, max_y = bbox.to_crs(BEV_CRS)

        logger.debug(f"Bbox in EPSG:3035: {min_x:.0f}, {min_y:.0f} to {max_x:.0f}, {max_y:.0f}")

        start_east = int(min_x // self.tile_size) * self.tile_size
        end_east = int(max_x // self.tile_size) * self.tile_size
        start_north = int(min_y // self.tile_size) * self.tile_size
        end_north = int(max_y // self.tile_size) * self.tile_size

        tiles = []
        for north in range(start_north, end_north + self.tile_size, self.tile_size):
            for east in range(start_east, end_east + self.tile_size, self.tile_size):
                tile_name = self.get_tile_name(north, east)
                tiles.append(tile_name)

        logger.info(f"Required tiles: {len(tiles)}")
        for tile in tiles:
            logger.debug(f"  {tile}")

        return tiles

    def get_download_urls(self, tile_name: str) -> List[str]:
        """Get list of possible download URLs for a tile."""
        match = re.search(r'N(\d+)E(\d+)', tile_name)
        if not match:
            return []

        north = match.group(1)
        east = match.group(2)

        urls = []
        for date_folder, filename_pattern in BEV_URL_PATTERNS:
            filename = filename_pattern.format(north=north, east=east)
            url = f"{BEV_DOWNLOAD_BASE}{date_folder}/{filename}"
            urls.append(url)

        return urls


class BEVDownloader:
    """Handles downloading and caching of BEV DEM tiles."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR, 
                 log_callback: Optional[Callable[[str], None]] = None,
                 progress_callback: Optional[Callable[[int], None]] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tile_calculator = BEVTileCalculator()
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self._cancelled = False
        
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Austria-DEM-Processor/1.0 (Wildlife Research)'
            })
        except ImportError:
            raise ImportError("requests is required. Install with: pip install requests")

    def _log(self, message: str):
        """Log a message."""
        logger.info(message)
        if self.log_callback:
            self.log_callback(message)

    def _progress(self, percent: int):
        """Report progress."""
        if self.progress_callback:
            self.progress_callback(percent)

    def cancel(self):
        """Cancel the download."""
        self._cancelled = True

    def _get_cache_path(self, tile_name: str) -> Path:
        """Get the cache path for a tile."""
        return self.cache_dir / f"{tile_name}.tif"

    def download_tile(self, tile_name: str, force: bool = False) -> Optional[Path]:
        """Download a single tile."""
        if self._cancelled:
            return None
            
        cache_path = self._get_cache_path(tile_name)

        if cache_path.exists() and not force:
            self._log(f"Using cached: {tile_name}")
            return cache_path

        urls = self.tile_calculator.get_download_urls(tile_name)

        for url in urls:
            if self._cancelled:
                return None
                
            self._log(f"Downloading: {tile_name}")
            logger.debug(f"  URL: {url}")

            try:
                response = self.session.get(url, stream=True, timeout=600)

                if response.status_code == 404:
                    logger.debug(f"  Not found at this URL, trying next...")
                    continue

                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                downloaded = 0
                with open(cache_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if self._cancelled:
                            f.close()
                            if cache_path.exists():
                                cache_path.unlink()
                            return None
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = int((downloaded / total_size) * 100)
                            logger.debug(f"  Progress: {percent}%")

                size_mb = cache_path.stat().st_size / 1024 / 1024
                self._log(f"Downloaded: {tile_name} ({size_mb:.1f} MB)")
                return cache_path

            except Exception as e:
                logger.debug(f"  Failed: {e}")
                continue

        self._log(f"Tile not found at any URL pattern: {tile_name}")
        if cache_path.exists():
            cache_path.unlink()
        return None

    def download_tiles_for_bbox(self, bbox: BoundingBox, force: bool = False) -> List[Path]:
        """Download all tiles needed for a bounding box."""
        self._cancelled = False
        tile_names = self.tile_calculator.get_required_tiles(bbox)

        downloaded_tiles = []
        total = len(tile_names)
        
        for i, tile_name in enumerate(tile_names):
            if self._cancelled:
                break
            self._progress(int((i / total) * 50))  # 0-50% for downloads
            tile_path = self.download_tile(tile_name, force=force)
            if tile_path:
                downloaded_tiles.append(tile_path)

        if not downloaded_tiles and not self._cancelled:
            self._log("No tiles were downloaded successfully")

        return downloaded_tiles


class DEMProcessor:
    """Processes DEM data - merging, clipping, and transforming."""

    def __init__(self, output_crs: str = DEFAULT_OUTPUT_CRS,
                 log_callback: Optional[Callable[[str], None]] = None,
                 progress_callback: Optional[Callable[[int], None]] = None):
        self.output_crs = output_crs
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self._cancelled = False

    def _log(self, message: str):
        """Log a message."""
        logger.info(message)
        if self.log_callback:
            self.log_callback(message)

    def _progress(self, percent: int):
        """Report progress."""
        if self.progress_callback:
            self.progress_callback(percent)

    def cancel(self):
        """Cancel processing."""
        self._cancelled = True

    def merge_and_clip(self, input_files: List[Path], output_file: Path,
                       bbox: BoundingBox) -> Optional[Path]:
        """Merge multiple GeoTIFF files and clip to bounding box."""
        try:
            import rasterio
            from rasterio.merge import merge
            from rasterio.mask import mask
            from rasterio.crs import CRS
            from shapely.geometry import box, mapping
        except ImportError as e:
            self._log(f"Missing required library: {e}")
            return None

        if not input_files:
            self._log("No input files to process")
            return None

        self._log(f"Processing {len(input_files)} tile(s)...")
        self._progress(55)

        try:
            # Open all files, forcing CRS to EPSG:3035
            src_files = []
            for f in input_files:
                src = rasterio.open(f)
                src_files.append(src)
            
            # Use PROJ4 string to avoid PROJ database version conflicts in QGIS
            # The BEV tiles are in EPSG:3035 but files don't encode it correctly
            src_crs = CRS.from_proj4(BEV_CRS_PROJ4)

            # Convert bbox to source CRS for clipping (using BEV_CRS string which maps to PROJ4)
            min_x, min_y, max_x, max_y = bbox.to_crs(BEV_CRS)
            clip_box = box(min_x, min_y, max_x, max_y)

            self._log(f"Clip bounds (EPSG:3035): {min_x:.1f}, {min_y:.1f} to {max_x:.1f}, {max_y:.1f}")

            if len(src_files) == 1:
                src = src_files[0]
                out_image, out_transform = mask(
                    src, [mapping(clip_box)], crop=True, all_touched=True
                )
                out_meta = src.meta.copy()
            else:
                mosaic, mosaic_transform = merge(src_files)

                merged_meta = src_files[0].meta.copy()
                merged_meta.update({
                    "driver": "GTiff",  # Explicitly set driver
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": mosaic_transform,
                    "crs": src_crs,  # Force correct CRS
                })

                temp_merged = output_file.parent / "temp_merged.tif"
                with rasterio.open(temp_merged, 'w', **merged_meta) as dst:
                    dst.write(mosaic)

                with rasterio.open(temp_merged) as src:
                    out_image, out_transform = mask(
                        src, [mapping(clip_box)], crop=True, all_touched=True
                    )
                    out_meta = src.meta.copy()

                temp_merged.unlink()

            # Force correct CRS in output metadata
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": src_crs,  # Force correct CRS
                "compress": "lzw"
            })

            with rasterio.open(output_file, "w", **out_meta) as dst:
                dst.write(out_image)

            for src in src_files:
                src.close()

            self._log(f"Clipped to: {output_file} ({out_image.shape[2]}x{out_image.shape[1]})")
            return output_file

        except Exception as e:
            self._log(f"Failed to process tiles: {e}")
            import traceback
            traceback.print_exc()
            return None

    def reproject_geotiff(self, input_file: Path, output_file: Path,
                          target_crs: Optional[str] = None) -> Optional[Path]:
        """Reproject a GeoTIFF to a different CRS."""
        try:
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject, Resampling
            from rasterio.crs import CRS
        except ImportError as e:
            self._log(f"Missing required library: {e}")
            return None

        if target_crs is None:
            target_crs = self.output_crs

        self._log(f"Reprojecting to {target_crs}...")
        self._progress(70)

        try:
            # Convert target CRS to PROJ4 to avoid PROJ database issues
            target_proj4 = get_proj4_for_crs(target_crs)
            dst_crs = CRS.from_proj4(target_proj4)
            
            with rasterio.open(input_file) as src:
                # Check if already in target CRS (compare PROJ4 representations)
                src_proj4 = src.crs.to_proj4() if src.crs else None
                if src_proj4 and src_proj4 == target_proj4:
                    self._log("Already in target CRS, copying...")
                    import shutil
                    shutil.copy2(input_file, output_file)
                    return output_file

                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )

                kwargs = src.meta.copy()
                kwargs.update({
                    'driver': 'GTiff',  # Explicitly set driver to avoid LIBERTIFF issues
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'compress': 'lzw'
                })

                with rasterio.open(output_file, 'w', **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear
                    )

                self._log(f"Reprojected to: {output_file}")
                return output_file

        except Exception as e:
            self._log(f"Failed to reproject GeoTIFF: {e}")
            return None


class GLTFMeshGenerator:
    """Generates GLTF meshes from DEM data."""

    def __init__(self, simplify_factor: int = 1,
                 log_callback: Optional[Callable[[str], None]] = None,
                 progress_callback: Optional[Callable[[int], None]] = None):
        self.simplify_factor = max(1, simplify_factor)
        self.log_callback = log_callback
        self.progress_callback = progress_callback

    def _log(self, message: str):
        """Log a message."""
        logger.info(message)
        if self.log_callback:
            self.log_callback(message)

    def _progress(self, percent: int):
        """Report progress."""
        if self.progress_callback:
            self.progress_callback(percent)

    def generate_mesh(self, geotiff_path: Path, output_path: Path,
                      metadata_path: Optional[Path] = None,
                      source_crs: Optional[str] = None) -> bool:
        """Generate a GLTF mesh from a GeoTIFF DEM.
        
        :param geotiff_path: Path to the input GeoTIFF
        :param output_path: Path for the output GLB file
        :param metadata_path: Optional path for metadata JSON
        :param source_crs: Optional CRS string (e.g., "EPSG:32633") to use if file CRS cannot be read
        """
        try:
            import rasterio
            from pyproj import Transformer, CRS as PyprojCRS
        except ImportError as e:
            self._log(f"Missing required library: {e}")
            return False

        self._log(f"Generating GLTF mesh from {geotiff_path.name}...")
        self._progress(80)

        try:
            with rasterio.open(geotiff_path) as src:
                elevation = src.read(1)

                nodata = src.nodata
                if nodata is not None:
                    elevation = np.where(elevation == nodata, np.nan, elevation)

                elevation = np.where(elevation == 0, np.nan, elevation)

                transform = src.transform
                crs = str(src.crs) if src.crs else None
                width = src.width
                height = src.height
                bounds = src.bounds

                # Origin is top-right corner of the DEM
                origin_x = bounds.left
                origin_y = bounds.bottom

                valid_elevations = elevation[~np.isnan(elevation)]
                if len(valid_elevations) == 0:
                    self._log("No valid elevation data")
                    return False

                min_elevation = float(np.nanmin(valid_elevations))
                origin_z = min_elevation

                # Use PROJ4 strings to avoid PROJ database version conflicts
                # Try multiple methods to get a valid source PROJ4 string
                src_proj4 = None
                
                # Method 1: Try to get PROJ4 from rasterio CRS
                if src.crs is not None:
                    try:
                        src_proj4 = src.crs.to_proj4()
                    except:
                        pass
                
                # Method 2: If that failed or returned None, try our mapping with file's CRS string
                if not src_proj4 and crs:
                    src_proj4 = get_proj4_for_crs(crs)
                
                # Method 3: Try to extract EPSG code and map it
                if not src_proj4 and src.crs is not None:
                    try:
                        epsg = src.crs.to_epsg()
                        if epsg:
                            src_proj4 = get_proj4_for_crs(f"EPSG:{epsg}")
                    except:
                        pass
                
                # Method 4: Use the passed source_crs parameter if available
                if not src_proj4 and source_crs:
                    self._log(f"Using provided source CRS: {source_crs}")
                    src_proj4 = get_proj4_for_crs(source_crs)
                
                # Method 5: Default to UTM 33N if all else fails (common for Austria)
                if not src_proj4:
                    self._log("Warning: Could not determine source CRS, assuming EPSG:32633")
                    src_proj4 = get_proj4_for_crs("EPSG:32633")
                
                dst_proj4 = get_proj4_for_crs(WGS84_CRS)
                
                transformer = Transformer.from_crs(
                    PyprojCRS.from_proj4(src_proj4),
                    PyprojCRS.from_proj4(dst_proj4),
                    always_xy=True
                )
                origin_lon, origin_lat = transformer.transform(origin_x, origin_y)

                pixel_size_x = abs(transform.a)
                pixel_size_y = abs(transform.e)

            self._progress(85)

            # Generate mesh
            step = self.simplify_factor
            rows = np.arange(0, height, step)
            cols = np.arange(0, width, step)

            vertices = []
            normals = []
            indices = []

            vertex_map = {}
            vertex_count = 0

            for row in rows:
                for col in cols:
                    if row < height and col < width:
                        z = elevation[row, col]
                        if np.isnan(z):
                            continue

                        x = col * pixel_size_x
                        y = (height - row - 1) * pixel_size_y
                        z_local = z - origin_z

                        # Z-up coordinate system: x=east, y=north, z=altitude
                        vertices.extend([x, y, z_local])
                        normals.extend([0, 0, 1])  # Normal pointing up in Z direction
                        vertex_map[(row, col)] = vertex_count
                        vertex_count += 1

            self._progress(90)

            for i, row in enumerate(rows[:-1]):
                for j, col in enumerate(cols[:-1]):
                    r0, c0 = row, col
                    r1, c1 = rows[i + 1] if i + 1 < len(rows) else row + step, cols[j + 1] if j + 1 < len(cols) else col + step

                    if r1 >= height:
                        r1 = height - 1
                    if c1 >= width:
                        c1 = width - 1

                    v00 = vertex_map.get((r0, c0))
                    v10 = vertex_map.get((r1, c0))
                    v01 = vertex_map.get((r0, c1))
                    v11 = vertex_map.get((r1, c1))

                    if v00 is not None and v10 is not None and v01 is not None:
                        indices.extend([v00, v10, v01])
                    if v10 is not None and v11 is not None and v01 is not None:
                        indices.extend([v10, v11, v01])

            if not vertices or not indices:
                self._log("No valid mesh data generated")
                return False

            self._progress(95)

            # Write metadata in expected format
            if metadata_path:
                # Use the source_crs parameter as the primary CRS (this is the output CRS we reprojected to)
                # Fall back to file CRS or default only if source_crs wasn't provided
                crs_string = source_crs if source_crs else (crs if crs else "EPSG:32633")
                
                # Convert Affine transform to list format [a, b, c, d, e, f, 0, 0, 1]
                transform_list = [
                    transform.a,  # scale x
                    transform.b,  # shear
                    transform.c,  # translation x
                    transform.d,  # shear
                    transform.e,  # scale y (usually negative)
                    transform.f,  # translation y
                    0.0,
                    0.0,
                    1.0
                ]
                
                # origin is in the projected coordinate system (e.g., EPSG:32633)
                # origin_x and origin_y come from bounds which are in the file's CRS
                metadata = {
                    "width": width,
                    "height": height,
                    "crs": crs_string,
                    "transform": transform_list,
                    "origin": [origin_x, origin_y, origin_z],
                    "origin_wgs84": {
                        "latitude": origin_lat,
                        "longitude": origin_lon,
                        "altitude": origin_z
                    }
                }
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                self._log(f"Created metadata: {metadata_path}")

            # Write GLTF
            self._write_gltf(output_path, vertices, normals, indices)
            return True

        except Exception as e:
            self._log(f"Failed to generate mesh: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _write_gltf(self, output_path: Path, vertices: list, normals: list, indices: list):
        """Write GLTF 2.0 binary file."""
        vertices_np = np.array(vertices, dtype=np.float32)
        normals_np = np.array(normals, dtype=np.float32)
        indices_np = np.array(indices, dtype=np.uint32)

        vertices_reshaped = vertices_np.reshape(-1, 3)
        v_min = vertices_reshaped.min(axis=0).tolist()
        v_max = vertices_reshaped.max(axis=0).tolist()

        normals_reshaped = normals_np.reshape(-1, 3)
        n_min = normals_reshaped.min(axis=0).tolist()
        n_max = normals_reshaped.max(axis=0).tolist()

        vertices_bytes = vertices_np.tobytes()
        normals_bytes = normals_np.tobytes()
        indices_bytes = indices_np.tobytes()

        def pad_to_4(data: bytes) -> bytes:
            padding = (4 - len(data) % 4) % 4
            return data + b'\x00' * padding

        vertices_bytes_padded = pad_to_4(vertices_bytes)
        normals_bytes_padded = pad_to_4(normals_bytes)
        indices_bytes_padded = pad_to_4(indices_bytes)

        vertices_offset = 0
        normals_offset = len(vertices_bytes_padded)
        indices_offset = normals_offset + len(normals_bytes_padded)

        buffer_data = vertices_bytes_padded + normals_bytes_padded + indices_bytes_padded
        buffer_length = len(buffer_data)

        gltf = {
            "asset": {"version": "2.0", "generator": "Austria-DEM-Processor"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4}]}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": len(vertices) // 3, "type": "VEC3", "min": v_min,
                 "max": v_max},
                {"bufferView": 1, "componentType": 5126, "count": len(normals) // 3, "type": "VEC3", "min": n_min,
                 "max": n_max},
                {"bufferView": 2, "componentType": 5125, "count": len(indices), "type": "SCALAR",
                 "min": [int(min(indices))], "max": [int(max(indices))]}
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": vertices_offset, "byteLength": len(vertices_bytes), "target": 34962},
                {"buffer": 0, "byteOffset": normals_offset, "byteLength": len(normals_bytes), "target": 34962},
                {"buffer": 0, "byteOffset": indices_offset, "byteLength": len(indices_bytes), "target": 34963}
            ],
            "buffers": [{"byteLength": buffer_length}]
        }

        output_glb = output_path.with_suffix('.glb')

        json_str = json.dumps(gltf, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')
        json_padding = (4 - len(json_bytes) % 4) % 4
        json_bytes_padded = json_bytes + b' ' * json_padding

        glb_length = 12 + 8 + len(json_bytes_padded) + 8 + buffer_length

        with open(output_glb, 'wb') as f:
            f.write(b'glTF')
            f.write(struct.pack('<I', 2))
            f.write(struct.pack('<I', glb_length))
            f.write(struct.pack('<I', len(json_bytes_padded)))
            f.write(b'JSON')
            f.write(json_bytes_padded)
            f.write(struct.pack('<I', buffer_length))
            f.write(b'BIN\x00')
            f.write(buffer_data)

        self._log(f"Created GLTF mesh: {output_glb}")
        self._log(f"  Vertices: {len(vertices) // 3}")
        self._log(f"  Triangles: {len(indices) // 3}")
        file_size_mb = output_glb.stat().st_size / 1024 / 1024
        self._log(f"  File size: {file_size_mb:.2f} MB")


class DEMDownloadWorker(QObject):
    """Worker for downloading DEM in a background thread."""
    
    finished = pyqtSignal(bool, str)  # success, message/path
    progress = pyqtSignal(int)  # percentage
    log = pyqtSignal(str)  # log message
    
    def __init__(self, csv_path: str, output_folder: str, padding: float = 30.0,
                 output_crs: str = DEFAULT_OUTPUT_CRS, simplify_factor: int = 2):
        super().__init__()
        self.csv_path = csv_path
        self.output_folder = output_folder
        self.padding = padding
        self.output_crs = output_crs
        self.simplify_factor = simplify_factor
        self._cancelled = False
        self._downloader = None
        self._processor = None

    def cancel(self):
        """Cancel the download."""
        self._cancelled = True
        if self._downloader:
            self._downloader.cancel()
        if self._processor:
            self._processor.cancel()

    def run(self):
        """Execute the DEM download process."""
        try:
            self.log.emit("Starting DEM download for Austria...")
            self.progress.emit(5)
            
            # Parse CSV to get bounding box
            self.log.emit(f"Reading GPS coordinates from: {os.path.basename(self.csv_path)}")
            bbox = BoundingBox.from_airdata_csv(self.csv_path, self.padding)
            
            if bbox is None:
                self.finished.emit(False, "Failed to extract GPS coordinates from CSV")
                return
            
            self.log.emit(f"Bounding box (with {self.padding}m padding):")
            self.log.emit(f"  SW: {bbox.min_lat:.6f}, {bbox.min_lon:.6f}")
            self.log.emit(f"  NE: {bbox.max_lat:.6f}, {bbox.max_lon:.6f}")
            self.progress.emit(10)
            
            # Initialize components
            self._downloader = BEVDownloader(
                log_callback=lambda msg: self.log.emit(msg),
                progress_callback=lambda p: self.progress.emit(10 + int(p * 0.4))  # 10-50%
            )
            
            self._processor = DEMProcessor(
                output_crs=self.output_crs,
                log_callback=lambda msg: self.log.emit(msg),
                progress_callback=lambda p: self.progress.emit(50 + int(p * 0.3))  # 50-80%
            )
            
            mesh_generator = GLTFMeshGenerator(
                simplify_factor=self.simplify_factor,
                log_callback=lambda msg: self.log.emit(msg),
                progress_callback=lambda p: self.progress.emit(80 + int(p * 0.2))  # 80-100%
            )
            
            # Download tiles
            self.log.emit("Downloading required DEM tiles...")
            tile_paths = self._downloader.download_tiles_for_bbox(bbox)
            
            if self._cancelled:
                self.finished.emit(False, "Download cancelled")
                return
            
            if not tile_paths:
                self.finished.emit(False, "No tiles downloaded - area may be outside Austria")
                return
            
            # Setup output path
            output_base = Path(self.output_folder) / "dem"
            output_base.parent.mkdir(parents=True, exist_ok=True)
            
            # Process in temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Merge and clip
                self.log.emit("Merging and clipping tiles...")
                clipped_file = temp_path / "clipped.tif"
                result = self._processor.merge_and_clip(tile_paths, clipped_file, bbox)
                
                if self._cancelled:
                    self.finished.emit(False, "Processing cancelled")
                    return
                
                if not result:
                    self.finished.emit(False, "Failed to merge and clip tiles")
                    return
                
                # Reproject
                self.log.emit("Reprojecting to output CRS...")
                reprojected_file = temp_path / "reprojected.tif"
                result = self._processor.reproject_geotiff(clipped_file, reprojected_file)
                
                if self._cancelled:
                    self.finished.emit(False, "Processing cancelled")
                    return
                
                if not result:
                    self.finished.emit(False, "Failed to reproject GeoTIFF")
                    return
                
                # Copy final GeoTIFF
                final_geotiff = output_base.with_suffix('.tif')
                import shutil
                shutil.copy2(reprojected_file, final_geotiff)
                self.log.emit(f"Created output GeoTIFF: {final_geotiff}")
                
                # Generate mesh
                self.log.emit("Generating GLTF mesh...")
                mesh_file = output_base.with_suffix('.glb')
                metadata_file = output_base.with_suffix('.json')
                
                # Pass the output_crs so mesh generator knows the CRS if it can't read it from file
                success = mesh_generator.generate_mesh(
                    final_geotiff, mesh_file, metadata_file, 
                    source_crs=self.output_crs
                )
                
                if not success:
                    self.finished.emit(False, "Failed to generate mesh")
                    return
            
            self.progress.emit(100)
            self.log.emit("=" * 50)
            self.log.emit("DEM download and processing complete!")
            self.log.emit(f"  GeoTIFF:  {final_geotiff}")
            self.log.emit(f"  Mesh:     {mesh_file}")
            self.log.emit(f"  Metadata: {metadata_file}")
            self.log.emit("=" * 50)
            
            self.finished.emit(True, str(mesh_file))
            
        except Exception as e:
            import traceback
            error_msg = f"Error during DEM download: {str(e)}"
            logger.error(error_msg)
            traceback.print_exc()
            self.finished.emit(False, error_msg)

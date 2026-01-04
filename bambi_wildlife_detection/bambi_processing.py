# -*- coding: utf-8 -*-
"""
BAMBI Wildlife Detection - Processing Module
=============================================

This module contains the processing logic for all pipeline steps.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict

from qgis.PyQt.QtCore import QObject, pyqtSignal


class ProcessingWorker(QObject):
    """Worker class for background processing."""
    
    finished = pyqtSignal(str, bool)  # step, success
    error = pyqtSignal(str, str)  # step, error_message
    progress = pyqtSignal(int)  # percentage
    log = pyqtSignal(str)  # log message
    
    def __init__(self, processor: 'BambiProcessor', step: str, config: Dict[str, Any]):
        super().__init__()
        self.processor = processor
        self.step = step
        self.config = config
        
    def run(self):
        """Execute the processing step."""
        try:
            if self.step == "extract_frames":
                self.processor.extract_frames(self.config, self.progress.emit, self.log.emit)
            elif self.step == "flight_route":
                self.processor.run_flight_route(self.config, self.progress.emit, self.log.emit)
            elif self.step == "detection":
                self.processor.run_detection(self.config, self.progress.emit, self.log.emit)
            elif self.step == "georeference":
                self.processor.run_georeference(self.config, self.progress.emit, self.log.emit)
            elif self.step == "calculate_fov":
                self.processor.run_calculate_fov(self.config, self.progress.emit, self.log.emit)
            elif self.step == "tracking":
                self.processor.run_tracking(self.config, self.progress.emit, self.log.emit)
            elif self.step == "orthomosaic":
                self.processor.run_orthomosaic(self.config, self.progress.emit, self.log.emit)
            elif self.step == "export_geotiffs":
                self.processor.run_export_geotiffs(self.config, self.progress.emit, self.log.emit)
            else:
                raise ValueError(f"Unknown step: {self.step}")
                
            self.finished.emit(self.step, True)
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(self.step, f"{str(e)}\n\n{tb}")
            self.finished.emit(self.step, False)


class BambiProcessor:
    """Main processing class for BAMBI wildlife detection pipeline."""
    
    DEFAULT_MODEL_REPO = "cpraschl/bambi-thermal-detection"
    DEFAULT_MODEL_FILENAME = "thermal_animal_detector.pt"
    
    def __init__(self):
        """Initialize the processor."""
        pass
        
    def download_default_model(self, target_folder: str, log_fn=None) -> str:
        """Download the default model from HuggingFace.
        
        :param target_folder: Folder to save the model
        :param log_fn: Optional logging function
        :return: Path to the downloaded model
        """
        import urllib.request
        import ssl
            
        models_folder = os.path.join(target_folder, "models")
        os.makedirs(models_folder, exist_ok=True)
        
        model_path = os.path.join(models_folder, self.DEFAULT_MODEL_FILENAME)
        
        if os.path.exists(model_path):
            # Check if file is not empty (incomplete download)
            if os.path.getsize(model_path) > 1000000:  # > 1MB
                if log_fn:
                    log_fn(f"Model already exists at {model_path}")
                return model_path
            else:
                # Remove incomplete file
                os.remove(model_path)
            
        if log_fn:
            log_fn(f"Downloading model from HuggingFace: {self.DEFAULT_MODEL_REPO}")
            log_fn("This may take a few minutes (model is ~6MB)...")
        
        # Direct download URL from HuggingFace
        url = f"https://huggingface.co/{self.DEFAULT_MODEL_REPO}/resolve/main/{self.DEFAULT_MODEL_FILENAME}"
        
        try:
            # Create SSL context that doesn't verify certificates (for corporate networks)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Download with progress reporting
            def report_progress(block_num, block_size, total_size):
                if log_fn and block_num % 100 == 0:
                    downloaded = block_num * block_size
                    if total_size > 0:
                        percent = min(100, (downloaded / total_size) * 100)
                        log_fn(f"Download progress: {percent:.1f}% ({downloaded // 1024} KB)")
            
            # Try with custom SSL context first
            try:
                opener = urllib.request.build_opener(
                    urllib.request.HTTPSHandler(context=ssl_context)
                )
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(url, model_path, reporthook=report_progress)
            except Exception as e1:
                if log_fn:
                    log_fn(f"Download with custom SSL failed: {e1}")
                    log_fn("Trying standard download...")
                # Try standard download
                urllib.request.urlretrieve(url, model_path, reporthook=report_progress)
            
            # Verify download
            if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
                if log_fn:
                    log_fn(f"Model downloaded successfully to: {model_path}")
                return model_path
            else:
                raise RuntimeError("Downloaded file is too small or missing")
                
        except Exception as e:
            # Clean up partial download
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                except:
                    pass
                    
            error_msg = (
                f"Failed to download model automatically.\n\n"
                f"Please download manually:\n"
                f"1. Go to: https://huggingface.co/{self.DEFAULT_MODEL_REPO}\n"
                f"2. Download: {self.DEFAULT_MODEL_FILENAME}\n"
                f"3. Save to: {model_path}\n\n"
                f"Or specify a local model path in the Parameters tab.\n\n"
                f"Error: {str(e)}"
            )
            raise RuntimeError(error_msg)
        
    def extract_frames(self, config: Dict[str, Any], progress_fn=None, log_fn=None):
        """Extract frames from drone videos with configurable sample rate.
        
        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        import cv2
        import datetime
        import numpy as np
        from typing import Any, List
        from pyproj import CRS, Transformer
        
        # Import bambi modules
        from bambi.video.calibrated_video_frame_accessor import CalibratedVideoFrameAccessor
        from bambi.webgl.timed_pose_extractor import TimedPoseExtractor
        from bambi.domain.camera import Camera
        from bambi.domain.drone import Drone
        from bambi.domain.sensor import SensorResolution
        from bambi.airdata.air_data_frame import AirDataFrame
        from bambi.airdata.air_data_interpolator import AirDataTimeInterpolator
        from bambi.srt.srt_parser import SrtParser
        from bambi.airdata.air_data_parser import AirDataParser
        
        if log_fn:
            log_fn("Initializing frame extraction...")
            
        target_folder = config["target_folder"]
        video_paths = config["video_paths"]
        srt_paths = config["srt_paths"]
        airdata_path = config["airdata_path"]
        calibration_path = config["calibration_path"]
        camera_name = config.get("camera", "T")
        target_epsg = config.get("target_epsg", 32633)
        sample_rate = config.get("sample_rate", 1)  # Extract every Nth frame
        
        # Create target folder and frames subfolder
        os.makedirs(target_folder, exist_ok=True)
        frames_folder = os.path.join(target_folder, "frames")
        os.makedirs(frames_folder, exist_ok=True)
        
        # Setup coordinate transformer
        input_crs = CRS.from_epsg(4326)
        target_crs = CRS.from_epsg(target_epsg)
        rel_transformer = Transformer.from_crs(input_crs, target_crs)
        
        # Load calibration
        if log_fn:
            log_fn(f"Loading calibration from {calibration_path}")
            
        with open(calibration_path, 'r') as f:
            calibration_res = json.load(f)
            
        # Get origin from DEM if available
        dem_path = config.get("dem_path")
        origin = None
        dem_origin_data = None
        
        if dem_path:
            dem_json_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
            if os.path.exists(dem_json_path):
                with open(dem_json_path, 'r') as f:
                    dem_json = json.load(f)
                if "origin_wgs84" in dem_json:
                    dem_origin_data = dem_json["origin_wgs84"]
                    origin = AirDataFrame()
                    origin.latitude = dem_origin_data.get("latitude")
                    origin.longitude = dem_origin_data.get("longitude")
                    origin.altitude = dem_origin_data.get("altitude", 0)
                    if log_fn:
                        log_fn(f"Using origin from DEM: {origin.latitude}, {origin.longitude}")
        
        # Create frame accessor
        accessor = CalibratedVideoFrameAccessor(calibration_res)
        camera = Camera.from_string(camera_name)
        drone = Drone.M30T
        
        if log_fn:
            log_fn(f"Extracting frames from {len(video_paths)} video(s) with sample rate {sample_rate}...")
            
        if progress_fn:
            progress_fn(5)
        
        # Parse SRT files to get frame timestamps
        srt_parser = SrtParser()
        all_srt_frames = []
        frame_to_video = []
        
        for vid_idx, srt_path in enumerate(srt_paths):
            frames = srt_parser.parse(srt_path)
            from dateutil import tz
            timezone = tz.gettz('Europe/Vienna')
            for frame in frames:
                frame.timestamp = frame.timestamp.replace(tzinfo=timezone)
            all_srt_frames.extend(frames)
            frame_to_video.extend([vid_idx] * len(frames))
        
        if log_fn:
            log_fn(f"Parsed {len(all_srt_frames)} SRT frames")
        
        # Parse AirData
        ad_parser = AirDataParser()
        ad_frames = ad_parser.parse(airdata_path)
        
        # Get first SRT timestamp for AirData alignment
        first_srt_timestamp = all_srt_frames[0].timestamp if all_srt_frames else None
        
        if progress_fn:
            progress_fn(10)
        
        # Extract frames with sampling
        image_files: List[str] = []
        image_timestamps: List[datetime.datetime] = []
        sampled_srt_frames: List[Any] = []
        
        total_frames = len(all_srt_frames)
        frames_to_extract = list(range(0, total_frames, sample_rate))
        
        if log_fn:
            log_fn(f"Will extract {len(frames_to_extract)} of {total_frames} frames (sample rate: {sample_rate})")
        
        current_video_idx = -1
        cap = None
        
        for extract_idx, frame_idx in enumerate(frames_to_extract):
            video_idx = frame_to_video[frame_idx]
            
            # Open new video if needed
            if video_idx != current_video_idx:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(video_paths[video_idx])
                current_video_idx = video_idx
                if log_fn:
                    log_fn(f"Processing video {video_idx + 1}/{len(video_paths)}: {os.path.basename(video_paths[video_idx])}")
            
            # Calculate frame position within current video
            frames_before = sum(1 for i, v in enumerate(frame_to_video[:frame_idx]) if v == video_idx)
            local_frame_idx = frame_idx - (frame_to_video.index(video_idx) if video_idx in frame_to_video else 0)
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                if log_fn:
                    log_fn(f"Warning: Could not read frame {frame_idx}")
                continue
            
            # Undistort frame
            undistorted = accessor.undistort(frame)
            
            # Get SRT frame info
            srt_frame = all_srt_frames[frame_idx]
            
            # Save frame
            filename = f"{len(image_files):06d}-{frame_idx}-{srt_frame.id}.jpg"
            filepath = os.path.join(frames_folder, filename)
            cv2.imwrite(filepath, undistorted)
            
            image_files.append(filename)
            image_timestamps.append(srt_frame.timestamp)
            sampled_srt_frames.append(srt_frame)
            
            # Update progress
            if progress_fn and extract_idx % 20 == 0:
                progress = 10 + int((extract_idx / len(frames_to_extract)) * 70)
                progress_fn(min(progress, 80))
        
        if cap is not None:
            cap.release()
        
        if progress_fn:
            progress_fn(80)
        
        if log_fn:
            log_fn(f"Extracted {len(image_files)} frames, building poses.json...")
        
        # Interpolate AirData for extracted frame timestamps
        try:
            interpolator = AirDataTimeInterpolator(ad_frames)
            interpolated_frames = interpolator(image_timestamps)
        except Exception as e:
            if log_fn:
                log_fn(f"Warning: AirData interpolation failed: {e}, using SRT data only")
            interpolated_frames = None
        
        # Build origin
        if origin is not None:
            origin_transformed = rel_transformer.transform(origin.latitude, origin.longitude)
            origin_altitude = origin.altitude or 0
        else:
            # Use first frame as origin
            if interpolated_frames and interpolated_frames[0]:
                origin = AirDataFrame()
                origin.latitude = interpolated_frames[0].latitude
                origin.longitude = interpolated_frames[0].longitude
                origin.altitude = interpolated_frames[0].altitude or 0
            elif sampled_srt_frames:
                origin = AirDataFrame()
                origin.latitude = sampled_srt_frames[0].latitude
                origin.longitude = sampled_srt_frames[0].longitude
                origin.altitude = sampled_srt_frames[0].altitude or 0
            origin_transformed = rel_transformer.transform(origin.latitude, origin.longitude)
            origin_altitude = origin.altitude or 0
        
        # Build poses
        images_data: List[dict] = []
        
        for i, (image_file, timestamp) in enumerate(zip(image_files, image_timestamps)):
            # Get frame data from interpolated AirData or SRT
            if interpolated_frames and i < len(interpolated_frames):
                frame_data = interpolated_frames[i]
                lat = frame_data.latitude
                lon = frame_data.longitude
                alt = frame_data.altitude or 0
                gimbal_pitch = frame_data.gimbal_pitch or -90
                compass_heading = frame_data.compass_heading or 0
            else:
                srt_frame = sampled_srt_frames[i]
                lat = srt_frame.latitude
                lon = srt_frame.longitude
                alt = srt_frame.altitude or 0
                gimbal_pitch = getattr(srt_frame, 'gimbal_pitch', -90) or -90
                compass_heading = getattr(srt_frame, 'compass_heading', 0) or 0
            
            frame_coord = rel_transformer.transform(lat, lon)
            location = [
                frame_coord[0] - origin_transformed[0],
                frame_coord[1] - origin_transformed[1],
                alt - origin_altitude,
            ]
            
            rotation = [
                (float(gimbal_pitch) + 90) % 360,
                0,  # roll
                compass_heading if compass_heading else 0.0,
            ]
            
            current_dict = {
                "imagefile": f"frames/{image_file}",
                "location": location,
                "rotation": rotation,
                "lat": lat,
                "lng": lon,
                "timestamp": timestamp.isoformat() if timestamp else None,
            }
            
            # Add fovy from calibration if available
            if hasattr(accessor, 'undistortion_parameters') and accessor.undistortion_parameters.fovy:
                current_dict["fovy"] = (accessor.undistortion_parameters.fovy,)
            else:
                current_dict["fovy"] = (50.0,)  # Default FOV
            
            images_data.append(current_dict)
        
        # Write mask image
        mask_filename = f"mask_{camera_name.lower()}.png"
        mask_path = os.path.join(target_folder, mask_filename)
        try:
            sr = SensorResolution(drone, camera)
            mask = accessor.create_distortion_mask(sr.width, sr.height)
            mask[mask < 255] = 0
            cv2.imwrite(mask_path, mask)
        except Exception as e:
            if log_fn:
                log_fn(f"Warning: Could not create mask: {e}")
            mask_filename = None
        
        # Build final poses dict
        poses_result = {
            "images": images_data,
            "origin": {
                "latitude": origin.latitude,
                "longitude": origin.longitude,
                "altitude": origin.altitude
            },
            "drone": Drone.product_name(drone) if drone else "Unknown",
            "camera": Camera.fullname(camera) if camera else "Unknown",
            "samplingRate": sample_rate,
        }
        
        if mask_filename:
            poses_result["mask"] = mask_filename
        
        # Write poses.json
        poses_file = os.path.join(target_folder, "poses.json")
        with open(poses_file, 'w', encoding='utf-8') as f:
            json.dump(poses_result, f, indent=2)
        
        if progress_fn:
            progress_fn(95)
        
        if log_fn:
            log_fn(f"Successfully extracted {len(image_files)} frames with sample rate {sample_rate}")
            log_fn(f"Output: {poses_file}")
        
        if progress_fn:
            progress_fn(100)
    
    def run_flight_route(self, config: Dict[str, Any], progress_fn=None, log_fn=None):
        """Generate a flight route polyline layer from camera positions.
        
        Creates a GeoJSON file with a LineString representing the drone's
        flight path based on the camera positions in poses.json.
        
        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        if log_fn:
            log_fn("Generating flight route...")
        
        target_folder = config["target_folder"]
        target_epsg = config.get("target_epsg", 32633)
        
        if progress_fn:
            progress_fn(10)
        
        # Load poses.json
        poses_file = os.path.join(target_folder, "poses.json")
        if not os.path.exists(poses_file):
            raise FileNotFoundError(f"poses.json not found at {poses_file}")
        
        with open(poses_file, 'r') as f:
            poses = json.load(f)
        
        images = poses.get("images", [])
        if len(images) < 2:
            raise RuntimeError("Need at least 2 frames to create a flight route")
        
        if log_fn:
            log_fn(f"Creating flight route from {len(images)} positions")
        
        if progress_fn:
            progress_fn(20)
        
        # Get coordinate offset from DEM metadata
        coord_offset_x = 0.0
        coord_offset_y = 0.0
        
        dem_metadata_path = config.get("ortho_dem_metadata_path")
        dem_path = config.get("dem_path")
        
        if dem_metadata_path and os.path.exists(dem_metadata_path):
            try:
                with open(dem_metadata_path, 'r') as f:
                    dem_metadata = json.load(f)
                origin = dem_metadata.get("origin", [0, 0, 0])
                coord_offset_x = float(origin[0])
                coord_offset_y = float(origin[1])
            except Exception:
                pass
        elif dem_path:
            auto_metadata_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
            if os.path.exists(auto_metadata_path):
                try:
                    with open(auto_metadata_path, 'r') as f:
                        dem_metadata = json.load(f)
                    origin = dem_metadata.get("origin", [0, 0, 0])
                    coord_offset_x = float(origin[0])
                    coord_offset_y = float(origin[1])
                except Exception:
                    pass
        
        if log_fn and (coord_offset_x != 0 or coord_offset_y != 0):
            log_fn(f"Using coordinate offset: X={coord_offset_x:.2f}, Y={coord_offset_y:.2f}")
        
        if progress_fn:
            progress_fn(40)
        
        # Build coordinates list for the flight path
        coordinates = []
        point_features = []
        
        for i, img_info in enumerate(images):
            location = img_info.get("location", [0, 0, 0])
            
            # Apply coordinate offset to get real-world coordinates
            x = location[0] + coord_offset_x
            y = location[1] + coord_offset_y
            z = location[2]
            
            coordinates.append([x, y, z])
            
            # Also create point features for each camera position
            point_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [x, y, z]
                },
                "properties": {
                    "frame_idx": i,
                    "imagefile": img_info.get("imagefile", ""),
                    "altitude": z,
                    "timestamp": img_info.get("timestamp", "")
                }
            })
        
        if progress_fn:
            progress_fn(60)
        
        # Create output folder
        route_folder = os.path.join(target_folder, "flight_route")
        os.makedirs(route_folder, exist_ok=True)
        
        # Create GeoJSON for the flight path line
        flight_line_geojson = {
            "type": "FeatureCollection",
            "name": "flight_route",
            "crs": {
                "type": "name",
                "properties": {
                    "name": f"urn:ogc:def:crs:EPSG::{target_epsg}"
                }
            },
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    },
                    "properties": {
                        "name": "Flight Route",
                        "total_frames": len(images),
                        "start_time": images[0].get("timestamp", ""),
                        "end_time": images[-1].get("timestamp", "")
                    }
                }
            ]
        }
        
        # Save flight route line
        route_line_file = os.path.join(route_folder, "flight_route.geojson")
        with open(route_line_file, 'w', encoding='utf-8') as f:
            json.dump(flight_line_geojson, f, indent=2)
        
        if log_fn:
            log_fn(f"Flight route line saved to: {route_line_file}")
        
        if progress_fn:
            progress_fn(80)
        
        # Create GeoJSON for camera positions (points)
        camera_points_geojson = {
            "type": "FeatureCollection",
            "name": "camera_positions",
            "crs": {
                "type": "name",
                "properties": {
                    "name": f"urn:ogc:def:crs:EPSG::{target_epsg}"
                }
            },
            "features": point_features
        }
        
        # Save camera positions
        camera_points_file = os.path.join(route_folder, "camera_positions.geojson")
        with open(camera_points_file, 'w', encoding='utf-8') as f:
            json.dump(camera_points_geojson, f, indent=2)
        
        if log_fn:
            log_fn(f"Camera positions saved to: {camera_points_file}")
            log_fn(f"Flight route generation complete")
        
        if progress_fn:
            progress_fn(100)
            
    def run_detection(self, config: Dict[str, Any], progress_fn=None, log_fn=None):
        """Run animal detection on extracted frames.
        
        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        import cv2
        
        from bambi.ai.models.ultralytics_yolo_detector import UltralyticsYoloDetector
        from bambi.ai.output.yolo_writer import YoloWriter
        
        target_folder = config["target_folder"]
        model_path = config.get("model_path")
        min_confidence = config.get("min_confidence", 0.5)
        
        # Download default model if not specified
        if not model_path:
            if log_fn:
                log_fn("No model specified, downloading default model...")
            model_path = self.download_default_model(target_folder, log_fn)
            
        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        if log_fn:
            log_fn(f"Using model: {model_path}")
            
        # Load poses to get frame list
        poses_file = os.path.join(target_folder, "poses.json")
        with open(poses_file, 'r') as f:
            poses = json.load(f)
            
        images = poses.get("images", [])
        total_frames = len(images)
        extraction_sample_rate = poses.get("samplingRate", 1)
        
        if total_frames == 0:
            raise RuntimeError("No frames found in poses.json")
            
        if log_fn:
            log_fn(f"Processing {total_frames} extracted frames (extraction sample rate was {extraction_sample_rate})")
            
        # Create output folder
        detections_folder = os.path.join(target_folder, "detections")
        os.makedirs(detections_folder, exist_ok=True)
        
        # Initialize detector
        detector = UltralyticsYoloDetector(
            model_path=model_path,
            labels=['animal'],
            detection_writer=YoloWriter(),
            min_confidence=min_confidence,
            verbose=False
        )
        
        # Process ALL extracted frames (sampling was already done during extraction)
        detection_results = []
        processed = 0
        
         # Write detection results
        output_file = os.path.join(detections_folder, "detections.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# frame x1 y1 x2 y2 confidence class_id\n")
            for idx, image_info in enumerate(images):
                imagefile = image_info.get("imagefile")
                if not imagefile:
                    continue
                    
                image_path = os.path.join(target_folder, imagefile)
                log_fn(f"Detecting frame {idx} / {total_frames}: {image_path}")
                
                if not os.path.exists(image_path):
                    if log_fn:
                        log_fn(f"Warning: Image not found: {image_path}")
                    continue
                    
                # Load and detect
                frame = cv2.imread(image_path)
                if frame is None:
                    continue
                    
                boxes = detector.detect_frame(idx, frame)
                
                for box in boxes:
                    f.write(f"{idx} {box.start_x:.2f} {box.start_y:.2f} "
                       f"{box.end_x:.2f} {box.end_y:.2f} {box.propability:.4f} "
                       f"{box.label}\n")
                
                processed += 1
                
                if progress_fn and processed % 10 == 0:
                    progress = int((processed / total_frames) * 100)
                    progress_fn(min(progress, 99))
                
                       
        if log_fn:
            log_fn(f"Detection complete: {len(detection_results)} detections in {processed} frames")
            
        if progress_fn:
            progress_fn(100)
            
    def run_georeference(self, config: Dict[str, Any], progress_fn=None, log_fn=None):
        """Geo-reference detections using DEM.
        
        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        import numpy as np
        import math
        
        from pyproj import CRS, Transformer
        from pyrr import Vector3, Quaternion
        from trimesh import Trimesh
        
        from alfspy.core.rendering import Resolution
        from alfspy.render.render import read_gltf, process_render_data, make_mgl_context, release_all
        from bambi.util.projection_util import label_to_world_coordinates
        
        target_folder = config["target_folder"]
        dem_path = config["dem_path"]
        target_epsg = config.get("target_epsg", 32633)
        
        # Correction factors
        translation = config.get("translation", {"x": 0, "y": 0, "z": 0})
        rotation = config.get("rotation", {"x": 0, "y": 0, "z": 0})
        
        if log_fn:
            log_fn("Loading DEM and poses data...")
            
        # Load DEM metadata
        dem_json_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
        with open(dem_json_path, 'r') as f:
            dem_json = json.load(f)
            
        x_offset = dem_json["origin"][0]
        y_offset = dem_json["origin"][1]
        z_offset = dem_json["origin"][2]
        
        # Load poses
        poses_file = os.path.join(target_folder, "poses.json")
        with open(poses_file, 'r') as f:
            poses = json.load(f)
            
        # Get input resolution from first image
        first_image = poses["images"][0]
        if "fovy" in first_image:
            # Assuming square images from calibration
            input_resolution = Resolution(512, 512)  # Will be updated from actual images
        else:
            input_resolution = Resolution(512, 512)
            
        # Load detections
        detections_file = os.path.join(target_folder, "detections", "detections.txt")
        if not os.path.exists(detections_file):
            raise FileNotFoundError("Detections file not found")
            
        detections = []
        with open(detections_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    detections.append({
                        "frame": int(parts[0]),
                        "x1": float(parts[1]),
                        "y1": float(parts[2]),
                        "x2": float(parts[3]),
                        "y2": float(parts[4]),
                        "confidence": float(parts[5]),
                        "class_id": int(parts[6]) if len(parts) > 6 else 0
                    })
                    
        if log_fn:
            log_fn(f"Loaded {len(detections)} detections to geo-reference")
            
        if progress_fn:
            progress_fn(10)
            
        # Create output folder
        georef_folder = os.path.join(target_folder, "georeferenced")
        os.makedirs(georef_folder, exist_ok=True)
        
        # Setup transformer
        input_crs = CRS.from_epsg(4326)
        target_crs = CRS.from_epsg(target_epsg)
        rel_transformer = Transformer.from_crs(input_crs, target_crs)
        
        # Load DEM mesh
        if log_fn:
            log_fn("Loading DEM mesh...")
            
        ctx = None
        try:
            mesh_data, texture_data = read_gltf(dem_path)
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)
            
            if progress_fn:
                progress_fn(30)
                
            # Process each detection
            georeferenced = []
            cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
            cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')
            
            total_dets = len(detections)
            
            for idx, det in enumerate(detections):
                frame_idx = det["frame"]
                
                if frame_idx >= len(poses["images"]):
                    continue
                    
                image_metadata = poses["images"][frame_idx]
                
                # Get camera for this frame
                fovy = image_metadata.get("fovy", [50])[0]
                position = Vector3(image_metadata["location"])
                rot = image_metadata["rotation"]
                rotation_eulers = (Vector3(
                    [np.deg2rad(val % 360.0) for val in rot]) - cor_rotation_eulers) * -1
                position += cor_translation
                rotation_quat = Quaternion.from_eulers(rotation_eulers)
                
                # Create camera-like object for projection
                from alfspy.core.rendering import Camera
                camera = Camera(fovy=fovy, aspect_ratio=1.0, position=position, rotation=rotation_quat)
                
                # Project bounding box corners
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                label_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                
                try:
                    world_coords = label_to_world_coordinates(
                        label_coords, input_resolution, tri_mesh, camera
                    )
                    
                    if len(world_coords) > 0:
                        xx = world_coords[:, 0] + x_offset
                        yy = world_coords[:, 1] + y_offset
                        zz = world_coords[:, 2] + z_offset
                        
                        georeferenced.append({
                            "frame": frame_idx,
                            "x1": min(xx), "y1": min(yy), "z1": min(zz),
                            "x2": max(xx), "y2": max(yy), "z2": max(zz),
                            "confidence": det["confidence"],
                            "class_id": det["class_id"]
                        })
                except Exception as e:
                    if log_fn:
                        log_fn(f"Warning: Could not geo-reference detection at frame {frame_idx}: {e}")
                    continue
                    
                if progress_fn and idx % 50 == 0:
                    progress = 30 + int((idx / total_dets) * 60)
                    progress_fn(min(progress, 95))
                    
        finally:
            if ctx:
                release_all(ctx)
            del mesh_data
            del texture_data
            del tri_mesh
            
        # Write georeferenced results
        output_file = os.path.join(georef_folder, "georeferenced.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# idx frame min_x min_y min_z max_x max_y max_z confidence class_id\n")
            for idx, det in enumerate(georeferenced):
                f.write(f"{idx} {det['frame']} {det['x1']:.6f} {det['y1']:.6f} {det['z1']:.6f} "
                       f"{det['x2']:.6f} {det['y2']:.6f} {det['z2']:.6f} "
                       f"{det['confidence']:.4f} {det['class_id']}\n")
                       
        if log_fn:
            log_fn(f"Geo-referencing complete: {len(georeferenced)} detections")
            
        if progress_fn:
            progress_fn(100)
            
    def run_calculate_fov(self, config: Dict[str, Any], progress_fn=None, log_fn=None):
        """Calculate and save Field of View (FoV) polygons for each frame.
        
        This geo-references the image corners for each frame to create a polygon
        showing the camera's field of view on the ground.
        
        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        import numpy as np
        import cv2
        
        from pyproj import CRS, Transformer
        from pyrr import Vector3, Quaternion
        from trimesh import Trimesh
        
        from alfspy.core.rendering import Resolution
        from alfspy.render.render import read_gltf, process_render_data, make_mgl_context, release_all
        from bambi.util.projection_util import label_to_world_coordinates
        
        target_folder = config["target_folder"]
        dem_path = config["dem_path"]
        target_epsg = config.get("target_epsg", 32633)
        
        # Correction factors
        translation = config.get("translation", {"x": 0, "y": 0, "z": 0})
        rotation = config.get("rotation", {"x": 0, "y": 0, "z": 0})
        
        # FoV mask options
        use_fov_mask = config.get("use_fov_mask", False)
        mask_path = config.get("fov_mask_path", "")
        mask_simplify_epsilon = config.get("mask_simplify_epsilon", 2.0)
        
        if log_fn:
            log_fn("Starting FoV calculation...")
            log_fn("Loading DEM and poses data...")
            
        # Load DEM metadata
        dem_json_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
        with open(dem_json_path, 'r') as f:
            dem_json = json.load(f)
            
        x_offset = dem_json["origin"][0]
        y_offset = dem_json["origin"][1]
        z_offset = dem_json["origin"][2]
        
        # Load poses
        poses_file = os.path.join(target_folder, "poses.json")
        with open(poses_file, 'r') as f:
            poses = json.load(f)
            
        # Get input resolution from first image
        first_image_path = os.path.join(target_folder, poses["images"][0].get("imagefile", ""))
        if os.path.exists(first_image_path):
            img = cv2.imread(first_image_path)
            if img is not None:
                input_resolution = Resolution(img.shape[1], img.shape[0])
            else:
                input_resolution = Resolution(512, 512)
        else:
            input_resolution = Resolution(512, 512)
            
        if log_fn:
            log_fn(f"Input resolution: {input_resolution.width}x{input_resolution.height}")
            
        if progress_fn:
            progress_fn(10)
            
        # Create output folder
        fov_folder = os.path.join(target_folder, "fov")
        os.makedirs(fov_folder, exist_ok=True)
        
        # Load or create FoV mask polygon
        mask_polygon = None
        if use_fov_mask and mask_path and os.path.exists(mask_path):
            if log_fn:
                log_fn(f"Loading FoV mask from {mask_path}...")
            mask_polygon = self._extract_mask_polygon(mask_path, mask_simplify_epsilon, log_fn)
            if mask_polygon:
                if log_fn:
                    log_fn(f"Loaded mask with {len(mask_polygon)} polygon points")
        
        # If no mask, use image corners
        if not mask_polygon:
            if log_fn:
                log_fn("Using image corners as FoV polygon...")
            w, h = input_resolution.width, input_resolution.height
            # Define corners with more points for better ground projection
            mask_polygon = [
                (0, 0), (w/2, 0), (w, 0),
                (w, h/2), (w, h),
                (w/2, h), (0, h),
                (0, h/2)
            ]
            
        if log_fn:
            log_fn("Loading DEM mesh...")
            
        # Load DEM mesh
        ctx = None
        try:
            mesh_data, texture_data = read_gltf(dem_path)
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)
            
            if progress_fn:
                progress_fn(20)
                
            # Process each frame
            total_frames = len(poses["images"])
            cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
            cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')
            
            if log_fn:
                log_fn(f"Calculating FoV for {total_frames} frames...")
            
            # Output file for FoV polygons
            output_file = os.path.join(fov_folder, "fov_polygons.txt")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# FoV polygon georeferenced data\n")
                f.write("# Format: frame_idx num_points x1 y1 z1 x2 y2 z2 ...\n")
                
                for frame_idx, image_metadata in enumerate(poses["images"]):
                    # Get camera for this frame
                    fovy = image_metadata.get("fovy", [50])[0]
                    position = Vector3(image_metadata["location"])
                    rot = image_metadata["rotation"]
                    rotation_eulers = (Vector3(
                        [np.deg2rad(val % 360.0) for val in rot]) - cor_rotation_eulers) * -1
                    position += cor_translation
                    rotation_quat = Quaternion.from_eulers(rotation_eulers)
                    
                    # Create camera for projection
                    from alfspy.core.rendering import Camera
                    camera = Camera(fovy=fovy, aspect_ratio=1.0, position=position, rotation=rotation_quat)
                    
                    # Georeference the mask polygon points
                    georef_points = self._georeference_polygon(
                        mask_polygon, input_resolution, tri_mesh, camera,
                        x_offset, y_offset, z_offset
                    )
                    
                    # Filter out None points
                    valid_points = [p for p in georef_points if p is not None]
                    
                    if len(valid_points) > 0:
                        coords_str = " ".join(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in valid_points)
                        f.write(f"{frame_idx} {len(valid_points)} {coords_str}\n")
                    else:
                        f.write(f"{frame_idx} 0\n")
                    
                    if progress_fn and frame_idx % 50 == 0:
                        progress = 20 + int((frame_idx / total_frames) * 75)
                        progress_fn(min(progress, 95))
                        
                    if log_fn and (frame_idx + 1) % 100 == 0:
                        log_fn(f"Processed {frame_idx + 1}/{total_frames} frames...")
                        
        finally:
            if ctx:
                release_all(ctx)
            del mesh_data
            del texture_data
            del tri_mesh
            
        if log_fn:
            log_fn(f"FoV calculation complete. Output: {output_file}")
            
        if progress_fn:
            progress_fn(100)
            
    def _extract_mask_polygon(self, mask_path: str, simplify_epsilon: float = 2.0, log_fn=None) -> Optional[List]:
        """Extract polygon from binary mask image.
        
        :param mask_path: Path to binary mask PNG image
        :param simplify_epsilon: Epsilon for polygon simplification
        :param log_fn: Logging callback function
        :return: List of (x, y) pixel coordinates or None
        """
        import cv2
        
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                if log_fn:
                    log_fn(f"Warning: Could not load mask image: {mask_path}")
                return None
                
            # Threshold to ensure binary mask
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return None
                
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Simplify the polygon
            if simplify_epsilon > 0:
                largest_contour = cv2.approxPolyDP(largest_contour, simplify_epsilon, True)
                
            # Convert to list of (x, y) tuples
            polygon_points = [(float(pt[0][0]), float(pt[0][1])) for pt in largest_contour]
            
            return polygon_points
            
        except Exception as e:
            if log_fn:
                log_fn(f"Warning: Error extracting mask polygon: {e}")
            return None
            
    def _georeference_polygon(
        self, 
        polygon_points: List, 
        input_resolution, 
        tri_mesh, 
        camera,
        x_offset: float, 
        y_offset: float, 
        z_offset: float
    ) -> List:
        """Georeference polygon points from pixel to world coordinates.
        
        :param polygon_points: List of (x, y) pixel coordinates
        :param input_resolution: Resolution object for input images
        :param tri_mesh: Trimesh object for ray casting
        :param camera: Camera object for the current frame
        :param x_offset, y_offset, z_offset: DEM origin offsets
        :return: List of (x, y, z) world coordinates or None for failed points
        """
        import numpy as np
        from bambi.util.projection_util import label_to_world_coordinates
        
        georeferenced_points = []
        
        for px, py in polygon_points:
            # Create a small bounding box around the point
            point_coords = [px, py, px + 1, py, px + 1, py + 1, px, py + 1]
            
            try:
                world_coordinates = label_to_world_coordinates(
                    point_coords, input_resolution, tri_mesh, camera
                )
                
                if len(world_coordinates) == 0:
                    georeferenced_points.append(None)
                    continue
                    
                # Take the mean of projected points
                xx = world_coordinates[:, 0] + x_offset
                yy = world_coordinates[:, 1] + y_offset
                zz = world_coordinates[:, 2] + z_offset
                
                georeferenced_points.append((float(np.mean(xx)), float(np.mean(yy)), float(np.mean(zz))))
                
            except Exception:
                georeferenced_points.append(None)
                
        return georeferenced_points

    def run_tracking(self, config: Dict[str, Any], progress_fn=None, log_fn=None):
        """Run tracking on geo-referenced detections.
        
        Supports multiple tracking backends:
        - Built-in: Default Hungarian IoU tracker
        - BoxMOT: Standard BoxMOT trackers (DeepOCSORT, BoTSORT, etc.)
        - GeoRef: Geo-referenced trackers (GeoNative, GeoHybrid)
        
        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        from .tracker_manager import get_tracker_manager, TrackerBackend, ReIDModel
        
        tracker_id = config.get("tracker_id", "builtin")
        
        if log_fn:
            log_fn(f"Starting tracking with backend: {tracker_id}")
            
        # Initialize tracker manager
        target_folder = config["target_folder"]
        models_folder = os.path.join(target_folder, "models")
        tracker_manager = get_tracker_manager(models_folder)
        
        # Check which backend to use
        if tracker_id == "builtin":
            # Use existing built-in tracking implementation
            self._run_builtin_tracking(config, progress_fn, log_fn)
        else:
            # Use advanced tracker (BoxMOT or GeoRef)
            self._run_advanced_tracking(config, tracker_manager, progress_fn, log_fn)
            
    def _run_builtin_tracking(self, config: Dict[str, Any], progress_fn=None, log_fn=None):
        """Run the built-in Hungarian IoU tracker.
        
        This is the original tracking implementation.
        """
        from collections import defaultdict
        from dataclasses import dataclass
        from enum import Enum
        from typing import List, Tuple
        import math
        
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        
        # Import tracking classes from bambi or define locally
        @dataclass
        class Detection:
            source_id: int
            frame: int
            x1: float
            y1: float
            z1: float
            x2: float
            y2: float
            z2: float
            conf: float
            cls: int
            interpolated: int = 0
            
        @dataclass
        class Track:
            tid: int
            cls: int
            x1: float
            y1: float
            z1: float
            x2: float
            y2: float
            z2: float
            last_frame: int
            age: int = 0
            hits: int = 0
            
        class TrackerMode(Enum):
            GREEDY = 1
            HUNGARIAN = 2
            CENTER = 3
            HUNGARIAN_CENTER = 4
            
        def iou(a, b) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0.0:
                return 0.0
            a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
            b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
            denom = a_area + b_area - inter
            return inter / denom if denom > 0 else 0.0
            
        target_folder = config["target_folder"]
        iou_threshold = config.get("iou_threshold", 0.3)
        max_age = config.get("max_age", -1)
        max_center_distance = config.get("max_center_distance", 0.2)
        tracker_mode_str = config.get("tracker_mode", "HUNGARIAN")
        class_aware = config.get("class_aware", True)
        interpolate = config.get("interpolate", True)
        
        tracker_mode = TrackerMode[tracker_mode_str]
        
        if log_fn:
            log_fn(f"Running built-in tracking with mode: {tracker_mode_str}")
            
        # Load georeferenced detections
        georef_folder = os.path.join(target_folder, "georeferenced")
        georef_file = os.path.join(georef_folder, "georeferenced.txt")
        
        if not os.path.exists(georef_file):
            raise FileNotFoundError("Georeferenced detections not found")
            
        # Parse detections
        frames: Dict[int, List[Detection]] = defaultdict(list)
        
        with open(georef_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    idx = int(parts[0])
                    frame = int(parts[1])
                    det = Detection(
                        source_id=idx,
                        frame=frame,
                        x1=float(parts[2]),
                        y1=float(parts[3]),
                        z1=float(parts[4]),
                        x2=float(parts[5]),
                        y2=float(parts[6]),
                        z2=float(parts[7]),
                        conf=float(parts[8]),
                        cls=int(parts[9])
                    )
                    frames[frame].append(det)
                    
        if log_fn:
            log_fn(f"Loaded {sum(len(v) for v in frames.values())} detections in {len(frames)} frames")
            
        if progress_fn:
            progress_fn(20)
            
        # Run tracking
        all_frames = sorted(frames.keys())
        active_tracks: List[Track] = []
        next_tid = 1
        results: List[Tuple[int, int, Detection]] = []
        
        for fidx, f in enumerate(all_frames):
            dets = frames[f]
            
            # Simple Hungarian matching on IoU
            nD, nT = len(dets), len(active_tracks)
            matches = []
            unmatched_dets = list(range(nD))
            unmatched_trks = list(range(nT))
            
            if nD > 0 and nT > 0:
                import numpy as np
                IoU = np.zeros((nD, nT), dtype=np.float32)
                
                for di, d in enumerate(dets):
                    for ti, t in enumerate(active_tracks):
                        if class_aware and t.cls is not None and d.cls != t.cls:
                            continue
                        IoU[di, ti] = iou(
                            (d.x1, d.y1, d.x2, d.y2),
                            (t.x1, t.y1, t.x2, t.y2)
                        )
                        
                # Hungarian assignment
                cost = 1.0 - IoU
                LARGE = 1e6
                cost[IoU < iou_threshold] = LARGE
                
                row_ind, col_ind = linear_sum_assignment(cost)
                
                det_used = set()
                trk_used = set()
                
                for di, ti in zip(row_ind, col_ind):
                    if IoU[di, ti] >= iou_threshold and cost[di, ti] < LARGE:
                        matches.append((di, ti))
                        det_used.add(di)
                        trk_used.add(ti)
                        
                unmatched_dets = [i for i in range(nD) if i not in det_used]
                unmatched_trks = [i for i in range(nT) if i not in trk_used]
                
            # Update matched tracks
            for di, ti in matches:
                d = dets[di]
                t = active_tracks[ti]
                t.x1, t.y1, t.z1 = d.x1, d.y1, d.z1
                t.x2, t.y2, t.z2 = d.x2, d.y2, d.z2
                t.last_frame = f
                t.hits += 1
                t.age = 0
                results.append((f, t.tid, d))
                
            # Create new tracks for unmatched detections
            for di in unmatched_dets:
                d = dets[di]
                t = Track(
                    tid=next_tid,
                    cls=d.cls if class_aware else None,
                    x1=d.x1, y1=d.y1, z1=d.z1,
                    x2=d.x2, y2=d.y2, z2=d.z2,
                    last_frame=f,
                    age=0, hits=1
                )
                active_tracks.append(t)
                results.append((f, t.tid, d))
                next_tid += 1
                
            # Age unmatched tracks
            survivors = []
            for ti in range(len(active_tracks)):
                if ti in unmatched_trks:
                    active_tracks[ti].age += 1
                if max_age < 0 or active_tracks[ti].age <= max_age:
                    survivors.append(active_tracks[ti])
            active_tracks = survivors
            
            if progress_fn and fidx % 50 == 0:
                progress = 20 + int((fidx / len(all_frames)) * 60)
                progress_fn(min(progress, 85))
                
        # Sort results
        results.sort(key=lambda r: (r[0], r[1]))
        
        # Interpolate missing frames if enabled
        if interpolate:
            if log_fn:
                log_fn("Interpolating missing frames...")
            results = self._interpolate_tracks(results, Detection)
            
        if progress_fn:
            progress_fn(90)
            
        # Create output folder and write results
        tracks_folder = os.path.join(target_folder, "tracks")
        os.makedirs(tracks_folder, exist_ok=True)
        
        output_file = os.path.join(tracks_folder, "tracks.csv")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for frame, tid, d in results:
                f.write(f"{frame:08d},{tid},{d.x1:.6f},{d.y1:.6f},{d.z1:.6f},"
                       f"{d.x2:.6f},{d.y2:.6f},{d.z2:.6f},"
                       f"{d.conf:.6f},{d.cls},{d.interpolated}\n")
                       
        # Count unique tracks
        unique_tracks = set(r[1] for r in results)
        
        if log_fn:
            log_fn(f"Tracking complete: {len(unique_tracks)} tracks, {len(results)} total detections")
            
        if progress_fn:
            progress_fn(100)
            
    def _run_advanced_tracking(self, config: Dict[str, Any], tracker_manager, progress_fn=None, log_fn=None):
        """Run advanced tracking with BoxMOT or GeoRef backends.
        
        This method handles:
        - Loading frames and detections
        - Creating the appropriate tracker
        - Running frame-by-frame tracking with the tracker's update method
        - Handling geo-referenced vs standard tracking
        """
        from .tracker_manager import TrackerBackend, ReIDModel
        import numpy as np
        import cv2
        from collections import defaultdict
        from dataclasses import dataclass
        
        @dataclass
        class Detection:
            source_id: int
            frame: int
            x1: float
            y1: float
            z1: float
            x2: float
            y2: float
            z2: float
            conf: float
            cls: int
            interpolated: int = 0
        
        target_folder = config["target_folder"]
        tracker_id = config.get("tracker_id", "builtin")
        reid_model_str = config.get("reid_model", "osnet")
        custom_reid_path = config.get("custom_reid_path", "")
        tracker_params_json = config.get("tracker_params_json", "")
        interpolate = config.get("interpolate", True)
        
        # Map ReID model string to enum
        reid_model_map = {
            "osnet": ReIDModel.OSNET,
            "bambi": ReIDModel.BAMBI,
            "custom": ReIDModel.CUSTOM
        }
        reid_model = reid_model_map.get(reid_model_str, ReIDModel.OSNET)
        
        if log_fn:
            log_fn(f"Creating tracker: {tracker_id}")
            log_fn(f"ReID model: {reid_model_str}")
            
        # Determine device
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                if log_fn:
                    log_fn("Using CUDA for tracking")
        except ImportError:
            pass
            
        # Create the tracker
        tracker, backend = tracker_manager.create_tracker(
            tracker_id=tracker_id,
            reid_model=reid_model,
            custom_reid_path=custom_reid_path,
            json_params=tracker_params_json,
            device=device,
            log_fn=log_fn
        )
        
        if log_fn:
            log_fn(f"Tracker created with backend: {backend.value}")
            
        # Load georeferenced detections
        georef_folder = os.path.join(target_folder, "georeferenced")
        georef_file = os.path.join(georef_folder, "georeferenced.txt")
        
        if not os.path.exists(georef_file):
            raise FileNotFoundError("Georeferenced detections not found")
            
        # Load poses.json for frame information
        poses_file = os.path.join(target_folder, "poses.json")
        if not os.path.exists(poses_file):
            raise FileNotFoundError("poses.json not found - run frame extraction first")
            
        with open(poses_file, 'r') as f:
            poses_data = json.load(f)
            
        images = poses_data.get("images", [])
        frame_to_path = {}
        for img in images:
            frame_idx = img.get("frame_index", img.get("id", 0))
            frame_to_path[frame_idx] = img.get("path", "")
            
        # Parse detections file
        frames_geo: Dict[int, List] = defaultdict(list)
        
        with open(georef_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    idx = int(parts[0])
                    frame = int(parts[1])
                    det = {
                        'source_id': idx,
                        'frame': frame,
                        'x1': float(parts[2]),
                        'y1': float(parts[3]),
                        'z1': float(parts[4]),
                        'x2': float(parts[5]),
                        'y2': float(parts[6]),
                        'z2': float(parts[7]),
                        'conf': float(parts[8]),
                        'cls': int(parts[9])
                    }
                    frames_geo[frame].append(det)
                    
        # Also load pixel-space detections if available
        detections_folder = os.path.join(target_folder, "detections")
        frames_pixel: Dict[int, List] = defaultdict(list)
        
        for det_file in os.listdir(detections_folder) if os.path.exists(detections_folder) else []:
            if det_file.endswith(".txt"):
                try:
                    # Extract frame number from filename
                    frame_num = int(det_file.replace(".txt", "").split("_")[-1])
                    det_path = os.path.join(detections_folder, det_file)
                    
                    with open(det_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                frames_pixel[frame_num].append({
                                    'cls': int(parts[0]),
                                    'x1': float(parts[1]),
                                    'y1': float(parts[2]),
                                    'x2': float(parts[3]),
                                    'y2': float(parts[4]),
                                    'conf': float(parts[5]) if len(parts) > 5 else 0.5
                                })
                except (ValueError, IndexError):
                    continue
                    
        if log_fn:
            log_fn(f"Loaded {sum(len(v) for v in frames_geo.values())} geo detections in {len(frames_geo)} frames")
            
        if progress_fn:
            progress_fn(20)
            
        # Run tracking frame by frame
        all_frames = sorted(set(frames_geo.keys()) | set(frames_pixel.keys()))
        results = []
        
        is_geo_tracker = backend in [TrackerBackend.GEOREF_NATIVE, TrackerBackend.GEOREF_HYBRID]
        
        for fidx, frame_num in enumerate(all_frames):
            # Load the frame image
            frame_path = frame_to_path.get(frame_num)
            img = None
            
            if frame_path and os.path.exists(frame_path):
                img = cv2.imread(frame_path)
            elif frame_path:
                # Try relative path from target folder
                full_path = os.path.join(target_folder, frame_path)
                if os.path.exists(full_path):
                    img = cv2.imread(full_path)
                    
            if img is None:
                # Create dummy image if we can't load the actual frame
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                
            # Get detections for this frame
            geo_dets = frames_geo.get(frame_num, [])
            pixel_dets = frames_pixel.get(frame_num, [])
            
            # Build detection arrays
            if pixel_dets:
                # Use pixel detections [x1, y1, x2, y2, conf, cls]
                dets = np.array([
                    [d['x1'], d['y1'], d['x2'], d['y2'], d['conf'], d['cls']]
                    for d in pixel_dets
                ])
            elif geo_dets:
                # Create dummy pixel detections from geo detections
                # This is a fallback - we use geo coords as pixel coords
                dets = np.array([
                    [0, 0, 100, 100, d['conf'], d['cls']]  # Placeholder pixel coords
                    for d in geo_dets
                ])
            else:
                dets = np.empty((0, 6))
                
            # Build geo detection array for geo-referenced trackers
            geodets = None
            if geo_dets and is_geo_tracker:
                # Format: [source_id, frame_id, x1, y1, z1, x2, y2, z2, conf, cls]
                geodets = np.array([
                    [d['source_id'], d['frame'], d['x1'], d['y1'], d['z1'],
                     d['x2'], d['y2'], d['z2'], d['conf'], d['cls']]
                    for d in geo_dets
                ])
                
            # Call the tracker's update method
            try:
                if is_geo_tracker and geodets is not None:
                    # Geo-referenced tracker workaround for BoxMOT v16+ decorator compatibility
                    # The @per_class_decorator doesn't forward extra kwargs, so we set them
                    # as instance attributes that the update method can read from self
                    tracker._current_frame_index = frame_num
                    tracker._current_geodets = geodets
                    # Call update with only the standard BoxMOT parameters
                    tracks = tracker.update(dets, img)
                else:
                    # Standard BoxMOT tracker
                    tracks = tracker.update(dets, img)
                    
            except Exception as e:
                if log_fn:
                    log_fn(f"Warning: Tracker update failed at frame {frame_num}: {e}")
                tracks = np.array([])
                
            # Process tracking results
            # BoxMOT returns: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
            if len(tracks) > 0:
                for track in tracks:
                    if len(track) >= 7:
                        track_id = int(track[4])
                        
                        # Find corresponding geo detection
                        geo_det = None
                        if geo_dets:
                            det_ind = int(track[7]) if len(track) > 7 else -1
                            if 0 <= det_ind < len(geo_dets):
                                geo_det = geo_dets[det_ind]
                            else:
                                # Find by matching (fallback)
                                for gd in geo_dets:
                                    if abs(gd['conf'] - track[5]) < 0.01:
                                        geo_det = gd
                                        break
                                        
                        if geo_det:
                            det_obj = Detection(
                                source_id=geo_det['source_id'],
                                frame=frame_num,
                                x1=geo_det['x1'],
                                y1=geo_det['y1'],
                                z1=geo_det['z1'],
                                x2=geo_det['x2'],
                                y2=geo_det['y2'],
                                z2=geo_det['z2'],
                                conf=geo_det['conf'],
                                cls=geo_det['cls'],
                                interpolated=0
                            )
                            results.append((frame_num, track_id, det_obj))
                            
            if progress_fn and fidx % 50 == 0:
                progress = 20 + int((fidx / len(all_frames)) * 60)
                progress_fn(min(progress, 85))
                
        # Sort results
        results.sort(key=lambda r: (r[0], r[1]))
        
        # Interpolate missing frames if enabled
        if interpolate and results:
            if log_fn:
                log_fn("Interpolating missing frames...")
            results = self._interpolate_tracks(results, Detection)
            
        if progress_fn:
            progress_fn(90)
            
        # Create output folder and write results
        tracks_folder = os.path.join(target_folder, "tracks")
        os.makedirs(tracks_folder, exist_ok=True)
        
        output_file = os.path.join(tracks_folder, "tracks.csv")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for frame, tid, d in results:
                f.write(f"{frame:08d},{tid},{d.x1:.6f},{d.y1:.6f},{d.z1:.6f},"
                       f"{d.x2:.6f},{d.y2:.6f},{d.z2:.6f},"
                       f"{d.conf:.6f},{d.cls},{d.interpolated}\n")
                       
        # Count unique tracks
        unique_tracks = set(r[1] for r in results)
        
        if log_fn:
            log_fn(f"Advanced tracking complete: {len(unique_tracks)} tracks, {len(results)} total detections")
            log_fn(f"Backend used: {backend.value}")
            
        if progress_fn:
            progress_fn(100)
            
    def _interpolate_tracks(self, results: List, detection_class) -> List:
        """Interpolate missing frames within each track."""
        from collections import defaultdict
        
        tracks: Dict[int, List] = defaultdict(list)
        for frame, tid, det in results:
            tracks[tid].append((frame, det))
            
        for tid in tracks:
            tracks[tid].sort(key=lambda x: x[0])
            
        new_results = []
        
        for tid, seq in tracks.items():
            if len(seq) < 2:
                for f, d in seq:
                    new_results.append((f, tid, d))
                continue
                
            # Add first detection
            prev_frame, prev_det = seq[0]
            new_results.append((prev_frame, tid, prev_det))
            
            for i in range(1, len(seq)):
                curr_frame, curr_det = seq[i]
                gap = curr_frame - prev_frame
                
                if gap > 1:
                    # Interpolate
                    for j in range(1, gap):
                        t = j / gap
                        interp_det = detection_class(
                            source_id=-1,
                            frame=prev_frame + j,
                            x1=prev_det.x1 + t * (curr_det.x1 - prev_det.x1),
                            y1=prev_det.y1 + t * (curr_det.y1 - prev_det.y1),
                            z1=prev_det.z1 + t * (curr_det.z1 - prev_det.z1),
                            x2=prev_det.x2 + t * (curr_det.x2 - prev_det.x2),
                            y2=prev_det.y2 + t * (curr_det.y2 - prev_det.y2),
                            z2=prev_det.z2 + t * (curr_det.z2 - prev_det.z2),
                            conf=(prev_det.conf + curr_det.conf) / 2,
                            cls=curr_det.cls,
                            interpolated=1
                        )
                        new_results.append((prev_frame + j, tid, interp_det))
                        
                new_results.append((curr_frame, tid, curr_det))
                prev_frame, prev_det = curr_frame, curr_det
                
        new_results.sort(key=lambda r: (r[0], r[1]))
        return new_results
    
    def run_orthomosaic(self, config: Dict[str, Any], progress_fn=None, log_fn=None):
        """Generate an orthomosaic from extracted frames and DEM.
        
        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        import math
        import cv2
        import numpy as np
        from dataclasses import dataclass
        from enum import Enum
        from typing import Tuple, Union
        
        if log_fn:
            log_fn("Initializing orthomosaic generation...")
            
        # Check for required dependencies
        try:
            from moderngl import Context as MglContext
            from pyrr import Quaternion, Vector3
            from trimesh import Trimesh
        except ImportError as e:
            raise ImportError(
                f"Required dependency not found: {e}\n\n"
                "Please install the following packages:\n"
                "  pip install moderngl pyrr trimesh\n\n"
                "Note: Orthomosaic generation requires OpenGL support."
            )
            
        # Check for alfspy (specialized rendering library)
        try:
            from alfspy.core.geo.aabb import AABB
            from alfspy.core.geo.transform import Transform
            from alfspy.core.rendering import Resolution, Camera, CtxShot, RenderResultMode, TextureData
            from alfspy.core.rendering.renderer import Renderer
            from alfspy.core.util.geo import get_aabb
            from alfspy.core.util.pyrrs import quaternion_from_eulers
            from alfspy.render.render import (
                make_mgl_context, read_gltf, process_render_data,
                make_shot_loader, release_all
            )
            HAS_ALFSPY = True
        except ImportError:
            HAS_ALFSPY = False
            if log_fn:
                log_fn("Warning: alfspy not available, using simplified orthomosaic generation")
        
        # Check for rasterio (GeoTIFF support)
        try:
            import rasterio
            from rasterio.transform import from_bounds
            from rasterio.crs import CRS
            HAS_RASTERIO = True
        except ImportError:
            HAS_RASTERIO = False
            if log_fn:
                log_fn("Warning: rasterio not installed, GeoTIFF output not available")
        
        # Get configuration parameters
        target_folder = config["target_folder"]
        dem_path = config["dem_path"]
        target_epsg = config.get("target_epsg", 32633)
        
        # Orthomosaic specific settings
        ground_resolution = config.get("ortho_ground_resolution", 0.05)
        dem_metadata_path = config.get("ortho_dem_metadata_path")
        blend_mode = config.get("ortho_blend_mode", "integral")
        use_all_frames = config.get("ortho_use_all_frames", True)
        start_frame = config.get("ortho_start_frame")
        end_frame = config.get("ortho_end_frame")
        crop_to_content = config.get("ortho_crop_to_content", True)
        create_overviews = config.get("ortho_create_overviews", True)
        max_tile_size = config.get("ortho_max_tile_size", 8192)
        
        if log_fn:
            log_fn(f"Ground resolution: {ground_resolution} m/px")
            log_fn(f"Blend mode: {blend_mode}")
            if use_all_frames:
                log_fn("Frame range: All frames")
            else:
                log_fn(f"Frame range: {start_frame} to {end_frame}")
        
        if progress_fn:
            progress_fn(5)
        
        # Load poses.json
        poses_file = os.path.join(target_folder, "poses.json")
        if not os.path.exists(poses_file):
            raise FileNotFoundError(f"poses.json not found at {poses_file}")
            
        with open(poses_file, 'r') as f:
            poses = json.load(f)
            
        images = poses.get("images", [])
        total_images = len(images)
        
        if total_images == 0:
            raise RuntimeError("No images found in poses.json")
            
        if log_fn:
            log_fn(f"Found {total_images} images in poses.json")
        
        # Apply frame filter
        if not use_all_frames and start_frame is not None and end_frame is not None:
            images = [img for i, img in enumerate(images) if start_frame <= i <= end_frame]
            if log_fn:
                log_fn(f"Filtered to {len(images)} images (frames {start_frame}-{end_frame})")
        
        if len(images) == 0:
            raise RuntimeError("No images remaining after filtering")
        
        if progress_fn:
            progress_fn(10)
        
        # Load DEM metadata if available
        dem_metadata = None
        coord_offset_x = 0.0
        coord_offset_y = 0.0
        
        if dem_metadata_path and os.path.exists(dem_metadata_path):
            if log_fn:
                log_fn(f"Loading DEM metadata from {dem_metadata_path}")
            try:
                with open(dem_metadata_path, 'r') as f:
                    dem_metadata = json.load(f)
                origin = dem_metadata.get("origin", [0, 0, 0])
                coord_offset_x = float(origin[0])
                coord_offset_y = float(origin[1])
                if log_fn:
                    log_fn(f"DEM origin offset: X={coord_offset_x:.2f}, Y={coord_offset_y:.2f}")
            except Exception as e:
                if log_fn:
                    log_fn(f"Warning: Could not load DEM metadata: {e}")
        else:
            # Try to find metadata in same folder as DEM
            auto_metadata_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
            if os.path.exists(auto_metadata_path):
                if log_fn:
                    log_fn(f"Found DEM metadata at {auto_metadata_path}")
                try:
                    with open(auto_metadata_path, 'r') as f:
                        dem_metadata = json.load(f)
                    origin = dem_metadata.get("origin", [0, 0, 0])
                    coord_offset_x = float(origin[0])
                    coord_offset_y = float(origin[1])
                except Exception as e:
                    if log_fn:
                        log_fn(f"Warning: Could not load auto-detected metadata: {e}")
        
        # Load mask if available
        mask_filename = poses.get("mask")
        mask_path = os.path.join(target_folder, mask_filename) if mask_filename else None
        
        if progress_fn:
            progress_fn(15)
        
        # Create output folder and determine output file
        # Check if a specific output file is requested (used by geotiff export)
        if config.get("ortho_output_file"):
            output_file = config["ortho_output_file"]
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        else:
            ortho_folder = os.path.join(target_folder, "orthomosaic")
            os.makedirs(ortho_folder, exist_ok=True)
            output_file = os.path.join(ortho_folder, "orthomosaic.tif")
        
        if HAS_ALFSPY:
            # Use full alfspy-based rendering pipeline
            self._run_orthomosaic_alfspy(
                config, images, poses, dem_path, mask_path,
                output_file, ground_resolution, blend_mode,
                coord_offset_x, coord_offset_y, target_epsg,
                crop_to_content, create_overviews, max_tile_size,
                progress_fn, log_fn
            )
        else:
            # Use simplified orthomosaic generation (projection without rendering)
            self._run_orthomosaic_simple(
                config, images, poses, dem_path, mask_path,
                output_file, ground_resolution, blend_mode,
                coord_offset_x, coord_offset_y, target_epsg,
                crop_to_content, create_overviews,
                progress_fn, log_fn
            )
        
        if log_fn:
            log_fn(f"Orthomosaic saved to: {output_file}")
            
        if progress_fn:
            progress_fn(100)
    

    def _create_tile_camera(self, global_camera, global_bounds, global_res, 
                           tx, ty, tw, th, Vector3, Camera):
        """Helper to create a sub-camera for tiling."""
        min_x, min_y, max_x, max_y = global_bounds
        
        pixel_to_world_x = (max_x - min_x) / global_res.width
        pixel_to_world_y = (max_y - min_y) / global_res.height
        
        # Tile bounds
        tile_min_x = min_x + tx * pixel_to_world_x
        tile_max_x = min_x + (tx + tw) * pixel_to_world_x
        tile_max_y = max_y - ty * pixel_to_world_y
        tile_min_y = max_y - (ty + th) * pixel_to_world_y
        
        tile_center_x = (tile_min_x + tile_max_x) / 2.0
        tile_center_y = (tile_min_y + tile_max_y) / 2.0
        tile_ortho_width = tile_max_x - tile_min_x
        tile_ortho_height = tile_max_y - tile_min_y # Note: height logic might vary, usually safe to keep relative sign
        
        return Camera(
            orthogonal=True,
            orthogonal_size=(tile_ortho_width, tile_ortho_height),
            position=Vector3([
                tile_center_x, tile_center_y, global_camera.transform.position.z
            ], dtype='f4'),
            rotation=global_camera.transform.rotation,
            near=global_camera.near,
            far=global_camera.far
        )

    def _render_sequential_alfspy(self, renderer, shots, mask, resolution, blend_mode, bounds):
        """Robust sequential rendering ported from orthomosaic.py"""
        import numpy as np
        from alfspy.core.rendering import RenderResultMode
        
        output = np.zeros((resolution.height, resolution.width, 4), dtype=np.uint8)
        
        # Initialize distance map for center blending
        if blend_mode == "center":
            distance_map = np.full((resolution.height, resolution.width), np.inf, dtype=np.float32)
            min_x, min_y, max_x, max_y = bounds
            
            # Pre-sort shots? In orthomosaic.py they sort farthest first.
            # We will process normally but check distance
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            shots = sorted(shots, key=lambda s: (s.camera.transform.position.x - center_x)**2 + 
                                                (s.camera.transform.position.y - center_y)**2, reverse=True)
                                                
        for shot in shots:
            try:
                results = list(renderer.project_shots_iter(
                    shot, RenderResultMode.ShotOnly, release_shots=False, mask=mask
                ))
                
                if not results: continue
                
                shot_result = results[0]
                shot_has_data = shot_result[:, :, 3] > 0
                
                if blend_mode == "first":
                    write_mask = shot_has_data & (output[:, :, 3] == 0)
                elif blend_mode == "last":
                    write_mask = shot_has_data
                elif blend_mode == "center":
                    # Simple center priority logic:
                    # Calculate per-pixel distance to this camera (simplified to camera center)
                    cam_pos = shot.camera.transform.position
                    
                    # Create coordinate grids
                    h, w = resolution.height, resolution.width
                    px_size_x = (max_x - min_x) / w
                    px_size_y = (max_y - min_y) / h
                    
                    # This calculation is heavy for Python, optimized version would be better
                    # But keeping it functional for now:
                    y_indices, x_indices = np.where(shot_has_data)
                    if len(x_indices) == 0: continue
                    
                    # Map pixels to world
                    px_world = min_x + x_indices * px_size_x
                    py_world = max_y - y_indices * px_size_y
                    
                    # Dist sq
                    dists = (px_world - cam_pos.x)**2 + (py_world - cam_pos.y)**2
                    
                    # Compare with existing
                    current_dists = distance_map[y_indices, x_indices]
                    better = dists < current_dists
                    
                    # Filter indices
                    valid_y = y_indices[better]
                    valid_x = x_indices[better]
                    
                    # Write
                    output[valid_y, valid_x] = shot_result[valid_y, valid_x]
                    distance_map[valid_y, valid_x] = dists[better]
                    write_mask = None # Handled manually above
                else:
                    write_mask = shot_has_data
                
                if write_mask is not None:
                    output[write_mask] = shot_result[write_mask]
                    
            except Exception:
                continue
                
        return output

    def _run_orthomosaic_alfspy(
        self, config, images, poses, dem_path, mask_path,
        output_file, ground_resolution, blend_mode,
        coord_offset_x, coord_offset_y, target_epsg,
        crop_to_content, create_overviews, max_tile_size,
        progress_fn, log_fn
    ):
        """Run orthomosaic generation using alfspy rendering pipeline with tiling support."""
        import math
        import cv2
        import numpy as np
        from pyrr import Quaternion, Vector3
        
        # Import alfspy components
        from alfspy.core.geo.transform import Transform
        from alfspy.core.rendering import Resolution, Camera, CtxShot, RenderResultMode, TextureData
        from alfspy.core.rendering.renderer import Renderer
        from alfspy.core.util.geo import get_aabb
        from alfspy.core.util.pyrrs import quaternion_from_eulers
        from alfspy.render.render import (
            make_mgl_context, read_gltf, process_render_data,
            make_shot_loader, release_all
        )
        
        target_folder = config["target_folder"]
        
        if log_fn:
            log_fn("Loading DEM mesh...")
        
        # 1. Load Mesh
        mesh_data, texture_data = read_gltf(dem_path)
        mesh_data, texture_data = process_render_data(mesh_data, texture_data)
        mesh_aabb = get_aabb(mesh_data.vertices)
        
        # 2. Setup Context & Mask
        ctx = make_mgl_context()
        mask = None
        if mask_path and os.path.exists(mask_path):
            if log_fn:
                log_fn(f"Loading mask from {mask_path}")
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_img is not None:
                mask = TextureData(CtxShot._cvt_img(mask_img))
        
        # 3. Prepare Corrections
        correction_translation = config.get("translation", {"x": 0, "y": 0, "z": 0})
        correction_rotation = config.get("rotation", {"x": 0, "y": 0, "z": 0})
        
        cor_translation = Vector3([
            correction_translation.get('x', 0),
            correction_translation.get('y', 0),
            correction_translation.get('z', 0)
        ], dtype='f4')
        
        cor_rotation_eulers = Vector3([
            np.deg2rad(correction_rotation.get('x', 0)),
            np.deg2rad(correction_rotation.get('y', 0)),
            np.deg2rad(correction_rotation.get('z', 0))
        ], dtype='f4')
        cor_quat = Quaternion.from_eulers(cor_rotation_eulers)
        correction = Transform(cor_translation, cor_quat)
        
        # 4. Load Shots
        if log_fn:
            log_fn(f"Loading {len(images)} shots...")
        
        shots = []
        default_fovy = 50.0
        
        for i, img_info in enumerate(images):
            image_file = img_info.get("imagefile")
            image_path = os.path.join(target_folder, image_file)
            
            if not os.path.exists(image_path):
                continue
            
            location = img_info.get("location", [0, 0, 0])
            rotation = img_info.get("rotation", [0, 0, 0])
            fovy = img_info.get("fovy", [default_fovy])
            
            camera_position = Vector3(location, dtype='f4')
            
            # Convert rotation
            if len(rotation) == 3:
                eulers = [np.deg2rad(val % 360.0) for val in rotation]
                camera_rotation = quaternion_from_eulers(eulers, 'zyx')
            elif len(rotation) == 4:
                camera_rotation = Quaternion(rotation)
            else:
                continue
                
            fov_value = fovy[0] if isinstance(fovy, (list, tuple)) else fovy
            
            try:
                shot = CtxShot(
                    ctx, image_path, camera_position, camera_rotation,
                    fov_value, 1, correction, lazy=True
                )
                shots.append(shot)
            except Exception as e:
                if log_fn:
                    log_fn(f"Warning: Failed to load shot {image_file}: {e}")

            if progress_fn and i % 50 == 0:
                progress_fn(10 + int((i / len(images)) * 10))

        if not shots:
            raise RuntimeError("No valid shots loaded")

        # 5. Compute Global Bounds (Merging Mesh AABB + Shot AABB)
        # Using the helper logic from orthomosaic.py
        shot_positions = np.array([shot.camera.transform.position for shot in shots])
        
        padding = 10.0
        min_x = min(shot_positions[:, 0].min(), float(mesh_aabb.p_min.x)) - padding
        min_y = min(shot_positions[:, 1].min(), float(mesh_aabb.p_min.y)) - padding
        max_x = max(shot_positions[:, 0].max(), float(mesh_aabb.p_max.x)) + padding
        max_y = max(shot_positions[:, 1].max(), float(mesh_aabb.p_max.y)) + padding
        
        global_bounds = (min_x, min_y, max_x, max_y)
        
        # 6. Compute Resolution & Camera
        width_meters = max_x - min_x
        height_meters = max_y - min_y
        width_pixels = int(math.ceil(width_meters / ground_resolution))
        height_pixels = int(math.ceil(height_meters / ground_resolution))
        
        global_resolution = Resolution(width_pixels, height_pixels)
        
        if log_fn:
            log_fn(f"Output resolution: {width_pixels} x {height_pixels}")
            log_fn(f"Global bounds: X[{min_x:.1f}, {max_x:.1f}] Y[{min_y:.1f}, {max_y:.1f}]")

        # Create global camera (using logic from orthomosaic.py)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        center_z = float(mesh_aabb.p_max.z) + 100.0 # Height above terrain
        
        ortho_width = max_x - min_x
        ortho_height = -(max_y - min_y) # Negative for GeoTIFF Y-axis orientation
        
        global_camera = Camera(
            orthogonal=True,
            orthogonal_size=(ortho_width, ortho_height),
            position=Vector3([center_x, center_y, center_z], dtype='f4'),
            rotation=Quaternion(), # Looking down
            near=0.1,
            far=10000.0
        )

        # 7. Render (Tiled or Single)
        # Check if tiling is needed based on max_tile_size
        needs_tiling = (width_pixels > max_tile_size or height_pixels > max_tile_size)
        
        if needs_tiling:
            if log_fn:
                log_fn(f"Resolution exceeds {max_tile_size}px. Using tiled rendering...")
            
            # Initialize large output array
            result = np.zeros((height_pixels, width_pixels, 4), dtype=np.uint8)
            
            # Generate tiles
            tile_overlap = 128
            effective_tile_size = max_tile_size - tile_overlap
            
            tiles = []
            for y in range(0, height_pixels, effective_tile_size):
                for x in range(0, width_pixels, effective_tile_size):
                    t_w = min(max_tile_size, width_pixels - x + tile_overlap)
                    t_h = min(max_tile_size, height_pixels - y + tile_overlap)
                    # Clamp
                    t_w = min(t_w, width_pixels - x)
                    t_h = min(t_h, height_pixels - y)
                    tiles.append((x, y, t_w, t_h))
            
            if log_fn:
                log_fn(f"Processing {len(tiles)} tiles...")
                
            for i, (tx, ty, tw, th) in enumerate(tiles):
                # Calculate tile specific camera
                tile_camera = self._create_tile_camera(
                    global_camera, global_bounds, global_resolution, 
                    tx, ty, tw, th, Vector3, Camera
                )
                
                # Render Tile
                tile_res = Resolution(tw, th)
                renderer = Renderer(tile_res, ctx, tile_camera, mesh_data, texture_data)
                
                if blend_mode == "integral":
                    shot_loader = make_shot_loader(shots)
                    tile_img = renderer.render_integral(
                        shot_loader, mask=mask, save=False, release_shots=False,
                        auto_contrast=True, alpha_threshold=0.5
                    )
                else:
                    # Calculate tile bounds for sequential logic
                    pixel_size_x = (max_x - min_x) / width_pixels
                    pixel_size_y = (max_y - min_y) / height_pixels
                    tile_bounds = (
                        min_x + tx * pixel_size_x,
                        max_y - (ty + th) * pixel_size_y,
                        min_x + (tx + tw) * pixel_size_x,
                        max_y - ty * pixel_size_y
                    )
                    tile_img = self._render_sequential_alfspy(
                        renderer, shots, mask, tile_res, blend_mode, tile_bounds
                    )
                
                # Copy to main result
                result[ty:ty+th, tx:tx+tw] = tile_img
                renderer.release()
                
                if progress_fn:
                    progress_fn(20 + int((i / len(tiles)) * 60))
        
        else:
            # Single pass rendering
            if log_fn:
                log_fn("Using single-pass rendering...")
            
            renderer = Renderer(global_resolution, ctx, global_camera, mesh_data, texture_data)
            
            if blend_mode == "integral":
                shot_loader = make_shot_loader(shots)
                result = renderer.render_integral(
                    shot_loader, mask=mask, save=False, release_shots=False,
                    auto_contrast=True, alpha_threshold=0.5
                )
            else:
                result = self._render_sequential_alfspy(
                    renderer, shots, mask, global_resolution, blend_mode, global_bounds
                )
            renderer.release()

        # 8. Post-Processing & Save
        if result.max() == 0:
            if log_fn:
                log_fn("WARNING: Rendered result is empty! (Check bounds/height)")
        
        # Crop to content
        if crop_to_content:
            result, global_bounds = self._crop_to_content(result, global_bounds)
        
        # Apply DEM coordinate offset
        geo_bounds = (
            global_bounds[0] + coord_offset_x,
            global_bounds[1] + coord_offset_y,
            global_bounds[2] + coord_offset_x,
            global_bounds[3] + coord_offset_y
        )
        
        if log_fn:
            log_fn(f"Saving to {output_file}")
            
        self._save_orthomosaic(
            result, output_file, geo_bounds, target_epsg,
            create_overviews, log_fn
        )
        
        # Cleanup
        release_all(ctx, shots)
    
    def _run_orthomosaic_simple(
        self, config, images, poses, dem_path, mask_path,
        output_file, ground_resolution, blend_mode,
        coord_offset_x, coord_offset_y, target_epsg,
        crop_to_content, create_overviews,
        progress_fn, log_fn
    ):
        """Simplified orthomosaic generation without alfspy.
        
        This method creates a basic mosaic by projecting frames
        based on their GPS positions and orientations.
        """
        import cv2
        import numpy as np
        import math
        
        target_folder = config["target_folder"]
        
        if log_fn:
            log_fn("Using simplified orthomosaic generation...")
        
        # Collect all image positions
        positions = []
        valid_images = []
        
        for img_info in images:
            location = img_info.get("location", [0, 0, 0])
            # Accept any location (even 0,0 might be valid for local coords)
            positions.append((location[0], location[1], location[2]))
            valid_images.append(img_info)
        
        if len(positions) == 0:
            raise RuntimeError("No valid image positions found")
        
        positions = np.array(positions)
        
        if log_fn:
            log_fn(f"Processing {len(valid_images)} images")
            log_fn(f"Position range: X=[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}], "
                  f"Y=[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}], "
                  f"Z=[{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")
        
        # Compute bounds with padding
        padding = 50.0  # meters
        min_x = positions[:, 0].min() - padding
        max_x = positions[:, 0].max() + padding
        min_y = positions[:, 1].min() - padding
        max_y = positions[:, 1].max() + padding
        
        if log_fn:
            log_fn(f"Canvas bounds: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]")
        
        # Compute output size
        width_meters = max_x - min_x
        height_meters = max_y - min_y
        width_pixels = int(math.ceil(width_meters / ground_resolution))
        height_pixels = int(math.ceil(height_meters / ground_resolution))
        
        # Limit size for memory
        max_dim = 16384
        if width_pixels > max_dim or height_pixels > max_dim:
            scale = min(max_dim / width_pixels, max_dim / height_pixels)
            width_pixels = int(width_pixels * scale)
            height_pixels = int(height_pixels * scale)
            # Adjust ground resolution accordingly
            ground_resolution = max(width_meters / width_pixels, height_meters / height_pixels)
            if log_fn:
                log_fn(f"Limiting output size, adjusted resolution: {ground_resolution:.4f} m/px")
        
        if log_fn:
            log_fn(f"Output size: {width_pixels} x {height_pixels} pixels")
        
        if progress_fn:
            progress_fn(30)
        
        # Create output canvas (use float for accumulation)
        canvas = np.zeros((height_pixels, width_pixels, 4), dtype=np.float32)
        count_map = np.zeros((height_pixels, width_pixels), dtype=np.float32)
        
        # Estimate footprint size based on altitude and FOV
        default_altitude = 50.0  # meters
        default_fov = 50.0  # degrees
        
        images_placed = 0
        
        for i, img_info in enumerate(valid_images):
            image_file = img_info.get("imagefile")
            image_path = os.path.join(target_folder, image_file)
            
            if not os.path.exists(image_path):
                if log_fn and i < 3:
                    log_fn(f"Warning: Image not found: {image_file}")
                continue
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                if log_fn and i < 3:
                    log_fn(f"Warning: Could not read: {image_file}")
                continue
            
            # Convert from BGR to RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img[:, :, 3] = 255
            
            # Get position
            location = img_info.get("location", [0, 0, 0])
            altitude = abs(location[2]) if location[2] != 0 else default_altitude
            
            # Get FOV
            fovy = img_info.get("fovy", [default_fov])
            fov_value = fovy[0] if isinstance(fovy, (list, tuple)) else fovy
            
            # Estimate footprint size
            footprint_y = 2 * altitude * math.tan(math.radians(fov_value / 2))
            aspect = img.shape[1] / img.shape[0]
            footprint_x = footprint_y * aspect
            
            # Calculate pixel position (center of image)
            px = int((location[0] - min_x) / ground_resolution)
            py = int((max_y - location[1]) / ground_resolution)  # Flip Y
            
            # Calculate output size for this frame
            out_w = int(footprint_x / ground_resolution)
            out_h = int(footprint_y / ground_resolution)
            
            # Clamp to reasonable sizes
            out_w = max(10, min(out_w, width_pixels // 2))
            out_h = max(10, min(out_h, height_pixels // 2))
            
            # Resize image
            resized = cv2.resize(img, (out_w, out_h)).astype(np.float32)
            
            # Calculate placement bounds (centered on camera position)
            x1 = max(0, px - out_w // 2)
            y1 = max(0, py - out_h // 2)
            x2 = min(width_pixels, x1 + out_w)
            y2 = min(height_pixels, y1 + out_h)
            
            # Calculate source region (if clipped)
            sx1 = max(0, (out_w // 2) - px) if px < out_w // 2 else 0
            sy1 = max(0, (out_h // 2) - py) if py < out_h // 2 else 0
            sx2 = sx1 + (x2 - x1)
            sy2 = sy1 + (y2 - y1)
            
            if x1 < x2 and y1 < y2 and sx1 < sx2 and sy1 < sy2 and sx2 <= out_w and sy2 <= out_h:
                try:
                    region = resized[sy1:sy2, sx1:sx2]
                    
                    if blend_mode == "integral":
                        # Accumulate for averaging
                        current_count = count_map[y1:y2, x1:x2]
                        for c in range(4):
                            canvas[y1:y2, x1:x2, c] = (
                                canvas[y1:y2, x1:x2, c] * current_count + region[:, :, c]
                            ) / (current_count + 1)
                        count_map[y1:y2, x1:x2] += 1
                    elif blend_mode == "first":
                        # Only write where empty
                        mask = canvas[y1:y2, x1:x2, 3] == 0
                        for c in range(4):
                            canvas[y1:y2, x1:x2, c][mask] = region[:, :, c][mask]
                    else:
                        # Last write wins (default)
                        canvas[y1:y2, x1:x2] = region
                    
                    images_placed += 1
                except Exception as e:
                    if log_fn and images_placed < 3:
                        log_fn(f"Warning: Error placing image {i}: {e}")
            
            if progress_fn and i % 20 == 0:
                progress = 30 + int((i / len(valid_images)) * 50)
                progress_fn(min(progress, 80))
        
        if log_fn:
            log_fn(f"Placed {images_placed} images on canvas")
        
        if progress_fn:
            progress_fn(80)
        
        # Convert to uint8 if needed
        result = canvas.astype(np.uint8)
        
        # Diagnostic: check result
        if log_fn:
            log_fn(f"Result shape: {result.shape}, min={result.min()}, max={result.max()}")
            non_zero = np.sum(result[:, :, 3] > 0)
            log_fn(f"Non-transparent pixels: {non_zero} / {result.shape[0] * result.shape[1]}")
        
        global_bounds = (min_x, min_y, max_x, max_y)
        
        # Crop to content if requested
        if crop_to_content:
            if log_fn:
                log_fn("Cropping to content area...")
            result, global_bounds = self._crop_to_content(result, global_bounds)
        
        # Apply coordinate offsets
        geo_bounds = (
            global_bounds[0] + coord_offset_x,
            global_bounds[1] + coord_offset_y,
            global_bounds[2] + coord_offset_x,
            global_bounds[3] + coord_offset_y
        )
        
        if progress_fn:
            progress_fn(85)
        
        # Save output
        self._save_orthomosaic(
            result, output_file, geo_bounds, target_epsg,
            create_overviews, log_fn
        )
        
        if progress_fn:
            progress_fn(95)
    
    def _render_sequential(
        self, renderer, shots, mask, resolution, blend_mode, global_bounds,
        progress_fn, log_fn
    ):
        """Render orthomosaic by processing shots sequentially."""
        import numpy as np
        from alfspy.core.rendering import RenderResultMode
        
        output = np.zeros((resolution.height, resolution.width, 4), dtype=np.uint8)
        
        total_shots = len(shots)
        
        for i, shot in enumerate(shots):
            try:
                results = list(renderer.project_shots_iter(
                    shot,
                    RenderResultMode.ShotOnly,
                    release_shots=False,
                    mask=mask
                ))
                
                if results and len(results) > 0:
                    shot_result = results[0]
                    shot_alpha = shot_result[:, :, 3]
                    shot_has_data = shot_alpha > 0
                    
                    if blend_mode == "first":
                        output_empty = output[:, :, 3] == 0
                        write_mask = shot_has_data & output_empty
                    else:
                        write_mask = shot_has_data
                    
                    for c in range(4):
                        output[:, :, c][write_mask] = shot_result[:, :, c][write_mask]
                        
            except Exception as e:
                if log_fn:
                    log_fn(f"Warning: Failed to render shot {i}: {e}")
                continue
            
            if progress_fn and i % 10 == 0:
                progress = 45 + int((i / total_shots) * 30)
                progress_fn(min(progress, 75))
        
        return output
    
    def _crop_to_content(self, image, bounds):
        """Crop image to minimal bounding box containing non-empty pixels."""
        import numpy as np
        
        min_x, min_y, max_x, max_y = bounds
        height, width = image.shape[:2]
        
        alpha = image[:, :, 3]
        non_empty = alpha > 0
        
        if not np.any(non_empty):
            return image, bounds
        
        rows = np.any(non_empty, axis=1)
        cols = np.any(non_empty, axis=0)
        
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        
        cropped = image[row_min:row_max + 1, col_min:col_max + 1]
        
        pixel_size_x = (max_x - min_x) / width
        pixel_size_y = (max_y - min_y) / height
        
        new_min_x = min_x + col_min * pixel_size_x
        new_max_x = min_x + (col_max + 1) * pixel_size_x
        new_max_y = max_y - row_min * pixel_size_y
        new_min_y = max_y - (row_max + 1) * pixel_size_y
        
        return cropped, (new_min_x, new_min_y, new_max_x, new_max_y)
    
    def _save_orthomosaic(
        self, image, output_file, bounds, crs_epsg,
        create_overviews, log_fn
    ):
        """Save orthomosaic image with georeferencing."""
        import cv2
        import numpy as np
        
        height, width = image.shape[:2]
        min_x, min_y, max_x, max_y = bounds
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Check if image has content
        if log_fn:
            log_fn(f"Saving image: {width}x{height}, dtype={image.dtype}")
            if len(image.shape) == 3:
                for i in range(min(4, image.shape[2])):
                    band = image[:, :, i]
                    log_fn(f"  Band {i}: min={band.min()}, max={band.max()}, mean={band.mean():.1f}")
        
        # Try to use rasterio for GeoTIFF
        try:
            import rasterio
            from rasterio.transform import from_bounds
            
            transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
            
            # Determine number of bands
            if len(image.shape) == 2:
                count = 1
                image = image[:, :, np.newaxis]
            else:
                count = image.shape[2]
            
            profile = {
                'driver': 'GTiff',
                'dtype': image.dtype,
                'width': width,
                'height': height,
                'count': count,
                'transform': transform,
                'compress': 'lzw',
            }
            
            # Try to set CRS using pyproj first (avoids PROJ database conflicts in QGIS)
            crs_set = False
            try:
                from pyproj import CRS as PyprojCRS
                pyproj_crs = PyprojCRS.from_epsg(crs_epsg)
                # Use WKT string which is more portable
                from rasterio.crs import CRS as RasterioCRS
                profile['crs'] = RasterioCRS.from_wkt(pyproj_crs.to_wkt())
                crs_set = True
                if log_fn:
                    log_fn(f"CRS set using pyproj: EPSG:{crs_epsg}")
            except Exception as e1:
                if log_fn:
                    log_fn(f"Warning: pyproj CRS failed: {e1}")
                # Try rasterio directly as fallback
                try:
                    from rasterio.crs import CRS as RasterioCRS
                    profile['crs'] = RasterioCRS.from_epsg(crs_epsg)
                    crs_set = True
                except Exception as e2:
                    if log_fn:
                        log_fn(f"Warning: rasterio CRS also failed: {e2}")
                        log_fn("GeoTIFF will be saved without embedded CRS - use .prj file")
            
            if width > 256 and height > 256:
                profile['tiled'] = True
                profile['blockxsize'] = 256
                profile['blockysize'] = 256
            
            if log_fn:
                log_fn(f"Saving GeoTIFF to {output_file}")
            
            with rasterio.open(output_file, 'w', **profile) as dst:
                for i in range(count):
                    dst.write(image[:, :, i], i + 1)
                
                if count == 4:
                    dst.descriptions = ('Red', 'Green', 'Blue', 'Alpha')
                elif count == 3:
                    dst.descriptions = ('Red', 'Green', 'Blue')
                
                if create_overviews:
                    if log_fn:
                        log_fn("Creating overviews...")
                    try:
                        overview_levels = [2, 4, 8, 16]
                        dst.build_overviews(overview_levels, rasterio.enums.Resampling.average)
                    except Exception as e:
                        if log_fn:
                            log_fn(f"Warning: Could not create overviews: {e}")
            
            # Always save world file as backup
            self._save_world_file(output_file, bounds, width, height)
            
            # Always save PRJ file for CRS (as backup and for compatibility)
            self._save_prj_file(output_file, crs_epsg, log_fn)
            
            if log_fn:
                log_fn("GeoTIFF saved successfully")
            
        except ImportError:
            # Fall back to OpenCV + world file
            if log_fn:
                log_fn("rasterio not available, saving with world file")
            
            self._save_with_worldfile(image, output_file, bounds, crs_epsg, log_fn)
        except Exception as e:
            if log_fn:
                log_fn(f"Warning: GeoTIFF save failed ({e}), falling back to world file")
            self._save_with_worldfile(image, output_file, bounds, crs_epsg, log_fn)
    
    def _save_with_worldfile(self, image, output_file, bounds, crs_epsg, log_fn):
        """Save image with world file (fallback method)."""
        import cv2
        
        height, width = image.shape[:2]
        min_x, min_y, max_x, max_y = bounds
        
        # Convert to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 4:
            result_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            result_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            result_bgr = image
        
        cv2.imwrite(output_file, result_bgr)
        
        # Save world file
        self._save_world_file(output_file, bounds, width, height)
        
        # Save PRJ file
        self._save_prj_file(output_file, crs_epsg, log_fn)
        
        if log_fn:
            log_fn(f"Image saved with world file")
    
    def _save_world_file(self, output_file, bounds, width, height):
        """Save a world file (.tfw) for georeferencing."""
        min_x, min_y, max_x, max_y = bounds
        
        # Determine world file extension
        if output_file.lower().endswith('.tif') or output_file.lower().endswith('.tiff'):
            world_file = output_file.rsplit('.', 1)[0] + '.tfw'
        elif output_file.lower().endswith('.png'):
            world_file = output_file.rsplit('.', 1)[0] + '.pgw'
        elif output_file.lower().endswith('.jpg') or output_file.lower().endswith('.jpeg'):
            world_file = output_file.rsplit('.', 1)[0] + '.jgw'
        else:
            world_file = output_file + '.wld'
        
        pixel_size_x = (max_x - min_x) / width
        pixel_size_y = -(max_y - min_y) / height  # Negative for Y
        
        with open(world_file, 'w') as f:
            f.write(f"{pixel_size_x:.10f}\n")
            f.write("0.0\n")
            f.write("0.0\n")
            f.write(f"{pixel_size_y:.10f}\n")
            f.write(f"{min_x + pixel_size_x / 2:.10f}\n")
            f.write(f"{max_y + pixel_size_y / 2:.10f}\n")
    
    def _save_prj_file(self, output_file, crs_epsg, log_fn):
        """Save a .prj file with CRS definition."""
        prj_file = output_file.rsplit('.', 1)[0] + '.prj'
        
        try:
            from pyproj import CRS
            crs = CRS.from_epsg(crs_epsg)
            with open(prj_file, 'w') as f:
                f.write(crs.to_wkt())
            if log_fn:
                log_fn(f"PRJ file saved: {prj_file}")
        except Exception as e:
            if log_fn:
                log_fn(f"Warning: Could not save PRJ file: {e}")

    def run_export_geotiffs(self, config: Dict[str, Any], progress_fn=None, log_fn=None):
        """Export each frame as an individual GeoTIFF using optimized projection.
        
        This method uses an efficient single-pass approach that loads the DEM mesh
        once and projects each frame individually without the overhead of the
        full orthomosaic pipeline.
        
        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        import numpy as np
        import cv2
        from PIL import Image
        from dataclasses import dataclass
        
        # Check for required dependencies
        try:
            from scipy.interpolate import griddata
            from scipy.signal import savgol_filter
        except ImportError:
            raise ImportError(
                "scipy is required for GeoTIFF export.\n"
                "Please install: pip install scipy"
            )
        
        try:
            from pyrr import Quaternion
        except ImportError:
            raise ImportError(
                "pyrr is required for GeoTIFF export.\n"
                "Please install: pip install pyrr"
            )
        
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "trimesh is required for GeoTIFF export.\n"
                "Please install: pip install trimesh"
            )
        
        if log_fn:
            log_fn("Initializing optimized frame export...")
        
        # Get configuration parameters
        target_folder = config["target_folder"]
        dem_path = config["dem_path"]
        target_epsg = config.get("target_epsg", 32633)
        use_all_frames = config.get("ortho_use_all_frames", True)
        start_frame = config.get("ortho_start_frame", 0)
        end_frame = config.get("ortho_end_frame", 999999)
        ground_resolution = config.get("ortho_ground_resolution", 0.1)
        dem_metadata_path = config.get("ortho_dem_metadata_path")
        apply_smoothing = config.get("geotiff_apply_smoothing", True)
        mesh_subsample = config.get("geotiff_mesh_subsample", 1)
        
        if log_fn:
            if use_all_frames:
                log_fn("Frame range: All frames")
            else:
                log_fn(f"Frame range: {start_frame} to {end_frame}")
            log_fn(f"Output resolution: {ground_resolution} m/px")
        
        if progress_fn:
            progress_fn(2)
        
        # ====== Load poses.json ======
        poses_file = os.path.join(target_folder, "poses.json")
        if not os.path.exists(poses_file):
            raise FileNotFoundError(f"poses.json not found at {poses_file}")
            
        with open(poses_file, 'r') as f:
            poses = json.load(f)
            
        all_images = poses.get("images", [])
        total_all = len(all_images)
        
        if total_all == 0:
            raise RuntimeError("No images found in poses.json")
            
        if log_fn:
            log_fn(f"Found {total_all} images in poses.json")
        
        # Apply pose smoothing if requested
        if apply_smoothing and total_all >= 5:
            window_length = min(11, total_all - 1 if (total_all - 1) % 2 == 1 else total_all - 2)
            if window_length >= 3:
                positions = np.array([img["location"] for img in all_images], dtype=float)
                smoothed = savgol_filter(positions, window_length=window_length,
                                        polyorder=2, axis=0, mode="interp")
                for img, loc in zip(all_images, smoothed):
                    img["location"] = loc.tolist()
                if log_fn:
                    log_fn(f"Applied pose smoothing (window={window_length})")
        
        # Build list of frame indices to export
        frame_indices = []
        for i in range(total_all):
            if use_all_frames or (start_frame <= i <= end_frame):
                frame_indices.append(i)
        
        if len(frame_indices) == 0:
            raise RuntimeError("No frames to export after filtering")
            
        if log_fn:
            log_fn(f"Will export {len(frame_indices)} frames as GeoTIFF")
        
        if progress_fn:
            progress_fn(5)
        
        # ====== Load DEM mesh ======
        if log_fn:
            log_fn(f"Loading DEM mesh from: {dem_path}")
        
        if not os.path.exists(dem_path):
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
        mesh = trimesh.load(dem_path, force='mesh')
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        
        # Apply subsampling if requested
        if mesh_subsample > 1:
            face_indices = np.arange(0, len(faces), mesh_subsample)
            faces = faces[face_indices]
            unique_vertex_indices = np.unique(faces.flatten())
            vertex_map = {old: new for new, old in enumerate(unique_vertex_indices)}
            vertices = vertices[unique_vertex_indices]
            faces = np.array([[vertex_map[v] for v in face] for face in faces])
            if log_fn:
                log_fn(f"Mesh subsampled by factor {mesh_subsample}")
        
        if log_fn:
            log_fn(f"Loaded {len(vertices)} vertices and {len(faces)} faces")
            log_fn(f"Mesh bounds (local): X=[{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}]")
            log_fn(f"                     Y=[{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}]")
        
        if progress_fn:
            progress_fn(10)
        
        # ====== Load DEM metadata (coordinate offset) ======
        coord_offset_x = 0.0
        coord_offset_y = 0.0
        coord_offset_z = 0.0
        
        # Try explicit metadata path first
        if dem_metadata_path and os.path.exists(dem_metadata_path):
            if log_fn:
                log_fn(f"Loading DEM metadata from: {dem_metadata_path}")
            with open(dem_metadata_path, 'r') as f:
                dem_metadata = json.load(f)
            origin = dem_metadata.get("origin", [0, 0, 0])
            coord_offset_x, coord_offset_y, coord_offset_z = float(origin[0]), float(origin[1]), float(origin[2])
        else:
            # Try to find metadata in same folder as DEM
            auto_metadata_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
            if os.path.exists(auto_metadata_path):
                if log_fn:
                    log_fn(f"Found DEM metadata at: {auto_metadata_path}")
                with open(auto_metadata_path, 'r') as f:
                    dem_metadata = json.load(f)
                origin = dem_metadata.get("origin", [0, 0, 0])
                coord_offset_x, coord_offset_y, coord_offset_z = float(origin[0]), float(origin[1]), float(origin[2])
        
        if log_fn:
            log_fn(f"DEM origin offset: ({coord_offset_x:.2f}, {coord_offset_y:.2f}, {coord_offset_z:.2f})")
        
        # ====== Load mask if available ======
        mask_array = None
        mask_filename = poses.get("mask")
        if mask_filename:
            mask_path = os.path.join(target_folder, mask_filename)
            if os.path.exists(mask_path):
                try:
                    mask_img = Image.open(mask_path).convert('L')
                    mask_array = np.array(mask_img)
                    if mask_array.max() <= 1:
                        mask_array = (mask_array * 255).astype(np.uint8)
                    if log_fn:
                        log_fn(f"Loaded mask from: {mask_path}")
                except Exception as e:
                    if log_fn:
                        log_fn(f"Warning: Could not load mask: {e}")
        
        if progress_fn:
            progress_fn(15)
        
        # ====== Create output folder ======
        geotiff_folder = os.path.join(target_folder, "geotiffs")
        os.makedirs(geotiff_folder, exist_ok=True)
        
        # ====== Process frames ======
        exported_count = 0
        total_frames = len(frame_indices)
        
        if log_fn:
            log_fn(f"Starting frame export...")
        
        for i, frame_idx in enumerate(frame_indices):
            try:
                # Get image info
                img_info = all_images[frame_idx]
                image_file = img_info.get("imagefile")
                image_path = os.path.join(target_folder, image_file)
                
                if not os.path.exists(image_path):
                    if log_fn and i < 5:
                        log_fn(f"Warning: Image not found: {image_file}")
                    continue
                
                # Load image
                img = Image.open(image_path).convert('RGB')
                image = np.array(img)
                img_h, img_w = image.shape[:2]
                
                # Apply mask
                if mask_array is not None:
                    if (mask_array.shape[0], mask_array.shape[1]) != (img_h, img_w):
                        mask_resized = np.array(Image.fromarray(mask_array).resize((img_w, img_h), Image.Resampling.NEAREST))
                    else:
                        mask_resized = mask_array
                    image_mask = mask_resized > 127
                else:
                    image_mask = np.ones((img_h, img_w), dtype=bool)
                
                # Get camera parameters
                position = np.array(img_info['location'], dtype=float)
                rotation_deg = np.array(img_info['rotation'], dtype=float)
                fovy = img_info.get('fovy', 45.0)
                if isinstance(fovy, list):
                    fovy = fovy[0]
                
                # Apply rotation correction and convert to radians
                rotation_rad = np.deg2rad((rotation_deg % 360.0)) * -1
                rotation = Quaternion.from_eulers(rotation_rad)
                rotation_matrix = np.array(rotation.matrix33)
                
                # Build camera transformation matrix
                cam_to_world = np.eye(4)
                cam_to_world[:3, :3] = rotation_matrix
                cam_to_world[:3, 3] = position
                world_to_cam = np.linalg.inv(cam_to_world)
                
                # Calculate FOV
                aspect_ratio = img_w / img_h
                fov_y_rad = np.deg2rad(fovy)
                fov_x_rad = 2 * np.arctan(aspect_ratio * np.tan(fov_y_rad / 2))
                half_width = np.tan(fov_x_rad / 2)
                half_height = np.tan(fov_y_rad / 2)
                
                # Project mesh vertices to image coordinates (vectorized)
                n_vertices = len(vertices)
                
                # Convert all vertices to homogeneous coordinates
                vertices_h = np.hstack([vertices, np.ones((n_vertices, 1))])
                
                # Transform all vertices to camera space at once
                vertices_cam = (world_to_cam @ vertices_h.T).T  # (N, 4)
                
                # Find vertices in front of camera (z < 0 in camera space)
                in_front = vertices_cam[:, 2] < 0
                
                # Project to normalized device coordinates (only for valid vertices)
                uvs = np.zeros((n_vertices, 2))
                valid_mask = np.zeros(n_vertices, dtype=bool)
                
                if np.any(in_front):
                    z_neg = -vertices_cam[in_front, 2]
                    x_ndc = vertices_cam[in_front, 0] / z_neg
                    y_ndc = vertices_cam[in_front, 1] / z_neg
                    
                    # Convert to UV coordinates [0, 1]
                    u = (x_ndc / half_width + 1) / 2
                    v = 1.0 - (y_ndc / half_height + 1) / 2
                    
                    # Check which are within valid UV range
                    uv_valid = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
                    
                    # Store results
                    in_front_indices = np.where(in_front)[0]
                    valid_indices_temp = in_front_indices[uv_valid]
                    valid_mask[valid_indices_temp] = True
                    uvs[valid_indices_temp, 0] = u[uv_valid]
                    uvs[valid_indices_temp, 1] = v[uv_valid]
                
                n_valid = np.sum(valid_mask)
                if n_valid < 10:
                    if log_fn and i < 5:
                        log_fn(f"Warning: Too few valid vertices for frame {frame_idx} ({n_valid})")
                    continue
                
                # Get valid vertices
                valid_indices = np.where(valid_mask)[0]
                valid_vertices = vertices[valid_indices]
                valid_uvs = uvs[valid_indices]
                
                # Calculate bounds in local coordinates
                local_min_x = valid_vertices[:, 0].min()
                local_max_x = valid_vertices[:, 0].max()
                local_min_y = valid_vertices[:, 1].min()
                local_max_y = valid_vertices[:, 1].max()
                
                # Calculate output dimensions
                width_m = local_max_x - local_min_x
                height_m = local_max_y - local_min_y
                out_width = int(np.ceil(width_m / ground_resolution))
                out_height = int(np.ceil(height_m / ground_resolution))
                
                if out_width <= 0 or out_height <= 0:
                    continue
                
                # Limit maximum dimensions
                max_dim = 8000
                if out_width > max_dim or out_height > max_dim:
                    scale = max_dim / max(out_width, out_height)
                    out_width = int(out_width * scale)
                    out_height = int(out_height * scale)
                
                # Create output grid
                x_coords = np.linspace(local_min_x, local_max_x, out_width)
                y_coords = np.linspace(local_max_y, local_min_y, out_height)  # Top to bottom
                grid_x, grid_y = np.meshgrid(x_coords, y_coords)
                grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
                
                # Interpolate UV coordinates at grid points
                interp_u = griddata(
                    valid_vertices[:, :2], valid_uvs[:, 0],
                    grid_points, method='linear', fill_value=-1
                ).reshape(out_height, out_width)
                
                interp_v = griddata(
                    valid_vertices[:, :2], valid_uvs[:, 1],
                    grid_points, method='linear', fill_value=-1
                ).reshape(out_height, out_width)
                
                # Convert UV to pixel coordinates
                pixel_x = interp_u * (img_w - 1)
                pixel_y = interp_v * (img_h - 1)
                
                # Valid output pixels
                valid_output = (interp_u >= 0) & (interp_u <= 1) & (interp_v >= 0) & (interp_v <= 1)
                
                # Create output arrays
                n_channels = image.shape[2] if len(image.shape) == 3 else 1
                if n_channels == 1:
                    output_image = np.zeros((out_height, out_width), dtype=np.uint8)
                else:
                    output_image = np.zeros((out_height, out_width, n_channels), dtype=np.uint8)
                output_valid = np.zeros((out_height, out_width), dtype=bool)
                
                # Sample pixels using vectorized bilinear interpolation
                valid_rows, valid_cols = np.where(valid_output)
                
                if len(valid_rows) > 0:
                    px = pixel_x[valid_rows, valid_cols]
                    py = pixel_y[valid_rows, valid_cols]
                    
                    x0 = np.floor(px).astype(int)
                    y0 = np.floor(py).astype(int)
                    x1 = np.minimum(x0 + 1, img_w - 1)
                    y1 = np.minimum(y0 + 1, img_h - 1)
                    
                    fx = px - x0
                    fy = py - y0
                    
                    # Check mask for all four corners
                    mask_valid = (
                        image_mask[y0, x0] & 
                        image_mask[y0, x1] & 
                        image_mask[y1, x0] & 
                        image_mask[y1, x1]
                    )
                    
                    # Filter to only valid pixels
                    valid_px_rows = valid_rows[mask_valid]
                    valid_px_cols = valid_cols[mask_valid]
                    x0_v = x0[mask_valid]
                    y0_v = y0[mask_valid]
                    x1_v = x1[mask_valid]
                    y1_v = y1[mask_valid]
                    fx_v = fx[mask_valid]
                    fy_v = fy[mask_valid]
                    
                    # Compute interpolation weights
                    w00 = (1 - fx_v) * (1 - fy_v)
                    w01 = fx_v * (1 - fy_v)
                    w10 = (1 - fx_v) * fy_v
                    w11 = fx_v * fy_v
                    
                    if n_channels == 1:
                        vals = (image[y0_v, x0_v] * w00 +
                                image[y0_v, x1_v] * w01 +
                                image[y1_v, x0_v] * w10 +
                                image[y1_v, x1_v] * w11)
                        output_image[valid_px_rows, valid_px_cols] = np.clip(vals, 0, 255).astype(np.uint8)
                    else:
                        for c in range(n_channels):
                            vals = (image[y0_v, x0_v, c] * w00 +
                                    image[y0_v, x1_v, c] * w01 +
                                    image[y1_v, x0_v, c] * w10 +
                                    image[y1_v, x1_v, c] * w11)
                            output_image[valid_px_rows, valid_px_cols, c] = np.clip(vals, 0, 255).astype(np.uint8)
                    
                    output_valid[valid_px_rows, valid_px_cols] = True
                
                # Convert bounds to UTM
                utm_bounds = (
                    local_min_x + coord_offset_x,
                    local_min_y + coord_offset_y,
                    local_max_x + coord_offset_x,
                    local_max_y + coord_offset_y
                )
                
                # Save GeoTIFF
                output_file = os.path.join(geotiff_folder, f"{frame_idx:08d}.tiff")
                self._save_frame_geotiff(output_image, output_valid, utm_bounds, output_file, target_epsg)
                
                if os.path.exists(output_file):
                    exported_count += 1
                    
            except Exception as e:
                if log_fn and i < 10:
                    import traceback
                    log_fn(f"Warning: Failed to export frame {frame_idx}: {e}")
                    if i < 3:
                        log_fn(traceback.format_exc())
            
            # Update progress
            if progress_fn:
                progress = 15 + int((i + 1) / total_frames * 80)
                progress_fn(min(progress, 95))
            
            # Periodic status update
            if log_fn and (i + 1) % 50 == 0:
                log_fn(f"Exported {exported_count}/{i + 1} frames...")
        
        if log_fn:
            log_fn(f"Successfully exported {exported_count} of {total_frames} frames")
            log_fn(f"Frame export complete. Files saved to: {geotiff_folder}")
            
        if progress_fn:
            progress_fn(100)
    
    def _save_frame_geotiff(self, image: 'np.ndarray', valid_mask: 'np.ndarray', 
                           bounds: tuple, output_path: str, crs_epsg: int):
        """Save a single frame as a georeferenced GeoTIFF.
        
        :param image: Image array (H, W) or (H, W, C)
        :param valid_mask: Boolean mask for valid pixels
        :param bounds: (min_x, min_y, max_x, max_y) in CRS coordinates
        :param output_path: Output file path
        :param crs_epsg: EPSG code for coordinate reference system
        """
        import numpy as np
        
        min_x, min_y, max_x, max_y = bounds
        height, width = image.shape[:2]
        
        # Determine number of bands
        if len(image.shape) == 2:
            count = 1
            data = image[np.newaxis, :, :]
        else:
            count = image.shape[2]
            data = np.moveaxis(image, -1, 0)
        
        # Set nodata for invalid pixels
        nodata = 0
        for band in range(count):
            data[band][~valid_mask] = nodata
        
        # Try to use rasterio
        try:
            import rasterio
            from rasterio.transform import from_bounds
            
            transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
            
            profile = {
                'driver': 'GTiff',
                'dtype': data.dtype,
                'width': width,
                'height': height,
                'count': count,
                'transform': transform,
                'compress': 'lzw',
                'nodata': nodata
            }
            
            # Set CRS
            try:
                from pyproj import CRS as PyprojCRS
                from rasterio.crs import CRS as RasterioCRS
                pyproj_crs = PyprojCRS.from_epsg(crs_epsg)
                profile['crs'] = RasterioCRS.from_wkt(pyproj_crs.to_wkt())
            except Exception:
                try:
                    from rasterio.crs import CRS as RasterioCRS
                    profile['crs'] = RasterioCRS.from_epsg(crs_epsg)
                except Exception:
                    pass  # Will save without CRS
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)
            
            # Save world file as backup
            self._save_world_file(output_path, bounds, width, height)
            
        except ImportError:
            # Fall back to OpenCV + world file
            import cv2
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                result_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                result_bgr = image
            
            cv2.imwrite(output_path, result_bgr)
            self._save_world_file(output_path, bounds, width, height)
            self._save_prj_file(output_path, crs_epsg, None)



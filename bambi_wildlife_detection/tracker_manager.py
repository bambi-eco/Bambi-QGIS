# -*- coding: utf-8 -*-
"""
BAMBI Wildlife Detection - Tracker Manager
===========================================

This module manages different tracking backends:
- Built-in tracker (default, no dependencies)
- BoxMOT trackers (optional, requires pip install boxmot)
- Geo-referenced trackers (optional, requires pip install georef-tracker)

The manager handles:
- Backend detection and availability checking
- ReID model selection and downloading
- JSON parameter configuration with validation
- Tracker instantiation with appropriate parameters
"""

import json
import inspect
import os
import urllib.request
import ssl
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Type
from enum import Enum
from dataclasses import dataclass


class TrackerBackend(Enum):
    """Available tracking backends."""
    BUILTIN = "builtin"           # Default built-in tracker (no dependencies)
    BOXMOT = "boxmot"             # Standard BoxMOT trackers
    GEOREF_NATIVE = "georef_native"    # Geo-native DeepOCSORT
    GEOREF_HYBRID = "georef_hybrid"    # Geo-hybrid DeepOCSORT


class ReIDModel(Enum):
    """Available ReID model options."""
    OSNET = "osnet"               # BoxMOT's standard osnet model
    BAMBI = "bambi"               # BAMBI Thermal-Omni ReID from HuggingFace
    CUSTOM = "custom"             # Custom user-provided weights


@dataclass
class TrackerInfo:
    """Information about an available tracker."""
    name: str
    backend: TrackerBackend
    description: str
    requires_reid: bool
    supports_geo: bool
    tracker_class: Optional[Type] = None


class TrackerManager:
    """
    Manages tracking backends and their configuration.
    
    This class handles:
    - Detection of available backends (boxmot, georef-tracker)
    - ReID model selection and downloading
    - JSON parameter parsing and validation
    - Tracker instantiation
    """
    
    # HuggingFace repository for BAMBI models
    BAMBI_REPO = "cpraschl/bambi-thermal-omni"
    
    # BAMBI Thermal-Omni ReID model (custom trained for thermal wildlife)
    # Based on osnet_x0_5 architecture - filename must include this for BoxMOT
    BAMBI_REID_FILENAME = "model.pt"  # The file at HuggingFace
    BAMBI_REID_LOCAL_NAME = "osnet_x0_5_bambi_thermal_omni.pt"  # Local filename includes architecture
    
    # OSNET ReID model (from BoxMOT/Google Drive)
    OSNET_REID_FILENAME = "osnet_x0_25_msmt17.pt"
    OSNET_GOOGLE_DRIVE_ID = "1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"
    
    def __init__(self, models_folder: Optional[str] = None):
        """
        Initialize the tracker manager.
        
        :param models_folder: Folder to store/load model weights
        """
        self.models_folder = models_folder or os.path.expanduser("~/.bambi/models")
        os.makedirs(self.models_folder, exist_ok=True)
        
        # Cache availability checks
        self._boxmot_available = None
        self._georef_available = None
        self._available_trackers: Optional[Dict[str, TrackerInfo]] = None
        
    def _ensure_stderr(self):
        """Ensure sys.stderr is not None (required for some libraries like loguru)."""
        import sys
        import io
        
        if sys.stderr is None:
            sys.stderr = io.StringIO()
            return True
        return False
    
    def _is_package_installed(self, package_name: str) -> bool:
        """Check if a package is installed without importing it."""
        import importlib.util
        return importlib.util.find_spec(package_name) is not None
    
    def refresh_available_trackers(self):
        """Force refresh of available trackers (clears cache)."""
        self._boxmot_available = None
        self._georef_available = None
        self._available_trackers = None
        # Clear any stored import errors
        if hasattr(self, '_boxmot_import_error'):
            delattr(self, '_boxmot_import_error')
        if hasattr(self, '_georef_import_error'):
            delattr(self, '_georef_import_error')
        if hasattr(self, '_boxmot_tracker_errors'):
            delattr(self, '_boxmot_tracker_errors')
    
    def is_boxmot_available(self) -> bool:
        """Check if BoxMOT is installed and available."""
        if self._boxmot_available is None:
            # First check if package is installed at all
            if not self._is_package_installed("boxmot"):
                self._boxmot_available = False
                return False
            
            # Check if boxmot was already successfully imported (e.g., from Python console)
            import sys
            if "boxmot" in sys.modules:
                self._boxmot_available = True
                return True
                
            try:
                # BoxMOT uses loguru which requires sys.stderr to not be None
                # In QGIS, sys.stderr can be None which causes import to fail
                self._ensure_stderr()
                
                import boxmot
                self._boxmot_available = True
                    
            except Exception as e:
                # Store the error for debugging
                self._boxmot_import_error = str(e)
                self._boxmot_available = False
                
        return self._boxmot_available
    
    def is_georef_available(self) -> bool:
        """Check if geo-referenced tracker is installed and available."""
        if self._georef_available is None:
            # First check if package is installed at all
            if not self._is_package_installed("georef_tracker"):
                self._georef_available = False
                return False
            
            # Check if georef_tracker was already successfully imported
            import sys
            if "georef_tracker" in sys.modules:
                self._georef_available = True
                return True
                
            try:
                # georef_tracker depends on boxmot, so may have same issue
                self._ensure_stderr()
                
                import georef_tracker
                self._georef_available = True
                    
            except Exception as e:
                # Store the error for debugging
                self._georef_import_error = str(e)
                self._georef_available = False
                
        return self._georef_available
    
    def get_import_errors(self) -> Dict[str, str]:
        """Get any import errors that occurred during availability checks."""
        errors = {}
        if hasattr(self, '_boxmot_import_error'):
            errors['boxmot'] = self._boxmot_import_error
        if hasattr(self, '_georef_import_error'):
            errors['georef_tracker'] = self._georef_import_error
        # Include individual tracker import errors
        if hasattr(self, '_boxmot_tracker_errors'):
            for tracker_id, error in self._boxmot_tracker_errors.items():
                errors[f'boxmot:{tracker_id}'] = error
        return errors
    
    def get_available_trackers(self) -> Dict[str, TrackerInfo]:
        """
        Get all available trackers based on installed packages.
        
        :return: Dictionary of tracker_id -> TrackerInfo
        """
        if self._available_trackers is not None:
            return self._available_trackers
            
        trackers = {}
        
        # Built-in tracker (always available)
        trackers["builtin"] = TrackerInfo(
            name="Built-in (Hungarian IoU)",
            backend=TrackerBackend.BUILTIN,
            description="Default tracker using Hungarian algorithm with IoU matching. No additional dependencies required.",
            requires_reid=False,
            supports_geo=True
        )
        
        # BoxMOT trackers (if available)
        if self.is_boxmot_available():
            boxmot_trackers = self._get_boxmot_trackers()
            trackers.update(boxmot_trackers)
            
        # Geo-referenced trackers (if available)
        if self.is_georef_available():
            georef_trackers = self._get_georef_trackers()
            trackers.update(georef_trackers)
            
        self._available_trackers = trackers
        return trackers
    
    def _get_boxmot_trackers(self) -> Dict[str, TrackerInfo]:
        """Get available BoxMOT trackers."""
        trackers = {}
        
        # Ensure stderr is available for loguru
        self._ensure_stderr()
        
        # Track import errors for individual trackers
        if not hasattr(self, '_boxmot_tracker_errors'):
            self._boxmot_tracker_errors = {}
        
        try:
            import importlib
            boxmot = importlib.import_module("boxmot")
            
            # Define tracker configurations with multiple possible class names
            # (class_names_to_try, tracker_id, display_name, description, requires_reid)
            # First name in list is preferred, others are fallbacks for older versions
            tracker_configs = [
                (["DeepOcSort", "DeepOCSORT"], "boxmot_deepocsort", "DeepOCSORT", 
                 "DeepOCSORT with observation-centric recovery and ReID", True),
                (["BotSort", "BoTSORT"], "boxmot_botsort", "BoTSORT", 
                 "BoTSORT with camera motion compensation", True),
                (["StrongSort", "StrongSORT"], "boxmot_strongsort", "StrongSORT", 
                 "StrongSORT with deep association", True),
                (["OcSort", "OCSORT"], "boxmot_ocsort", "OCSORT", 
                 "OCSORT - Observation-Centric SORT", False),
                (["ByteTrack", "BYTETracker"], "boxmot_bytetrack", "BYTETrack", 
                 "BYTETracker - high-performance MOT", False),
                (["HybridSort", "HybridSORT"], "boxmot_hybridsort", "HybridSORT", 
                 "Hybrid approach combining multiple strategies", True),
                (["BoostTrack"], "boxmot_boosttrack", "BoostTrack", 
                 "BoostTrack - enhanced tracking with boosting", True),
            ]
            
            for class_names, tracker_id, display_name, description, requires_reid in tracker_configs:
                tracker_class = None
                
                # Try each possible class name
                for class_name in class_names:
                    # Try direct attribute on boxmot module
                    if hasattr(boxmot, class_name):
                        tracker_class = getattr(boxmot, class_name)
                        break
                    
                    # Try importing from boxmot.trackers.{name}.{name}
                    # e.g., from boxmot.trackers.botsort.botsort import BotSort
                    try:
                        submod_name = class_name.lower()
                        submod = importlib.import_module(f"boxmot.trackers.{submod_name}.{submod_name}")
                        if hasattr(submod, class_name):
                            tracker_class = getattr(submod, class_name)
                            break
                    except ImportError:
                        pass
                    
                    # Try importing from boxmot.trackers.{name}
                    try:
                        submod_name = class_name.lower()
                        submod = importlib.import_module(f"boxmot.trackers.{submod_name}")
                        if hasattr(submod, class_name):
                            tracker_class = getattr(submod, class_name)
                            break
                    except ImportError:
                        pass
                
                if tracker_class is not None:
                    trackers[tracker_id] = TrackerInfo(
                        name=f"BoxMOT: {display_name}",
                        backend=TrackerBackend.BOXMOT,
                        description=description,
                        requires_reid=requires_reid,
                        supports_geo=False,
                        tracker_class=tracker_class
                    )
                else:
                    self._boxmot_tracker_errors[tracker_id] = f"Class not found (tried: {class_names})"
                    
        except Exception as e:
            self._boxmot_tracker_errors["_general"] = str(e)
            
        return trackers
    
    def _get_georef_trackers(self) -> Dict[str, TrackerInfo]:
        """Get available geo-referenced trackers."""
        trackers = {}
        
        try:
            # Ensure stderr is available for loguru (georef_tracker depends on boxmot)
            self._ensure_stderr()
            
            from georef_tracker import GeoNativeDeepOcSort, GeoHybridDeepOcSort
            
            trackers["georef_native"] = TrackerInfo(
                name="GeoNative DeepOCSORT",
                backend=TrackerBackend.GEOREF_NATIVE,
                description="DeepOCSORT operating entirely in geo-referenced coordinates. "
                           "Provides stable tracking independent of camera motion.",
                requires_reid=True,
                supports_geo=True,
                tracker_class=GeoNativeDeepOcSort
            )
            
            trackers["georef_hybrid"] = TrackerInfo(
                name="GeoHybrid DeepOCSORT",
                backend=TrackerBackend.GEOREF_HYBRID,
                description="Standard DeepOCSORT with an additional geo-referenced association stage "
                           "for recovering lost tracks using world coordinates.",
                requires_reid=True,
                supports_geo=True,
                tracker_class=GeoHybridDeepOcSort
            )
            
        except Exception:
            # georef_tracker import can fail
            pass
            
        return trackers
    
    def get_tracker_constructor_params(self, tracker_id: str) -> Dict[str, Any]:
        """
        Get the valid constructor parameters for a tracker.
        
        :param tracker_id: The tracker identifier
        :return: Dictionary of param_name -> (type, default, description)
        """
        trackers = self.get_available_trackers()
        
        if tracker_id not in trackers:
            return {}
            
        info = trackers[tracker_id]
        
        if info.backend == TrackerBackend.BUILTIN:
            # Built-in tracker parameters
            return {
                "iou_threshold": (float, 0.3, "IoU threshold for matching"),
                "max_age": (int, -1, "Max frames without detection (-1=unlimited)"),
                "max_center_distance": (float, 0.2, "Max center distance in meters"),
                "class_aware": (bool, True, "Match only same class"),
                "interpolate": (bool, True, "Interpolate missing frames"),
            }
            
        if info.tracker_class is None:
            return {}
            
        # Inspect the constructor of the tracker class
        params = {}
        try:
            sig = inspect.signature(info.tracker_class.__init__)
            for name, param in sig.parameters.items():
                if name in ('self', 'args', 'kwargs'):
                    continue
                    
                # Determine type from annotation or default value
                param_type = type(param.default) if param.default is not inspect.Parameter.empty else str
                if param.annotation is not inspect.Parameter.empty:
                    param_type = param.annotation
                    
                default = param.default if param.default is not inspect.Parameter.empty else None
                
                params[name] = (param_type, default, f"Parameter: {name}")
                
        except Exception:
            pass
            
        return params
    
    def validate_and_filter_params(
        self, 
        tracker_id: str, 
        params: Dict[str, Any],
        log_fn=None
    ) -> Dict[str, Any]:
        """
        Validate and filter parameters for a tracker constructor.
        
        Only parameters that are valid for the tracker will be returned.
        Invalid parameters are logged and ignored.
        
        :param tracker_id: The tracker identifier
        :param params: User-provided parameters
        :param log_fn: Optional logging function
        :return: Filtered dictionary of valid parameters
        """
        valid_params = self.get_tracker_constructor_params(tracker_id)
        filtered = {}
        
        for key, value in params.items():
            if key in valid_params:
                expected_type, default, _ = valid_params[key]
                
                # Try to convert the value to the expected type
                try:
                    if expected_type == bool:
                        # Handle string booleans
                        if isinstance(value, str):
                            value = value.lower() in ('true', '1', 'yes')
                        else:
                            value = bool(value)
                    elif expected_type == int:
                        value = int(value)
                    elif expected_type == float:
                        value = float(value)
                    else:
                        value = expected_type(value)
                        
                    filtered[key] = value
                    
                except (ValueError, TypeError) as e:
                    if log_fn:
                        log_fn(f"Warning: Invalid value for parameter '{key}': {value} (expected {expected_type.__name__})")
            else:
                if log_fn:
                    log_fn(f"Warning: Parameter '{key}' is not valid for tracker '{tracker_id}', ignoring")
                    
        return filtered
    
    def parse_json_params(self, json_str: str, log_fn=None) -> Dict[str, Any]:
        """
        Parse JSON parameter string.
        
        :param json_str: JSON string containing parameters
        :param log_fn: Optional logging function
        :return: Parsed dictionary or empty dict on error
        """
        if not json_str or not json_str.strip():
            return {}
            
        try:
            params = json.loads(json_str)
            if not isinstance(params, dict):
                if log_fn:
                    log_fn("Warning: JSON parameters must be an object/dictionary")
                return {}
            return params
        except json.JSONDecodeError as e:
            if log_fn:
                log_fn(f"Warning: Invalid JSON parameters: {e}")
            return {}
    
    def get_reid_weights_path(
        self, 
        reid_model: ReIDModel, 
        custom_path: Optional[str] = None,
        log_fn=None
    ) -> Optional[str]:
        """
        Get the path to ReID weights, downloading if necessary.
        
        :param reid_model: The ReID model type
        :param custom_path: Path for custom weights
        :param log_fn: Optional logging function
        :return: Path to weights file or None
        """
        if reid_model == ReIDModel.CUSTOM:
            if custom_path and os.path.exists(custom_path):
                return custom_path
            if log_fn:
                log_fn("Warning: Custom ReID weights path not found")
            return None
            
        if reid_model == ReIDModel.OSNET:
            return self._download_osnet_reid(log_fn)
            
        if reid_model == ReIDModel.BAMBI:
            return self._download_bambi_reid(log_fn)
            
        return None
    
    def _download_file(self, url: str, target_path: str, description: str, min_size: int = 100000, log_fn=None) -> bool:
        """
        Download a file from URL to target path.
        Uses the same approach as the detection model download.
        
        :param url: URL to download from
        :param target_path: Local path to save file
        :param description: Description for logging
        :param min_size: Minimum file size to consider valid
        :param log_fn: Optional logging function
        :return: True if successful, False otherwise
        """
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
                        log_fn(f"{description} download progress: {percent:.1f}% ({downloaded // 1024} KB)")
            
            # Try with custom SSL context first
            try:
                opener = urllib.request.build_opener(
                    urllib.request.HTTPSHandler(context=ssl_context)
                )
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(url, target_path, reporthook=report_progress)
            except Exception as e1:
                if log_fn:
                    log_fn(f"Download with custom SSL failed: {e1}")
                    log_fn("Trying standard download...")
                # Try standard download
                urllib.request.urlretrieve(url, target_path, reporthook=report_progress)
            
            # Verify download
            if os.path.exists(target_path) and os.path.getsize(target_path) > min_size:
                if log_fn:
                    log_fn(f"{description} downloaded successfully to: {target_path}")
                return True
            else:
                raise RuntimeError("Downloaded file is too small or missing")
                
        except Exception as e:
            # Clean up partial download
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except:
                    pass
            if log_fn:
                log_fn(f"Failed to download {description}: {e}")
            return False
    
    def _download_osnet_reid(self, log_fn=None) -> Optional[str]:
        """
        Download the OSNET ReID model from Google Drive (BoxMOT source).
        
        :param log_fn: Optional logging function
        :return: Path to downloaded model or None
        """
        model_path = os.path.join(self.models_folder, self.OSNET_REID_FILENAME)
        
        # Check if already downloaded
        if os.path.exists(model_path) and os.path.getsize(model_path) > 100000:
            if log_fn:
                log_fn(f"OSNET ReID model already exists at {model_path}")
            return model_path
            
        if log_fn:
            log_fn(f"Downloading OSNET ReID model from Google Drive...")
            log_fn("This may take a moment...")
        
        # Google Drive direct download URL (same source as BoxMOT uses)
        url = f"https://drive.google.com/uc?id={self.OSNET_GOOGLE_DRIVE_ID}&export=download"
        
        if self._download_file(url, model_path, "OSNET ReID", min_size=100000, log_fn=log_fn):
            return model_path
        
        # If download failed, fall back to letting BoxMOT handle it
        if log_fn:
            log_fn("Direct download failed. Falling back to BoxMOT default download...")
        return self.OSNET_REID_FILENAME
    
    def _download_bambi_reid(self, log_fn=None) -> Optional[str]:
        """
        Download the BAMBI Thermal-Omni ReID model from HuggingFace.
        
        :param log_fn: Optional logging function
        :return: Path to downloaded model or None
        """
        # Save locally with a clear name
        model_path = os.path.join(self.models_folder, self.BAMBI_REID_LOCAL_NAME)
        
        # Check if already downloaded
        if os.path.exists(model_path) and os.path.getsize(model_path) > 100000:
            if log_fn:
                log_fn(f"BAMBI Thermal-Omni ReID model already exists at {model_path}")
            return model_path
            
        if log_fn:
            log_fn(f"Downloading BAMBI Thermal-Omni ReID model from HuggingFace...")
            log_fn("This may take a moment...")
            
        # Direct download URL from HuggingFace
        url = f"https://huggingface.co/{self.BAMBI_REPO}/resolve/main/{self.BAMBI_REID_FILENAME}"
        
        if self._download_file(url, model_path, "BAMBI Thermal-Omni ReID", min_size=100000, log_fn=log_fn):
            return model_path
            
        # Download failed
        if log_fn:
            log_fn(f"BAMBI Thermal-Omni ReID model not available at: {url}")
            log_fn(f"Use 'custom' ReID option with a local file path instead.")
                
        return None
    
    def create_tracker(
        self,
        tracker_id: str,
        reid_model: ReIDModel = ReIDModel.OSNET,
        custom_reid_path: Optional[str] = None,
        json_params: str = "",
        device: str = "cpu",
        log_fn=None
    ) -> Tuple[Any, TrackerBackend]:
        """
        Create a tracker instance with the specified configuration.
        
        :param tracker_id: The tracker identifier
        :param reid_model: ReID model to use
        :param custom_reid_path: Path to custom ReID weights
        :param json_params: JSON string with additional parameters
        :param device: Device to run on ('cpu', 'cuda', etc.)
        :param log_fn: Optional logging function
        :return: Tuple of (tracker instance, backend type)
        """
        # Ensure stderr is available for libraries that use loguru
        self._ensure_stderr()
        
        trackers = self.get_available_trackers()
        
        if tracker_id not in trackers:
            raise ValueError(f"Unknown tracker: {tracker_id}. Available: {list(trackers.keys())}")
            
        info = trackers[tracker_id]
        
        # Parse and validate parameters
        user_params = self.parse_json_params(json_params, log_fn)
        valid_params = self.validate_and_filter_params(tracker_id, user_params, log_fn)
        
        if log_fn:
            log_fn(f"Creating tracker: {info.name}")
            if valid_params:
                log_fn(f"Using parameters: {valid_params}")
                
        # Handle built-in tracker specially
        if info.backend == TrackerBackend.BUILTIN:
            return self._create_builtin_tracker(valid_params), TrackerBackend.BUILTIN
            
        # For BoxMOT and georef trackers, get ReID weights if needed
        reid_weights = None
        if info.requires_reid:
            reid_weights = self.get_reid_weights_path(reid_model, custom_reid_path, log_fn)
            
            # If the requested model failed, fall back to OSNET
            if not reid_weights and reid_model != ReIDModel.OSNET:
                if log_fn:
                    log_fn(f"Failed to get {reid_model.value} weights, falling back to OSNET...")
                reid_weights = self.get_reid_weights_path(ReIDModel.OSNET, None, log_fn)
            
            # If we still don't have weights and they're required, raise an error
            if not reid_weights:
                raise RuntimeError(
                    f"ReID weights are required for {info.name} but could not be obtained. "
                    f"Please ensure you have internet access or provide custom weights."
                )
                
            if log_fn:
                log_fn(f"Using ReID weights: {reid_weights}")
                
        # Create the tracker
        try:
            import torch
            device_obj = torch.device(device)
        except ImportError:
            device_obj = device
            
        # Build constructor arguments
        constructor_args = {}
        
        if info.requires_reid and reid_weights:
            constructor_args["reid_weights"] = reid_weights
            
        if hasattr(info.tracker_class, '__init__'):
            sig = inspect.signature(info.tracker_class.__init__)
            if 'device' in sig.parameters:
                constructor_args["device"] = device_obj
            if 'half' in sig.parameters:
                constructor_args["half"] = False
                
        # Add user parameters
        constructor_args.update(valid_params)
        
        if log_fn:
            log_fn(f"Initializing {info.name} with args: {list(constructor_args.keys())}")
            
        tracker = info.tracker_class(**constructor_args)
        
        return tracker, info.backend
    
    def _create_builtin_tracker(self, params: Dict[str, Any]) -> "BuiltinTracker":
        """Create a built-in tracker wrapper."""
        return BuiltinTracker(**params)


class BuiltinTracker:
    """
    Wrapper class for the built-in Hungarian IoU tracker.
    
    This provides a consistent interface with BoxMOT trackers.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = -1,
        max_center_distance: float = 0.2,
        class_aware: bool = True,
        interpolate: bool = True
    ):
        """
        Initialize the built-in tracker.
        
        :param iou_threshold: IoU threshold for matching
        :param max_age: Max frames without detection (-1 = unlimited)
        :param max_center_distance: Max center distance for matching (meters)
        :param class_aware: Whether to enforce class matching
        :param interpolate: Whether to interpolate missing frames
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.max_center_distance = max_center_distance
        self.class_aware = class_aware
        self.interpolate = interpolate
        
        # Will be populated during tracking
        self.tracks = []
        self.results = []
        
    def get_config(self) -> Dict[str, Any]:
        """Get configuration as dictionary (for use by processing module)."""
        return {
            "iou_threshold": self.iou_threshold,
            "max_age": self.max_age,
            "max_center_distance": self.max_center_distance,
            "class_aware": self.class_aware,
            "interpolate": self.interpolate
        }


def get_tracker_manager(models_folder: Optional[str] = None) -> TrackerManager:
    """
    Get a TrackerManager instance.
    
    :param models_folder: Optional custom models folder
    :return: TrackerManager instance
    """
    return TrackerManager(models_folder)


def diagnose_trackers():
    """
    Diagnostic function to debug tracker discovery issues.
    Call from QGIS Python console:
        from bambi_wildlife_detection.tracker_manager import diagnose_trackers
        diagnose_trackers()
    """
    import sys
    print("=" * 60)
    print("BAMBI Tracker Diagnostics")
    print("=" * 60)
    
    # Check sys.stderr status
    print(f"\nsys.stderr: {sys.stderr}")
    if sys.stderr is None:
        print("  WARNING: sys.stderr is None - this can cause issues with loguru")
    
    # Check package availability
    print("\n--- Package Availability ---")
    import importlib.util
    for pkg in ["boxmot", "georef_tracker", "loguru", "torch"]:
        spec = importlib.util.find_spec(pkg)
        status = "INSTALLED" if spec else "NOT FOUND"
        print(f"  {pkg}: {status}")
    
    # Try importing boxmot
    print("\n--- BoxMOT Import Test ---")
    try:
        import io
        if sys.stderr is None:
            sys.stderr = io.StringIO()
            print("  (Fixed sys.stderr)")
        
        import boxmot
        print(f"  boxmot imported successfully")
        print(f"  boxmot location: {boxmot.__file__}")
        print(f"  boxmot version: {getattr(boxmot, '__version__', 'unknown')}")
        
        # List available attributes
        attrs = [a for a in dir(boxmot) if not a.startswith('_')]
        print(f"  Available attributes: {attrs}")
        
        # Check for specific tracker classes (both old and new naming)
        tracker_classes = [
            ("DeepOcSort", "DeepOCSORT"),  # (new name, old name)
            ("BotSort", "BoTSORT"),
            ("StrongSort", "StrongSORT"),
            ("OcSort", "OCSORT"),
            ("ByteTrack", "BYTETracker"),
            ("HybridSort", "HybridSORT"),
            ("BoostTrack", None),
        ]
        print("\n--- Tracker Class Availability ---")
        for names in tracker_classes:
            new_name = names[0]
            old_name = names[1] if len(names) > 1 else None
            
            if hasattr(boxmot, new_name):
                print(f"  {new_name}: FOUND")
            elif old_name and hasattr(boxmot, old_name):
                print(f"  {old_name}: FOUND (legacy name)")
            else:
                print(f"  {new_name}: NOT FOUND")
                    
    except Exception as e:
        print(f"  FAILED to import boxmot: {e}")
        import traceback
        traceback.print_exc()
    
    # Test TrackerManager
    print("\n--- TrackerManager Test ---")
    try:
        manager = TrackerManager()
        
        print(f"  BoxMOT available: {manager.is_boxmot_available()}")
        print(f"  GeoRef available: {manager.is_georef_available()}")
        
        trackers = manager.get_available_trackers()
        print(f"\n  Available trackers ({len(trackers)}):")
        for tid, info in trackers.items():
            print(f"    - {tid}: {info.name}")
            
        errors = manager.get_import_errors()
        if errors:
            print(f"\n  Import errors:")
            for key, error in errors.items():
                print(f"    - {key}: {error}")
        else:
            print(f"\n  No import errors")
            
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

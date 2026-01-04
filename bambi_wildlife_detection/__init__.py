# -*- coding: utf-8 -*-
"""
BAMBI Wildlife Detection QGIS Plugin
=====================================

A comprehensive plugin for detecting and tracking wildlife in drone thermal videos.

This plugin provides:
- Frame extraction from drone videos
- Animal detection using YOLO models
- Geo-referencing of detections
- Multi-object tracking
- QGIS layer creation for visualization
"""


def classFactory(iface):
    """Load BambiWildlifeDetection class from file bambi_wildlife_detection.
    
    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from .bambi_wildlife_detection import BambiWildlifeDetection
    return BambiWildlifeDetection(iface)

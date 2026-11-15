# -*- coding: utf-8 -*-
"""The Pre-Processing and Processing tabs: the steps, in the order they run.

Every step is a button, the camera it works on and a line that says how far
it is. This part builds both tabs - together, because the steps are made in
the order of the pipeline and that order alternates between the two - and
holds what the buttons do: check that the step can run, ask what has to be
asked, and hand it to the step runner.

What a finished step puts on the map is in ``result_layers`` and
``flight_route_layers``.
"""

import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QFileDialog,
    QLabel, QComboBox, QCheckBox, QMessageBox, QFrame,
)
from qgis.core import QgsSettings

from ..bambi_processing import BambiProcessor
from ..core import store as _store_kinds
from ..gui_utils import PLUGIN_SCOPE
from .common import _CLASSIFICATION_INPUTS, _CLASSIFICATION_TASKS


def _camera_name(modality: str) -> str:
    """The camera a modality suffix means, as a user reads it."""
    return "Thermal" if modality == "t" else "RGB"


class StepTabsMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_step_tabs(self, main_tabs):
        """The Pre-Processing and the Processing tab: the step buttons with
        their status marks and "Add ... to QGIS" buttons, and the export.

        Both in one place because their steps are made in the order of the
        pipeline, which alternates between the two tabs.
        """
        # =====================================================================
        # MAIN TAB 3: PROCESSING
        # =====================================================================
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)
        main_tabs.addTab(processing_tab, "Pre-Processing")

        # Info button row. Each tab explains its own steps, so the popup does
        # not describe steps that live on the other one.
        self._preprocessing_info_btn = self._make_info_button(
            self._show_preprocessing_info)
        processing_info_row = QHBoxLayout()
        processing_info_row.setContentsMargins(0, 0, 0, 0)
        processing_info_row.addStretch()
        processing_info_row.addWidget(self._preprocessing_info_btn)
        processing_layout.addLayout(processing_info_row)

        # Step buttons. The split mirrors the dependency graph of the steps
        # (core.stages): everything here derives from the poses and
        # the DEM and is independent of any animal, while the Processing tab
        # holds everything that depends on detections. Presenting them as one
        # numbered list implied a sequence that does not exist.
        steps_group = QGroupBox("Pre-Processing Steps")
        steps_btn_layout = QVBoxLayout(steps_group)

        proc_steps_group = QGroupBox("Detection and Tracking")
        proc_steps_layout = QVBoxLayout(proc_steps_group)

        # One camera selector per stage instead of one per step (9.0.0).
        # The per-step combos still exist - the steps, the configuration
        # and the status checks read them - but are not shown; the stage
        # selector sets them all.
        self.preprocessing_camera_combo = QComboBox()
        self.preprocessing_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.preprocessing_camera_combo.setToolTip(
            "The camera every pre-processing step works on: frame "
            "extraction, flight route, field of view, ALFS, GeoTIFFs and "
            "the orthomosaic. The step statuses show this camera's outputs.")
        pre_camera_row = QHBoxLayout()
        pre_camera_row.addWidget(QLabel("Camera:"))
        pre_camera_row.addWidget(self.preprocessing_camera_combo, 1)
        steps_btn_layout.addLayout(pre_camera_row)

        self.processing_camera_combo = QComboBox()
        self.processing_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.processing_camera_combo.setToolTip(
            "The camera detection, geo-referencing, the perpendicular "
            "distances and tracking work on. Matching (A3) always uses "
            "both cameras.")
        proc_camera_row = QHBoxLayout()
        proc_camera_row.addWidget(QLabel("Camera:"))
        proc_camera_row.addWidget(self.processing_camera_combo, 1)
        proc_steps_layout.addLayout(proc_camera_row)

        # ----- P1: Extract Frames -----
        step1_row = QHBoxLayout()
        self.extract_btn = QPushButton("P1. Extract Frames")
        self.extract_btn.clicked.connect(self.run_extract_frames)
        self.extract_btn.setToolTip(
            "Extract frames from the selected camera's videos:\n"
            "Thermal (_T_) → frames_t/, RGB (_W_/_V_) → frames_w/"
        )
        self.extract_camera_combo = QComboBox()
        self.extract_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.extract_camera_combo.setFixedWidth(100)
        self.extract_camera_combo.setToolTip("Select camera source for frame extraction")
        self.extract_status = QLabel("⚪ Not started")
        step1_row.addWidget(self.extract_btn)
        step1_row.addWidget(self.extract_status)
        steps_btn_layout.addLayout(step1_row)

        correction_info_label = QLabel(
            "If not already done: Use the correction tooling to determine "
            "positional and orientation errors after frame extraction."
        )
        correction_info_label.setWordWrap(True)
        correction_info_label.setStyleSheet("color: gray; font-size: 10px;")
        steps_btn_layout.addWidget(correction_info_label)

        # Separator after extraction
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        steps_btn_layout.addWidget(separator1)

        # ----- P2: Generate Flight Route -----
        step2_row = QHBoxLayout()
        self.flight_route_btn = QPushButton("P2. Generate Flight Route")
        self.flight_route_btn.clicked.connect(self.run_flight_route)
        self.flight_route_btn.setToolTip("Generate flight route polyline from camera positions")
        self.flight_route_camera_combo = QComboBox()
        self.flight_route_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.flight_route_camera_combo.setFixedWidth(100)
        self.flight_route_camera_combo.setToolTip("Select camera source for poses")
        self.flight_route_status = QLabel("⚪ Not started")
        step2_row.addWidget(self.flight_route_btn)
        step2_row.addWidget(self.flight_route_status)
        steps_btn_layout.addLayout(step2_row)

        # -> Add Flight Route to QGIS
        add_flight_route_row = QHBoxLayout()
        self.add_flight_route_btn = QPushButton("   → Add Flight Route to QGIS")
        self.add_flight_route_btn.clicked.connect(self.add_flight_route_to_qgis)
        self.add_flight_route_status = QLabel("⚪")
        add_flight_route_row.addWidget(self.add_flight_route_btn)
        add_flight_route_row.addWidget(self.add_flight_route_status)
        steps_btn_layout.addLayout(add_flight_route_row)

        # ----- A1: Detect Animals -----
        step3_row = QHBoxLayout()
        self.detect_btn = QPushButton("A1. Detect Animals")
        self.detect_btn.clicked.connect(self.run_detection)
        self.detection_camera_combo = QComboBox()
        self.detection_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.detection_camera_combo.setFixedWidth(100)
        self.detection_camera_combo.setToolTip("Select camera source for frames")
        self.detect_status = QLabel("⚪ Not started")
        step3_row.addWidget(self.detect_btn)
        step3_row.addWidget(self.detect_status)
        proc_steps_layout.addLayout(step3_row)

        # -> Geo-Reference Detections (sub-step of detection)
        step4_row = QHBoxLayout()
        self.georef_btn = QPushButton("   → Geo-Reference Detections")
        self.georef_btn.clicked.connect(self.run_georeference)
        self.georef_status = QLabel("⚪ Not started")
        step4_row.addWidget(self.georef_btn)
        step4_row.addWidget(self.georef_status)
        proc_steps_layout.addLayout(step4_row)

        # -> Add Detections to QGIS
        add_detections_row = QHBoxLayout()
        self.add_frame_detections_btn = QPushButton("   → Add Detections to QGIS")
        self.add_frame_detections_btn.clicked.connect(self.add_frame_detections_to_qgis)
        self.add_frame_detections_btn.setToolTip("Add geo-referenced bounding boxes as layers (one per frame)")
        self.frame_detections_status = QLabel("⚪")
        add_detections_row.addWidget(self.add_frame_detections_btn)
        add_detections_row.addWidget(self.frame_detections_status)
        proc_steps_layout.addLayout(add_detections_row)

        # -> Calculate Perpendicular
        perp_calc_row = QHBoxLayout()
        self.perpendicular_btn = QPushButton("   → Calculate Perpendicular")
        self.perpendicular_btn.clicked.connect(self.run_perpendicular)
        self.perpendicular_btn.setToolTip(
            "For each detection, find the nearest point on the flight route and\n"
            "calculate the perpendicular distance. Requires both flight route and\n"
            "geo-referenced detections."
        )
        self.perpendicular_status = QLabel("⚪")
        perp_calc_row.addWidget(self.perpendicular_btn)
        perp_calc_row.addWidget(self.perpendicular_status)
        proc_steps_layout.addLayout(perp_calc_row)

        # -> Add Perpendicular Lines to QGIS
        perp_add_row = QHBoxLayout()
        self.add_perpendicular_btn = QPushButton("   → Add Perpendicular Lines to QGIS")
        self.add_perpendicular_btn.clicked.connect(self.add_perpendicular_to_qgis)
        self.add_perpendicular_btn.setToolTip(
            "Add line layers connecting each detection to its nearest point on the flight route."
        )
        self.add_perpendicular_status = QLabel("⚪")
        perp_add_row.addWidget(self.add_perpendicular_btn)
        perp_add_row.addWidget(self.add_perpendicular_status)
        proc_steps_layout.addLayout(perp_add_row)

        # ----- A2: Track Animals Or Import -----
        step5_row = QHBoxLayout()
        self.track_btn = QPushButton("A2. Track Animals Or Import")
        self.track_btn.setToolTip(
            "Track geo-referenced detections.\n"
            "If a TRex NPZ folder is set (Config → Tracking tab), the pre-computed\n"
            "TRex tracklets are imported and geo-referenced instead."
        )
        self.track_btn.clicked.connect(self.run_tracking)
        self.tracking_camera_combo = QComboBox()
        self.tracking_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.tracking_camera_combo.setFixedWidth(100)
        self.tracking_camera_combo.setToolTip("Select camera source for geo-referenced detections")
        self.track_status = QLabel("⚪ Not started")
        step5_row.addWidget(self.track_btn)
        step5_row.addWidget(self.track_status)
        proc_steps_layout.addLayout(step5_row)

        # -> Add Tracks to QGIS
        add_tracks_row = QHBoxLayout()
        self.add_layers_btn = QPushButton("   → Add Tracks to QGIS")
        self.add_layers_btn.clicked.connect(self.add_tracks_to_qgis)
        self.layers_status = QLabel("⚪")
        add_tracks_row.addWidget(self.add_layers_btn)
        add_tracks_row.addWidget(self.layers_status)
        proc_steps_layout.addLayout(add_tracks_row)

        # -> Calculate Track Perpendicular
        track_perp_calc_row = QHBoxLayout()
        self.track_perpendicular_btn = QPushButton("   → Calculate Track Perpendicular")
        self.track_perpendicular_btn.clicked.connect(self.run_track_perpendicular)
        self.track_perpendicular_btn.setToolTip(
            "For each track, find the nearest point on the flight route to the\n"
            "last bounding box and calculate the perpendicular distance.\n"
            "Requires both flight route and tracked animals."
        )
        self.track_perpendicular_status = QLabel("⚪")
        track_perp_calc_row.addWidget(self.track_perpendicular_btn)
        track_perp_calc_row.addWidget(self.track_perpendicular_status)
        proc_steps_layout.addLayout(track_perp_calc_row)

        # -> Add Track Perpendicular Lines to QGIS
        track_perp_add_row = QHBoxLayout()
        self.add_track_perpendicular_btn = QPushButton("   → Add Track Perpendicular Lines to QGIS")
        self.add_track_perpendicular_btn.clicked.connect(self.add_track_perpendicular_to_qgis)
        self.add_track_perpendicular_btn.setToolTip(
            "Add line layers connecting the last detection of each track\n"
            "to its nearest point on the flight route."
        )
        self.add_track_perpendicular_status = QLabel("⚪")
        track_perp_add_row.addWidget(self.add_track_perpendicular_btn)
        track_perp_add_row.addWidget(self.add_track_perpendicular_status)
        proc_steps_layout.addLayout(track_perp_add_row)

        # ----- A3: Match RGB <-> Thermal tracks -----
        # A statement about the tracks - which two of them are one animal -
        # so it belongs with tracking. Classification is one consumer of the
        # answer, not what the step is for.
        match_row = QHBoxLayout()
        self.track_matching_btn = QPushButton("A3. Match RGB ↔ Thermal Tracks")
        self.track_matching_btn.clicked.connect(self.run_track_matching)
        self.track_matching_btn.setToolTip(
            "Work out which thermal track and which RGB track are the same "
            "animal.\n\n"
            "Needs both cameras tracked. An animal confirmed by both is a real "
            "animal; a track only one camera saw is either an animal the other "
            "sensor cannot see, or noise."
        )
        self.track_matching_status = QLabel("⚪ Not started")
        match_row.addWidget(self.track_matching_btn)
        match_row.addWidget(self.track_matching_status)
        proc_steps_layout.addLayout(match_row)

        # -> Add matched pairs to QGIS
        add_matches_row = QHBoxLayout()
        self.add_matches_btn = QPushButton("   → Add Matched Pairs to QGIS")
        self.add_matches_btn.clicked.connect(self.add_matches_to_qgis)
        self.add_matches_btn.setToolTip(
            "Draw a line between each matched pair's geo-referenced positions, "
            "attributed with the shared frame count and median distance - the "
            "quickest way to see whether the gate is set sensibly."
        )
        self.add_matches_status = QLabel("⚪")
        add_matches_row.addWidget(self.add_matches_btn)
        add_matches_row.addWidget(self.add_matches_status)
        proc_steps_layout.addLayout(add_matches_row)

        # ----- P3: Calculate Field of View -----
        step6_row = QHBoxLayout()
        self.calculate_fov_btn = QPushButton("P3. Calculate Field of View")
        self.calculate_fov_btn.clicked.connect(self.run_calculate_fov)
        self.calculate_fov_btn.setToolTip("Calculate and save camera FoV footprints for each frame")
        self.fov_camera_combo = QComboBox()
        self.fov_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.fov_camera_combo.setFixedWidth(100)
        self.fov_camera_combo.setToolTip("Select camera source for poses and calibration")
        self.calculate_fov_status = QLabel("⚪ Not started")
        step6_row.addWidget(self.calculate_fov_btn)
        step6_row.addWidget(self.calculate_fov_status)
        steps_btn_layout.addLayout(step6_row)

        # -> Add FoV Layers to QGIS
        add_fov_row = QHBoxLayout()
        self.add_fov_btn = QPushButton("   → Add FoV Layers to QGIS")
        self.add_fov_btn.clicked.connect(self.add_fov_to_qgis)
        self.add_fov_btn.setToolTip("Add FoV polygons as QGIS layers (one per frame)")
        self.add_fov_status = QLabel("⚪")
        add_fov_row.addWidget(self.add_fov_btn)
        add_fov_row.addWidget(self.add_fov_status)
        steps_btn_layout.addLayout(add_fov_row)

        # -> Add Merged FoV to QGIS
        add_merged_fov_row = QHBoxLayout()
        self.add_merged_fov_btn = QPushButton("   → Add Merged FoV to QGIS")
        self.add_merged_fov_btn.clicked.connect(self.add_merged_fov_to_qgis)
        self.add_merged_fov_btn.setToolTip("Add union of all FoV polygons as single layer (for area calculation)")
        self.add_merged_fov_status = QLabel("⚪")
        add_merged_fov_row.addWidget(self.add_merged_fov_btn)
        add_merged_fov_row.addWidget(self.add_merged_fov_status)
        steps_btn_layout.addLayout(add_merged_fov_row)

        # ----- P4: Generate ALFS -----
        step7_row = QHBoxLayout()
        self.alfs_btn = QPushButton("P4. Generate ALFS")
        self.alfs_btn.clicked.connect(self.run_alfs)
        self.alfs_camera_combo = QComboBox()
        self.alfs_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.alfs_camera_combo.setFixedWidth(100)
        self.alfs_camera_combo.setToolTip("Select camera source for frames and poses")
        self.alfs_status = QLabel("⚪ Not started")
        step7_row.addWidget(self.alfs_btn)
        step7_row.addWidget(self.alfs_status)
        steps_btn_layout.addLayout(step7_row)

        # -> Add ALFS to QGIS
        add_alfs_row = QHBoxLayout()
        self.add_alfs_btn = QPushButton("   → Add ALFS to QGIS")
        self.add_alfs_btn.clicked.connect(self.add_alfs_to_qgis)
        self.add_alfs_status = QLabel("⚪")
        add_alfs_row.addWidget(self.add_alfs_btn)
        add_alfs_row.addWidget(self.add_alfs_status)
        steps_btn_layout.addLayout(add_alfs_row)

        # ----- P5: Export Frames as GeoTIFF -----
        step8_row = QHBoxLayout()
        self.export_geotiffs_btn = QPushButton("P5. Export Frames as GeoTIFF")
        self.export_geotiffs_btn.clicked.connect(self.run_export_geotiffs)
        self.geotiff_camera_combo = QComboBox()
        self.geotiff_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.geotiff_camera_combo.setFixedWidth(100)
        self.geotiff_camera_combo.setToolTip("Select camera source for frames and poses")
        self.export_geotiffs_status = QLabel("⚪ Not started")
        step8_row.addWidget(self.export_geotiffs_btn)
        step8_row.addWidget(self.export_geotiffs_status)
        steps_btn_layout.addLayout(step8_row)

        # -> Add Frame GeoTIFFs to QGIS
        add_geotiffs_row = QHBoxLayout()
        self.add_geotiffs_btn = QPushButton("   → Add Frame GeoTIFFs to QGIS")
        self.add_geotiffs_btn.clicked.connect(self.add_geotiffs_to_qgis)
        self.add_geotiffs_status = QLabel("⚪")
        add_geotiffs_row.addWidget(self.add_geotiffs_btn)
        add_geotiffs_row.addWidget(self.add_geotiffs_status)
        steps_btn_layout.addLayout(add_geotiffs_row)

        # ----- P6: Generate Orthomosaic (merge exported GeoTIFFs) -----
        ortho_step_row = QHBoxLayout()
        self.orthomosaic_btn = QPushButton("P6. Generate Orthomosaic")
        self.orthomosaic_btn.setToolTip(
            "Merge the exported frame GeoTIFFs (P5) into a single orthomosaic"
        )
        self.orthomosaic_btn.clicked.connect(self.run_orthomosaic)
        self.ortho_camera_combo = QComboBox()
        self.ortho_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.ortho_camera_combo.setFixedWidth(100)
        self.ortho_camera_combo.setToolTip("Select camera source (GeoTIFFs) for the orthomosaic")
        self.orthomosaic_status = QLabel("⚪ Not started")
        ortho_step_row.addWidget(self.orthomosaic_btn)
        ortho_step_row.addWidget(self.orthomosaic_status)
        steps_btn_layout.addLayout(ortho_step_row)

        # -> Add Orthomosaic to QGIS
        add_ortho_row = QHBoxLayout()
        self.add_orthomosaic_btn = QPushButton("   → Add Orthomosaic to QGIS")
        self.add_orthomosaic_btn.clicked.connect(self.add_orthomosaic_to_qgis)
        self.add_orthomosaic_status = QLabel("⚪")
        add_ortho_row.addWidget(self.add_orthomosaic_btn)
        add_ortho_row.addWidget(self.add_orthomosaic_status)
        steps_btn_layout.addLayout(add_ortho_row)

        # =====================================================================
        # Classification
        # =====================================================================
        # Its own section: everything here works on animals the steps above
        # already found, and none of it is needed to produce detections or
        # tracks.
        classify_steps_group = QGroupBox("Classification")
        classify_steps_layout = QVBoxLayout(classify_steps_group)

        classify_note = QLabel(
            "ℹ Tracks you annotated in the labelling tool are never changed "
            "by these steps. A hand annotation outranks a model, so the "
            "classifiers fill in what is missing and leave your own work "
            "alone."
        )
        classify_note.setWordWrap(True)
        classify_note.setStyleSheet(
            "color: #31708f; background: #d9edf7; border: 1px solid #bce8f1; "
            "border-radius: 4px; padding: 5px; font-size: 10px;")
        classify_steps_layout.addWidget(classify_note)

        self.classification_camera_combo = QComboBox()
        for label, value in _CLASSIFICATION_INPUTS:
            self.classification_camera_combo.addItem(label, value)
        self.classification_camera_combo.setCurrentIndex(2)
        self.classification_camera_combo.setToolTip(
            "Which camera's crops the embeddings (C1) and the classifiers "
            "(C2-C5) read.\n\n'Matched' reads both cameras fused, which is "
            "where the two sensors complement each other most - it needs "
            "the tracks matched (A3).")
        classify_camera_row = QHBoxLayout()
        classify_camera_row.addWidget(QLabel("Input:"))
        classify_camera_row.addWidget(self.classification_camera_combo, 1)
        classify_steps_layout.addLayout(classify_camera_row)

        # ----- C1: Compute DINOv3 embeddings -----
        embeddings_row = QHBoxLayout()
        self.embeddings_btn = QPushButton("C1. Compute DINOv3 Embeddings")
        self.embeddings_btn.clicked.connect(self.run_embeddings)
        self.embeddings_btn.setToolTip(
            "Embed every tracked animal's crop with DINOv3, once, so the "
            "classifiers can reuse the vectors.\n\n"
            "This is the expensive step. The vectors are written beside the "
            "frames, and a re-run only embeds what is missing - an "
            "interrupted run resumes rather than starting again."
        )
        self.embeddings_camera_combo = QComboBox()
        for label, value in _CLASSIFICATION_INPUTS:
            self.embeddings_camera_combo.addItem(label, value)
        self.embeddings_camera_combo.setCurrentIndex(2)
        self.embeddings_camera_combo.setFixedWidth(100)
        self.embeddings_camera_combo.setToolTip(
            "Which camera's crops to embed. The same three choices as the "
            "classifiers below, so whatever they are set to can be embedded "
            "in one run - 'Matched' does both cameras."
        )
        self.embeddings_status = QLabel("⚪ Not started")
        embeddings_row.addWidget(self.embeddings_btn)
        embeddings_row.addWidget(self.embeddings_status)
        classify_steps_layout.addLayout(embeddings_row)

        # ----- C2-C5: one step per classifier -----
        # Their Input combos are not shown: the stage's one "Input" selector
        # above sets them all (9.0.0).
        inputs = self._ensure_classification_input_combos()
        # Separate steps rather than one: they take very different times, and
        # each reads what the earlier ones stored, so they can be run on
        # different days or repeated one at a time.
        classify_row = QHBoxLayout()
        self.classify_occlusion_btn = QPushButton(
            "C2. Occlusion Classification")
        self.classify_occlusion_btn.clicked.connect(
            lambda: self.run_classification("occlusion"))
        self.classify_occlusion_btn.setToolTip(
            "Label each frame of each animal clear or occluded.\n\n"
            "A quality filter rather than a verdict about the animal: it "
            "decides which frames the species and sex votes may use. Optional "
            "- those run either way."
        )
        self.classify_occlusion_status = QLabel("⚪ Not started")
        classify_row.addWidget(self.classify_occlusion_btn)
        classify_row.addWidget(self.classify_occlusion_status)
        classify_steps_layout.addLayout(classify_row)

        species_row = QHBoxLayout()
        self.classify_species_btn = QPushButton(
            "C3. Species Classification")
        self.classify_species_btn.clicked.connect(
            lambda: self.run_classification("species"))
        self.classify_species_btn.setToolTip(
            "Decide what each animal is, by a vote across its frames.\n\n"
            "Sex and age read this answer to choose their own model, so run "
            "it before them."
        )
        self.classify_species_status = QLabel("⚪ Not started")
        species_row.addWidget(self.classify_species_btn)
        species_row.addWidget(self.classify_species_status)
        classify_steps_layout.addLayout(species_row)

        sex_row = QHBoxLayout()
        self.classify_sex_btn = QPushButton("C4. Sex Classification")
        self.classify_sex_btn.clicked.connect(
            lambda: self.run_classification("sex"))
        self.classify_sex_btn.setToolTip(
            "Decide each animal's sex, by a vote across exactly the frames "
            "species used.\n\n"
            "The model is chosen by species - the cue is species-specific - so "
            "species has to have run first."
        )
        self.classify_sex_status = QLabel("⚪ Not started")
        sex_row.addWidget(self.classify_sex_btn)
        sex_row.addWidget(self.classify_sex_status)
        classify_steps_layout.addLayout(sex_row)

        # ----- C5: Age -----
        life_stage_row = QHBoxLayout()
        self.life_stage_btn = QPushButton("C5. Age Classification")
        self.life_stage_btn.clicked.connect(self.run_life_stage)
        self.life_stage_btn.setToolTip(
            "Decide whether each animal is a juvenile or an adult.\n\n"
            "Uses a classifier where one is configured for the species, and "
            "otherwise body size within this flight - a juvenile cannot be "
            "told from an adult female by appearance at survey resolution, so "
            "size is what separates them. The size estimate needs no models.\n\n"
            "Each camera is measured as its own cohort. Where tracks have been "
            "matched across cameras, a juvenile found on one camera is carried "
            "onto its partner track on the other - the thermal cohort often "
            "cannot confirm a calf that stands out clearly among the RGB deer."
        )
        self.life_stage_status = QLabel("⚪ Not started")
        life_stage_row.addWidget(self.life_stage_btn)
        life_stage_row.addWidget(self.life_stage_status)
        classify_steps_layout.addLayout(life_stage_row)

        # ----- Sync labels across the match -----
        # Not a numbered step: it runs on demand, on results the steps above
        # already produced. It is also the way out of a run where the two
        # cameras were classified by different stages - sync the species over
        # and the demographic heads on this side have a model to pick.
        sync_row = QHBoxLayout()
        self.sync_labels_btn = QPushButton("   → Sync Labels Across the Match")
        self.sync_labels_btn.clicked.connect(self.sync_labels)
        self.sync_labels_btn.setToolTip(
            "Copy species, sex and age from one camera's animals onto their "
            "matched partners on the other, and per-frame occlusion onto the "
            "paired detections.\n\n"
            "Only matched pairs are touched (run A3 first). An animal that "
            "already carries an answer keeps it, and hand annotations from the "
            "labelling tool are never overwritten."
        )
        self.sync_labels_combo = QComboBox()
        self.sync_labels_combo.addItems(["Thermal → RGB", "RGB → Thermal"])
        self.sync_labels_combo.setFixedWidth(120)
        self.sync_labels_combo.setToolTip(
            "Which way the labels travel: the first camera is the one that "
            "already has them.")
        self.sync_labels_status = QLabel("⚪")
        sync_row.addWidget(self.sync_labels_btn)
        sync_row.addWidget(self.sync_labels_combo)
        sync_row.addWidget(self.sync_labels_status)
        classify_steps_layout.addLayout(sync_row)

        # =====================================================================
        # Segmentation
        # =====================================================================
        # Stands apart from the detect-track-classify chain: it reads the
        # frames directly, nothing downstream consumes it, and it calls out to
        # a third-party service. Its own section stops it reading as the next
        # step in a sequence it is not part of.
        extra_steps_group = QGroupBox("Segmentation")
        extra_steps_layout = QVBoxLayout(extra_steps_group)

        extra_info = QLabel(
            "Independent of the steps above - it neither needs them, nor "
            "feeds into them."
        )
        extra_info.setWordWrap(True)
        extra_info.setStyleSheet("color: gray; font-size: 10px;")
        extra_steps_layout.addWidget(extra_info)

        # ----- Segmentation tool -----
        # Prompting is interactive - a click on the frame, a mask to look
        # at, another click - so it lives in a window of its own rather than
        # in a step button. This just opens it.
        seg_row = QHBoxLayout()
        self.sam3_segment_btn = QPushButton("S1. Open Segmentation Tool…")
        self.sam3_segment_btn.clicked.connect(self.open_segmentation_tool)
        self.sam3_segment_btn.setToolTip(
            "Segment the extracted frames with SAM3 / SAM 3.1 by text or "
            "point prompts, on single frames or tracked across a sequence; "
            "geo-reference the masks, add them as layers, export GeoJSON")
        self.sam3_segment_status = QLabel("⚪ Not started")
        seg_row.addWidget(self.sam3_segment_btn)
        seg_row.addWidget(self.sam3_segment_status)
        extra_steps_layout.addLayout(seg_row)

        # Re-evaluate step statuses when a camera selection changes so the
        # indicators always reflect the selected modality's outputs.
        for camera_combo in (self.extract_camera_combo, self.flight_route_camera_combo,
                             self.detection_camera_combo, self.tracking_camera_combo,
                             self.fov_camera_combo, self.alfs_camera_combo,
                             self.geotiff_camera_combo, self.ortho_camera_combo):
            camera_combo.currentIndexChanged.connect(self._on_step_camera_changed)

        # (stage selector, the step combo it is read back from, all the
        # step combos it sets)
        self._stage_cameras = [
            (self.preprocessing_camera_combo, self.extract_camera_combo,
             [self.extract_camera_combo, self.flight_route_camera_combo,
              self.fov_camera_combo, self.alfs_camera_combo,
              self.geotiff_camera_combo, self.ortho_camera_combo]),
            (self.processing_camera_combo, self.detection_camera_combo,
             [self.detection_camera_combo, self.tracking_camera_combo]),
            (self.classification_camera_combo, inputs["species"],
             [self.embeddings_camera_combo] + [inputs[task] for task in _CLASSIFICATION_TASKS]),
        ]
        for stage_combo, _lead, step_combos in self._stage_cameras:
            stage_combo.currentIndexChanged.connect(
                lambda index, combos=step_combos:
                self._push_stage_camera(index, combos))
            # a step combo set from code (configuration load, a script)
            # carries its stage with it, so the steps cannot disagree
            for combo in step_combos:
                combo.currentIndexChanged.connect(stage_combo.setCurrentIndex)

        processing_layout.addWidget(steps_group)
        # Satellite, land-cover, terrain and weather context of the flight -
        # camera-independent, each source its own sub-step.
        processing_layout.addWidget(self._build_environment_steps_group())
        processing_layout.addStretch()

        # =====================================================================
        # MAIN TAB 3b: PROCESSING (animal-specific steps)
        # =====================================================================
        animal_tab = QWidget()
        animal_layout = QVBoxLayout(animal_tab)
        main_tabs.addTab(animal_tab, "Processing")

        self._processing_info_btn = self._make_info_button(
            self._show_processing_info)
        animal_info_row = QHBoxLayout()
        animal_info_row.setContentsMargins(0, 0, 0, 0)
        animal_info_row.addStretch()
        animal_info_row.addWidget(self._processing_info_btn)
        animal_layout.addLayout(animal_info_row)

        animal_layout.addWidget(proc_steps_group)
        animal_layout.addWidget(classify_steps_group)
        animal_layout.addWidget(extra_steps_group)

        # ----- Export -----
        from ..core import exporters as _exporters

        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        export_info = QLabel(
            "Write the detections, tracks and their attributes out in a "
            "standard format. Custom fields travel with them where the format "
            "has room; species names and enum values are resolved, so nothing "
            "receives an internal id."
        )
        export_info.setWordWrap(True)
        export_info.setStyleSheet("color: gray; font-size: 10px;")
        export_layout.addWidget(export_info)

        export_form = QHBoxLayout()
        export_form.addWidget(QLabel("Format:"))
        self.export_format_combo = QComboBox()
        for key, (label, _fn, _folder) in _exporters.EXPORTERS.items():
            # The segmentation GeoJSON is exported from the Segmentation
            # tool, next to the masks it belongs to.
            if key == "geojson_segmentation":
                continue
            self.export_format_combo.addItem(label, key)
        self.export_format_combo.currentIndexChanged.connect(
            self._on_export_format_changed)
        export_form.addWidget(self.export_format_combo, 1)

        self.export_camera_combo = QComboBox()
        self.export_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.export_camera_combo.setFixedWidth(100)
        export_form.addWidget(self.export_camera_combo)
        export_layout.addLayout(export_form)

        self.export_false_positives_check = QCheckBox(
            "Include detections labelled 'not-an-animal'")
        self.export_false_positives_check.setToolTip(
            "Survey formats keep them as a record of what was looked at and "
            "rejected; training formats drop them. Darwin Core always excludes "
            "them - a rejected detection is not an occurrence of anything."
        )
        export_layout.addWidget(self.export_false_positives_check)

        self.export_images_check = QCheckBox("Include images")
        self.export_images_check.setChecked(True)
        self.export_images_check.setToolTip(
            "Copy the frames the export refers to alongside it. YOLO and MOT "
            "are folder layouts that do not resolve without their images, and "
            "a Camtrap DP package is only self-contained with its media - but "
            "the frames are usually the heaviest part of a project, so turn "
            "this off when the annotations are all you need. Only frames "
            "carrying a detection are copied. Formats that reference no image "
            "(GeoJSON, TRex) ignore it."
        )
        export_layout.addWidget(self.export_images_check)

        self.export_btn = QPushButton("Export…")
        self.export_btn.clicked.connect(self.run_export)
        export_layout.addWidget(self.export_btn)

        animal_layout.addWidget(export_group)
        self._on_export_format_changed()
        animal_layout.addStretch()

    def validate_inputs(self, required_fields: list) -> bool:
        """Validate that required input fields are filled."""
        # A target folder typed in without a flight yet: name the flight now
        # (asks for the name), since every result belongs to a named flight.
        folder = self.target_folder_edit.text().strip()
        if folder and not self._flight_owns(folder):
            self._adopt_target_folder(folder)
            if self.target_folder_edit.text().strip() != folder:
                # The folder was declined or refused - and said so. No step
                # starts in the folder the field has gone back to.
                return False
            if getattr(self, "_adoption_loaded_config", False):
                # The form has just been replaced by what the folder had
                # stored: nothing is started on settings nobody has seen.
                QMessageBox.information(
                    self, "Stored configuration loaded",
                    "The target folder's stored configuration has been "
                    "loaded.\n\nCheck the inputs and settings, then start "
                    "the step again.")
                return False

        config = self.get_config()
        missing = []
        flight = self._active_flight()
        if folder and not (flight and (flight.get("name") or "").strip()):
            missing.append("Flight name (Input tab)")

        field_labels = {
            "thermal_video_paths": "Thermal video files",
            "thermal_srt_paths": "Thermal SRT files",
            "thermal_calibration_path": "Thermal calibration file",
            "rgb_video_paths": "RGB video files",
            "rgb_srt_paths": "RGB SRT files",
            "rgb_calibration_path": "RGB calibration file",
            "airdata_path": "AirData CSV",
            "dem_path": "DEM file",
            "target_folder": "Target folder"
        }

        for field in required_fields:
            value = config.get(field)
            if not value or (isinstance(value, list) and len(value) == 0):
                missing.append(field_labels.get(field, field))

        if missing:
            QMessageBox.warning(
                self,
                "Missing Inputs",
                "Please provide the following required inputs:\n\n• " + "\n• ".join(missing)
            )
            return False
        return True

    #: Shared opening for both pipeline info popups: the two tabs are split by
    #: what they depend on, not by an order they have to be run in.
    _PIPELINE_INTRO = (
        "The steps are split across two tabs, following what they depend on. "
        "<b>Pre-Processing</b> derives from the drone poses and the DEM and is "
        "independent of any animal; <b>Processing</b> depends on the "
        "detections. Only frame extraction is a prerequisite for both - "
        "re-running detection marks the Processing steps out of date and "
        "leaves the ALFS and orthomosaic alone.<br><br>"
    )

    def _show_preprocessing_info(self):
        """Info popup for the Pre-Processing steps."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Pre-Processing")
        msg.setIcon(QMessageBox.Icon.Information)
        body = (
            "<b>P1 - Extract Frames</b><br>"
            "Decodes and undistorts frames for the selected camera (thermal or RGB); "
            "matches GPS positions from the AirData log via SRT timestamps (video) or "
            "EXIF timestamps (photo). Run once per camera you need.<br><br>"
            "<b>P2 - Generate Flight Route</b><br>"
            "Creates a GPS flight-path line layer and per-frame camera-position points "
            "from the extracted pose data.<br><br>"
            "<b>P3 - Calculate Field of View</b><br>"
            "Computes per-frame camera footprint polygons on the ground using the DEM.<br><br>"
            "<b>P4 - Generate ALFS</b><br>"
            "Projects all frames onto the DEM surface and blends them into a "
            "georeferenced GeoTIFF mosaic.<br><br>"
            "<b>P5 - Export Frames as GeoTIFF</b><br>"
            "Exports individual frames as separate georeferenced GeoTIFFs.<br><br>"
            "<b>P6 - Generate Orthomosaic</b><br>"
            "Merges the exported frame GeoTIFFs (P5) into a single true "
            "orthomosaic, using all frames or a selected range and a configurable "
            "overlap merge mode.<br><br>"

            "<b>Several flights in one project</b><br>"
            "Each flight needs its own target folder - the result files are per "
            "folder, so two flights sharing one would overwrite each other. Add "
            "them with the <b>+</b> button beside the flight selector on the "
            "Input tab; each keeps its own configuration and its own QGIS layer "
            "group, and the Survey Analytics tools can combine them."
        )
        msg.setText(f"<b>Pre-Processing</b><br><br>{self._PIPELINE_INTRO}{body}")
        msg.exec()

    def _show_processing_info(self):
        """Info popup for the animal-specific steps."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Processing")
        msg.setIcon(QMessageBox.Icon.Information)
        body = (
            "<b>A1 - Detect Animals</b><br>"
            "Runs YOLO-based detection on every extracted frame. The thermal model is "
            "downloaded automatically on first use.<br><br>"
            "<b>→ Geo-Reference Detections</b><br>"
            "Projects pixel-space bounding boxes to UTM coordinates by ray-casting "
            "against the DEM mesh. A detection that cannot be placed is recorded "
            "with the reason rather than dropped.<br><br>"
            "<b>→ Calculate Perpendicular</b><br>"
            "Measures the perpendicular distance from each geo-referenced detection "
            "to the flight route line - useful for transect-based surveys.<br><br>"
            "<b>A2 - Track Animals Or Import</b><br>"
            "Links detections across frames into continuous tracks using the selected "
            "tracking backend. With a TRex tracklet folder configured this imports "
            "those instead, and runs no tracker.<br><br>"
            "<b>A3 - Match RGB ↔ Thermal Tracks</b><br>"
            "Works out which thermal track and which RGB track are the same "
            "animal, by comparing where their boxes sit once the two views are "
            "registered onto each other. Needs both cameras tracked. An animal "
            "both cameras saw is a confirmed animal; a track only one saw is "
            "either an animal the other sensor cannot make out, or noise. The "
            "classifiers and the label sync both read the answer.<br><br>"
            "<b>S1 - Open Segmentation Tool</b><br>"
            "Opens the Segmentation window (also on the toolbar): prompt "
            "SAM3 / SAM 3.1 with text or clicked points on the extracted "
            "frames, on single frames or tracked across a sequence - through "
            "the Roboflow API, transformers, or Meta's sam3 package. The "
            "window also geo-references the masks onto the DEM, adds them as "
            "QGIS layers and exports them as GeoJSON.<br><br>"

            "<b>Export</b><br>"
            "Writes the detections and tracks out as COCO, YOLO, MOT, TRex "
            "tracklets, GeoJSON, a Camtrap DP package or a Darwin Core Archive "
            "for GBIF. Species names, enum values and custom fields are "
            "resolved, so nothing receives an internal id."
        )
        msg.setText(f"<b>Processing</b><br><br>{self._PIPELINE_INTRO}{body}")
        msg.exec()

    def run_trex_import(self):
        """Import TRex tracklets and geo-reference them against the DEM."""
        config = self.get_config()
        camera = config.get("tracking_camera", "T")
        target_folder = config.get("target_folder", "")
        npz_dir = config.get("trex_npz_dir", "")

        if not npz_dir or not os.path.isdir(npz_dir):
            QMessageBox.warning(
                self, "Missing Input",
                "Please specify a valid NPZ folder in the Tracking config tab."
            )
            return

        poses_suffix = "t" if camera == "T" else "w"
        poses_file = os.path.join(target_folder, f"poses_{poses_suffix}.json")
        if not os.path.isfile(poses_file):
            camera_name = "Thermal" if camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Please run Step 1 (Extract Frames) for {camera_name} first."
            )
            return

        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("trex_import")

    def _on_export_format_changed(self):
        """Reflect the chosen format's own defaults in the options."""
        from ..core import exporters

        key = self.export_format_combo.currentData()
        if key is None:
            return
        # Darwin Core has no say in this, so the checkbox is disabled rather
        # than silently ignored.
        is_dwca = key == "dwca"
        self.export_false_positives_check.setEnabled(not is_dwca)
        self.export_false_positives_check.setChecked(
            (not is_dwca) and key not in exporters.TRAINING_FORMATS)

        # Disabled rather than hidden, so it is visible that this format
        # references no image at all rather than that the option was missed.
        self.export_images_check.setEnabled(
            key in exporters.SUPPORTS_IMAGES)

    def run_export(self):
        """Write the current project out in the selected format."""
        from qgis.PyQt.QtWidgets import QApplication

        from ..core import exporters

        folder = self.target_folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Export", "Select a target folder first.")
            return

        key = self.export_format_combo.currentData()
        label, _function, is_folder = exporters.EXPORTERS[key]
        modality = "t" if self.export_camera_combo.currentIndex() == 0 else "w"

        epsg = None
        if key in exporters.NEEDS_CRS:
            epsg = self._parse_epsg_from_text(
                self.target_crs_edit.text().strip().upper())
            if not epsg:
                QMessageBox.warning(
                    self, "Export",
                    f"{label} publishes latitude/longitude, so the target CRS "
                    "must be set before exporting.")
                return

        if is_folder:
            output = QFileDialog.getExistingDirectory(
                self, f"Export {label} - choose a folder", folder)
        else:
            suggested = os.path.join(
                folder, exporters.DEFAULT_FILENAME.get(key, "export.json"))
            output, _filter = QFileDialog.getSaveFileName(
                self, f"Export {label}", suggested)
        if not output:
            return

        self.log(f"Exporting {label} ({'thermal' if modality == 't' else 'RGB'})…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            exporters.run_export(
                key, folder, modality, output, epsg=epsg,
                include_not_an_animal=(
                    self.export_false_positives_check.isChecked()
                    if self.export_false_positives_check.isEnabled() else None),
                include_images=self.export_images_check.isChecked(),
                log_fn=self.log)
        except exporters.ExportError as exc:
            QApplication.restoreOverrideCursor()
            self.log(f"Export failed: {exc}")
            QMessageBox.warning(self, "Export", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surfaced to the user below
            QApplication.restoreOverrideCursor()
            self.log(f"Export failed: {exc}")
            QMessageBox.critical(
                self, "Export", f"The export could not be completed:\n\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        QMessageBox.information(
            self, "Export", f"{label} written to:\n{output}")

    def _push_stage_camera(self, index: int, step_combos):
        """Set every step combo of a stage to the stage's selection.

        Each step combo re-evaluates the step statuses when it changes; a
        stage has up to six, so the statuses were scanned (and logged) six
        times per switch. They are held back here and scanned once.
        """
        self._stage_push_depth = getattr(self, "_stage_push_depth", 0) + 1
        changed = False
        try:
            for combo in step_combos:
                if combo.currentIndex() != index:
                    combo.setCurrentIndex(index)
                    changed = True
        finally:
            self._stage_push_depth -= 1
        if changed and self._stage_push_depth == 0:
            self._on_step_camera_changed()

    def _sync_stage_cameras(self):
        """Bring the stage selectors and their step combos back into step
        after values were set with signals blocked or one by one - e.g. a
        project of an earlier version that stored a camera per step. The
        stage takes its first step's camera (species for classification)."""
        for stage_combo, lead, step_combos in getattr(
                self, "_stage_cameras", []):
            index = lead.currentIndex()
            if stage_combo.currentIndex() != index:
                stage_combo.setCurrentIndex(index)
            self._push_stage_camera(index, step_combos)

    def _on_step_camera_changed(self):
        """Re-evaluate step statuses when a step's camera selection changes."""
        if getattr(self, "_stage_push_depth", 0):
            return  # a stage is setting its steps: scanned once afterwards
        if self.worker is not None:
            return  # don't clobber the status of a currently running step
        target_folder = self.target_folder_edit.text().strip()
        if target_folder and os.path.isdir(target_folder):
            self._check_existing_outputs(target_folder)

    # Processing functions
    def run_extract_frames(self):
        """Run frame extraction for the camera selected in the step's combo."""
        if self.extract_camera_combo.currentIndex() == 0:
            self.run_extract_thermal_frames()
        else:
            self.run_extract_rgb_frames()

    def run_extract_thermal_frames(self):
        """Run frame extraction for thermal modality."""
        config = self.get_config()

        if config["input_mode"] == "photo":
            # Photo mode validation
            thermal_calib_ok = (
                config.get("thermal_photo_calibration_data") is not None or  # noqa: W503, W504
                config.get("thermal_photo_calibration_path")
            )
            if not config.get("thermal_photo_dir") or not thermal_calib_ok:
                QMessageBox.warning(
                    self,
                    "Missing Thermal Photo Inputs",
                    "Please provide thermal photo inputs:\n\n"
                    "• Thermal photo folder\n"
                    "• Thermal calibration (preset or custom file)"
                )
                return
        else:
            # Video mode validation
            thermal_calib_ok = (
                config.get("thermal_calibration_data") is not None or  # noqa: W503, W504
                config.get("thermal_calibration_path")
            )
            srt_ok = self.embedded_srt_check.isChecked() or bool(config.get("thermal_srt_paths"))
            if not (config.get("thermal_video_paths") and srt_ok and thermal_calib_ok):
                QMessageBox.warning(
                    self,
                    "Missing Thermal Inputs",
                    "Please provide thermal video inputs:\n\n"
                    "• Thermal video files\n" +  # noqa: W503, W504
                    ("" if self.embedded_srt_check.isChecked() else "• Thermal SRT files\n") +  # noqa: W503, W504
                    "• Thermal calibration (preset or custom file)"
                )
                return

        # Validate common inputs (airdata optional in photo mode - EXIF fallback)
        required = ["target_folder"]
        if config["input_mode"] != "photo":
            required.append("airdata_path")
        if not self.validate_inputs(required):
            return
        if not self._confirm_duplicate_flag_change(config):
            return

        self.start_worker("extract_thermal_frames")

    def _confirm_duplicate_flag_change(self, config) -> bool:
        """Re-extracting thermal video with the duplicated-frames setting
        switched renumbers every frame; say so before it happens."""
        from ..core import poses

        if config.get("input_mode") == "photo":
            return True
        before = poses.flag_setting(config.get("target_folder", ""), "t")
        now = bool(config.get("flag_duplicate_frames"))
        if before is None or before == now:
            return True
        was = "kept and flagged" if before else "dropped"
        will = "kept and flagged" if now else "dropped"
        answer = QMessageBox.question(
            self, "Frame numbers will change",
            f"The existing thermal frames were extracted with duplicated "
            f"(frozen) frames {was}; this extraction has them {will}.\n\n"
            "That shifts the frame numbers. Everything computed from the "
            "current thermal frames - detections, tracks, labels, per-frame "
            "corrections, field of view, GeoTIFFs, embeddings - refers to "
            "the old numbers and has to be re-run (labels and frame-range "
            "corrections have to be checked by hand).\n\nExtract anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def run_extract_rgb_frames(self):
        """Run frame extraction for RGB modality."""
        config = self.get_config()

        if config["input_mode"] == "photo":
            # Photo mode validation
            rgb_calib_ok = (
                config.get("rgb_photo_calibration_data") is not None or  # noqa: W503, W504
                config.get("rgb_photo_calibration_path")
            )
            if not config.get("rgb_photo_dir") or not rgb_calib_ok:
                QMessageBox.warning(
                    self,
                    "Missing RGB Photo Inputs",
                    "Please provide RGB photo inputs:\n\n"
                    "• RGB photo folder\n"
                    "• RGB calibration (preset or custom file)"
                )
                return
        else:
            # Video mode validation
            rgb_calib_ok = (
                config.get("rgb_calibration_data") is not None or  # noqa: W503, W504
                config.get("rgb_calibration_path")
            )
            srt_ok = self.embedded_srt_check.isChecked() or bool(config.get("rgb_srt_paths"))
            if not (config.get("rgb_video_paths") and srt_ok and rgb_calib_ok):
                QMessageBox.warning(
                    self,
                    "Missing RGB Inputs",
                    "Please provide RGB video inputs:\n\n"
                    "• RGB video files\n" +  # noqa: W503, W504
                    ("" if self.embedded_srt_check.isChecked() else "• RGB SRT files\n") +  # noqa: W503, W504
                    "• RGB calibration (preset or custom file)"
                )
                return

        # Validate common inputs (airdata optional in photo mode - EXIF fallback)
        required = ["target_folder"]
        if config["input_mode"] != "photo":
            required.append("airdata_path")
        if not self.validate_inputs(required):
            return

        self.start_worker("extract_rgb_frames")

    def _confirm_ultralytics_license(self) -> bool:
        """Show the Ultralytics license notice before running detection.

        Returns True if the user accepted (or previously chose to skip the
        notice), False if they cancelled. The skip decision is only stored
        when the user accepts, so cancelling can never silently disable
        detection in future runs.
        """
        settings = QgsSettings()
        settings_key = f"{PLUGIN_SCOPE}/skipUltralyticsLicenseNotice"
        if settings.value(settings_key, False, type=bool):
            return True

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Ultralytics License Notice")
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setText(
            "Wildlife detection uses the Ultralytics framework, which is "
            "licensed under AGPL-3.0."
        )
        msg_box.setInformativeText(
            "Commercial use requires a separate Ultralytics Enterprise "
            "License. Please make sure your usage complies with the license "
            "terms before proceeding.\n\n"
            "See https://www.ultralytics.com/license for details."
        )
        msg_box.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QMessageBox.StandardButton.Ok)
        remember_check = QCheckBox("Remember my decision")
        remember_check.setChecked(False)
        msg_box.setCheckBox(remember_check)

        msg_box.exec()
        # exec() returns a plain int under PyQt6, so resolve the clicked
        # button to a StandardButton for a binding-agnostic comparison.
        clicked = msg_box.standardButton(msg_box.clickedButton())
        if clicked != QMessageBox.StandardButton.Ok:
            return False

        if remember_check.isChecked():
            settings.setValue(settings_key, True)
        return True

    def run_detection(self):
        """Run animal detection step."""
        config = self.get_config()
        camera = config.get("detection_camera", "T")

        # Check if frames exist for selected camera
        target_folder = config["target_folder"]
        frames_folder = os.path.join(target_folder, f"frames_{'t' if camera == 'T' else 'w'}")
        poses_file = os.path.join(target_folder, f"poses_{'t' if camera == 'T' else 'w'}.json")

        if not os.path.exists(poses_file) or not os.path.exists(frames_folder):
            camera_name = "Thermal" if camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Please run Step 1 (Extract Frames) for {camera_name} first."
            )
            return

        from ..core import custom_models
        suffix = "t" if camera == "T" else "w"
        from ..core import raster_sources
        rasters = BambiProcessor._raster_source(
            config, target_folder, suffix,
            config.get("detection_projection", "perspective"))
        if rasters is not None:
            camera_name = "Thermal" if camera == "T" else "RGB"
            if not rasters.frames():
                QMessageBox.warning(
                    self, "Missing Prerequisites",
                    raster_sources.missing_message(
                        rasters.kind, camera_name, "Detection"))
                return
            if not self.validate_inputs(["dem_path"]):
                return
        uses_custom = custom_models.is_custom(
            (config.get("custom_implementations") or {}).get(
                f"detector_{suffix}"))
        # the Ultralytics licence concerns the built-in detector only
        if not uses_custom and not self._confirm_ultralytics_license():
            return

        self.start_worker("detection")

    def run_georeference(self):
        """Run geo-referencing step."""
        config = self.get_config()
        det_camera = config.get("detection_camera", "T")
        det_suffix = "t" if det_camera == "T" else "w"

        # Whether there are detections is what the store says - not whether
        # the folder of the legacy text file exists, which a flight copied
        # without it, or one that only has hand-labelled boxes, has not.
        if not self.has_stage(_store_kinds.DETECTIONS, det_suffix,
                              config["target_folder"]):
            camera_name = "Thermal" if det_camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} detection has not been completed.\nPlease run Step 3 (Detect Animals) first."
            )
            return

        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("georeference")

    def run_calculate_fov(self):
        """Run Field of View calculation step."""
        config = self.get_config()
        camera = config.get("fov_camera", "T")

        # Check if poses exist for selected camera
        target_folder = config["target_folder"]
        poses_file = os.path.join(target_folder, f"poses_{'t' if camera == 'T' else 'w'}.json")

        if not os.path.exists(poses_file):
            camera_name = "Thermal" if camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Please run Step 1 (Extract Frames) for {camera_name} first."
            )
            return

        # Check DEM path
        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("calculate_fov")

    def run_tracking(self):
        """Run tracking step, or import TRex tracklets if a tracklet folder is set."""
        config = self.get_config()

        # If a TRex NPZ folder is configured, import the pre-computed tracklets
        # instead of running the classic tracker.
        from ..core import custom_models
        if custom_models.is_custom(
                (config.get("custom_implementations") or {}).get("tracker")):
            # A custom tracker says itself whether it needs geo-referenced
            # detections; the step reports what is missing.
            if self.validate_inputs(["target_folder"]):
                self.start_worker("tracking")
            return

        if config.get("trex_npz_dir", "").strip():
            self.run_trex_import()
            return

        trk_camera = config.get("tracking_camera", "T")
        trk_suffix = "t" if trk_camera == "T" else "w"

        # Geo-referenced detections: in the store, wherever the legacy text
        # file's folder is or is not.
        if not self.has_stage(_store_kinds.GEOREFERENCED, trk_suffix,
                              config["target_folder"]):
            camera_name = "Thermal" if trk_camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} geo-referencing has not been completed.\n"
                f"Please run Geo-Reference Detections (under step A1) first."
            )
            return

        self.start_worker("tracking")

    def run_alfs(self):
        """Run alfs generation step."""
        config = self.get_config()
        camera = config.get("alfs_camera", "T")

        # Check if frames exist for selected camera
        target_folder = config["target_folder"]
        poses_file = os.path.join(target_folder, f"poses_{'t' if camera == 'T' else 'w'}.json")
        frames_folder = os.path.join(target_folder, f"frames_{'t' if camera == 'T' else 'w'}")

        if not os.path.exists(poses_file) or not os.path.exists(frames_folder):
            camera_name = "Thermal" if camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Please run Step 1 (Extract Frames) for {camera_name} first."
            )
            return

        # Check DEM path
        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("alfs")

    def run_export_geotiffs(self):
        """Run frame GeoTIFF export step."""
        config = self.get_config()
        camera = config.get("geotiff_camera", "T")

        # Check if frames exist for selected camera
        target_folder = config["target_folder"]
        poses_file = os.path.join(target_folder, f"poses_{'t' if camera == 'T' else 'w'}.json")
        frames_folder = os.path.join(target_folder, f"frames_{'t' if camera == 'T' else 'w'}")

        if not os.path.exists(poses_file) or not os.path.exists(frames_folder):
            camera_name = "Thermal" if camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Please run Step 1 (Extract Frames) for {camera_name} first."
            )
            return

        # Check DEM path
        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("export_geotiffs")

    def run_orthomosaic(self):
        """Merge exported frame GeoTIFFs into a single orthomosaic."""
        config = self.get_config()
        camera = config.get("ortho_camera", "T")
        suffix = "t" if camera == "T" else "w"
        camera_name = "Thermal" if camera == "T" else "RGB"

        geotiff_folder = os.path.join(config["target_folder"], f"geotiffs_{suffix}")
        has_geotiffs = os.path.isdir(geotiff_folder) and any(
            f.lower().endswith((".tif", ".tiff")) for f in os.listdir(geotiff_folder)
        )
        if not has_geotiffs:
            QMessageBox.warning(
                self,
                "Missing GeoTIFFs",
                f"{camera_name} frame GeoTIFFs have not been exported.\n"
                f"Please run Step 7 (Export Frames as GeoTIFF) first."
            )
            return

        self.start_worker("orthomosaic")

    def run_track_matching(self):
        """Match thermal and RGB tracks (step A3)."""
        from ..core import store, track_store

        config = self.get_config()
        target_folder = config["target_folder"]

        # Both modalities are required - this is the one step that is
        # inherently about the pair, so there is no camera combo to choose.
        missing = [name for name, modality in (("Thermal", "t"), ("RGB", "w"))
                   if not os.path.isfile(
                       store.stage_path(target_folder, store.TRACKS, modality))]
        if missing:
            QMessageBox.warning(
                self, "Missing Prerequisites",
                f"{' and '.join(missing)} tracking has not been completed.\n\n"
                "Cross-modal matching compares the two cameras, so both have "
                "to be detected, geo-referenced and tracked first."
            )
            return

        empty = [name for name, modality in (("Thermal", "t"), ("RGB", "w"))
                 if not track_store.load_pixel_tracks(target_folder, modality)]
        if empty:
            QMessageBox.warning(
                self, "No Tracks",
                f"{' and '.join(empty)} tracking produced no tracks, so there "
                "is nothing to match."
            )
            return

        self.start_worker("track_matching")

    def run_embeddings(self):
        """Compute DINOv3 embeddings for the selected cameras (step C1).

        'Matched' embeds both, so every camera it covers is checked before any
        of them is started - half an embedding run is the worst outcome here.
        """
        from ..core import classification, geo_crops, hf_access, store
        from ..core import track_store

        config = self.get_config()
        target_folder = config["target_folder"]
        projection = config.get("classification_projection", "non_geo")
        targets = classification.INPUT_TARGETS.get(
            config.get("embeddings_camera", "matched"), ("t", "w"))

        for suffix in targets:
            camera_name = _camera_name(suffix)
            if not os.path.isfile(
                    store.stage_path(target_folder, store.TRACKS, suffix)):
                QMessageBox.warning(
                    self, "Missing Prerequisites",
                    f"{camera_name} tracking has not been completed.\n"
                    "Embeddings are computed for tracked animals, so run "
                    f"tracking for {camera_name} first."
                )
                return

            if not track_store.load_pixel_tracks(target_folder, suffix):
                QMessageBox.warning(
                    self, "No Tracks",
                    f"{camera_name} tracking produced no tracks, so there is "
                    "nothing to embed."
                )
                return

            # An orthorectified projection crops the exported GeoTIFFs, so say
            # so here rather than letting the worker raise seconds in.
            from ..core import raster_sources
            rasters = BambiProcessor._raster_source(
                config, target_folder, suffix, projection)
            if rasters is not None and not rasters.frames():
                QMessageBox.warning(
                    self, "Missing GeoTIFFs"
                    if rasters.kind == raster_sources.GEOTIFF
                    else "Missing ALFS Integrals",
                    raster_sources.missing_message(
                        rasters.kind, camera_name,
                        "The classification projection"))
                return

        # A gated repository is the usual failure, and finding out after the
        # first frames have been cropped wastes the user's time.
        token, _source = hf_access.resolve_token(
            self.hf_token_edit.text() if hasattr(self, 'hf_token_edit') else "")
        if not token:
            answer = QMessageBox.question(
                self, "No Hugging Face Token",
                "No Hugging Face token is configured, and the DINOv3 backbone "
                "is a gated model.\n\n"
                "This will only work if the model is already in the local "
                "cache. Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return

        self.start_worker("embeddings")

    def run_classification(self, task: str = ""):
        """Run one classifier on the cameras its Input selects (steps C2–C4).

        *task* empty runs everything enabled, which is what a scripted run of
        the whole block does.
        """
        from ..core import classification, classification_store, hf_access

        # Carried in the config rather than the step name, so the worker's
        # entry point stays uniform.
        self._classification_only = (task,) if task else ()
        config = self.get_config()
        target_folder = config["target_folder"]

        models = config.get("classification_models") or {}
        wanted = (task,) if task else hf_access.TASKS
        enabled = [name for name in wanted
                   if (models.get(name) or {}).get("model", "off") != "off"]
        if not enabled:
            label = (hf_access.TASK_LABELS.get(task, task).lower() if task
                     else "")
            QMessageBox.warning(
                self, "No Classifier Enabled",
                f"The {label} classifier is switched off.\n\n"
                "Choose a model for it in the Classification configuration "
                "tab." if task else
                "No classifier is switched on.\n\n"
                "Choose a model for one of them in the Classification "
                "configuration tab."
            )
            return

        # Every camera any enabled stage lands on has to be embedded, because
        # each is its own pass over that camera's tracks.
        targets = sorted({m for name in enabled
                          for m in classification.targets_of(models[name])})
        # A custom implementation may read the cut-out boxes instead of
        # embeddings, and only its setup() says which - so its stages are
        # not held back here; the step reports missing embeddings itself.
        from ..core import custom_models
        embedded_targets = sorted({
            m for name in enabled if not custom_models.is_custom(models[name])
            for m in classification.targets_of(models[name])})
        for suffix in embedded_targets:
            if classification_store.active_embedding_run(
                    target_folder, suffix) is None:
                QMessageBox.warning(
                    self, "Missing Embeddings",
                    f"No {_camera_name(suffix)} embeddings have been "
                    "computed.\n\n"
                    "This step is set to classify that camera's animals, so "
                    "run 'Compute DINOv3 Embeddings' for it first - the "
                    "classifiers read those vectors rather than the images."
                )
                return

        # Sex and age pick their model by species, so the species has to be
        # known on each camera they are about - classified there, hand
        # annotated, or synced across from the other camera.
        if task in hf_access.PER_SPECIES_TASKS:
            unknown = [suffix for suffix in targets
                       if not self._species_is_known(target_folder, suffix)]
            if unknown:
                names = " and ".join(_camera_name(s) for s in unknown)
                QMessageBox.warning(
                    self, "Species Not Classified",
                    f"No species has been assigned to the {names} animals "
                    "yet.\n\n"
                    f"The {hf_access.TASK_LABELS.get(task, task).lower()} "
                    "classifier is chosen per species, so run Species "
                    "Classification on that camera first - or sync the labels "
                    "across from the other one."
                )
                return

        self.start_worker(f"classify_{task}" if task else "classification")

    def _species_is_known(self, target_folder: str, suffix: str) -> bool:
        """True when this camera's animals carry a species from any source."""
        from ..core import classification_store, track_store

        if classification_store.track_predictions(
                target_folder, suffix, "species"):
            return True
        return bool(track_store.track_species(target_folder, suffix))

    def run_life_stage(self):
        """Decide each animal's age class (step C5).

        Box areas are compared within one camera, so a stage set to Matched
        measures each camera's animals against their own cohort.
        """
        from ..core import classification, store, track_store

        config = self.get_config()
        target_folder = config["target_folder"]
        spec = (config.get("classification_models") or {}).get(
            "life_stage") or {}
        minimum = config.get("life_stage_min_individuals", 4)

        for suffix in classification.targets_of(spec):
            camera_name = _camera_name(suffix)
            if not os.path.isfile(
                    store.stage_path(target_folder, store.TRACKS, suffix)):
                QMessageBox.warning(
                    self, "Missing Prerequisites",
                    f"{camera_name} tracking has not been completed.\n\n"
                    "Life stage is read from each animal's box size, so the "
                    "animals have to be tracked first."
                )
                return

            tracks = {row["track_id"] for row
                      in track_store.load_pixel_tracks(target_folder, suffix)}
            if len(tracks) < minimum:
                QMessageBox.warning(
                    self, "Too Few Individuals",
                    f"The {camera_name} camera has {len(tracks)} tracked "
                    f"individual(s), and the size comparison needs at least "
                    f"{minimum}.\n\n"
                    "Sizes are compared within one camera's animals, so a "
                    "small set gives the test nothing to compare against. "
                    "Lower the minimum in the Classification tab if you want "
                    "it to run anyway."
                )
                return

        self.start_worker("life_stage")

    def sync_labels(self):
        """Carry labels from one camera's animals onto their matched partners."""
        from ..core import label_sync, match_store

        config = self.get_config()
        target_folder = config["target_folder"]
        # "Thermal → RGB" is index 0, so the source is the first named camera.
        source = "t" if self.sync_labels_combo.currentIndex() == 0 else "w"
        destination = "w" if source == "t" else "t"

        if not match_store.track_matches(target_folder):
            QMessageBox.warning(
                self, "No Matches",
                "No matched track pairs were found.\n\n"
                "Labels travel along the match, so run 'Match RGB ↔ Thermal "
                "Tracks' first."
            )
            return

        try:
            self.update_status("sync_labels", "🟡 Syncing...")
            counts = label_sync.sync_all(
                target_folder, source, destination, log_fn=self.log)
        except Exception as exc:
            self.log(f"Error syncing labels: {exc}")
            self.update_status("sync_labels", "🔴 Error")
            QMessageBox.critical(
                self, "Error", f"Failed to sync the labels: {exc}")
            return

        self.update_status("sync_labels", "🟢 Synced")
        if not any(counts.values()):
            self.log(
                f"Nothing to sync from {_camera_name(source)} to "
                f"{_camera_name(destination)}: the matched animals already "
                "carry an answer, or the source has none to give.")
        else:
            summary = ", ".join(f"{value} {key.replace('_', ' ')}"
                                for key, value in counts.items() if value)
            self.log(f"Synced {_camera_name(source)} → "
                     f"{_camera_name(destination)}: {summary}")

    def open_segmentation_tool(self):
        """Open the Segmentation window (the toolbar action owns it)."""
        action = getattr(self, "_segmentation_tool_action", None)
        if action is not None:
            action.trigger()
            return
        from ..bambi_segmentation_tool import SegmentationToolDialog
        dlg = getattr(self, "_segmentation_tool_dlg", None)
        if dlg is None:
            dlg = SegmentationToolDialog(self.iface, dock_widget=self, parent=self)
            dlg.finished.connect(
                lambda _: setattr(self, "_segmentation_tool_dlg", None))
            self._segmentation_tool_dlg = dlg
        else:
            dlg.apply_dock_defaults()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def run_flight_route(self):
        """Run flight route generation step."""
        config = self.get_config()
        camera = config.get("flight_route_camera", "T")

        # Poses are optional: without them only the AirData route line is
        # generated (no camera positions / frame markers / image labels)
        target_folder = config["target_folder"]
        poses_file = os.path.join(target_folder, f"poses_{'t' if camera == 'T' else 'w'}.json")
        airdata_path = config.get("airdata_path", "")

        if not os.path.exists(poses_file):
            camera_name = "Thermal" if camera == "T" else "RGB"
            if not (airdata_path and os.path.exists(airdata_path)):
                QMessageBox.warning(
                    self,
                    "Missing Prerequisites",
                    f"{camera_name} frame extraction has not been completed and "
                    f"no AirData CSV is selected.\n"
                    f"Run Step 1 (Extract Frames) for {camera_name}, or select an "
                    f"AirData file to generate the flight route line only."
                )
                return
            QMessageBox.warning(
                self,
                "Camera Positions Not Available",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Only the flight route line (from AirData) will be generated - "
                f"camera positions, frame/distance markers and image labels "
                f"will not be available.\n\n"
                f"Run Step 1 (Extract Frames) for {camera_name} and re-run this "
                f"step later to add them."
            )

        self.start_worker("flight_route")

    def run_perpendicular(self):
        """Run perpendicular distance calculation step."""
        config = self.get_config()
        target_folder = config["target_folder"]
        fr_suffix = "t" if config.get("flight_route_camera", "T") == "T" else "w"
        det_suffix = "t" if config.get("detection_camera", "T") == "T" else "w"

        route_file = os.path.join(target_folder, f"flight_route_{fr_suffix}", "flight_route.geojson")

        missing = []
        if not os.path.exists(route_file):
            missing.append("Flight route (run Generate Flight Route first)")
        if not self.has_stage(_store_kinds.GEOREFERENCED, det_suffix,
                              target_folder):
            missing.append("Geo-referenced detections (run Geo-Reference Detections first)")

        if missing:
            QMessageBox.warning(
                self, "Missing Prerequisites",
                "The following are required:\n\n" + "\n".join(f"• {m}" for m in missing)
            )
            return

        self.start_worker("perpendicular")

    def run_track_perpendicular(self):
        """Run track perpendicular distance calculation step."""
        config = self.get_config()
        target_folder = config["target_folder"]
        fr_suffix = "t" if config.get("flight_route_camera", "T") == "T" else "w"
        trk_suffix = "t" if config.get("tracking_camera", "T") == "T" else "w"

        route_file = os.path.join(target_folder, f"flight_route_{fr_suffix}", "flight_route.geojson")

        missing = []
        if not os.path.exists(route_file):
            missing.append("Flight route (run Generate Flight Route first)")
        if not self.has_stage(_store_kinds.TRACKS, trk_suffix, target_folder):
            missing.append("Tracked animals (run Track Animals first)")

        if missing:
            QMessageBox.warning(
                self, "Missing Prerequisites",
                "The following are required:\n\n" + "\n".join(f"• {m}" for m in missing)
            )
            return

        self.start_worker("track_perpendicular")

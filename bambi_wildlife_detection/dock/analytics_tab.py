# -*- coding: utf-8 -*-
"""The Survey Analytics tab: what the flights say together.

The survey - the ticked flights, their survey area, the species that are
counted - and the analyses that run on it: census, track inventory, density
and coverage maps, the survey orthomosaic, distance sampling and the
population estimate, each with the dialog that shows its result.

The transect layers are in ``transect_layers``; the Habitat Context group is
in ``bambi_environment_panel``.
"""

import json
import os
from typing import Dict, List

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QSpinBox,
    QDoubleSpinBox, QLabel, QTextEdit, QCheckBox, QMessageBox, QListWidget,
    QListWidgetItem, QDialog, QDialogButtonBox, QAbstractItemView,
)
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY, QgsField,
    QgsRasterLayer,
)

from ..qgis_compat import FieldType
from ..bambi_survey_area import SurveyAreaPanel
from ..bambi_survey_panel import CensusTableDialog, SurveyFlightsPanel
from ..core import store as _store_kinds, quiet


class AnalyticsTabMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_analytics_tab(self, main_tabs):
        """The Survey Analytics tab."""
        # =====================================================================
        # MAIN TAB 4: SURVEY ANALYTICS
        # =====================================================================
        # The survey tools are not tied to the flight selected on the
        # processing tabs: the Survey box at the top says which processed
        # flights, which camera and which output folder (core.survey).
        analytics_tab = QWidget()
        analytics_tab_layout = QVBoxLayout(analytics_tab)
        main_tabs.addTab(analytics_tab, "Survey Analytics")

        analytics_info = QLabel(
            "Population-level products of the flights ticked below, pooled "
            "into one survey. Tracks count each animal once; detections use "
            "every box. The run log is shown below the tabs."
        )
        analytics_info.setWordWrap(True)
        analytics_info.setStyleSheet("color: gray; font-size: 10px;")
        analytics_tab_layout.addWidget(analytics_info)

        # ----- Survey: flights, camera, folder -----
        self.survey_panel = SurveyFlightsPanel(self)
        self.survey_panel.changed.connect(self._on_survey_changed)
        analytics_tab_layout.addWidget(self.survey_panel)

        # ----- Species filter -----
        # One control for the whole tab: a density map of roe deer and an
        # abundance estimate of everything are not comparable, so the filter
        # belongs to the question rather than to one analytic. Species are
        # matched by name across the flights (ids are per flight).
        species_group = QGroupBox("Species")
        species_layout = QVBoxLayout(species_group)

        self.analytics_all_species_check = QCheckBox("All species")
        self.analytics_all_species_check.setChecked(True)
        self.analytics_all_species_check.setToolTip(
            "Count every species the flights define. 'not-an-animal' is "
            "never counted either way.")
        self.analytics_all_species_check.toggled.connect(
            self._on_analytics_species_mode_changed)
        species_layout.addWidget(self.analytics_all_species_check)

        self.analytics_species_list = QListWidget()
        self.analytics_species_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self.analytics_species_list.setMaximumHeight(96)
        self.analytics_species_list.setEnabled(False)
        self.analytics_species_list.setToolTip(
            "Species to count, from every ticked flight. All tools on this "
            "tab use this list and record it alongside the result. The "
            "coverage map counts frames rather than animals, so it is "
            "unaffected.")
        self.analytics_species_list.itemChanged.connect(
            lambda _item: self._update_analytics_species_status())
        species_layout.addWidget(self.analytics_species_list)

        self.analytics_per_species_check = QCheckBox(
            "Separate result per species")
        self.analytics_per_species_check.setToolTip(
            "Off: the selected species are pooled into one result.\n"
            "On: the density heatmap, distance sampling and population "
            "estimation run once per species and write one result each, named "
            "after the species. A species with nothing to count is skipped, "
            "and one too thin to fit is reported rather than failing the run."
        )
        self.analytics_per_species_check.toggled.connect(
            lambda _on: self._update_analytics_species_status())
        species_layout.addWidget(self.analytics_per_species_check)

        self.analytics_species_status = QLabel("")
        self.analytics_species_status.setWordWrap(True)
        self.analytics_species_status.setStyleSheet(
            "color: gray; font-size: 10px;")
        species_layout.addWidget(self.analytics_species_status)

        analytics_tab_layout.addWidget(species_group)

        # ----- Survey area (plot + exclusion zones) -----
        self.survey_area_panel = SurveyAreaPanel(self)
        analytics_tab_layout.addWidget(self.survey_area_panel)

        # ----- Census Summary -----
        census_group = QGroupBox("Census Summary")
        census_layout = QVBoxLayout(census_group)
        census_desc = QLabel(
            "Counts per species and flight - inside the survey area when one "
            "is applied - with animals per km² of the survey areas and of the "
            "ground the frames actually covered, the sex and age breakdown, "
            "and the groups the animals formed (how far apart the animals of "
            "a group may be is set per species in the Project Schema). "
            "Minimum counts: no detection probability is modelled."
        )
        census_desc.setWordWrap(True)
        census_desc.setStyleSheet("color: gray; font-size: 10px;")
        census_layout.addWidget(census_desc)

        census_run_row = QHBoxLayout()
        self.census_btn = QPushButton("→ Build Census Summary")
        self.census_btn.clicked.connect(self.run_census_summary)
        self.census_btn.setToolTip(
            "Write analytics_{camera}/census_summary_{source}.csv and .json "
            "into the survey folder and show the table.")
        self.census_status = QLabel("⚪")
        census_run_row.addWidget(self.census_btn)
        census_run_row.addWidget(self.census_status)
        census_layout.addLayout(census_run_row)

        census_show_row = QHBoxLayout()
        self.census_show_btn = QPushButton("→ Show Census Table")
        self.census_show_btn.clicked.connect(self.show_census_table)
        census_show_row.addWidget(self.census_show_btn)
        census_show_row.addStretch()
        census_layout.addLayout(census_show_row)

        groups_add_row = QHBoxLayout()
        self.add_groups_btn = QPushButton("→ Add Groups to QGIS")
        self.add_groups_btn.clicked.connect(self.add_groups_to_qgis)
        self.add_groups_btn.setToolTip(
            "Outline every group of two or more animals the census found, "
            "in the colour of its species and with the number of animals "
            "in it. The groups are listed in analytics_{camera}/"
            "groups_{source}.csv.")
        self.add_groups_status = QLabel("⚪")
        groups_add_row.addWidget(self.add_groups_btn)
        groups_add_row.addWidget(self.add_groups_status)
        census_layout.addLayout(groups_add_row)

        analytics_tab_layout.addWidget(census_group)

        # ----- Grid Density (dock/grid_density.py) -----
        analytics_tab_layout.addWidget(self._build_grid_group())

        # ----- Compare Visits (dock/trend.py) -----
        analytics_tab_layout.addWidget(self._build_trend_group())

        # ----- Track Inventory -----
        inventory_group = QGroupBox("Track Inventory")
        inventory_layout = QVBoxLayout(inventory_group)

        inventory_desc = QLabel(
            "One row per tracked individual of every ticked flight, with "
            "everything the project knows about it: first and last frame and "
            "capture time, start and end position (project CRS and WGS84), "
            "species, sex and age with the votes behind them, occluded versus "
            "total boxes, the matched track on the other camera, "
            "labelling-tool annotations and the flight-route distance. "
            "Written as CSV and JSON into the survey folder."
        )
        inventory_desc.setWordWrap(True)
        inventory_desc.setStyleSheet("color: gray; font-size: 10px;")
        inventory_layout.addWidget(inventory_desc)

        inventory_params_row = QHBoxLayout()
        self.inventory_columns_btn = QPushButton("Columns…")
        self.inventory_columns_btn.setToolTip(
            "Choose which columns the report shows for each animal. Saved "
            "with the project; the written CSV and JSON keep every column.")
        self.inventory_columns_btn.clicked.connect(self.choose_inventory_columns)
        inventory_params_row.addWidget(self.inventory_columns_btn)
        inventory_params_row.addStretch()
        inventory_layout.addLayout(inventory_params_row)
        #: Columns the report shows; empty = all (core.config_schema "json").
        self._inventory_columns = []

        inventory_run_row = QHBoxLayout()
        self.track_inventory_btn = QPushButton("→ List Tracked Individuals")
        self.track_inventory_btn.clicked.connect(self.run_track_inventory)
        self.track_inventory_btn.setToolTip(
            "Write analytics_{camera}/track_inventory.csv and .json. Runs on "
            "whatever has been computed so far - columns for steps that have "
            "not run stay empty.")
        self.track_inventory_status = QLabel("⚪")
        inventory_run_row.addWidget(self.track_inventory_btn)
        inventory_run_row.addWidget(self.track_inventory_status)
        inventory_layout.addLayout(inventory_run_row)

        inventory_report_row = QHBoxLayout()
        self.track_inventory_report_btn = QPushButton("→ Show Report")
        self.track_inventory_report_btn.clicked.connect(
            self.show_track_inventory_report)
        self.track_inventory_report_btn.setToolTip(
            "Open the inventory in a window: every individual with every "
            "column, sortable and filterable, with copy and save.")
        inventory_report_row.addWidget(self.track_inventory_report_btn)
        self.review_queue_btn = QPushButton("→ Review Queue…")
        self.review_queue_btn.clicked.connect(self.open_review_queue)
        self.review_queue_btn.setToolTip(
            "The tracks worth a human look - faint, very short, of no or "
            "doubtful species, or seen differently by the two cameras - the "
            "most doubtful first, to approve, rule out or correct one after "
            "another in the inspector (dock/review_queue.py).")
        inventory_report_row.addWidget(self.review_queue_btn)
        inventory_report_row.addStretch()
        inventory_layout.addLayout(inventory_report_row)

        inventory_add_row = QHBoxLayout()
        self.add_track_inventory_btn = QPushButton("→ Add Inventory to QGIS")
        self.add_track_inventory_btn.clicked.connect(self.add_track_inventory_to_qgis)
        self.add_track_inventory_btn.setToolTip(
            "Load the inventory as a point layer at each animal's last "
            "position, carrying every column of the table. Animals without a "
            "ground position are kept as rows without geometry.")
        self.add_track_inventory_status = QLabel("⚪")
        inventory_add_row.addWidget(self.add_track_inventory_btn)
        inventory_add_row.addWidget(self.add_track_inventory_status)
        inventory_layout.addLayout(inventory_add_row)

        analytics_tab_layout.addWidget(inventory_group)

        # ----- Density Heatmap -----
        density_group = QGroupBox("Density Heatmap")
        density_layout = QVBoxLayout(density_group)

        density_desc = QLabel(
            "Kernel-density estimate raster (points per hectare) of animal "
            "locations. Requires geo-referenced detections or tracks."
        )
        density_desc.setWordWrap(True)
        density_desc.setStyleSheet("color: gray; font-size: 10px;")
        density_layout.addWidget(density_desc)

        density_params_row = QHBoxLayout()
        density_params_row.addWidget(QLabel("Cell (m):"))
        self.density_cell_spin = QDoubleSpinBox()
        self.density_cell_spin.setRange(0.5, 500.0)
        self.density_cell_spin.setValue(5.0)
        self.density_cell_spin.setSingleStep(1.0)
        self.density_cell_spin.setToolTip("Output raster cell size in metres")
        density_params_row.addWidget(self.density_cell_spin)
        density_params_row.addWidget(QLabel("Bandwidth (m):"))
        self.density_bandwidth_spin = QDoubleSpinBox()
        self.density_bandwidth_spin.setRange(1.0, 2000.0)
        self.density_bandwidth_spin.setValue(25.0)
        self.density_bandwidth_spin.setSingleStep(5.0)
        self.density_bandwidth_spin.setToolTip(
            "Gaussian kernel bandwidth (smoothing radius) in metres")
        density_params_row.addWidget(self.density_bandwidth_spin)
        density_params_row.addStretch()
        density_layout.addLayout(density_params_row)

        density_run_row = QHBoxLayout()
        self.density_btn = QPushButton("→ Generate Density Heatmap")
        self.density_btn.clicked.connect(self.run_density_heatmap)
        self.density_btn.setToolTip(
            "Kernel-density estimate raster (points per hectare) of animal locations."
        )
        self.density_status = QLabel("⚪")
        density_run_row.addWidget(self.density_btn)
        density_run_row.addWidget(self.density_status)
        density_layout.addLayout(density_run_row)

        density_add_row = QHBoxLayout()
        self.add_density_btn = QPushButton("→ Add Density Heatmap to QGIS")
        self.add_density_btn.clicked.connect(self.add_density_heatmap_to_qgis)
        self.add_density_btn.setToolTip("Load the density raster with a graduated colour ramp.")
        self.add_density_status = QLabel("⚪")
        density_add_row.addWidget(self.add_density_btn)
        density_add_row.addWidget(self.add_density_status)
        density_layout.addLayout(density_add_row)

        analytics_tab_layout.addWidget(density_group)

        # ----- Coverage Map -----
        coverage_group = QGroupBox("Coverage Map")
        coverage_layout = QVBoxLayout(coverage_group)

        coverage_desc = QLabel(
            "Combines the exported frame GeoTIFFs of every ticked flight like "
            "the orthomosaic, but counts how many frames cover each pixel "
            "(survey effort/overlap). Requires 'Export Frames as GeoTIFF' on "
            "the Processing tab."
        )
        coverage_desc.setWordWrap(True)
        coverage_desc.setStyleSheet("color: gray; font-size: 10px;")
        coverage_layout.addWidget(coverage_desc)

        coverage_params_row = QHBoxLayout()
        coverage_params_row.addWidget(QLabel("Cell (m):"))
        self.coverage_cell_spin = QDoubleSpinBox()
        self.coverage_cell_spin.setRange(0.0, 100.0)
        self.coverage_cell_spin.setValue(1.0)
        self.coverage_cell_spin.setSingleStep(0.5)
        self.coverage_cell_spin.setSpecialValueText("native")
        self.coverage_cell_spin.setToolTip(
            "Output raster cell size in metres.\n"
            "0 = native resolution of the exported GeoTIFFs (larger output).")
        coverage_params_row.addWidget(self.coverage_cell_spin)
        coverage_params_row.addStretch()
        coverage_layout.addLayout(coverage_params_row)

        coverage_run_row = QHBoxLayout()
        self.coverage_btn = QPushButton("→ Generate Coverage Map")
        self.coverage_btn.clicked.connect(self.run_coverage_map)
        self.coverage_btn.setToolTip(
            "Count, per ground pixel, the number of overlapping frame GeoTIFFs.\n"
            "1 = seen once, N = seen in N frames, transparent = never covered."
        )
        self.coverage_status = QLabel("⚪")
        coverage_run_row.addWidget(self.coverage_btn)
        coverage_run_row.addWidget(self.coverage_status)
        coverage_layout.addLayout(coverage_run_row)

        coverage_add_row = QHBoxLayout()
        self.add_coverage_btn = QPushButton("→ Add Coverage Map to QGIS")
        self.add_coverage_btn.clicked.connect(self.add_coverage_map_to_qgis)
        self.add_coverage_btn.setToolTip(
            "Load the coverage raster with a graduated colour ramp.")
        self.add_coverage_status = QLabel("⚪")
        coverage_add_row.addWidget(self.add_coverage_btn)
        coverage_add_row.addWidget(self.add_coverage_status)
        coverage_layout.addLayout(coverage_add_row)

        analytics_tab_layout.addWidget(coverage_group)

        # ----- Survey Orthomosaic -----
        survey_ortho_group = QGroupBox("Survey Orthomosaic")
        survey_ortho_layout = QVBoxLayout(survey_ortho_group)

        survey_ortho_desc = QLabel(
            "One orthomosaic over every ticked flight: the same merge of the "
            "exported frame GeoTIFFs as 'Generate Orthomosaic' on the "
            "Pre-Processing tab, with the merge mode of "
            "Configuration → Orthomosaic - only across flights. The flights "
            "are merged in the order of the list above (what 'first' and "
            "'last' refer to). Requires 'Export Frames as GeoTIFF' for each "
            "flight; the frame range of the Orthomosaic tab belongs to a "
            "single flight and is not applied."
        )
        survey_ortho_desc.setWordWrap(True)
        survey_ortho_desc.setStyleSheet("color: gray; font-size: 10px;")
        survey_ortho_layout.addWidget(survey_ortho_desc)

        survey_ortho_run_row = QHBoxLayout()
        self.survey_ortho_btn = QPushButton("→ Generate Survey Orthomosaic")
        self.survey_ortho_btn.clicked.connect(self.run_survey_orthomosaic)
        self.survey_ortho_btn.setToolTip(
            "Merge the frame GeoTIFFs of all ticked flights into one "
            "orthomosaic (survey camera).")
        self.survey_ortho_status = QLabel("⚪")
        survey_ortho_run_row.addWidget(self.survey_ortho_btn)
        survey_ortho_run_row.addWidget(self.survey_ortho_status)
        survey_ortho_layout.addLayout(survey_ortho_run_row)

        survey_ortho_add_row = QHBoxLayout()
        self.add_survey_ortho_btn = QPushButton(
            "→ Add Survey Orthomosaic to QGIS")
        self.add_survey_ortho_btn.clicked.connect(
            self.add_survey_orthomosaic_to_qgis)
        self.add_survey_ortho_btn.setToolTip(
            "Load the survey orthomosaic into the survey's layer group.")
        self.add_survey_ortho_status = QLabel("⚪")
        survey_ortho_add_row.addWidget(self.add_survey_ortho_btn)
        survey_ortho_add_row.addWidget(self.add_survey_ortho_status)
        survey_ortho_layout.addLayout(survey_ortho_add_row)

        analytics_tab_layout.addWidget(survey_ortho_group)

        # ----- Distance Sampling -----
        ds_group = QGroupBox("Distance Sampling")
        ds_layout = QVBoxLayout(ds_group)

        ds_desc = QLabel(
            "Line-transect density/abundance with 95% confidence intervals, "
            "fitted from the perpendicular distances of every ticked flight "
            "(their route lengths summed into the effort). Run 'Calculate "
            "Perpendicular' (or 'Calculate Track Perpendicular') on the "
            "Processing tab first."
        )
        ds_desc.setWordWrap(True)
        ds_desc.setStyleSheet("color: gray; font-size: 10px;")
        ds_layout.addWidget(ds_desc)

        ds_params_row = QHBoxLayout()
        ds_params_row.addWidget(QLabel("Truncation (m):"))
        self.ds_truncation_spin = QDoubleSpinBox()
        self.ds_truncation_spin.setRange(0.0, 100000.0)
        self.ds_truncation_spin.setValue(0.0)
        self.ds_truncation_spin.setSingleStep(5.0)
        self.ds_truncation_spin.setSpecialValueText("auto (95th pct)")
        self.ds_truncation_spin.setToolTip(
            "Discard observations beyond this distance. 0 = automatic "
            "(95th percentile of distances).")
        ds_params_row.addWidget(self.ds_truncation_spin)
        ds_params_row.addStretch()
        ds_layout.addLayout(ds_params_row)

        self.ds_covariates_check = QCheckBox(
            "Let canopy cover and flight height explain detectability")
        self.ds_covariates_check.setToolTip(
            "Fit, beside the plain models, detection functions whose width "
            "depends on the canopy over an animal and on the height it was "
            "flown over at, and use the one with the lowest AIC. A "
            "condition that explains nothing is left out.\n"
            "Canopy cover is the tree cover around the animal (within the "
            "habitat sampling radius): it needs 'Land Cover' of the "
            "Environmental Context on every flight. Flight height is the "
            "camera's height above the animal: it needs the flight's DEM.\n"
            "At least 20 observations are needed.")
        ds_layout.addWidget(self.ds_covariates_check)

        ds_run_row = QHBoxLayout()
        self.distance_sampling_btn = QPushButton("→ Estimate Density (Distance Sampling)")
        self.distance_sampling_btn.clicked.connect(self.run_distance_sampling)
        self.distance_sampling_btn.setToolTip(
            "Fit half-normal & hazard-rate detection functions to the perpendicular\n"
            "distances and estimate density/abundance with 95% confidence intervals.\n"
            "Requires 'Calculate Perpendicular' (or track perpendicular) first."
        )
        self.distance_sampling_status = QLabel("⚪")
        ds_run_row.addWidget(self.distance_sampling_btn)
        ds_run_row.addWidget(self.distance_sampling_status)
        ds_layout.addLayout(ds_run_row)

        analytics_tab_layout.addWidget(ds_group)

        # ----- Reaction to the Drone (dock/reaction.py) -----
        analytics_tab_layout.addWidget(self._build_reaction_group())

        # ----- Position Uncertainty (dock/position.py) -----
        analytics_tab_layout.addWidget(self._build_position_group())

        # ----- Population Estimation (Transects) -----
        pop_group = QGroupBox("Population Estimation (Transects)")
        pop_layout = QVBoxLayout(pop_group)

        pop_desc = QLabel(
            "Density from the transects of the ticked flights, pooled into one "
            "count/area table: each track is assigned to the transect nearest "
            "in perpendicular distance, and the monitored area of a transect "
            "is the union of its frames' field-of-view footprints. Requires "
            "transects (Transect Splitting Tool), 'Calculate Field of View', "
            "'Calculate Track Perpendicular' and each flight's DEM."
        )
        pop_desc.setWordWrap(True)
        pop_desc.setStyleSheet("color: gray; font-size: 10px;")
        pop_layout.addWidget(pop_desc)

        pop_params_row = QHBoxLayout()
        pop_params_row.addWidget(QLabel("Truncation (m):"))
        self.pop_truncation_spin = QDoubleSpinBox()
        self.pop_truncation_spin.setRange(0.0, 100000.0)
        self.pop_truncation_spin.setValue(0.0)
        self.pop_truncation_spin.setSingleStep(5.0)
        self.pop_truncation_spin.setSpecialValueText("off (assign all)")
        self.pop_truncation_spin.setToolTip(
            "Maximum perpendicular distance between a track and its transect.\n"
            "Tracks farther from every transect are not counted.\n"
            "0 = assign every track to its nearest transect.")
        pop_params_row.addWidget(self.pop_truncation_spin)
        pop_params_row.addStretch()
        pop_layout.addLayout(pop_params_row)

        # Study area: the region the density is extrapolated to. It is NOT the
        # monitored area (the searched strips, i.e. the density's denominator)
        # - setting it to that just returns the animals already counted.
        pop_area_row = QHBoxLayout()
        pop_area_row.addWidget(QLabel("Study area (ha):"))
        self.pop_study_area_spin = QDoubleSpinBox()
        self.pop_study_area_spin.setRange(0.0, 10_000_000.0)
        self.pop_study_area_spin.setValue(0.0)
        self.pop_study_area_spin.setSingleStep(10.0)
        self.pop_study_area_spin.setSpecialValueText("off (density only)")
        self.pop_study_area_spin.setToolTip(
            "Size of the region the density is extrapolated to an abundance for.\n"
            "This is not the monitored area (the searched strips) - extrapolating\n"
            "to that would just return the animals already counted.\n"
            "0 = report densities only.")
        pop_area_row.addWidget(self.pop_study_area_spin)
        self.pop_study_area_auto_check = QCheckBox("Use flight FoV area")
        self.pop_study_area_auto_check.setToolTip(
            "Derive the study area from the flights' total field-of-view coverage:\n"
            "the union of every frame's footprint. Unlike the summed transect areas\n"
            "it counts ground seen by two transects only once, and it also includes\n"
            "frames that belong to no transect.")
        self.pop_study_area_survey_check = QCheckBox("Use survey area")
        self.pop_study_area_survey_check.setToolTip(
            "Use the survey areas minus the exclusion zones (Survey Area box\n"
            "above) as the study area.")
        self.pop_study_area_auto_check.toggled.connect(
            self._on_pop_study_area_mode_changed)
        self.pop_study_area_survey_check.toggled.connect(
            self._on_pop_study_area_mode_changed)
        pop_area_row.addWidget(self.pop_study_area_auto_check)
        pop_area_row.addWidget(self.pop_study_area_survey_check)
        pop_area_row.addStretch()
        pop_layout.addLayout(pop_area_row)

        pop_methods_row = QHBoxLayout()
        pop_methods_row.addWidget(QLabel("Methods:"))
        self.pop_naive_check = QCheckBox("Naive")
        self.pop_naive_check.setChecked(True)
        self.pop_naive_check.setToolTip(
            "Total animals divided by the total monitored area "
            "(sum(count) / sum(ha) × 100).")
        self.pop_bootstrap_check = QCheckBox("Bootstrap")
        self.pop_bootstrap_check.setChecked(True)
        self.pop_bootstrap_check.setToolTip(
            "Resample the transects with replacement, recompute the naive\n"
            "density each time and report its mean, SE and percentile 95% CI.")
        self.pop_zinb_check = QCheckBox("ZINB")
        self.pop_zinb_check.setChecked(True)
        self.pop_zinb_check.setToolTip(
            "Zero-inflated negative binomial regression (count ~ ha, constant\n"
            "zero-inflation), for transect counts with many zeros and\n"
            "overdispersion. Needs at least 4 transects of differing area.\n"
            "Its standard error is that of the summed fitted counts, with a\n"
            "log-normal 95% interval - wider than the reference R analysis\n"
            "reports, which adds the transects' uncertainties up as if they\n"
            "were independent.")
        pop_methods_row.addWidget(self.pop_naive_check)
        pop_methods_row.addWidget(self.pop_bootstrap_check)
        pop_methods_row.addWidget(self.pop_zinb_check)
        pop_methods_row.addSpacing(12)
        pop_methods_row.addWidget(QLabel("Bootstrap iterations:"))
        self.pop_n_boot_spin = QSpinBox()
        self.pop_n_boot_spin.setRange(10, 100000)
        self.pop_n_boot_spin.setValue(999)
        self.pop_n_boot_spin.setSingleStep(100)
        self.pop_n_boot_spin.setToolTip("Number of bootstrap resamples.")
        pop_methods_row.addWidget(self.pop_n_boot_spin)
        pop_methods_row.addWidget(QLabel("Seed:"))
        self.pop_seed_spin = QSpinBox()
        self.pop_seed_spin.setRange(0, 2_147_483_647)
        self.pop_seed_spin.setValue(42)
        self.pop_seed_spin.setToolTip(
            "Random seed of the bootstrap, so a run is reproducible.")
        pop_methods_row.addWidget(self.pop_seed_spin)
        pop_methods_row.addStretch()
        pop_layout.addLayout(pop_methods_row)

        pop_run_row = QHBoxLayout()
        self.population_btn = QPushButton("→ Estimate Population (Transects)")
        self.population_btn.clicked.connect(self.run_population_estimation)
        self.population_btn.setToolTip(
            "Assign the tracks to the transects, measure each transect's monitored\n"
            "area from its FoV footprints and estimate the density with the\n"
            "selected methods."
        )
        self.population_status = QLabel("⚪")
        pop_run_row.addWidget(self.population_btn)
        pop_run_row.addWidget(self.population_status)
        pop_layout.addLayout(pop_run_row)

        pop_add_row = QHBoxLayout()
        self.add_transect_areas_btn = QPushButton("→ Add Transect Areas to QGIS")
        self.add_transect_areas_btn.clicked.connect(self.add_transect_areas_to_qgis)
        self.add_transect_areas_btn.setToolTip(
            "Load the monitored area of every transect (the merged FoV footprint) "
            "as a labelled polygon layer.")
        self.add_transect_areas_status = QLabel("⚪")
        pop_add_row.addWidget(self.add_transect_areas_btn)
        pop_add_row.addWidget(self.add_transect_areas_status)
        pop_layout.addLayout(pop_add_row)

        analytics_tab_layout.addWidget(pop_group)
        analytics_tab_layout.addWidget(self._build_habitat_group())
        analytics_tab_layout.addStretch()

    def _on_pop_study_area_mode_changed(self, checked: bool):
        """'Use flight FoV area' and 'Use survey area' exclude each other."""
        sender = self.sender()
        if checked:
            other = (self.pop_study_area_survey_check
                     if sender is self.pop_study_area_auto_check
                     else self.pop_study_area_auto_check)
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
        derived = any(check.isChecked() for check in (
            self.pop_study_area_auto_check, self.pop_study_area_survey_check))
        self.pop_study_area_spin.setDisabled(derived)

    def _refresh_survey_area(self):
        """Re-read the survey area into the panel."""
        if hasattr(self, 'survey_area_panel'):
            self.survey_area_panel.refresh()

    def _refresh_survey(self):
        """Rebuild the survey's flight list (the flights changed)."""
        if hasattr(self, 'survey_panel'):
            self.survey_panel.refresh()
            self._on_survey_changed()

    def _on_survey_changed(self):
        """Flights, camera or folder changed: species and area follow."""
        if not hasattr(self, 'analytics_species_list'):
            return
        self.refresh_analytics_species()
        self._refresh_survey_area()
        if not getattr(self, "_loading_config", False):
            try:
                self.survey_panel.save(QgsProject.instance())
            except Exception:  # saved again with the project
                quiet.problem("The survey could not be saved to the project now; it is saved again with the project")

    def current_survey(self):
        """The :class:`core.survey.Survey` the tab describes right now."""
        return self.survey_panel.survey(
            species=self.analytics_species_names(),
            per_species=self.analytics_per_species_check.isChecked(),
            area_enabled=self.survey_area_panel.is_enabled())

    def _survey_config(self) -> dict:
        return self.current_survey().to_dict()

    def _checked_survey(self, check_species: bool = True):
        """The survey, validated - or ``None`` after telling the user why.

        *check_species* is off for the coverage map: it counts frames, not
        animals, so the species filter does not apply to it.
        """
        from ..core.survey import SurveyError

        if check_species and not self._check_analytics_species():
            return None
        survey = self.current_survey()
        try:
            return survey.validate()
        except SurveyError as exc:
            QMessageBox.warning(self, "Survey Analytics", str(exc))
            return None

    def _survey_prerequisites(self, survey, checks) -> bool:
        """Run *checks(flight) -> [missing, ...]* on every flight; warn and
        return False when anything is missing."""
        problems = {}
        for flight in survey.flights:
            missing = checks(flight)
            if missing:
                problems[flight.name] = missing
        if problems:
            self._warn_missing_project_files(problems)
            return False
        return True

    def _selected_population_methods(self) -> list:
        """The population estimators ticked on the Survey Analytics tab."""
        checks = (
            ("naive", "pop_naive_check"),
            ("bootstrap", "pop_bootstrap_check"),
            ("zinb", "pop_zinb_check"),
        )
        methods = [
            name for name, attr in checks
            if not hasattr(self, attr) or getattr(self, attr).isChecked()
        ]
        return methods

    def _warn_missing_project_files(self, problems: Dict[str, List[str]]):
        """Report which required files are missing in which flight(s).

        *problems* maps a flight name to what it lacks.
        """
        blocks = []
        for name, missing in problems.items():
            bullets = "\n".join(f"      • {m}" for m in missing)
            blocks.append(f"  {name}\n{bullets}")
        body = "\n\n".join(blocks)
        QMessageBox.warning(
            self, "Missing Prerequisites",
            "The analysis was aborted - required files are missing in the "
            "following flight(s):\n\n" + body)

    def _survey_source(self) -> str:
        """What the survey's tools count: ``"tracks"`` (each animal once) or
        ``"detections"`` (every box) - the Survey box's one setting."""
        if hasattr(self, "survey_panel"):
            return self.survey_panel.source()
        return "tracks"

    # ------------------------------------------------------------------
    # Which species the analytics count
    # ------------------------------------------------------------------
    def _on_analytics_species_mode_changed(self, all_species: bool):
        self.analytics_species_list.setEnabled(not all_species)
        if not all_species and self.analytics_species_list.count() == 0:
            self.refresh_analytics_species()
        self._update_analytics_species_status()

    def refresh_analytics_species(self):
        """Fill the list from the vocabularies of the survey's flights.

        Species are listed by name: ids are assigned per flight, so roe deer
        may be 3 in one flight and 7 in another, and the analytics translate
        the names into each flight's own ids.
        """
        previous = set(self._analytics_species_selection())
        self.analytics_species_list.clear()
        names = []
        if hasattr(self, 'survey_panel'):
            try:
                names = self.survey_panel.survey().species_names()
            except Exception:  # noqa: BLE001 - an empty list is the answer
                names = []

        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Keep a species ticked across a refresh, by name.
            item.setCheckState(Qt.CheckState.Checked if name in previous
                               else Qt.CheckState.Unchecked)
            self.analytics_species_list.addItem(item)

        self._update_analytics_species_status()
        return len(names)

    def _analytics_species_selection(self):
        """Names of the ticked species."""
        selection = []
        for index in range(self.analytics_species_list.count()):
            item = self.analytics_species_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                selection.append(item.text())
        return selection

    def analytics_species_names(self):
        """Species names for the analytics, or ``None`` to count everything.

        ``None`` rather than "every name" on purpose: it is what the analytics
        mean by no filter, and it keeps a species added later included
        without the selection having to be revisited.
        """
        if self.analytics_all_species_check.isChecked():
            return None
        return self._analytics_species_selection()

    def _update_analytics_species_status(self):
        separate = self.analytics_per_species_check.isChecked()
        if self.analytics_all_species_check.isChecked():
            self.analytics_species_status.setText(
                ("One result per species, for every species with something to "
                 "count." if separate else
                 "Counting every species together. 'not-an-animal' is "
                 "excluded from every analytic either way."))
            return
        if self.analytics_species_list.count() == 0:
            self.analytics_species_status.setText(
                "No species found - tick processed flights in the Survey box "
                "above.")
            return
        chosen = self._analytics_species_selection()
        if not chosen:
            self.analytics_species_status.setText(
                "Tick at least one species, or choose 'All species'.")
        elif separate:
            names = ", ".join(chosen)
            self.analytics_species_status.setText(
                f"One result each for: {names}")
        else:
            names = ", ".join(chosen)
            self.analytics_species_status.setText(
                f"Counting together: {names}")

    def _check_analytics_species(self) -> bool:
        """False when the filter would exclude everything, with a reason."""
        names = self.analytics_species_names()
        if names is not None and not names:
            QMessageBox.warning(
                self, "Survey Analytics",
                "No species are selected, so there would be nothing to "
                "count. Tick at least one species, or choose 'All species'.")
            return False
        return True

    def analytics_outputs(self, analytics_folder: str, stem: str, ext: str):
        """``(species, path)`` for the results the current mode produced.

        Keyed off the checkbox rather than off whatever is in the folder, so a
        leftover file from an earlier per-species run is not mistaken for part
        of this one.
        """
        if not self.analytics_per_species_check.isChecked():
            path = os.path.join(analytics_folder, f"{stem}.{ext}")
            return [("", path)] if os.path.isfile(path) else []

        if not os.path.isdir(analytics_folder):
            return []
        found = []
        for name in sorted(os.listdir(analytics_folder)):
            if not (name.startswith(f"{stem}_") and name.endswith(f".{ext}")):
                continue
            path = os.path.join(analytics_folder, name)
            found.append((self._analytics_species_of(path, stem, ext, name),
                          path))
        return found

    @staticmethod
    def _analytics_species_of(path: str, stem: str, ext: str, name: str) -> str:
        """The species a result belongs to - from the file, not the filename.

        The name is a slug ("roe-deer"), so the readable name is read back out
        of the result itself; the slug is only the fallback.
        """
        document = path if ext == "json" else f"{path[:-len(ext)]}json"
        try:
            with open(document, "r", encoding="utf-8") as fh:
                species = json.load(fh).get("species")
            if species:
                return str(species)
        except (OSError, ValueError):
            pass
        return name[len(stem) + 1:-(len(ext) + 1)]

    # ------------------------------------------------------------------ #
    # Survey analytics
    # ------------------------------------------------------------------ #
    def run_density_heatmap(self):
        """Run the density-heatmap generation step over the survey."""
        survey = self._checked_survey()
        if survey is None:
            return
        source = self._survey_source()
        kind = (_store_kinds.TRACKS if source == "tracks"
                else _store_kinds.GEOREFERENCED)
        msg = ("Tracks (run 'Track Animals' first)" if source == "tracks"
               else "Geo-referenced detections (run 'Geo-Reference "
               "Detections' first)")

        def checks(flight):
            ok = self.has_stage(kind, survey.modality, flight.target_folder)
            return [] if ok else [f"{survey.camera_label} {msg}"]

        if self._survey_prerequisites(survey, checks):
            self.start_worker("density_heatmap")

    def add_density_heatmap_to_qgis(self):
        """Load the survey's density raster(s) with a graduated colour ramp."""
        survey = self.current_survey()
        source = self._survey_source()
        rasters = self.analytics_outputs(
            survey.analytics_folder, f"density_{source}", "tif")
        if not rasters:
            QMessageBox.warning(
                self, "Missing Data",
                "Density heatmap has not been generated.\n"
                "Please run 'Generate Density Heatmap' first."
            )
            return

        try:
            self.update_status("add_density", "🟡 Loading...")
            for species, raster_file in rasters:
                layer_name = (f"BAMBI Density {source.capitalize()} "
                              f"({survey.camera_label})")
                if species:
                    layer_name = (f"BAMBI Density {species} "
                                  f"{source.capitalize()} "
                                  f"({survey.camera_label})")
                layer = QgsRasterLayer(raster_file, layer_name)
                if not layer.isValid():
                    raise RuntimeError(f"Failed to load raster: {raster_file}")
                self._apply_density_style(layer)
                self._add_layer_to_survey(layer)
                self.log(f"Added density heatmap layer: {layer_name}")
            self.update_status("add_density", "🟢 Added")
            self.iface.mapCanvas().refresh()
        except Exception as e:
            self.update_status("add_density", "🔴 Error")
            self.log(f"Error adding density heatmap: {e}")
            QMessageBox.critical(self, "Error", f"Failed to add density heatmap: {e}")

    def _apply_density_style(self, layer):
        """Apply a single-band pseudocolor (blue→red) ramp scaled to the data."""
        try:
            from qgis.core import (
                QgsSingleBandPseudoColorRenderer, QgsColorRampShader,
                QgsRasterShader,
            )
            stats = layer.dataProvider().bandStatistics(1)
            vmin, vmax = stats.minimumValue, stats.maximumValue
            if vmax <= vmin:
                vmax = vmin + 1.0

            ramp = QgsColorRampShader(vmin, vmax)
            ramp.setColorRampType(QgsColorRampShader.Type.Interpolated)
            steps = [
                (0.0, QColor(44, 123, 182)),
                (0.25, QColor(171, 217, 233)),
                (0.5, QColor(255, 255, 191)),
                (0.75, QColor(253, 174, 97)),
                (1.0, QColor(215, 25, 28)),
            ]
            items = [
                QgsColorRampShader.ColorRampItem(
                    vmin + frac * (vmax - vmin), color, f"{vmin + frac * (vmax - vmin):.2f}")
                for frac, color in steps
            ]
            ramp.setColorRampItemList(items)

            shader = QgsRasterShader()
            shader.setRasterShaderFunction(ramp)
            renderer = QgsSingleBandPseudoColorRenderer(
                layer.dataProvider(), 1, shader)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
        except Exception as e:  # nosec B110
            self.log(f"Warning: could not style density layer: {e}")

    def _apply_coverage_style(self, layer):
        """Colour the coverage map: classes over the frames that saw the
        ground, and a colour of its own for 0 - ground of the survey area
        that no frame saw (core.raster_styles)."""
        from ..core import raster_styles
        try:
            from qgis.core import (
                QgsSingleBandPseudoColorRenderer, QgsColorRampShader,
                QgsRasterShader,
            )
            stats = layer.dataProvider().bandStatistics(1)
            entries = raster_styles.coverage_classes(
                stats.minimumValue, stats.maximumValue)
            ramp = QgsColorRampShader(entries[0][0], entries[-1][0])
            ramp.setColorRampType(QgsColorRampShader.Type.Discrete)
            ramp.setColorRampItemList([
                QgsColorRampShader.ColorRampItem(upper, QColor(*colour), label)
                for upper, colour, label in entries])
            shader = QgsRasterShader()
            shader.setRasterShaderFunction(ramp)
            layer.setRenderer(QgsSingleBandPseudoColorRenderer(
                layer.dataProvider(), 1, shader))
            layer.triggerRepaint()
        except Exception as e:  # noqa: BLE001 - the layer is still added
            self.log(f"Warning: could not style coverage layer: {e}")

    def run_coverage_map(self):
        """Run the coverage-map generation step over the survey."""
        survey = self._checked_survey(check_species=False)
        if survey is None:
            return

        def checks(flight):
            folder = os.path.join(flight.target_folder,
                                  f"geotiffs_{survey.modality}")
            if os.path.isdir(folder):
                return []
            return [f"{survey.camera_label} frame GeoTIFFs "
                    "(run 'Export Frames as GeoTIFF' first)"]

        if self._survey_prerequisites(survey, checks):
            self.start_worker("coverage_map")

    def run_survey_orthomosaic(self):
        """Merge the frame GeoTIFFs of every ticked flight into one
        orthomosaic - the flight's orthomosaic step, across flights."""
        survey = self._checked_survey(check_species=False)
        if survey is None:
            return

        def checks(flight):
            folder = os.path.join(flight.target_folder,
                                  f"geotiffs_{survey.modality}")
            if os.path.isdir(folder):
                return []
            return [f"{survey.camera_label} frame GeoTIFFs "
                    "(run 'Export Frames as GeoTIFF' first)"]

        if self._survey_prerequisites(survey, checks):
            self.start_worker("survey_orthomosaic")

    def add_survey_orthomosaic_to_qgis(self):
        """Load the survey's orthomosaic into the survey layer group."""
        survey = self.current_survey()
        raster_file = os.path.join(survey.analytics_folder, "orthomosaic.tif")
        if not os.path.exists(raster_file):
            QMessageBox.warning(
                self, "Missing Data",
                "The survey orthomosaic has not been generated.\n"
                "Please run 'Generate Survey Orthomosaic' first."
            )
            return

        try:
            self.update_status("add_survey_orthomosaic", "🟡 Loading...")
            layer_name = f"BAMBI Survey Orthomosaic ({survey.camera_label})"
            layer = QgsRasterLayer(raster_file, layer_name)
            if not layer.isValid():
                raise RuntimeError(f"Failed to load raster: {raster_file}")
            self._add_layer_to_survey(layer)
            self.update_status("add_survey_orthomosaic", "🟢 Added")
            self.iface.mapCanvas().refresh()
            self.log(f"Added survey orthomosaic layer: {layer_name}")
        except Exception as e:
            self.update_status("add_survey_orthomosaic", "🔴 Error")
            self.log(f"Error adding survey orthomosaic: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to add survey orthomosaic: {e}")

    def add_coverage_map_to_qgis(self):
        """Load the survey's coverage raster with a graduated colour ramp."""
        survey = self.current_survey()
        raster_file = os.path.join(survey.analytics_folder, "coverage_map.tif")
        if not os.path.exists(raster_file):
            QMessageBox.warning(
                self, "Missing Data",
                "Coverage map has not been generated.\n"
                "Please run 'Generate Coverage Map' first."
            )
            return

        try:
            self.update_status("add_coverage", "🟡 Loading...")
            layer_name = f"BAMBI Coverage Map ({survey.camera_label})"
            layer = QgsRasterLayer(raster_file, layer_name)
            if not layer.isValid():
                raise RuntimeError(f"Failed to load raster: {raster_file}")
            self._apply_coverage_style(layer)
            self._add_layer_to_survey(layer)
            self.update_status("add_coverage", "🟢 Added")
            self.iface.mapCanvas().refresh()
            self.log(f"Added coverage map layer: {layer_name}")
        except Exception as e:
            self.update_status("add_coverage", "🔴 Error")
            self.log(f"Error adding coverage map: {e}")
            QMessageBox.critical(self, "Error", f"Failed to add coverage map: {e}")

    def _tracks_check(self, survey):
        """Prerequisite check: the flight has tracks of the survey camera."""
        def checks(flight):
            if self.has_stage(_store_kinds.TRACKS, survey.modality,
                              flight.target_folder):
                return []
            return [f"{survey.camera_label} tracks (run 'Track Animals' for "
                    "this camera first)"]
        return checks

    def run_track_inventory(self):
        """List every tracked individual of the survey's flights."""
        survey = self._checked_survey()
        if survey is None:
            return
        if self._survey_prerequisites(survey, self._tracks_check(survey)):
            self.start_worker("track_inventory")

    def run_census_summary(self):
        """Count the animals per species and flight (core.census)."""
        survey = self._checked_survey()
        if survey is None:
            return
        source = self._survey_source()
        if source == "tracks":
            checks = self._tracks_check(survey)
        else:
            def checks(flight):
                if self.has_stage(_store_kinds.DETECTIONS, survey.modality,
                                  flight.target_folder):
                    return []
                return [f"{survey.camera_label} detections (run 'Detect "
                        "Animals' first)"]
        if self._survey_prerequisites(survey, checks):
            self.start_worker("census_summary")

    def show_census_table(self):
        """Show the last census summary of the survey."""
        from ..core import census

        survey = self.current_survey()
        source = self._survey_source()
        path = os.path.join(survey.analytics_folder,
                            f"census_summary_{source}.json")
        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "Missing Data",
                "The census summary has not been built yet.\n"
                "Run 'Build Census Summary' first.")
            return
        try:
            result = census.read_json(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Census Summary",
                                 f"Could not read {path}: {exc}")
            return
        dialog = CensusTableDialog(
            result, csv_path=path[:-len(".json")] + ".csv", parent=self)
        dialog.setModal(False)
        dialog.show()
        self._census_dialog = dialog

    def choose_inventory_columns(self):
        """Choose the report's columns; saved with the project."""
        from ..bambi_track_inventory_dialog import BambiInventoryColumnsDialog
        from ..core import track_inventory

        dialog = BambiInventoryColumnsDialog(
            self._inventory_columns or None, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dialog.selected_columns()
        self._inventory_columns = (
            [] if set(chosen) >= set(track_inventory.COLUMNS) else chosen)
        columns = ", ".join(self._inventory_columns) if self._inventory_columns else "all"
        self.log("Track inventory columns: " + columns)

    def show_track_inventory_report(self):
        """Open the live inventory of the survey's flights in a report."""
        from qgis.PyQt.QtWidgets import QApplication
        from ..bambi_click_tool import open_track_in_viewer
        from ..bambi_track_inventory_dialog import BambiTrackInventoryDialog
        from ..core import track_inventory, track_store

        survey = self._checked_survey()
        if survey is None:
            return
        if not self._survey_prerequisites(survey, self._tracks_check(survey)):
            return
        suffix = survey.modality
        camera_label = survey.camera_label
        epsg = survey.epsg
        by_name = {f.name: f for f in survey.flights}

        # Built from the stores as they stand now, not from the files of the
        # last run: a classification applied since would otherwise be
        # missing from the report while present everywhere else.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            rows = []
            for flight in survey.flights:
                ids = survey.species_ids(flight)
                if ids is not None and not ids:
                    continue
                for row in track_inventory.build_inventory(
                        flight.target_folder, suffix, epsg=epsg,
                        species_ids=ids):
                    if survey.several:
                        row = dict(row, flight=flight.name)
                    rows.append(row)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Track Inventory",
                                 f"Could not build the inventory: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        csv_path = os.path.join(survey.analytics_folder, "track_inventory.csv")
        single = survey.flights[0]

        def _flight(name):
            return by_name.get(name, single) if name else single

        # A start or end frame in the report opens the animal in the
        # inspector, exactly as clicking its track on the map does.
        def _open_track(track_id, frame, flight_name=None):
            flight = _flight(flight_name)
            return open_track_in_viewer(
                self.iface.mainWindow(), flight.target_folder, suffix,
                int(track_id), dem_path=flight.dem_path,
                correction_path=flight.correction_path, frame=frame,
                epsg=epsg)

        def _set_approved(track_id, approved, flight_name=None):
            flight = _flight(flight_name)
            changed = track_store.set_track_attribute(
                flight.target_folder, suffix, int(track_id), "approved",
                True if approved else None)
            if changed:
                self.log(f"Track {track_id} ({flight.name}, {camera_label}) "
                         f"{'approved' if approved else 'no longer approved'}")

        def _columns_changed(columns):
            self._inventory_columns = list(columns)
            self.log("Track inventory columns: " + (", ".join(columns) if columns else "all"))

        dialog = BambiTrackInventoryDialog(
            rows, camera_label,
            csv_path=csv_path if os.path.isfile(csv_path) else None,
            open_track=_open_track, set_approved=_set_approved,
            columns=self._inventory_columns or None,
            on_columns_changed=_columns_changed, parent=self)
        dialog.setModal(False)
        dialog.show()
        # Keep a reference: a non-modal dialog with only a local name would
        # be collected the moment this method returns.
        self._track_inventory_dialog = dialog
        self._track_inventory_context = (
            {f.key: f.name for f in survey.flights}, suffix, epsg,
            survey.several)
        # What the reviewer does in the inspector - approve, delete a box,
        # delete a track - shows up in the open report.
        self._connect_viewer_signals()

    def _connect_viewer_signals(self):
        """Follow the inspector's review actions. The viewer is a singleton
        and is connected once - once per viewer: should the one that was
        followed be gone, the one that took its place is followed."""
        from ..bambi_feature_viewer import FeatureViewerDialog

        viewer = FeatureViewerDialog.get_instance(self.iface.mainWindow())
        if getattr(self, "_viewer_followed", None) is viewer:
            return
        viewer.trackDeleted.connect(self._on_viewer_track_deleted)
        viewer.detectionDeleted.connect(self._on_viewer_detection_deleted)
        viewer.trackApproved.connect(self._on_viewer_track_approved)
        self._viewer_followed = viewer

    def _inventory_dialog_for(self, target_folder: str, modality: str):
        """``(dialog, flight name)`` when the open report shows this camera
        of this flight, else ``(None, None)``."""
        dialog = getattr(self, "_track_inventory_dialog", None)
        context = getattr(self, "_track_inventory_context", None)
        if dialog is None or context is None or not dialog.isVisible():
            return None, None
        flights, suffix, _epsg, several = context
        name = flights.get(os.path.normcase(os.path.abspath(target_folder)))
        if suffix != modality or name is None:
            return None, None
        return dialog, (name if several else None)

    def _on_viewer_track_deleted(self, target_folder: str, modality: str,
                                 track_id: int):
        camera_label = "Thermal" if modality == "t" else "RGB"
        self.log(f"Track {track_id} ({camera_label}) deleted from the project "
                 "in the inspector")
        self._review_queue_forget(target_folder, modality, track_id)
        dialog, flight = self._inventory_dialog_for(target_folder, modality)
        if dialog is not None:
            dialog.remove_track(track_id, flight)

    def _on_viewer_detection_deleted(self, target_folder: str, modality: str,
                                     detection_id: int, track_id: int):
        camera_label = "Thermal" if modality == "t" else "RGB"
        note = f" (track {track_id})" if track_id >= 0 else ""
        self.log(f"Detection {detection_id} ({camera_label}){note} deleted "
                 "from the project in the inspector")
        dialog, flight = self._inventory_dialog_for(target_folder, modality)
        if dialog is None or track_id < 0:
            return
        from ..core import track_inventory

        epsg = self._track_inventory_context[2]
        try:
            rows = track_inventory.build_inventory(
                target_folder, modality, epsg=epsg, track_ids=[track_id])
        except Exception as exc:  # noqa: BLE001 - the report is a view
            self.log(f"Could not refresh track {track_id} in the report: {exc}")
            return
        if rows:
            row = dict(rows[0], flight=flight) if flight else rows[0]
            dialog.update_track(row)
        else:
            dialog.remove_track(track_id, flight)

    def _on_viewer_track_approved(self, target_folder: str, modality: str,
                                  track_id: int, approved: bool):
        camera_label = "Thermal" if modality == "t" else "RGB"
        self.log(f"Track {track_id} ({camera_label}) "
                 f"{'approved' if approved else 'no longer approved'}")
        if approved:
            self._review_queue_forget(target_folder, modality, track_id)
        dialog, flight = self._inventory_dialog_for(target_folder, modality)
        if dialog is not None:
            dialog.set_track_approved(track_id, approved, flight)

    def add_track_inventory_to_qgis(self):
        """Load the survey inventory as a point layer at each animal's last
        position, with a ``flight`` column."""
        from ..core import track_inventory

        survey = self.current_survey()
        json_path = os.path.join(survey.analytics_folder,
                                 "track_inventory.json")
        if not os.path.isfile(json_path):
            QMessageBox.warning(
                self, "Missing Data",
                "The track inventory has not been written yet.\n"
                "Run 'List Tracked Individuals' first.")
            return

        camera_label = survey.camera_label
        columns = ["flight"] + list(track_inventory.COLUMNS)
        try:
            self.update_status("add_track_inventory", "🟡 Loading...")
            rows = track_inventory.read_json(json_path)
            epsg = survey.epsg or self.get_config().get("target_epsg", 32633)
            layer = QgsVectorLayer(
                f"Point?crs=EPSG:{epsg}",
                f"BAMBI Track Inventory ({camera_label})", "memory")
            provider = layer.dataProvider()

            def _field(name):
                sample = next((r[name] for r in rows
                               if r.get(name) is not None), None)
                if isinstance(sample, bool):
                    return QgsField(name, FieldType.Bool)
                if isinstance(sample, int):
                    return QgsField(name, FieldType.LongLong)
                if isinstance(sample, float):
                    return QgsField(name, FieldType.Double)
                return QgsField(name, FieldType.String)

            provider.addAttributes([_field(name) for name in columns])
            layer.updateFields()

            features = []
            for row in rows:
                feature = QgsFeature(layer.fields())
                x, y = row.get("end_x"), row.get("end_y")
                if x is None:
                    x, y = row.get("start_x"), row.get("start_y")
                if x is not None and y is not None:
                    feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                feature.setAttributes([
                    row.get(name) if row.get(name) is not None else None
                    for name in columns])
                features.append(feature)
            provider.addFeatures(features)
            layer.updateExtents()

            layer = self._persist_memory_layer(
                layer, f"Track_Inventory_{survey.modality}", "analytics_layers",
                display_name=f"BAMBI Track Inventory ({camera_label})",
                base_folder=survey.folder)
            layer.setCustomProperty("bambi_layer_type", "track_inventory")
            self._add_layer_to_survey(layer)
            self.update_status("add_track_inventory", "🟢 Added")
            self.iface.mapCanvas().refresh()
            self.log(f"Added track inventory layer with {len(features)} "
                     "individual(s)")
        except Exception as e:
            self.update_status("add_track_inventory", "🔴 Error")
            self.log(f"Error adding track inventory: {e}")
            QMessageBox.critical(self, "Error",
                                 f"Failed to add the track inventory: {e}")

    def run_distance_sampling(self):
        """Run the distance-sampling estimation over the survey's flights."""
        from ..core.survey import route_suffix

        survey = self._checked_survey()
        if survey is None:
            return
        source = self._survey_source()
        m = survey.modality
        if source == "tracks":
            perp_name = f"perpendicular_tracks_{m}.json"
            perp_msg = ("Track perpendicular distances "
                        "(run 'Calculate Track Perpendicular')")
        else:
            perp_name = f"perpendicular_{m}.json"
            perp_msg = "Perpendicular distances (run 'Calculate Perpendicular')"

        def checks(flight):
            missing = []
            if route_suffix(flight.target_folder, m, perp_name) is None:
                missing.append(f"{survey.camera_label} {perp_msg}")
            if route_suffix(flight.target_folder, m) is None:
                missing.append("Flight route (run 'Generate Flight Route')")
            return missing

        if self._survey_prerequisites(survey, checks):
            self.start_worker("distance_sampling")

    def _show_distance_sampling_results(self):
        """Read the distance-sampling JSON(s) and show a summary dialog."""
        survey = self.current_survey()
        source = self._survey_source()
        results = self.analytics_outputs(
            survey.analytics_folder, f"distance_sampling_{source}", "json")
        if not results:
            return

        sections = []
        for species, result_file in results:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    r = json.load(f)
            except Exception as e:
                self.log(f"Could not read distance-sampling results: {e}")
                continue
            if species:
                sections.append(f"<h2>{species}</h2>")
            sections.append(
                self._distance_sampling_html(r, result_file))
        if not sections:
            return

        self._show_analytics_dialog(
            "Distance-Sampling Results", "".join(sections))

    def _show_analytics_dialog(self, title: str, html: str):
        """One read-only dialog for an analytic's summary."""
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(480)
        layout = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml(html)
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()

    def _distance_sampling_html(self, r, result_file):
        """The summary table for one distance-sampling result."""
        d_ci = r.get("density_ci95", [0, 0])
        n_ci = r.get("abundance_ci95", [0, 0])
        rows = "".join(
            f"<tr><td>{m['name']}</td><td align='right'>{m['esw_m']:.2f}</td>"
            f"<td align='right'>{m['aic']:.2f}</td></tr>"
            for m in r.get("models", [])
        )

        projects = r.get("projects", [])
        projects_section = ""
        if len(projects) > 1:
            proj_rows = "".join(
                f"<tr><td>{p.get('name') or os.path.basename(os.path.normpath(p['target_folder']))}</td>"
                f"<td align='right'>{p.get('n', 0)}</td>"
                f"<td align='right'>{p.get('n_lines', 1)}</td>"
                f"<td align='right'>{p.get('transect_length_m', 0):.0f}</td></tr>"
                for p in projects
            )
            projects_section = f"""
        <h4>Flights pooled ({len(projects)})</h4>
        <table cellpadding='4'>
          <tr><th align='left'>Flight</th><th align='right'>Observations</th>
              <th align='right'>Lines</th>
              <th align='right'>Surveyed line (m)</th></tr>
          {proj_rows}
        </table>
        """

        # What counted as line, and how its encounter rate was judged
        # (core.distance_effort). Absent from results written before 9.1.
        lines_text = ""
        if r.get("n_lines"):
            lines_text = f" in {r['n_lines']} line(s)"
        transit_row = (
            f"<tr><td><b>Seen in transit</b></td><td align='right' "
            f"style='color:#c80'>{r['n_off_effort']} (not counted)</td></tr>"
            if r.get("n_off_effort") else "")
        encounter_row = ""
        if r.get("encounter_rate_cv") is not None:
            how = ("measured between the lines"
                   if r.get("encounter_rate_variance") == "between lines"
                   else "Poisson - a single line")
            encounter_row = (
                f"<tr><td><b>Encounter-rate CV</b></td><td align='right'>"
                f"{r['encounter_rate_cv'] * 100:.0f} % ({how})</td></tr>")

        # An observation is a group of animals that were together
        # (core.groups); results written before 9.1 know nothing of them.
        grouped = (r.get("n_animals") or 0) > (r.get("n") or 0)
        groups_rows = ""
        if grouped:
            groups_rows = (
                "<tr><td><b>Animals in them</b></td><td align='right'>"
                f"{r['n_animals']}</td></tr>"
                "<tr><td><b>Mean group size</b></td><td align='right'>"
                f"{r.get('mean_group_size') or 0:.2f} "
                f"(SE {r.get('group_size_se') or 0:.2f}, largest "
                f"{r.get('largest_group')})</td></tr>")
        group_density_row = (
            "<tr><td><b>Density of groups</b></td><td align='right'>"
            f"{r.get('group_density_per_km2') or 0:.3f} /km²</td></tr>"
            if grouped else "")
        what = " groups" if grouped else ""
        of_animals = " of animals" if grouped else ""
        detectability = self._distance_covariates_html(r.get("covariates"))

        html = f"""
        <h3>Distance-Sampling Estimate ({r.get('source', '')})</h3>
        <table cellpadding='4'>
          <tr><td><b>Observations (n)</b></td><td align='right'>{r.get('n', 0)}{what}
              (of {r.get('n_before_truncation', 0)})</td></tr>
          {groups_rows}
          <tr><td><b>Surveyed line (L)</b></td>
              <td align='right'>{r.get('transect_length_m', 0):.0f} m{lines_text}</td></tr>
          {transit_row}
          {encounter_row}
          <tr><td><b>Truncation (w)</b></td>
              <td align='right'>{r.get('truncation_m', 0):.2f} m</td></tr>
          <tr><td><b>Best model</b></td><td align='right'>{r.get('best_model', '')}</td></tr>
          <tr><td><b>Effective strip width</b></td>
              <td align='right'>{r.get('effective_strip_width_m', 0):.2f} m</td></tr>
          <tr><td><b>Detection probability (p)</b></td>
              <td align='right'>{r.get('detection_probability', 0):.3f}</td></tr>
          {group_density_row}
          <tr><td><b>Density{of_animals}</b></td>
              <td align='right'>{r.get('density_per_km2', 0):.3f} /km²
              (95% CI {d_ci[0]:.3f}–{d_ci[1]:.3f})</td></tr>
          <tr><td><b>Abundance in covered area</b></td>
              <td align='right'>{r.get('abundance_in_covered_area', 0):.1f}
              (95% CI {n_ci[0]:.1f}–{n_ci[1]:.1f})</td></tr>
          <tr><td><b>Covered area</b></td>
              <td align='right'>{r.get('covered_area_km2', 0):.3f} km²</td></tr>
        </table>
        {detectability}
        <h4>Model comparison</h4>
        <table cellpadding='4'>
          <tr><th align='left'>Model</th><th align='right'>ESW (m)</th>
              <th align='right'>AIC</th></tr>
          {rows}
        </table>
        {projects_section}
        <p style='color:gray;font-size:11px'>{r.get('notes', '')}</p>
        <p style='color:gray;font-size:11px'>Saved to: {result_file}</p>
        """
        return html

    @staticmethod
    def _distance_covariates_html(covariates) -> str:
        """What canopy cover and flight height did to detectability
        (core.detection_function); nothing where they were not asked for."""
        if not covariates:
            return ""
        from ..core import detection_function

        lines = [detection_function.effect_sentence(row)
                 for row in covariates.get("effects") or []]
        used = set(covariates.get("in_best_model") or [])
        for name in covariates.get("considered") or {}:
            if name not in used:
                lines.append(
                    f"{detection_function.label(name).capitalize()}: "
                    "offered to the model, and the distances are described "
                    "as well without it.")
        for name, reason in (covariates.get("not_used") or {}).items():
            lines.append(f"{detection_function.label(name).capitalize()} "
                         f"is not used: {reason}.")
        if not lines:
            return ""
        from html import escape

        items = "".join(f"<li>{escape(line, quote=False)}</li>"
                        for line in lines)
        return f"<h4>What explains detectability</h4><ul>{items}</ul>"

    def run_population_estimation(self):
        """Run the transect population estimation over the survey's flights."""
        from ..core.survey import route_suffix

        survey = self._checked_survey()
        if survey is None:
            return
        if not self.get_config().get("pop_methods"):
            QMessageBox.warning(
                self, "No Method Selected",
                "Please tick at least one estimation method "
                "(Naive, Bootstrap or ZINB).")
            return

        from ..core import analytics_source

        m = survey.modality
        label = survey.camera_label
        # What the survey counts decides which distances are needed.
        if self._survey_source() == "tracks":
            perp_name = f"perpendicular_tracks_{m}.json"
            perp_missing = (f"{label} track perpendicular distances "
                            "(run 'Calculate Track Perpendicular')")
        else:
            perp_name = f"perpendicular_{m}.json"
            perp_missing = (f"{label} perpendicular distances of the "
                            "detections (run 'Calculate Perpendicular')")

        def checks(flight):
            folder = flight.target_folder
            missing = []
            if not os.path.exists(os.path.join(
                    folder, f"transects_{m}", "transects.json")):
                missing.append(f"{label} transect definitions "
                               "(define them with the Transect Splitting Tool)")
            if not self.has_stage(_store_kinds.FOV, m, folder):
                missing.append(f"{label} FoV footprints "
                               "(run 'Calculate Field of View')")
            # A flight on which nothing was found has no distances and can
            # have none; it counts with zero animals.
            if route_suffix(folder, m, perp_name) is None \
                    and not analytics_source.has_no_animals(folder, m):
                missing.append(perp_missing)
            if not os.path.exists(os.path.join(folder, f"poses_{m}.json")):
                missing.append(f"{label} camera poses (poses_{m}.json - run "
                               "frame extraction)")
            if not flight.dem and not flight.dem_path:
                missing.append("DEM metadata (dem.json) - set the DEM on the "
                               "flight's Processing tab")
            return missing

        if self._survey_prerequisites(survey, checks):
            self.start_worker("population_estimation")

    def _show_population_results(self):
        """Read the population-estimate JSON(s) and show a summary dialog."""
        survey = self.current_survey()
        results = self.analytics_outputs(
            survey.analytics_folder, "population_estimate", "json")
        if not results:
            return

        sections = []
        for species, result_file in results:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    r = json.load(f)
            except Exception as e:
                self.log(f"Could not read population-estimation results: {e}")
                continue
            if species:
                sections.append(f"<h2>{species}</h2>")
            sections.append(self._population_html(r, result_file))
        if not sections:
            return

        self._show_analytics_dialog(
            "Population Estimation Results", "".join(sections))

    def _population_html(self, r, result_file):
        """The summary table for one population estimate."""
        study_ha = r.get("study_area_ha", 0) or 0
        method_labels = {"naive": "Naive", "bootstrap": "Bootstrap", "zinb": "ZINB"}
        est_rows = ""
        for key, est in r.get("estimates", {}).items():
            label = method_labels.get(key, key)
            density = est.get("density_per_100ha")
            if density is None:
                est_rows += (
                    f"<tr><td>{label}</td><td colspan='3' style='color:#c33'>"
                    f"failed - {est.get('error', 'unknown error')}</td></tr>")
                continue
            ci = est.get("ci95")
            ci_txt = f"{ci[0]:.2f} – {ci[1]:.2f}" if ci else "–"
            abundance = est.get("abundance_study_area")
            ab_txt = f"{abundance:.1f}" if abundance is not None else "–"
            note = (f"<br><span style='color:#c80'>{est['error']}</span>"
                    if est.get("error") else "")
            est_rows += (
                f"<tr><td>{label}{note}</td>"
                f"<td align='right'>{density:.2f}</td>"
                f"<td align='right'>{ci_txt}</td>"
                f"<td align='right'>{ab_txt}</td></tr>")

        transect_rows = "".join(
            f"<tr><td>{t['name']}</td>"
            f"<td align='right'>{t['start_frame']}–{t['end_frame']}</td>"
            f"<td align='right'>{t['length_m']:.0f}</td>"
            f"<td align='right'>{t['area_ha']:.2f}</td>"
            f"<td align='right'>{t['count']}</td></tr>"
            for t in r.get("transects", [])
        )

        source_labels = {
            "flight_fov": "from the flights' FoV coverage",
            "survey_area": "the survey areas minus exclusion zones",
            "manual": "entered manually",
        }
        source = source_labels.get(r.get("study_area_source", ""), "")
        study_row = (
            f"<tr><td><b>Study area</b></td>"
            f"<td align='right'>{study_ha:.1f} ha"
            f"<span style='color:gray'> ({source})</span></td></tr>"
            if study_ha > 0 else "")
        flight_ha = r.get("flight_fov_area_ha")
        flight_row = (
            f"<tr><td><b>Flight FoV coverage</b></td>"
            f"<td align='right'>{flight_ha:.2f} ha</td></tr>"
            if flight_ha else "")
        no_area = r.get("n_transects_without_area", 0)
        no_area_row = (
            f"<tr><td><b>Transects without area</b></td>"
            f"<td align='right' style='color:#c80'>{no_area} (excluded)</td></tr>"
            if no_area else "")

        projects = r.get("projects", [])
        projects_section = ""
        if len(projects) > 1:
            proj_rows = "".join(
                f"<tr><td>{p.get('name', '')}</td>"
                f"<td align='right'>{p.get('n_transects', 0)}</td>"
                f"<td align='right'>{int(p.get('total_count', 0))}</td>"
                f"<td align='right'>{p.get('total_ha', 0):.2f}</td>"
                f"<td align='right'>{p.get('dem_origin_source', '')}</td></tr>"
                for p in projects
            )
            projects_section = f"""
        <h4>Flights pooled ({len(projects)})</h4>
        <table cellpadding='4'>
          <tr><th align='left'>Flight</th><th align='right'>Transects</th>
              <th align='right'>Counted</th><th align='right'>Area (ha)</th>
              <th align='right'>DEM origin</th></tr>
          {proj_rows}
        </table>
        <p style='color:gray;font-size:11px'>DEM origin: where each flight's
        mesh-to-world origin was read from (its dem.json, or the JSON beside
        its DEM mesh).</p>
        """

        html = f"""
        <h3>Transect Population Estimate ({r.get('camera', '')})</h3>
        <table cellpadding='4'>
          <tr><td><b>Transects</b></td>
              <td align='right'>{r.get('n_transects', 0)}
              ({r.get('n_zero_transects', 0)} with zero counts)</td></tr>
          <tr><td><b>Animals (tracks)</b></td>
              <td align='right'>{int(r.get('total_count', 0))} counted
              (of {r.get('n_tracks', 0)})</td></tr>
          <tr><td><b>Not counted</b></td>
              <td align='right'>
              {r.get('n_tracks_outside_fov', 0)} outside every field of view,
              {r.get('n_tracks_truncated', 0)} beyond the truncation</td></tr>
          <tr><td><b>Monitored area</b></td>
              <td align='right'>{r.get('total_ha', 0):.2f} ha
              <span style='color:gray'>(sum of the transects)</span></td></tr>
          {flight_row}
          <tr><td><b>Truncation</b></td>
              <td align='right'>{'off' if not r.get('truncation_m') else f"{r['truncation_m']:.1f} m"}</td></tr>
          {study_row}
          {no_area_row}
        </table>
        <h4>Estimates</h4>
        <table cellpadding='4'>
          <tr><th align='left'>Method</th>
              <th align='right'>Density (/100 ha)</th>
              <th align='right'>95% CI</th>
              <th align='right'>{'Abundance' if study_ha > 0 else '-'}</th></tr>
          {est_rows}
        </table>
        {projects_section}
        <h4>Per transect</h4>
        <table cellpadding='4'>
          <tr><th align='left'>Name</th><th align='right'>Frames</th>
              <th align='right'>Length (m)</th><th align='right'>Area (ha)</th>
              <th align='right'>Count</th></tr>
          {transect_rows}
        </table>
        <p style='color:gray;font-size:11px'>{r.get('notes', '')}</p>
        <p style='color:gray;font-size:11px'>Saved to: {result_file}</p>
        """
        return html

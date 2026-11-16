# -*- coding: utf-8 -*-
"""
BAMBI Wildlife Detection - Dock Widget
=======================================

The plugin's main panel. When the plugin runs, ``BambiDockWidget`` is one
widget and one class: every control is an attribute of it, and the tools,
the tests and the settings table (``core.config_schema``) reach them there.

Its code is kept by tab, in the ``dock`` package - one module per tab or
concern, each a class that is mixed in below. See ``dock/__init__.py`` for
what is where. What is left here is the frame around the tabs.
"""

import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar,
    QScrollArea, QToolButton,
)

from .bambi_processing import BambiProcessor
from .bambi_environment_panel import EnvironmentDockMixin
from .gui_utils import FitTabWidget
from . import gui_utils as gui_utils_hf

from .dock.input_tab import InputTabMixin
from .dock.flights import FlightsMixin
from .dock.configuration_tab import ConfigurationTabMixin
from .dock.correction_settings import CorrectionSettingsMixin
from .dock.tracking_settings import TrackingSettingsMixin
from .dock.classification_settings import ClassificationSettingsMixin
from .dock.step_tabs import StepTabsMixin
from .dock.step_status import StepStatusMixin
from .dock.analytics_tab import AnalyticsTabMixin
from .dock.transect_layers import TransectLayersMixin
from .dock.group_layers import GroupLayersMixin
from .dock.grid_density import GridDensityMixin
from .dock.trend import TrendMixin
from .dock.accuracy import AccuracyMixin
from .dock.review_queue import ReviewQueueMixin
from .dock.result_layers import ResultLayersMixin
from .dock.flight_route_layers import FlightRouteLayersMixin
from .dock.layer_groups import LayerGroupsMixin
from .dock.settings import SettingsMixin
from .dock.inspectors import InspectorsMixin
from .dock.run_panel import RunPanelMixin
from .dock.step_runner import StepRunner

# What was importable from this module before its code was split by tab, and
# still is: the tests and the tools take these names from here.
from .bambi_calibrations import THERMAL_CALIBRATIONS, RGB_CALIBRATIONS  # noqa: F401
from .core.hf_access import DEFAULT_BACKBONE as HF_DEFAULT_BACKBONE  # noqa: F401
from .gui_utils import PLUGIN_SCOPE  # noqa: F401
from .dock.range_dialogs import CorrectionRangeDialog, FrameRangeDialog  # noqa: F401

# The Hugging Face token lives in the QGIS settings rather than in the project
# configuration: it is a user credential, and a project file gets shared.
_HF_TOKEN_SETTING = gui_utils_hf.HF_TOKEN_SETTING  # one key with the Segmentation tool


class BambiDockWidget(
        InputTabMixin,
        FlightsMixin,
        ConfigurationTabMixin,
        CorrectionSettingsMixin,
        TrackingSettingsMixin,
        ClassificationSettingsMixin,
        StepTabsMixin,
        StepStatusMixin,
        AnalyticsTabMixin,
        TransectLayersMixin,
        GroupLayersMixin,
        GridDensityMixin,
        TrendMixin,
        AccuracyMixin,
        ReviewQueueMixin,
        ResultLayersMixin,
        FlightRouteLayersMixin,
        LayerGroupsMixin,
        SettingsMixin,
        InspectorsMixin,
        RunPanelMixin,
        EnvironmentDockMixin,
        QDockWidget):
    """Main dock widget for the BAMBI Wildlife Detection plugin."""

    def __init__(self, iface, parent=None):
        """Initialize the dock widget.

        :param iface: QGIS interface instance
        :param parent: Parent widget
        """
        super().__init__("Bambi - QGIS Integration", parent)
        self.iface = iface
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        # Processing state: the steps themselves, and what runs them one at
        # a time on a thread of its own (self.worker and self.worker_thread
        # are the runner's - see dock/run_panel.py).
        self.processor = BambiProcessor()
        self.steps = StepRunner()

        # Track if initial check has been done
        self._initial_check_done = False

        # Flag to prevent recursive saves during config loading
        self._loading_config = False

        # Inspector map tools (lazy-created on first activation)
        self._click_tool = None
        self._fov_click_tool = None
        self._fov_georef_click_tool = None

        # Toolbar QAction references - set via set_inspector_actions()
        self._inspector_action = None
        self._fov_inspector_action = None
        self._fov_georef_inspector_action = None
        self._segmentation_tool_action = None

        # Setup UI
        self.setup_ui()

        # Snapshot pristine widget state so a new project can reset to defaults
        self._snapshot_default_config()
        # ... and the same by settings key: what a flight added with
        # "default configuration" starts from (core.flights.new_flight_values).
        self._pristine_config = self._current_config_values()

        # Connect to project signals for config persistence
        self._connect_project_signals()

        # Keep the inspector button in sync when the user switches to another map tool
        self.iface.mapCanvas().mapToolSet.connect(self._on_map_tool_changed)

        # Load config if project already has saved config
        self._loading_config = True
        try:
            self.load_config_from_project()
        finally:
            self._loading_config = False
        self._refresh_survey_area()

    def showEvent(self, event):
        """Handle widget show event to check for existing layers."""
        super().showEvent(event)

        # Check for existing QGIS layers when first shown
        if not self._initial_check_done:
            self._initial_check_done = True
            self._check_existing_qgis_layers()

            # Also check target folder if already set
            target_folder = self.target_folder_edit.text().strip()
            if target_folder and os.path.isdir(target_folder):
                self._check_existing_outputs(target_folder)

    def setup_ui(self):
        """Setup the user interface."""
        # Main widget and layout
        main_widget = QWidget()
        self.setWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Progress bar with abort button - pinned above the tabs (outside the
        # scroll area) so the running step's progress is always in view.
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar, stretch=1)

        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setToolTip("Cancel the current running process")
        self.abort_btn.setEnabled(False)
        self.abort_btn.setFixedWidth(70)
        self.abort_btn.clicked.connect(self._abort_current_process)
        progress_layout.addWidget(self.abort_btn)

        main_layout.addLayout(progress_layout)

        # Create scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        # Create main tab widget
        # Fitted to the page in front (gui_utils.FitTabWidget): a plain tab
        # widget is as tall as its longest page, which put a long stretch of
        # nothing between every shorter tab and the log below.
        main_tabs = FitTabWidget()
        self.main_tabs = main_tabs
        scroll_layout.addWidget(main_tabs)

        # One builder per tab, in the order of the tabs. Each makes its
        # widgets as attributes of this dock widget, as they always were.
        self._build_input_tab(main_tabs)
        self._build_configuration_tab(main_tabs)
        self._build_step_tabs(main_tabs)
        self._build_run_panel(scroll_layout)
        self._build_analytics_tab(main_tabs)

    def _make_info_button(self, slot) -> QToolButton:
        """A round "?" button opening *slot*, styled like the original one."""
        button = QToolButton()
        button.setText("?")
        button.setFixedSize(20, 20)
        button.setStyleSheet(
            "QToolButton {"
            "  border-radius: 10px;"
            "  border: 1px solid palette(mid);"
            "  background: palette(button);"
            "  font-weight: bold;"
            "  font-size: 11px;"
            "}"
            "QToolButton:hover { background: palette(light); }"
            "QToolButton:pressed { background: palette(mid); }"
        )
        button.clicked.connect(slot)
        return button

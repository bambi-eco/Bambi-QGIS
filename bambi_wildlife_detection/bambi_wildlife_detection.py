# -*- coding: utf-8 -*-
"""
BAMBI Wildlife Detection - Main Plugin Class
=============================================

This module contains the main plugin class that integrates with QGIS.
"""

import os
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .bambi_dock_widget import BambiDockWidget


class BambiWildlifeDetection:
    """QGIS Plugin Implementation for wildlife detection in drone videos."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)

        # Initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            f'BambiWildlifeDetection_{locale}.qm')

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr('&Bambi - QGIS Integration')
        self.toolbar = self.iface.addToolBar('BambiWildlifeDetection')
        self.toolbar.setObjectName('BambiWildlifeDetection')

        # Dock widget
        self.dock_widget = None
        self.dock_widget_action = None

        # Inspector toolbar actions (created in initGui)
        self.inspector_action = None
        self.fov_inspector_action = None
        self.fov_georef_inspector_action = None
        self.correction_wizard_action = None
        self.camera_calibration_action = None
        self.thermal_viewer_action = None
        self._thermal_viewer_dlg = None
        self.dependency_manager_action = None
        self._dependency_manager_dlg = None
        self.flight_planner_action = None
        self._flight_planner_dlg = None

    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        return QCoreApplication.translate('BambiWildlifeDetection', message)

    def add_action(
            self,
            icon_path,
            text,
            callback,
            enabled_flag=True,
            add_to_menu=True,
            add_to_toolbar=True,
            status_tip=None,
            whats_this=None,
            parent=None,
            checkable=False):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action.
        :param text: Text that should be shown in menu items for this action.
        :param callback: Function to be called when the action is triggered.
        :param enabled_flag: A flag indicating if the action should be enabled.
        :param add_to_menu: Flag indicating whether the action should be added to the menu.
        :param add_to_toolbar: Flag indicating whether the action should be added to the toolbar.
        :param status_tip: Optional text to show in a popup when mouse hovers over the action.
        :param whats_this: Optional text to show in the status bar.
        :param parent: Parent widget for the new action.
        :param checkable: If True, the action will be checkable.

        :returns: The action that was created.
        """
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)
        action.setCheckable(checkable)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(self.menu, action)

        self.actions.append(action)
        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        icon_path = os.path.join(self.plugin_dir, 'icons', 'icon.png')

        # Add main action to show/hide the dock widget
        self.dock_widget_action = self.add_action(
            icon_path,
            text=self.tr('Bambi - QGIS Integration'),
            callback=self.toggle_dock_widget,
            parent=self.iface.mainWindow(),
            checkable=True,
            status_tip=self.tr('Open Bambi - QGIS Integration panel'))

        # Camera Calibration Wizard
        _calib_icon = os.path.join(self.plugin_dir, 'icons', 'icon_camera_calib.png')
        if not os.path.isfile(_calib_icon):
            _calib_icon = os.path.join(self.plugin_dir, 'icons', 'icon_calibration.png')
        self.camera_calibration_action = self.add_action(
            _calib_icon,
            text=self.tr('Camera Calibration Wizard'),
            callback=self._on_camera_calibration,
            parent=self.iface.mainWindow(),
            add_to_menu=True,
            status_tip=self.tr(
                'Open the camera calibration wizard (single-camera SfM or '
                'stereo RGB+thermal)'))

        # Thermal Image Viewer
        _thermal_icon = os.path.join(self.plugin_dir, 'icons', 'icon_thermal.png')
        if not os.path.isfile(_thermal_icon):
            _thermal_icon = icon_path
        self.thermal_viewer_action = self.add_action(
            _thermal_icon,
            text=self.tr('Thermal Image Viewer'),
            callback=self._on_thermal_viewer,
            parent=self.iface.mainWindow(),
            add_to_menu=True,
            status_tip=self.tr('Open the DJI radiometric thermal image viewer'))

        # Correction Wizard (between main icon and inspector tools)
        self.correction_wizard_action = self.add_action(
            os.path.join(self.plugin_dir, 'icons', 'icon_correction.png'),
            text=self.tr('Correction Wizard'),
            callback=self._on_correction_wizard,
            parent=self.iface.mainWindow(),
            add_to_menu=False,
            status_tip=self.tr(
                'Open the correction calibration wizard'))

        # Inspector: Detection / Track
        self.inspector_action = self.add_action(
            os.path.join(self.plugin_dir, 'icons', 'icon_inspector_det.png'),
            text=self.tr('Inspector: Click Detection / Track'),
            callback=self._on_inspector_toggled,
            parent=self.iface.mainWindow(),
            checkable=True,
            add_to_menu=False,
            status_tip=self.tr(
                'Click a detection or track bounding box to view the frame image'))

        # Inspector: FoV (simple viewer)
        self.fov_inspector_action = self.add_action(
            os.path.join(self.plugin_dir, 'icons', 'icon_inspector_fov.png'),
            text=self.tr('Inspector: Click FoV'),
            callback=self._on_fov_inspector_toggled,
            parent=self.iface.mainWindow(),
            checkable=True,
            add_to_menu=False,
            status_tip=self.tr(
                'Click a Field of View polygon to view the corresponding frame image'))

        # Inspector: FoV with geo-referenced click projection
        self.fov_georef_inspector_action = self.add_action(
            os.path.join(self.plugin_dir, 'icons', 'icon_inspector_fov_georef.png'),
            text=self.tr('Inspector: Click FoV (Geo-Referenced)'),
            callback=self._on_fov_georef_inspector_toggled,
            parent=self.iface.mainWindow(),
            checkable=True,
            add_to_menu=False,
            status_tip=self.tr(
                'Click a Field of View polygon to view the frame image with '
                'the clicked map position projected into the image space'))

        # Flight Strategy Planner
        self.flight_planner_action = self.add_action(
            os.path.join(self.plugin_dir, 'icons', 'icon_flight_planner.png'),
            text=self.tr('Random Flight Strategy Planner'),
            callback=self._on_flight_planner,
            parent=self.iface.mainWindow(),
            add_to_menu=True,
            status_tip=self.tr('Open the random flight strategy planner'))

        # Dependency Manager
        self.dependency_manager_action = self.add_action(
            os.path.join(self.plugin_dir, 'icons', 'icon_dependencies.png'),
            text=self.tr('Dependency Manager'),
            callback=self._on_dependency_manager,
            parent=self.iface.mainWindow(),
            add_to_menu=True,
            status_tip=self.tr('Install or update BAMBI plugin dependencies'))

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(self.tr('&Bambi - QGIS Integration'), action)
            self.iface.removeToolBarIcon(action)

        # Remove the toolbar
        del self.toolbar

        # Close thermal viewer if open
        if self._thermal_viewer_dlg is not None:
            self._thermal_viewer_dlg.close()
            self._thermal_viewer_dlg = None

        # Close flight planner if open
        if self._flight_planner_dlg is not None:
            self._flight_planner_dlg.close()
            self._flight_planner_dlg = None

        # Close dependency manager if open
        if self._dependency_manager_dlg is not None:
            self._dependency_manager_dlg.close()
            self._dependency_manager_dlg = None

        # Disconnect project signals and remove dock widget
        if self.dock_widget:
            # Disconnect project signals to prevent issues
            self.dock_widget.disconnect_project_signals()

            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget.deleteLater()
            self.dock_widget = None

    def _ensure_dock_widget(self):
        """Create and register the dock widget if it does not yet exist."""
        if self.dock_widget is None:
            self.dock_widget = BambiDockWidget(self.iface)
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)
            # addDockWidget shows the widget by default; keep it hidden until
            # the user explicitly opens it via the main toolbar button.
            self.dock_widget.hide()
            self.dock_widget.visibilityChanged.connect(self.on_dock_visibility_changed)
            # Hand the toolbar actions to the dock widget so it can keep their
            # checked state in sync when the map tool changes externally.
            self.dock_widget.set_inspector_actions(
                self.inspector_action,
                self.fov_inspector_action,
                self.fov_georef_inspector_action,
            )

    def toggle_dock_widget(self):
        """Toggle the visibility of the dock widget."""
        self._ensure_dock_widget()
        if self.dock_widget.isVisible():
            self.dock_widget.hide()
            self.dock_widget_action.setChecked(False)
        else:
            self.dock_widget.show()
            self.dock_widget_action.setChecked(True)

    def _on_inspector_toggled(self, checked: bool):
        """Toolbar action: toggle the detection/track inspector."""
        self._ensure_dock_widget()
        self.dock_widget._toggle_inspector(checked)

    def _on_fov_inspector_toggled(self, checked: bool):
        """Toolbar action: toggle the FoV inspector (simple viewer)."""
        self._ensure_dock_widget()
        self.dock_widget._toggle_fov_inspector(checked)

    def _on_fov_georef_inspector_toggled(self, checked: bool):
        """Toolbar action: toggle the FoV geo-referenced inspector."""
        self._ensure_dock_widget()
        self.dock_widget._toggle_fov_georef_inspector(checked)

    def _on_correction_wizard(self):
        """Toolbar action: open the correction calibration wizard."""
        self._ensure_dock_widget()
        self.dock_widget.open_correction_wizard()

    def _on_camera_calibration(self):
        """Toolbar action: open the camera calibration wizard."""
        from .bambi_camera_calibration import CameraCalibrationWizard
        dlg = CameraCalibrationWizard(self.iface.mainWindow())
        dlg.exec_()

    def _on_thermal_viewer(self):
        """Toolbar action: open the thermal image viewer (non-modal)."""
        from .bambi_thermal_viewer import ThermalViewerDialog
        if self._thermal_viewer_dlg is None:
            self._thermal_viewer_dlg = ThermalViewerDialog(self.iface.mainWindow())
        self._thermal_viewer_dlg.show()
        self._thermal_viewer_dlg.raise_()
        self._thermal_viewer_dlg.activateWindow()

    def _on_flight_planner(self):
        """Toolbar action: open the random flight strategy planner (non-modal)."""
        from .bambi_flight_planner import FlightPlannerDialog
        if self._flight_planner_dlg is None:
            self._flight_planner_dlg = FlightPlannerDialog(
                self.iface, parent=self.iface.mainWindow()
            )
            self._flight_planner_dlg.finished.connect(
                lambda _: setattr(self, '_flight_planner_dlg', None)
            )
        self._flight_planner_dlg.show()
        self._flight_planner_dlg.raise_()
        self._flight_planner_dlg.activateWindow()

    def _on_dependency_manager(self):
        """Toolbar action: open the dependency manager (non-modal)."""
        from .bambi_dependency_manager import DependencyManagerDialog
        if self._dependency_manager_dlg is None:
            self._dependency_manager_dlg = DependencyManagerDialog(
                self.iface.mainWindow(),
                plugin_dir=self.plugin_dir,
            )
        self._dependency_manager_dlg.show()
        self._dependency_manager_dlg.raise_()
        self._dependency_manager_dlg.activateWindow()

    def on_dock_visibility_changed(self, visible):
        """Handle dock widget visibility changes."""
        self.dock_widget_action.setChecked(visible)

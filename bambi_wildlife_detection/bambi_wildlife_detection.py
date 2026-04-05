# -*- coding: utf-8 -*-
"""
BAMBI Wildlife Detection - Main Plugin Class
=============================================

This module contains the main plugin class that integrates with QGIS.
"""

import os
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QDockWidget
from qgis.core import QgsProject

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

        # Inspector: FoV
        self.fov_inspector_action = self.add_action(
            os.path.join(self.plugin_dir, 'icons', 'icon_inspector_fov.png'),
            text=self.tr('Inspector: Click FoV'),
            callback=self._on_fov_inspector_toggled,
            parent=self.iface.mainWindow(),
            checkable=True,
            add_to_menu=False,
            status_tip=self.tr(
                'Click a Field of View polygon to view the corresponding frame image'))

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(self.tr('&Bambi - QGIS Integration'), action)
            self.iface.removeToolBarIcon(action)
        
        # Remove the toolbar
        del self.toolbar
        
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
                self.inspector_action, self.fov_inspector_action
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
        """Toolbar action: toggle the FoV inspector."""
        self._ensure_dock_widget()
        self.dock_widget._toggle_fov_inspector(checked)

    def on_dock_visibility_changed(self, visible):
        """Handle dock widget visibility changes."""
        self.dock_widget_action.setChecked(visible)

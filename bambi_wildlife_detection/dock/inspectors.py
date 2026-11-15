# -*- coding: utf-8 -*-
"""The inspectors: click on the map, see what is there.

The three map tools of the toolbar - detection or track, field of view,
geo-referenced field of view - of which one is active at a time, and the
toolbar buttons that show which.
"""

from ..bambi_click_tool import BambiClickTool


class InspectorsMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def set_inspector_actions(self, inspector_action, fov_inspector_action,
                              fov_georef_inspector_action=None):
        """Receive the toolbar QAction references from the main plugin class.

        Called once, immediately after the dock widget is created.  The dock
        widget uses these to keep the toolbar button checked-states in sync.
        """
        self._inspector_action = inspector_action
        self._fov_inspector_action = fov_inspector_action
        self._fov_georef_inspector_action = fov_georef_inspector_action

    def set_segmentation_tool_action(self, action):
        """The toolbar action that opens the Segmentation window.

        The S1 button on the Processing tab triggers it, so the panel and
        the toolbar share one window instead of each opening their own.
        """
        self._segmentation_tool_action = action

    def _toggle_inspector(self, checked: bool):
        """Activate or deactivate the detection/track inspector map tool."""
        canvas = self.iface.mapCanvas()
        if checked:
            self._set_fov_inspector_off()
            self._set_fov_georef_inspector_off()
            if self._click_tool is None:
                self._click_tool = BambiClickTool(self.iface, mode="detection_track")
            canvas.setMapTool(self._click_tool)
        else:
            if self._click_tool is not None:
                canvas.unsetMapTool(self._click_tool)
        if self._inspector_action is not None:
            self._inspector_action.blockSignals(True)
            self._inspector_action.setChecked(checked)
            self._inspector_action.blockSignals(False)

    def _toggle_fov_inspector(self, checked: bool):
        """Activate or deactivate the FoV inspector map tool (simple, no geo-referencing)."""
        canvas = self.iface.mapCanvas()
        if checked:
            self._set_inspector_off()
            self._set_fov_georef_inspector_off()
            if self._fov_click_tool is None:
                self._fov_click_tool = BambiClickTool(self.iface, mode="fov")
            canvas.setMapTool(self._fov_click_tool)
        else:
            if self._fov_click_tool is not None:
                canvas.unsetMapTool(self._fov_click_tool)
        if self._fov_inspector_action is not None:
            self._fov_inspector_action.blockSignals(True)
            self._fov_inspector_action.setChecked(checked)
            self._fov_inspector_action.blockSignals(False)

    def _toggle_fov_georef_inspector(self, checked: bool):
        """Activate or deactivate the FoV geo-referenced inspector map tool."""
        canvas = self.iface.mapCanvas()
        if checked:
            self._set_inspector_off()
            self._set_fov_inspector_off()
            if self._fov_georef_click_tool is None:
                self._fov_georef_click_tool = BambiClickTool(self.iface, mode="fov_georef")
            canvas.setMapTool(self._fov_georef_click_tool)
        else:
            if self._fov_georef_click_tool is not None:
                canvas.unsetMapTool(self._fov_georef_click_tool)
        if self._fov_georef_inspector_action is not None:
            self._fov_georef_inspector_action.blockSignals(True)
            self._fov_georef_inspector_action.setChecked(checked)
            self._fov_georef_inspector_action.blockSignals(False)

    def _set_inspector_off(self):
        """Silently deactivate the detection/track inspector."""
        if self._inspector_action is not None:
            self._inspector_action.blockSignals(True)
            self._inspector_action.setChecked(False)
            self._inspector_action.blockSignals(False)

    def _set_fov_inspector_off(self):
        """Silently deactivate the FoV inspector."""
        if self._fov_inspector_action is not None:
            self._fov_inspector_action.blockSignals(True)
            self._fov_inspector_action.setChecked(False)
            self._fov_inspector_action.blockSignals(False)

    def _set_fov_georef_inspector_off(self):
        """Silently deactivate the FoV geo-referenced inspector."""
        if self._fov_georef_inspector_action is not None:
            self._fov_georef_inspector_action.blockSignals(True)
            self._fov_georef_inspector_action.setChecked(False)
            self._fov_georef_inspector_action.blockSignals(False)

    def _on_map_tool_changed(self, new_tool, old_tool):
        """Keep toolbar action states in sync when the user switches map tools."""
        if self._click_tool is not None and new_tool is not self._click_tool:
            self._set_inspector_off()
        if self._fov_click_tool is not None and new_tool is not self._fov_click_tool:
            self._set_fov_inspector_off()
        if self._fov_georef_click_tool is not None and new_tool is not self._fov_georef_click_tool:
            self._set_fov_georef_inspector_off()

# -*- coding: utf-8 -*-
"""Where a layer goes in the layer tree.

Every flight has a group, with a sub-group per camera; what belongs to the
survey as a whole has one group of its own. A layer that is added again
takes the place of the one it replaces. Layers made in memory are written
to the flight's folder so they are still there when the project is opened
again.
"""

import os
from contextlib import contextmanager
from typing import Optional

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsLayerTreeGroup, QgsLayerTreeLayer,
    QgsLayerTree,
)
from qgis.PyQt import sip


class LayerGroupsMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _flight_group(self, create: bool = True, or_root: bool = True):
        """The layer group of the active flight, or the tree root.

        Layers are grouped per flight so it is obvious which outputs belong
        together. Renaming a flight renames this group rather than
        leaving a stale one behind, which is why it is looked up by the current
        name every time instead of being cached.
        """
        from ..core import flights

        root = QgsProject.instance().layerTreeRoot()
        flight = self._active_flight()
        # A survey tool adding one flight's layers (see _layers_for_flight).
        override = getattr(self, "_layer_folder_override", None)
        if override:
            index = flights.find_by_folder(self._flights(), override)
            if index is not None:
                flight = self._flights()[index]
        if not flight:
            return root

        name = flights.group_name(flight)
        group = root.findGroup(name)
        if group is None and create:
            group = root.insertGroup(0, name)
        if group is None and not or_root:
            # A flight that has no group yet has no layers of its own; the
            # tree root would stand in for it - and hold the other flights'.
            return None
        return group or root

    @contextmanager
    def _layers_for_flight(self, folder: str):
        """Place layers in *folder*'s flight group and files, not the active
        flight's - for survey tools that add per-flight layers."""
        previous = getattr(self, "_layer_folder_override", None)
        self._layer_folder_override = folder
        try:
            yield
        finally:
            self._layer_folder_override = previous

    #: Root layer group of the Survey Analytics products.
    SURVEY_GROUP = "BAMBI Survey"

    def _survey_group(self):
        root = QgsProject.instance().layerTreeRoot()
        group = root.findGroup(self.SURVEY_GROUP)
        if group is None:
            group = root.insertGroup(0, self.SURVEY_GROUP)
        return group

    @staticmethod
    def _insert_node(parent, node, name: str, at_top: bool = True):
        """Put *node* into *parent* at its place in the plugin's fixed layer
        order (:mod:`core.layer_order`); names the plugin does not order go
        to the top (or bottom), and nothing already there is moved."""
        from ..core import layer_order
        siblings = [child.name() for child in parent.children()]
        parent.insertChildNode(
            layer_order.insert_index(siblings, name, at_top), node)
        return node

    @staticmethod
    def _insert_group(parent, name: str, at_top: bool = True):
        """Create a group called *name* in *parent* at its place in the
        fixed layer order.

        Made with ``insertGroup`` rather than ``insertChildNode(
        QgsLayerTreeGroup(...))``: a group built in Python comes back from
        ``children()`` / ``parent()`` as a bare ``QgsLayerTreeNode`` once
        its wrapper is gone, so it would no longer be recognised as a group.
        """
        from ..core import layer_order
        siblings = [child.name() for child in parent.children()]
        return parent.insertGroup(
            layer_order.insert_index(siblings, name, at_top), name)

    @staticmethod
    def _parent_group(node, fallback):
        """*node*'s parent as a group (PyQGIS may hand back a bare
        ``QgsLayerTreeNode``), or *fallback* when it has none."""
        parent = node.parent()
        if parent is None or not QgsLayerTree.isGroup(parent):
            return fallback
        return sip.cast(parent, QgsLayerTreeGroup)

    def _camera_group(self, name: str, create: bool = True):
        """The group a flight product called *name* belongs in: the flight's
        "Thermal" / "RGB" subgroup when the name carries a camera label,
        else the flight group itself.

        Without a flight (the tree root) there are no camera subgroups.
        """
        from ..core import layer_order
        flight_group = self._flight_group(create=create)
        root = QgsProject.instance().layerTreeRoot()
        modality = layer_order.modality(name)
        if modality is None or flight_group is root:
            return flight_group
        label = layer_order.MODALITY_GROUPS[modality]
        for child in flight_group.children():
            if QgsLayerTree.isGroup(child) and child.name() == label:
                return sip.cast(child, QgsLayerTreeGroup)
        if not create:
            return flight_group
        return self._insert_group(flight_group, label)

    @staticmethod
    def _take_out(parent, name: str, keep=None) -> int:
        """Take what is called *name* out of *parent* - a layer, or a group
        with everything in it - and out of the project.

        "Add ... to QGIS" shows a product as it is now. Pressed a second
        time it used to add the product again beside the first: a second
        group of the same name (the first one emptied, where its layers'
        files were rewritten), a second copy of a raster layer.

        *keep* is a layer that must stay (the one about to be added).
        Returns how many nodes were removed.
        """
        removed = 0
        for child in list(parent.children()):
            if child.name() != name:
                continue
            if QgsLayerTree.isGroup(child):
                group = sip.cast(child, QgsLayerTreeGroup)
                layer_ids = [node.layerId() for node in group.findLayers()
                             if node.layerId()]
                if layer_ids:
                    QgsProject.instance().removeMapLayers(layer_ids)
                parent.removeChildNode(child)
                removed += 1
            elif QgsLayerTree.isLayer(child):
                layer_id = sip.cast(child, QgsLayerTreeLayer).layerId()
                if keep is not None and layer_id == keep.id():
                    continue
                if layer_id and QgsProject.instance().mapLayer(layer_id):
                    # takes its node out of the tree with it
                    QgsProject.instance().removeMapLayer(layer_id)
                else:
                    parent.removeChildNode(child)
                removed += 1
        return removed

    def _add_layer_to_survey(self, layer, at_top: bool = True):
        """Add a survey-wide layer (density, coverage, inventory, survey
        area) to the survey group - it belongs to no single flight. A layer
        of the same name that is already there is replaced."""
        group = self._survey_group()
        self._take_out(group, layer.name(), keep=layer)
        QgsProject.instance().addMapLayer(layer, False)
        self._insert_node(group, QgsLayerTreeLayer(layer),
                          layer.name(), at_top)
        return layer

    def _add_layer_to_flight(self, layer, at_top: bool = True):
        """Add a single layer to the active flight's group.

        ``QgsProject.addMapLayer(layer)`` puts it at the root of the layer
        tree, outside the flight it belongs to - so the layer has to be added
        without a tree node first and then placed by hand. Use this for the
        products that are one layer and get no subgroup of their own
        (orthomosaic, ALFS, merged FoV). A layer named for a camera goes
        into that camera's subgroup, at its place in the fixed order. A
        layer - or group - of the same name that is already there is
        replaced.
        """
        group = self._camera_group(layer.name())
        self._take_out(group, layer.name(), keep=layer)
        QgsProject.instance().addMapLayer(layer, False)
        self._insert_node(group, QgsLayerTreeLayer(layer), layer.name(),
                          at_top)
        return layer

    def _remove_layer_group(self, group_name: str) -> bool:
        """Remove a layer group and every layer in it from the project.

        Re-adding a group would otherwise duplicate it, and its layers hold
        their GeoPackages open - which on Windows blocks rewriting them.
        Looked up anywhere below the flight group, so a group from before
        the camera subgroups existed is found as well.
        """
        # Only inside this flight's own group. Falling back on the tree root
        # for a flight that has no group yet found - and removed - the
        # same-named group of another flight.
        root = self._flight_group(create=False, or_root=False)
        if root is None:
            return False
        group = root.findGroup(group_name)
        if group is None:
            return False

        layer_ids = [node.layerId() for node in group.findLayers()
                     if node.layerId()]
        if layer_ids:
            QgsProject.instance().removeMapLayers(layer_ids)
        self._parent_group(group, root).removeChildNode(group)
        return True

    def _create_layer_group(self, group_name: str, at_top: bool = True) -> QgsLayerTreeGroup:
        """Create a layer group in the active flight's group - in the
        camera subgroup its name belongs to, at its place in the fixed order.

        A group of that name which is already there - from pressing the
        same "Add ... to QGIS" before - is replaced, with its layers.

        :param group_name: Name for the new group
        :param at_top: for names the plugin does not order: top or bottom
        :return: The created QgsLayerTreeGroup
        """
        # anywhere in this flight's group: one from before the camera
        # subgroups existed sits a level higher
        self._remove_layer_group(group_name)
        parent = self._camera_group(group_name)
        self._take_out(parent, group_name)
        return self._insert_group(parent, group_name, at_top)

    def _persist_memory_layer(self, mem_layer: QgsVectorLayer, file_name: str,
                              subfolder: str = "qgis_layers",
                              display_name: Optional[str] = None,
                              base_folder: Optional[str] = None) -> QgsVectorLayer:
        """Save a memory layer to GeoPackage file and return the file-based layer.

        If no target folder is set, returns the original memory layer unchanged.

        :param mem_layer: The memory layer to persist
        :param file_name: Unique name for the GeoPackage file (without extension)
        :param subfolder: Subfolder within target folder for storing layers
        :param display_name: Display name for the layer in QGIS (defaults to mem_layer's name)
        :param base_folder: folder to save under instead of the active
            flight's target folder (a survey folder, another flight)
        :return: File-based QgsVectorLayer if saved successfully, otherwise original memory layer
        """
        from qgis.core import QgsVectorFileWriter, QgsCoordinateTransformContext

        # Use memory layer's name as display name if not specified
        if display_name is None:
            display_name = mem_layer.name()

        # Check if target folder is set
        candidates = (base_folder,
                      getattr(self, "_layer_folder_override", None),
                      self.target_folder_edit.text().strip())
        target_folder = next((c for c in candidates if c), "")
        if target_folder and base_folder:
            os.makedirs(target_folder, exist_ok=True)
        if not target_folder or not os.path.isdir(target_folder):
            # No target folder, return memory layer as-is
            self.log(f"Warning: No target folder set, layer '{display_name}' will be temporary")
            return mem_layer

        try:
            # Create output folder
            gpkg_folder = os.path.join(target_folder, subfolder)
            os.makedirs(gpkg_folder, exist_ok=True)

            # Sanitize file name for filesystem
            safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in file_name)
            gpkg_path = os.path.join(gpkg_folder, f"{safe_name}.gpkg")

            # If file already exists, release any QGIS layers holding it open
            # before deleting - on Windows, open file handles cause WinError 32.
            exists = os.path.exists(gpkg_path)
            if exists:
                norm_path = os.path.normcase(os.path.abspath(gpkg_path))
                layers_to_remove = [
                    lid for lid, lyr in QgsProject.instance().mapLayers().items()
                    if isinstance(lyr, QgsVectorLayer) and  # noqa: W503, W504
                    os.path.normcase(os.path.abspath(lyr.source().split("|")[0])) == norm_path
                ]
                if layers_to_remove:
                    QgsProject.instance().removeMapLayers(layers_to_remove)
                try:
                    os.remove(gpkg_path)
                    exists = False
                except OSError as e:
                    # GDAL pools its datasets, so the handle can outlive the
                    # layer and Windows then refuses the delete (WinError 32).
                    # No need to delete: a GeoPackage is SQLite, so the layer
                    # inside it can be rewritten in place while a reader holds
                    # the file open.
                    self.log(f"Note: could not replace "
                             f"{os.path.basename(gpkg_path)} ({e}); "
                             "overwriting the layer inside it instead.")

            # Write to GeoPackage
            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = "GPKG"
            options.fileEncoding = "UTF-8"
            options.layerName = safe_name
            if exists:
                options.actionOnExistingFile = (
                    QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer)

            error = QgsVectorFileWriter.writeAsVectorFormatV3(
                mem_layer,
                gpkg_path,
                QgsCoordinateTransformContext(),
                options
            )

            if error[0] != QgsVectorFileWriter.WriterError.NoError:
                self.log(f"Warning: Could not save layer '{display_name}' to GeoPackage: {error[1]}")
                return mem_layer

            # Load the persisted layer with the display name
            persisted_layer = QgsVectorLayer(gpkg_path, display_name, "ogr")

            if not persisted_layer.isValid():
                self.log(f"Warning: Persisted layer '{display_name}' is not valid, using memory layer")
                return mem_layer

            return persisted_layer

        except Exception as e:
            self.log(f"Warning: Failed to persist layer '{display_name}': {str(e)}")
            return mem_layer

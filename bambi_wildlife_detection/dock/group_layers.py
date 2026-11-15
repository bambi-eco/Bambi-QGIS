# -*- coding: utf-8 -*-
"""Survey Analytics > Add Groups to QGIS: the groups on the map.

The census divides the counted animals into groups (``core.groups``) and
writes them down, one row each. This draws them: an outline around the
members of every group of two or more, in the colour of the species, with
the number of animals in it. An animal that was alone is not drawn - it is
on the map already, as itself.
"""

import os

from qgis.PyQt.QtGui import QColor, QFont
from qgis.PyQt.QtWidgets import QMessageBox
from qgis.core import (
    QgsCategorizedSymbolRenderer, QgsFeature, QgsField, QgsFillSymbol,
    QgsGeometry, QgsPalLayerSettings, QgsPointXY, QgsRendererCategory,
    QgsTextBufferSettings, QgsTextFormat, QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)

from ..qgis_compat import FieldType

#: Metres between the outermost members of a group and its outline: two
#: animals are a line, and a line has no inside to see.
OUTLINE_MARGIN_M = 3.0

_FIELDS = (
    ("flight", FieldType.String), ("group", FieldType.LongLong),
    ("species", FieldType.String), ("size", FieldType.Int),
    ("counted", FieldType.Int), ("radius_m", FieldType.Double),
    ("first_time", FieldType.String), ("last_time", FieldType.String),
    ("members", FieldType.String),
)


class GroupLayersMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def add_groups_to_qgis(self):
        """Draw the groups of the last census: one outline per group of two
        or more animals."""
        from ..core import groups, map_symbols

        survey = self.current_survey()
        source = self._survey_source()
        path = os.path.join(survey.analytics_folder, f"groups_{source}.json")
        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "Missing Data",
                "The groups have not been worked out yet.\n"
                "Run 'Build Census Summary' first.")
            return

        camera_label = survey.camera_label
        name = f"BAMBI Groups ({camera_label})"
        try:
            self.update_status("add_groups", "🟡 Loading...")
            table = groups.read_table_json(path)
            epsg = survey.epsg or self.get_config().get("target_epsg", 32633)
            layer = QgsVectorLayer(f"Polygon?crs=EPSG:{epsg}", name, "memory")
            provider = layer.dataProvider()
            provider.addAttributes(
                [QgsField(field, kind) for field, kind in _FIELDS])
            layer.updateFields()

            features, alone = [], 0
            for row in table.get("groups", []):
                outline = self._group_outline(row.get("positions") or [])
                if outline is None:
                    alone += 1
                    continue
                feature = QgsFeature(layer.fields())
                feature.setGeometry(outline)
                feature.setAttributes([
                    row.get("flight"), row.get("group"), row.get("species"),
                    row.get("size"), row.get("counted"), row.get("radius_m"),
                    row.get("first_time"), row.get("last_time"),
                    " ".join(str(m) for m in row.get("members") or [])])
                features.append(feature)
            provider.addFeatures(features)
            layer.updateExtents()

            layer = self._persist_memory_layer(
                layer, f"Groups_{survey.modality}_{source}",
                "analytics_layers", display_name=name,
                base_folder=survey.folder)
            layer.setCustomProperty("bambi_layer_type", "groups")
            # Each flight has its own schema; the first one's colours are
            # the survey's.
            colours = {}
            for flight in survey.flights:
                colours = map_symbols.species_colours(flight.target_folder)
                if colours:
                    break
            self._style_groups_layer(layer, colours)
            self._add_layer_to_survey(layer)
            self.update_status("add_groups", "🟢 Added")
            self.iface.mapCanvas().refresh()
            self.log(f"Added {len(features)} group(s) of two or more "
                     f"animals; {alone} animal(s) were alone and are not "
                     "outlined")
        except Exception as e:
            self.update_status("add_groups", "🔴 Error")
            self.log(f"Error adding the groups: {e}")
            QMessageBox.critical(self, "Error",
                                 f"Failed to add the groups: {e}")

    @staticmethod
    def _group_outline(positions):
        """The outline of a group: around its members, a little way off.
        ``None`` for fewer than two - one animal is no group to draw."""
        if len(positions) < 2:
            return None
        points = QgsGeometry.fromMultiPointXY(
            [QgsPointXY(float(x), float(y)) for x, y in positions])
        return points.convexHull().buffer(OUTLINE_MARGIN_M, 12)

    @staticmethod
    def _style_groups_layer(layer, colours) -> None:
        """An outline per species colour, the number of animals inside."""
        from ..core import map_symbols

        index = layer.fields().indexOf("species")
        names = sorted(str(value) for value in layer.uniqueValues(index))
        categories = []
        for species in names:
            colour = QColor(map_symbols.colour_of(colours, species))
            fill = QColor(colour)
            fill.setAlpha(40)
            symbol = QgsFillSymbol.createSimple({
                "color": f"{fill.red()},{fill.green()},{fill.blue()},"
                         f"{fill.alpha()}",
                "outline_color": colour.name(),
                "outline_width": "0.6",
            })
            categories.append(QgsRendererCategory(species, symbol, species))
        layer.setRenderer(QgsCategorizedSymbolRenderer("species", categories))

        settings = QgsPalLayerSettings()
        settings.fieldName = "size"
        text = QgsTextFormat()
        font = QFont()
        font.setBold(True)
        text.setFont(font)
        text.setSize(10)
        text.setColor(QColor("#202020"))
        halo = QgsTextBufferSettings()
        halo.setEnabled(True)
        halo.setSize(1.0)
        halo.setColor(QColor("#ffffff"))
        text.setBuffer(halo)
        settings.setFormat(text)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
        layer.setLabelsEnabled(True)
        layer.triggerRepaint()

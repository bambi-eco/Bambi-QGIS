# -*- coding: utf-8 -*-
"""QGIS symbology of the animal layers (detections and tracks).

One rule-based style for every box layer, visible and editable in the
ordinary Symbology dialog:

* **Zoomed in**, the geo-referenced bounding box, outlined in the colour of
  its species.
* **Zoomed out**, where a one-metre box is a speck, a pin of fixed screen
  size at the box's centre, in the same colour, with a short sex / age
  indicator beside it ("♀ juv").

The two are top-level rules with a scale range each, so the switch-over
scale is changed in two places in the dialog, however many species there
are; below each sits one rule per species present in the layer, which is
also what the legend lists.

What the colours and indicators *are* is decided in :mod:`core.map_symbols`;
this module only draws them.
"""

from typing import Dict, Iterable, List, Optional

from qgis.core import (
    QgsCentroidFillSymbolLayer, QgsField, QgsFillSymbol,
    QgsFontMarkerSymbolLayer, QgsLineSymbol, QgsMarkerSymbol, QgsProperty,
    QgsRuleBasedRenderer, QgsSimpleFillSymbolLayer,
    QgsSimpleMarkerSymbolLayer, QgsSingleSymbolRenderer, QgsSymbolLayer,
)
from qgis.PyQt.QtCore import QPointF, Qt, QVariant
from qgis.PyQt.QtGui import QColor, QFont

from .core import map_symbols

#: Custom property marking a layer this module styled, with the kind of
#: geometry it holds, so it can be found and restyled.
STYLE_PROPERTY = "bambi_animal_style"
BOX = "box"
PATH = "path"

PIN_HEAD_MM = 4.4
PIN_OUTLINE = QColor(255, 255, 255)
BADGE_COLOR = QColor(20, 20, 20)
BADGE_SIZE_MM = 3.2


def animal_fields() -> List[QgsField]:
    """The fields a styled layer carries (see ``core.map_symbols.FIELDS``)."""
    return [QgsField(name, QVariant.String) for name in map_symbols.FIELDS]


def animal_values(attributes: Optional[Dict[str, str]]) -> List[str]:
    """*attributes* in field order; an animal nothing is known about is an
    undetermined one, not a missing value."""
    attributes = attributes or map_symbols.attributes()
    return [attributes.get(name, "") for name in map_symbols.FIELDS]


def _property(name: str):
    """A data-defined symbol-layer property, by its old (QGIS 3) or its
    scoped (QGIS 4) name."""
    scoped = getattr(QgsSymbolLayer, "Property", None)
    short = name[len("Property"):]
    if scoped is not None and hasattr(scoped, short):
        return getattr(scoped, short)
    return getattr(QgsSymbolLayer, name)


def _shape(name: str):
    """A simple-marker shape, by its old or its scoped name."""
    try:
        from qgis.core import Qgis
        return getattr(Qgis.MarkerShape, name)
    except (ImportError, AttributeError):
        return getattr(QgsSimpleMarkerSymbolLayer, name)


def box_symbol(colour: str) -> QgsFillSymbol:
    """The bounding box: a firm outline, a faint fill."""
    line = QColor(colour)
    fill = QColor(colour)
    fill.setAlpha(60)
    layer = QgsSimpleFillSymbolLayer(fill)
    layer.setStrokeColor(line)
    layer.setStrokeWidth(0.8)
    symbol = QgsFillSymbol()
    symbol.changeSymbolLayer(0, layer)
    return symbol


def pin_marker(colour: str) -> QgsMarkerSymbol:
    """A map pin whose tip is the animal's position, with the sex / age
    indicator to its upper right."""
    fill = QColor(colour)
    head_centre = -(PIN_HEAD_MM * 0.95)

    # The point of the pin: a triangle from the head down to the position.
    tip = QgsSimpleMarkerSymbolLayer(_shape("Triangle"))
    tip.setSize(PIN_HEAD_MM * 0.72)
    tip.setAngle(180)
    tip.setColor(fill)
    tip.setStrokeColor(PIN_OUTLINE)
    tip.setStrokeWidth(0.3)
    # The offset turns with the marker, so "up" is positive here.
    tip.setOffset(QPointF(0, PIN_HEAD_MM * 0.36))

    head = QgsSimpleMarkerSymbolLayer(_shape("Circle"))
    head.setSize(PIN_HEAD_MM)
    head.setColor(fill)
    head.setStrokeColor(PIN_OUTLINE)
    head.setStrokeWidth(0.45)
    head.setOffset(QPointF(0, head_centre))

    eye = QgsSimpleMarkerSymbolLayer(_shape("Circle"))
    eye.setSize(PIN_HEAD_MM * 0.36)
    eye.setColor(PIN_OUTLINE)
    eye.setStrokeStyle(Qt.PenStyle.NoPen)
    eye.setOffset(QPointF(0, head_centre))

    # Sex and age, read from the feature, to the right of the head.
    # Drawn twice: a white, outlined copy as a halo and the dark text
    # on top - an outline on the text itself is painted over the letters.
    badges = []
    for text_colour, outline in ((PIN_OUTLINE, 0.7), (BADGE_COLOR, 0.0)):
        badge = QgsFontMarkerSymbolLayer()
        # The class default is a dingbat font, which has no letters.
        badge.setFontFamily(QFont().family())
        badge.setFontStyle("Bold")
        badge.setCharacter("")
        badge.setSize(BADGE_SIZE_MM)
        badge.setColor(text_colour)
        badge.setStrokeColor(PIN_OUTLINE)
        badge.setStrokeWidth(outline)
        badge.setOffset(QPointF(PIN_HEAD_MM, head_centre - 0.9))
        # The text is centred on its offset, so a longer one has to start
        # further out or it would run across the pin's head.
        badge.setDataDefinedProperty(
            _property("PropertyOffset"),
            QgsProperty.fromExpression(
                f"to_string({PIN_HEAD_MM * 0.68:.2f} + "
                f"length(\"indicator\") * {BADGE_SIZE_MM * 0.3:.2f})"
                f" || ',{head_centre - 0.9:.2f}'"))
        badge.setDataDefinedProperty(
            _property("PropertyCharacter"),
            QgsProperty.fromField("indicator"))
        badge.setDataDefinedProperty(
            _property("PropertyLayerEnabled"),
            QgsProperty.fromExpression(
                "coalesce(\"indicator\", '') <> ''"))
        badges.append(badge)

    marker = QgsMarkerSymbol()
    marker.changeSymbolLayer(0, tip)
    for layer in (head, eye, *badges):
        marker.appendSymbolLayer(layer)
    return marker


def pin_symbol(colour: str) -> QgsFillSymbol:
    """A pin at the centre of a box polygon."""
    centroid = QgsCentroidFillSymbolLayer()
    centroid.setSubSymbol(pin_marker(colour))
    centroid.setPointOnSurface(False)
    symbol = QgsFillSymbol()
    symbol.changeSymbolLayer(0, centroid)
    return symbol


def _quoted(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _species_rules(parent, names: Iterable[str], colours: Dict[str, str],
                   make_symbol) -> None:
    """One child rule per species, and one for everything else."""
    for name in names:
        parent.appendChild(QgsRuleBasedRenderer.Rule(
            make_symbol(map_symbols.colour_of(colours, name)),
            0, 0, f'"species" = {_quoted(name)}', name))
    rest = QgsRuleBasedRenderer.Rule(
        make_symbol(map_symbols.FALLBACK_COLOUR), 0, 0, "ELSE",
        "undetermined")
    rest.setIsElse(True)
    parent.appendChild(rest)


def layer_species(layer) -> List[str]:
    """The species named in a layer's ``species`` field."""
    index = layer.fields().indexFromName("species")
    if index < 0:
        return []
    return map_symbols.species_present(layer.uniqueValues(index))


def box_renderer(names: Iterable[str], colours: Dict[str, str],
                 switch_scale: float = map_symbols.DEFAULT_SWITCH_SCALE,
                 pins: bool = True) -> QgsRuleBasedRenderer:
    """The renderer of a box layer.

    :param names: species to give a rule of their own
    :param colours: ``species name -> #rrggbb``
    :param switch_scale: map scale (1:n) beyond which pins replace boxes
    :param pins: off draws the boxes at every scale
    """
    names = list(names)
    root = QgsRuleBasedRenderer.Rule(None)
    if not pins or switch_scale <= 0:
        _species_rules(root, names, colours, box_symbol)
        return QgsRuleBasedRenderer(root)

    scale = float(switch_scale)
    boxes = QgsRuleBasedRenderer.Rule(
        None, 0, 0, "", f"Boxes (closer than 1:{scale:,.0f})")
    # "Minimum scale" is the most zoomed-out scale a rule is drawn at.
    boxes.setMinimumScale(scale)
    _species_rules(boxes, names, colours, box_symbol)
    far = QgsRuleBasedRenderer.Rule(
        None, 0, 0, "", f"Pins (beyond 1:{scale:,.0f})")
    far.setMaximumScale(scale)
    _species_rules(far, names, colours, pin_symbol)
    root.appendChild(boxes)
    root.appendChild(far)
    return QgsRuleBasedRenderer(root)


def style_boxes(layer, colours: Dict[str, str],
                switch_scale: float = map_symbols.DEFAULT_SWITCH_SCALE,
                pins: bool = True) -> None:
    """Style a polygon layer of animal boxes by species, with pins beyond
    *switch_scale*."""
    layer.setRenderer(box_renderer(
        layer_species(layer), colours, switch_scale, pins))
    layer.setCustomProperty(STYLE_PROPERTY, BOX)
    layer.triggerRepaint()


def style_path(layer, colours: Dict[str, str]) -> None:
    """Style a track's path line in the colour of the track's species."""
    names = layer_species(layer)
    colour = QColor(map_symbols.colour_of(
        colours, names[0] if names else ""))
    symbol = QgsLineSymbol.createSimple({
        "width": "1.2", "capstyle": "round", "joinstyle": "round"})
    symbol.setColor(colour)
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))
    layer.setCustomProperty(STYLE_PROPERTY, PATH)
    layer.triggerRepaint()


def restyle(layers: Iterable, colours: Dict[str, str],
            switch_scale: float = map_symbols.DEFAULT_SWITCH_SCALE,
            pins: bool = True) -> int:
    """Apply the current colours and switch-over scale to layers this
    module styled before. Returns how many were restyled."""
    count = 0
    for layer in layers:
        kind = layer.customProperty(STYLE_PROPERTY, "")
        if kind == BOX:
            style_boxes(layer, colours, switch_scale, pins)
        elif kind == PATH:
            style_path(layer, colours)
        else:
            continue
        count += 1
    return count

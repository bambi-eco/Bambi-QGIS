# -*- coding: utf-8 -*-
"""QGIS symbology of the animal layers (detections and tracks).

One rule-based style for every box layer, visible and editable in the
ordinary Symbology dialog:

* **Zoomed in**, the geo-referenced bounding box, in the colour of the
  individual - a track's own colour, so neighbours can be told apart.
* **Zoomed out**, where a one-metre box is a speck, a teardrop pin of fixed
  screen size whose tip is the box's centre, in the colour of the species.
  The sex is the mark in its middle ("?" when unknown) and the age a small
  badge on its lower right, so both stay with their pin however close the
  next one is.

The two are top-level rules with a scale range each. "Boxes" is one symbol
that reads its colour from the feature; below "Pins" sits one rule per
species present in the layer, which is also what the legend lists.

What the colours and marks *are* is decided in :mod:`core.map_symbols`;
this module only draws them.
"""

import base64
import os
from typing import Dict, Iterable, List, Optional

from qgis.core import (
    QgsCentroidFillSymbolLayer, QgsField, QgsFillSymbol,
    QgsFontMarkerSymbolLayer, QgsLineSymbol, QgsMarkerSymbol, QgsProperty,
    QgsRuleBasedRenderer, QgsSimpleFillSymbolLayer,
    QgsSimpleMarkerSymbolLayer, QgsSingleSymbolRenderer, QgsSvgMarkerSymbolLayer,
    QgsSymbolLayer,
)
from qgis.PyQt.QtCore import QPointF

from .qgis_compat import FieldType
from qgis.PyQt.QtGui import QColor, QFont

from .core import map_symbols

#: Custom property marking a layer this module styled, with the kind of
#: geometry it holds, so it can be found and restyled.
STYLE_PROPERTY = "bambi_animal_style"
BOX = "box"
PATH = "path"

#: Width of a pin; its height follows from the drawing (100 x 128).
PIN_WIDTH_MM = 8.5
PIN_HEIGHT_MM = PIN_WIDTH_MM * 1.28
PIN_OUTLINE = QColor(255, 255, 255)
BADGE_COLOR = QColor(32, 32, 32)
#: Diameter of the age badge on the pin's lower right.
AGE_BADGE_MM = 4.0

#: The teardrop, bundled with the plugin.
PIN_SVG = os.path.join(os.path.dirname(__file__), "icons", "animal_pin.svg")


def pin_svg_path() -> str:
    """The pin drawing as a path QGIS can store in a style.

    Embedded rather than referenced: a project opened on another computer,
    or by someone without the plugin, would otherwise look for a file in
    this installation's folder and draw a question mark.
    """
    with open(PIN_SVG, "rb") as fh:
        return "base64:" + base64.b64encode(fh.read()).decode("ascii")


def animal_fields() -> List[QgsField]:
    """The fields a styled layer carries (see ``core.map_symbols.FIELDS``)."""
    return [QgsField(name, FieldType.String) for name in map_symbols.FIELDS]


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


def _filled(field: str) -> QgsProperty:
    """True for a feature whose *field* is not empty."""
    return QgsProperty.fromExpression(f"coalesce(\"{field}\", '') <> ''")


def _letter(field: str, size: float, colour: QColor,
            offset: QPointF) -> QgsFontMarkerSymbolLayer:
    """One character read from *field*, drawn only where there is one."""
    letter = QgsFontMarkerSymbolLayer()
    # The class default is a dingbat font, which has no letters.
    letter.setFontFamily(QFont().family())
    letter.setFontStyle("Bold")
    letter.setCharacter("")
    letter.setSize(size)
    letter.setColor(colour)
    letter.setOffset(offset)
    letter.setDataDefinedProperty(
        _property("PropertyCharacter"), QgsProperty.fromField(field))
    letter.setDataDefinedProperty(
        _property("PropertyLayerEnabled"), _filled(field))
    return letter


def box_symbol(colour: str = map_symbols.UNTRACKED_COLOUR) -> QgsFillSymbol:
    """The bounding box: a firm outline and a faint fill, in the colour
    the feature names in ``box_colour`` (*colour* where it names none)."""
    line = QColor(colour)
    fill = QColor(colour)
    fill.setAlpha(60)
    layer = QgsSimpleFillSymbolLayer(fill)
    layer.setStrokeColor(line)
    layer.setStrokeWidth(0.8)
    own = f"coalesce(nullif(\"box_colour\", ''), '{line.name()}')"
    layer.setDataDefinedProperty(
        _property("PropertyStrokeColor"), QgsProperty.fromExpression(own))
    layer.setDataDefinedProperty(
        _property("PropertyFillColor"),
        QgsProperty.fromExpression(f"set_color_part({own}, 'alpha', 60)"))
    symbol = QgsFillSymbol()
    symbol.changeSymbolLayer(0, layer)
    return symbol


def pin_marker(colour: str) -> QgsMarkerSymbol:
    """A teardrop pin whose tip is the animal's position.

    In its middle the sex as a mark - "?" when it is not known - and on its
    lower right a badge with the first letter of the age class, ringed in
    the pin's colour. Both are part of the pin, so they cannot be read as a
    neighbour's.
    """
    fill = QColor(colour)
    mark_colour = QColor(map_symbols.reads_on(colour))

    body = QgsSvgMarkerSymbolLayer(pin_svg_path())
    body.setSize(PIN_WIDTH_MM)
    body.setFillColor(fill)
    body.setStrokeColor(PIN_OUTLINE)
    body.setStrokeWidth(0.45)
    # The drawing's bottom edge is 4/128 below the tip.
    body.setVerticalAnchorPoint(
        QgsSvgMarkerSymbolLayer.VerticalAnchorPoint.Bottom)
    body.setOffset(QPointF(0, PIN_HEIGHT_MM * 4.0 / 128.0))

    # The round part of the teardrop is centred 78/128 above the tip.
    # The mark sits a little above it, clear of the age badge.
    middle = QPointF(0, -PIN_HEIGHT_MM * 78.0 / 128.0 - 0.9)
    sex = _letter("sex_mark", PIN_WIDTH_MM * 0.5, mark_colour, middle)
    sex.setDataDefinedProperty(
        _property("PropertyCharacter"), QgsProperty.fromExpression(
            "coalesce(nullif(\"sex_mark\", ''), "
            f"'{map_symbols.UNKNOWN_SEX_MARK}')"))
    sex.setDataDefinedProperty(
        _property("PropertyLayerEnabled"), QgsProperty.fromValue(True))

    # Age: a white disc ringed in the pin's colour, on its lower right.
    corner = QPointF(PIN_WIDTH_MM * 0.37, -PIN_HEIGHT_MM * 0.22)
    badge = QgsSimpleMarkerSymbolLayer(_shape("Circle"))
    badge.setSize(AGE_BADGE_MM)
    badge.setColor(PIN_OUTLINE)
    badge.setStrokeColor(fill)
    badge.setStrokeWidth(0.5)
    badge.setOffset(corner)
    badge.setDataDefinedProperty(
        _property("PropertyLayerEnabled"), _filled("age_mark"))
    age = _letter("age_mark", AGE_BADGE_MM * 0.66, BADGE_COLOR, corner)

    marker = QgsMarkerSymbol()
    marker.changeSymbolLayer(0, body)
    for layer in (sex, badge, age):
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

    :param names: species to give a pin rule of their own
    :param colours: ``species name -> #rrggbb``
    :param switch_scale: map scale (1:n) beyond which pins replace boxes
    :param pins: off draws the boxes at every scale
    """
    root = QgsRuleBasedRenderer.Rule(None)
    if not pins or switch_scale <= 0:
        root.appendChild(QgsRuleBasedRenderer.Rule(
            box_symbol(), 0, 0, "", "Boxes"))
        return QgsRuleBasedRenderer(root)

    scale = float(switch_scale)
    boxes = QgsRuleBasedRenderer.Rule(
        box_symbol(), 0, 0, "", f"Boxes (closer than 1:{scale:,.0f})")
    # "Minimum scale" is the most zoomed-out scale a rule is drawn at.
    boxes.setMinimumScale(scale)
    far = QgsRuleBasedRenderer.Rule(
        None, 0, 0, "", f"Pins (beyond 1:{scale:,.0f})")
    far.setMaximumScale(scale)
    _species_rules(far, list(names), colours, pin_symbol)
    root.appendChild(boxes)
    root.appendChild(far)
    return QgsRuleBasedRenderer(root)


def style_boxes(layer, colours: Dict[str, str],
                switch_scale: float = map_symbols.DEFAULT_SWITCH_SCALE,
                pins: bool = True) -> None:
    """Style a polygon layer of animal boxes: each box in its individual's
    colour, and pins in the species' colour beyond *switch_scale*."""
    layer.setRenderer(box_renderer(
        layer_species(layer), colours, switch_scale, pins))
    layer.setCustomProperty(STYLE_PROPERTY, BOX)
    layer.triggerRepaint()


def layer_box_colour(layer) -> str:
    """The individual colour a single-animal layer names (the neutral one
    when it names none)."""
    index = layer.fields().indexFromName("box_colour")
    values = [v for v in layer.uniqueValues(index)
              if v not in (None, "")] if index >= 0 else []
    return str(values[0]) if values else map_symbols.UNTRACKED_COLOUR


def style_path(layer) -> None:
    """Style a track's path line in the colour of its individual, like the
    track's box."""
    symbol = QgsLineSymbol.createSimple({
        "width": "1.2", "capstyle": "round", "joinstyle": "round"})
    symbol.setColor(QColor(layer_box_colour(layer)))
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
            style_path(layer)
        else:
            continue
        count += 1
    return count

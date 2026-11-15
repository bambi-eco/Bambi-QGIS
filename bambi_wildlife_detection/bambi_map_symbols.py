# -*- coding: utf-8 -*-
"""QGIS symbology of the animal layers (detections and tracks).

One rule-based style for every box layer, visible and editable in the
ordinary Symbology dialog:

* **Zoomed in**, the geo-referenced bounding box, in the colour of the
  individual - a track's own colour, so neighbours can be told apart.
* **Zoomed out**, where a one-metre box is a speck, a pin of fixed screen
  size at the box's centre, in the colour of the species. The sex is a mark
  inside the pin's head and the age a small badge attached to it, so both
  stay with their pin however close the next one is.

The two are top-level rules with a scale range each. "Boxes" is one symbol
that reads its colour from the feature; below "Pins" sits one rule per
species present in the layer, which is also what the legend lists.

What the colours and marks *are* is decided in :mod:`core.map_symbols`;
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

PIN_HEAD_MM = 6.0
PIN_OUTLINE = QColor(255, 255, 255)
BADGE_COLOR = QColor(20, 20, 20)
#: Diameter of the age badge on the pin's shoulder.
AGE_BADGE_MM = 3.3


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


def _filled(field: str) -> QgsProperty:
    """True for a feature whose *field* is not empty."""
    return QgsProperty.fromExpression(f"coalesce(\"{field}\", '') <> ''")


def _empty(field: str) -> QgsProperty:
    return QgsProperty.fromExpression(f"coalesce(\"{field}\", '') = ''")


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
    """A map pin whose tip is the animal's position.

    The head carries the sex as a mark (a plain dot when it is unknown) and,
    on its shoulder, a badge with the first letter of the age class - both
    part of the pin, so they cannot be read as a neighbour's.
    """
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

    # Sex unknown: the plain dot of an ordinary pin.
    eye = QgsSimpleMarkerSymbolLayer(_shape("Circle"))
    eye.setSize(PIN_HEAD_MM * 0.3)
    eye.setColor(PIN_OUTLINE)
    eye.setStrokeStyle(Qt.PenStyle.NoPen)
    eye.setOffset(QPointF(0, head_centre))
    eye.setDataDefinedProperty(
        _property("PropertyLayerEnabled"), _empty("sex_mark"))

    # Sex known: its mark in the head, dark under white so it reads on
    # every species colour.
    # The sex glyphs hang below their baseline; lifted a little they sit
    # in the middle of the head.
    centre = QPointF(0, head_centre - PIN_HEAD_MM * 0.07)
    sex_shadow = _letter("sex_mark", PIN_HEAD_MM * 0.74, BADGE_COLOR, centre)
    sex_shadow.setStrokeColor(BADGE_COLOR)
    sex_shadow.setStrokeWidth(0.5)
    sex = _letter("sex_mark", PIN_HEAD_MM * 0.74, PIN_OUTLINE, centre)

    # Age: a badge on the head's upper right, overlapping it.
    shoulder = QPointF(PIN_HEAD_MM * 0.42, head_centre - PIN_HEAD_MM * 0.42)
    badge = QgsSimpleMarkerSymbolLayer(_shape("Circle"))
    badge.setSize(AGE_BADGE_MM)
    badge.setColor(PIN_OUTLINE)
    badge.setStrokeColor(BADGE_COLOR)
    badge.setStrokeWidth(0.3)
    badge.setOffset(shoulder)
    badge.setDataDefinedProperty(
        _property("PropertyLayerEnabled"), _filled("age_mark"))
    age = _letter("age_mark", AGE_BADGE_MM * 0.72, BADGE_COLOR, shoulder)

    marker = QgsMarkerSymbol()
    marker.changeSymbolLayer(0, tip)
    for layer in (head, eye, sex_shadow, sex, badge, age):
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

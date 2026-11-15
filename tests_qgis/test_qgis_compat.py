# -*- coding: utf-8 -*-
"""Field types for the QGIS at hand (``qgis_compat``).

A ``QgsField`` takes ``QVariant.Int`` up to QGIS 3.36 and
``QMetaType.Type.Int`` since 3.38; Qt 6 has no ``QVariant.Int`` at all, and
the plugin made every one of its layers with it.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import pytest

from qgis.core import Qgis, QgsField, QgsVectorLayer
from qgis.PyQt.QtCore import QMetaType, QVariant

from bambi_wildlife_detection import qgis_compat
from bambi_wildlife_detection.qgis_compat import FieldType

#: What the memory provider calls a field of each type.
NAMES = {"Int": "integer", "LongLong": "int8", "Double": "double",
         "String": "string", "Bool": "boolean"}


@pytest.mark.parametrize("name", sorted(NAMES))
def test_every_type_makes_a_field_of_a_layer(name):
    layer = QgsVectorLayer("Point?crs=EPSG:32633", "probe", "memory")
    provider = layer.dataProvider()
    assert provider.addAttributes([QgsField("value", getattr(FieldType, name))])
    layer.updateFields()

    field = layer.fields().field("value")
    assert field.typeName().lower() == NAMES[name]


def test_they_are_what_this_qgis_takes():
    if Qgis.QGIS_VERSION_INT >= 33800:
        assert FieldType.Int == QMetaType.Type.Int
    else:
        assert FieldType.Int == QVariant.Int
        assert FieldType.String == QVariant.String


def test_a_newer_qgis_gets_the_newer_spelling(monkeypatch):
    monkeypatch.setattr(Qgis, "QGIS_VERSION_INT", 33800, raising=False)
    types = qgis_compat._field_types()
    assert types == {"Int": QMetaType.Type.Int,
                     "LongLong": QMetaType.Type.LongLong,
                     "Double": QMetaType.Type.Double,
                     "String": QMetaType.Type.QString,
                     "Bool": QMetaType.Type.Bool}


def test_an_older_qgis_keeps_the_older_one(monkeypatch):
    monkeypatch.setattr(Qgis, "QGIS_VERSION_INT", 32200, raising=False)
    types = qgis_compat._field_types()
    assert types["Int"] == QVariant.Int and types["Bool"] == QVariant.Bool


def test_no_layer_is_made_with_the_qt5_only_spelling():
    import os
    import re

    root = os.path.dirname(qgis_compat.__file__)
    for folder, _dirs, files in os.walk(root):
        if os.sep + "ext" in folder:
            continue
        for name in files:
            if name.endswith(".py") and name != "qgis_compat.py":
                with open(os.path.join(folder, name), encoding="utf-8") as handle:
                    found = re.findall(
                        r"\bQVariant\.(?:Int|LongLong|Double|String|Bool)\b",
                        handle.read())
                assert found == [], name

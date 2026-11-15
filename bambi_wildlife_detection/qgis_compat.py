# -*- coding: utf-8 -*-
"""What differs between the QGIS versions the plugin runs in.

**Field types.** A ``QgsField`` is given its type as ``QVariant.Int`` up to
QGIS 3.36, and as ``QMetaType.Type.Int`` since 3.38. Qt 6 - QGIS 4 - has no
``QVariant.Int`` at all. :class:`FieldType` is the type under the names the
plugin always used, for the QGIS it finds itself in.
"""


def _field_types():
    from qgis.PyQt import QtCore

    try:
        from qgis.core import Qgis
        version = int(Qgis.QGIS_VERSION_INT)
    except Exception:  # noqa: BLE001 - no QGIS to ask: the older spelling
        version = 0
    meta = getattr(getattr(QtCore, "QMetaType", None), "Type", None)
    variant = getattr(QtCore, "QVariant", None)
    old_style = all(hasattr(variant, name) for name in ("Int", "String"))
    if meta is not None and (version >= 33800 or not old_style):
        return {"Int": meta.Int, "LongLong": meta.LongLong,
                "Double": meta.Double, "String": meta.QString,
                "Bool": meta.Bool}
    return {name: getattr(variant, name)
            for name in ("Int", "LongLong", "Double", "String", "Bool")}


class FieldType:
    """``QgsField`` types: ``QgsField("count", FieldType.Int)``."""

    Int = LongLong = Double = String = Bool = None


for _name, _value in _field_types().items():
    setattr(FieldType, _name, _value)

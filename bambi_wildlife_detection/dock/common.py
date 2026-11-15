# -*- coding: utf-8 -*-
"""What more than one part of the dock uses.

Small helpers and the lists of choices that the form and ``get_config`` have
to agree on.
"""

from qgis.PyQt.QtWidgets import QSizePolicy

# The classification tasks, in the order they run: occlusion decides which
# frames the other two may vote over, and sex reuses exactly the species
# frames.
_CLASSIFICATION_TASKS = ("occlusion", "species", "sex", "life_stage")
_CLASSIFICATION_INPUTS = (("Thermal", "thermal"), ("RGB", "rgb"),
                          ("Matched", "matched"))


def _cap_vertical_growth(widget):
    """Stop a height-capped widget from advertising vertical expansion.

    Qt propagates a layout's Expanding direction up to the widget holding it,
    so a QTextEdit/QListWidget with a small ``setMaximumHeight()`` still makes
    its group box grow whenever a config tab page is taller than its contents
    (every page is stretched to the tallest one in the tab widget). The group
    box cannot pass that surplus to the capped widget, so it lands on the only
    other child that may grow - the word-wrapped info label - which then
    centres one line of text in a few hundred pixels.
    """
    policy = widget.sizePolicy()
    policy.setVerticalPolicy(QSizePolicy.Policy.Preferred)
    widget.setSizePolicy(policy)

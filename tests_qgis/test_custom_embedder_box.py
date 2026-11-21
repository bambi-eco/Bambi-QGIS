# -*- coding: utf-8 -*-
"""Configuration > Classification > Custom embedding model: the box, what
it stores and when the step asks before running the file.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
from bambi_wildlife_detection.core import custom_models


def test_the_classification_tab_has_the_box(dock):
    box = dock.custom_embedder_box

    assert box.kind == custom_models.EMBEDDER
    assert box.title() == "Custom embedding model"
    assert dock._custom_boxes["embedder"] is box
    assert dock.config_sub_tabs.tabText(dock.config_sub_tabs.indexOf(
        _page_of(dock, box))) == "Classification"


def _page_of(dock, widget):
    tabs = dock.config_sub_tabs
    for index in range(tabs.count()):
        if tabs.widget(index).isAncestorOf(widget):
            return tabs.widget(index)
    raise AssertionError("not on a configuration sub tab")


def test_dinov3_is_used_unless_it_is_ticked(dock):
    spec = dock.get_config()["custom_implementations"].get("embedder")

    assert not custom_models.is_custom(spec)


def test_the_choice_reaches_the_step(dock, tmp_path):
    path = tmp_path / "mine.py"
    path.write_text("class E:\n    def setup(self, c):\n        pass\n"
                    "    def run(self, crops, items):\n        pass\n")
    dock.custom_embedder_box.set_spec({
        "source": custom_models.SOURCE, "path": str(path), "class": "E"})
    dock._on_custom_implementation_changed()      # as a click on it does

    spec = dock.get_config()["custom_implementations"]["embedder"]

    assert custom_models.is_custom(spec)
    assert spec["path"] == str(path)
    # the embedding step asks before the file runs; other steps do not
    assert dock._custom_specs_for_step("embeddings", dock.get_config()) == \
        [spec]
    assert spec not in dock._custom_specs_for_step(
        "detection", dock.get_config())

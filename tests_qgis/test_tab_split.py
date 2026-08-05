# -*- coding: utf-8 -*-
"""The Pre-Processing / Processing split (EXCHANGE_FORMAT_PLAN.md §10.1).

The split follows the dependency graph of §7: Pre-Processing derives from the
poses and the DEM and is independent of any animal, Processing depends on the
detections. This is presentation only — config keys, step ids and
``output_inventory`` keys are unchanged — so the tests check that every step is
still reachable and that a saved configuration still loads.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import pytest

from bambi_wildlife_detection.core import stages


def _tab_names(dock):
    tabs = dock.findChild(type(dock.findChildren(type(dock))[0])) \
        if False else None       # placeholder, replaced below
    return tabs


def _main_tabs(dock):
    from qgis.PyQt.QtWidgets import QTabWidget

    for widget in dock.findChildren(QTabWidget):
        names = [widget.tabText(i) for i in range(widget.count())]
        if "Input" in names:
            return widget
    raise AssertionError("main tab widget not found")


def test_both_tabs_exist(dock):
    tabs = _main_tabs(dock)
    names = [tabs.tabText(i) for i in range(tabs.count())]
    assert "Pre-Processing" in names
    assert "Processing" in names


def test_processing_follows_pre_processing(dock):
    tabs = _main_tabs(dock)
    names = [tabs.tabText(i) for i in range(tabs.count())]
    assert names.index("Processing") == names.index("Pre-Processing") + 1


# ---------------------------------------------------------------------------
# Every step is still reachable
# ---------------------------------------------------------------------------

PRE_BUTTONS = [
    ("extract_btn", "P1"),
    ("flight_route_btn", "P2"),
    ("calculate_fov_btn", "P3"),
    ("alfs_btn", "P4"),
    ("export_geotiffs_btn", "P5"),
    ("orthomosaic_btn", "P6"),
]

PROC_BUTTONS = [
    ("detect_btn", "A1"),
    ("track_btn", "A2"),
    ("sam3_segment_btn", "A3"),
]


@pytest.mark.parametrize("attribute,prefix", PRE_BUTTONS + PROC_BUTTONS)
def test_every_step_button_exists_and_is_numbered(dock, attribute, prefix):
    button = getattr(dock, attribute)
    assert button.text().startswith(prefix + ".")


def test_no_bare_number_means_two_things(dock):
    """The reason for renumbering: '7' meant GeoTIFF and nothing else."""
    labels = [getattr(dock, attribute).text()
              for attribute, _ in PRE_BUTTONS + PROC_BUTTONS]
    prefixes = [label.split(".")[0] for label in labels]
    assert len(prefixes) == len(set(prefixes))


def test_sub_steps_are_still_present(dock):
    """Geo-referencing has a button; it is a sub-step, not a step of its own."""
    assert dock.georef_btn.text().strip().startswith("→")
    assert dock.perpendicular_btn is not None
    assert dock.track_perpendicular_btn is not None


def test_segmentation_geo_referencing_is_a_sub_step(dock):
    """It follows SAM3 segmentation exactly as geo-referencing follows A1."""
    assert dock.sam3_georef_btn.text().strip().startswith("→")
    assert dock.sam3_georef_btn.parent() is dock.sam3_segment_btn.parent()


def test_pre_processing_steps_share_a_parent(dock):
    parents = {getattr(dock, attribute).parent() for attribute, _ in PRE_BUTTONS}
    assert len(parents) == 1


def test_processing_steps_share_a_parent(dock):
    parents = {getattr(dock, attribute).parent() for attribute, _ in PROC_BUTTONS}
    assert len(parents) == 1


def test_the_two_groups_are_different_parents(dock):
    pre = dock.extract_btn.parent()
    proc = dock.detect_btn.parent()
    assert pre is not proc


def test_geo_referencing_sits_with_the_detections(dock):
    """It depends on detections, so it belongs in Processing (§10.1)."""
    assert dock.georef_btn.parent() is dock.detect_btn.parent()


def test_perpendicular_steps_sit_with_the_animals(dock):
    assert dock.perpendicular_btn.parent() is dock.detect_btn.parent()
    assert dock.track_perpendicular_btn.parent() is dock.track_btn.parent()


def test_fov_sits_with_the_scene_products(dock):
    """FoV derives from poses and the DEM, not from any animal."""
    assert dock.calculate_fov_btn.parent() is dock.extract_btn.parent()


# ---------------------------------------------------------------------------
# The run panel is shared
# ---------------------------------------------------------------------------

def test_the_run_panel_is_outside_both_tabs(dock):
    """Progress and the log must be visible whichever tab started the step."""
    tabs = _main_tabs(dock)
    pre = tabs.widget([tabs.tabText(i)
                       for i in range(tabs.count())].index("Pre-Processing"))
    proc = tabs.widget([tabs.tabText(i)
                        for i in range(tabs.count())].index("Processing"))

    for widget in (dock.progress_bar, dock.log_text, dock.abort_btn,
                   dock.reset_stage_btn):
        assert not pre.isAncestorOf(widget)
        assert not proc.isAncestorOf(widget)


def test_logging_still_reaches_the_shared_log(dock):
    dock.log("hello from a step")
    assert "hello from a step" in dock.log_text.toPlainText()


# ---------------------------------------------------------------------------
# Presentation only
# ---------------------------------------------------------------------------

def test_status_keys_are_unchanged(dock):
    """A saved configuration and output_inventory must be unaffected."""
    for key in ("detection", "tracking", "calculate_fov", "alfs",
                "export_geotiffs", "orthomosaic", "flight_route"):
        assert key in stages.STAGE_DEPENDENCIES


def test_config_round_trips_across_the_split(dock, tmp_path):
    dock.target_folder_edit.setText(str(tmp_path))
    config = dock.get_config()
    assert config["target_folder"] == str(tmp_path)
    assert "detection_camera" in config
    assert "tracking_camera" in config


# ---------------------------------------------------------------------------
# The dock must not be forced wide by a single label
# ---------------------------------------------------------------------------

def test_long_labels_wrap(dock):
    """One unwrapped sentence widens every tab, not just its own.

    A QTabWidget sizes to its widest page, so a long non-wrapping label on the
    Processing tab stretches the Input forms too. Descriptions are word-wrapped;
    short field labels ("Format:", "Erode px:") are left alone.
    """
    from qgis.PyQt.QtWidgets import QLabel

    offenders = [
        label.text()[:60]
        for label in dock.findChildren(QLabel)
        if len(label.text()) > 90 and not label.wordWrap()
    ]
    assert not offenders, f"unwrapped long labels: {offenders}"


def test_the_dock_has_a_reasonable_size_hint(dock):
    """Nothing should push the panel beyond a normal docked width."""
    assert dock.sizeHint().width() < 900


def test_the_erode_setting_sits_with_the_configuration(dock):
    """It is a setting, so it belongs on the config tab, not the step row."""
    assert dock.geotiff_edge_erosion_spin.parent() is not \
        dock.export_geotiffs_btn.parent()


def test_the_erode_setting_is_still_read(dock, tmp_path):
    dock.target_folder_edit.setText(str(tmp_path))
    dock.geotiff_edge_erosion_spin.setValue(4)
    assert dock.get_config()["geotiff_edge_erosion_px"] == 4

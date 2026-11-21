# -*- coding: utf-8 -*-
"""The sub tabs of the Survey Analytics tab (dock/analytics_tab.py): the
tools sorted by the question they answer, under what holds for all of
them.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
from qgis.PyQt.QtWidgets import QGroupBox

from bambi_wildlife_detection.dock.analytics_tab import ANALYTICS_PAGES
from bambi_wildlife_detection.gui_utils import FitTabWidget


def _titles(tabs):
    return [tabs.tabText(index) for index in range(tabs.count())]


def _groups(page):
    """The titles of the sections of a page, top to bottom."""
    layout = page.layout()
    widgets = [layout.itemAt(index).widget()
               for index in range(layout.count())]
    return [widget.title() for widget in widgets
            if isinstance(widget, QGroupBox)]


def test_the_tools_are_on_four_sub_tabs(dock):
    tabs = dock.analytics_sub_tabs

    assert isinstance(tabs, FitTabWidget)
    # "&&" is how a tab says "&"
    assert _titles(tabs) == ["Overview", "Maps", "Population && Behaviour",
                             "Quality"]


def test_each_holds_the_tools_of_its_question(dock):
    tabs = dock.analytics_sub_tabs
    held = {tabs.tabText(index): _groups(tabs.widget(index))
            for index in range(tabs.count())}

    assert held["Overview"] == [
        "Census Summary", "Track Inventory", "Survey Report"]
    assert held["Maps"] == [
        "Density Heatmap", "Grid Density", "Coverage Map",
        "Survey Orthomosaic"]
    assert held["Population && Behaviour"][:4] == [
        "Distance Sampling", "Population Estimation (Transects)",
        "Compare Visits", "Reaction to the Drone"]
    assert len(held["Population && Behaviour"]) == 5       # and the habitat
    assert held["Quality"] == ["Review Queue", "Position Uncertainty"]


def test_every_section_is_on_one_page_and_one_only(dock):
    placed = [name for _title, held in ANALYTICS_PAGES for name in held]

    assert sorted(placed) == sorted(dock.analytics_sections)
    assert len(placed) == len(set(placed))


def test_what_holds_for_every_tool_stays_above_them(dock):
    tabs = dock.analytics_sub_tabs
    for panel in (dock.survey_panel, dock.survey_area_panel,
                  dock.analytics_species_list):
        assert not tabs.isAncestorOf(panel)
    tab = tabs.parentWidget()
    assert tab.isAncestorOf(dock.survey_panel)
    assert tab.layout().indexOf(dock.survey_area_panel) < tab.layout(
    ).indexOf(tabs)


def test_a_short_sub_tab_is_not_as_tall_as_the_longest(dock):
    tabs = dock.analytics_sub_tabs
    heights = []
    for index in range(tabs.count()):
        tabs.setCurrentIndex(index)
        heights.append(tabs.sizeHint().height())

    assert max(heights) > min(heights) + 100

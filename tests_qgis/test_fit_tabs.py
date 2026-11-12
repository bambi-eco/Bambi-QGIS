# -*- coding: utf-8 -*-
"""The dock's tabs are as tall as the page in front (gui_utils.FitTabWidget).

A plain QTabWidget is as tall as its longest page, which left a long stretch
of nothing between every shorter tab and the log below it.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import pytest

from qgis.PyQt.QtWidgets import (
    QApplication, QLabel, QScrollArea, QVBoxLayout, QWidget)

from bambi_wildlife_detection.gui_utils import FitTabWidget


def _settle():
    for _ in range(6):
        QApplication.processEvents()


def _page(height):
    page = QWidget()
    page.setMinimumHeight(height)
    return page


def test_the_hint_follows_the_current_page():
    tabs = FitTabWidget()
    tabs.addTab(_page(100), "short")
    tabs.addTab(_page(900), "long")
    short = tabs.sizeHint().height()
    assert 100 <= short < 200
    assert 100 <= tabs.minimumSizeHint().height() < 200
    tabs.setCurrentIndex(1)
    assert tabs.sizeHint().height() == short + 800
    # the width is the widest page's whichever page is in front
    tabs.widget(1).setMinimumWidth(700)
    tabs.setCurrentIndex(0)
    assert tabs.minimumSizeHint().width() >= 700


def test_word_wrapped_pages_are_measured_alone():
    tabs = FitTabWidget()
    for text in ("short", "long " * 400):
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)
        tabs.addTab(page, text[:5])
    assert tabs.hasHeightForWidth()
    short = tabs.heightForWidth(300)
    tabs.setCurrentIndex(1)
    assert tabs.heightForWidth(300) > short + 200
    assert FitTabWidget().sizeHint().isValid()      # no page: no crash


@pytest.fixture
def shown(dock):
    dock.resize(640, 900)
    dock.show()
    _settle()
    return dock, dock.findChild(QScrollArea)


def _select(tabs, title):
    index = [tabs.tabText(i) for i in range(tabs.count())].index(title)
    tabs.setCurrentIndex(index)
    _settle()


def test_a_short_tab_is_not_followed_by_empty_space(shown):
    dock, area = shown
    _select(dock.main_tabs, "Survey Analytics")
    long_content = area.widget().height()
    _select(dock.main_tabs, "Pre-Processing")
    content = area.widget().height()
    assert content < long_content - 800
    # the tab widget takes what its page asks for, and no more
    assert dock.main_tabs.height() <= dock.main_tabs.heightForWidth(
        dock.main_tabs.width())


def test_a_configuration_sub_tab_resizes_the_whole_dock(shown):
    dock, area = shown
    _select(dock.main_tabs, "Configuration")
    _select(dock.config_sub_tabs, "Classification")
    long_content = area.widget().height()
    _select(dock.config_sub_tabs, "Field of View")
    # the short sub-tab fits the dock: nothing left to scroll
    assert area.widget().height() == area.viewport().height()
    assert long_content > area.viewport().height() + 500
    _select(dock.config_sub_tabs, "Classification")
    assert area.widget().height() == long_content


def test_every_tab_widget_of_the_dock_fits_its_page(dock):
    from qgis.PyQt.QtWidgets import QTabWidget

    plain = [widget for widget in dock.findChildren(QTabWidget)
             if not isinstance(widget, FitTabWidget)]
    assert plain == []

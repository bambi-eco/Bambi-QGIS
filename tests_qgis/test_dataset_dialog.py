# -*- coding: utf-8 -*-
"""Download Example Flights: the list in which the flights of the BAMBI
dataset are ticked (bambi_dataset_dialog.py).

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import pytest

from qgis.core import QgsApplication
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QAbstractItemView, QFileDialog

from bambi_wildlife_detection import bambi_dataset_dialog as module
from bambi_wildlife_detection.core import bambi_dataset as bd


def flight(number, split="", labelled=True, raw=True):
    def archive(record):
        return bd.Archive(record, f"flight_{number}.zip")

    return bd.Flight(number, archive(111) if raw else None,
                     archive(222) if labelled else None, split)


FLIGHTS = [flight(306, "val"), flight(9, "train"), flight(10, "train"),
           flight(23, labelled=False), flight(500, "test", raw=False)]
RECORDS = {111: {"kind": "raw", "title": "The BAMBI Raw Dataset (Part 1)",
                 "doi": "10.5281/zenodo.111"},
           222: {"kind": "labelled", "title": "The BAMBI Dataset (Part 1)",
                 "doi": "10.5281/zenodo.222"}}


@pytest.fixture
def dialog():
    made = module.DatasetFlightsDialog(FLIGHTS, folder="/data/flights",
                                       records=RECORDS)
    yield made
    made.deleteLater()


def column(dialog, index):
    return [dialog.table.item(row, index).text()
            for row in range(dialog.table.rowCount())
            if not dialog.table.isRowHidden(row)]


def tick(dialog, number, on=True):
    for row in range(dialog.table.rowCount()):
        item = dialog.table.item(row, 0)
        if item.text() == str(number):
            item.setCheckState(Qt.CheckState.Checked if on
                               else Qt.CheckState.Unchecked)
            return
    raise AssertionError(f"no flight {number} in the list")


def test_every_flight_by_its_number(dialog):
    assert column(dialog, 0) == ["9", "10", "23", "306", "500"]   # numbers
    assert column(dialog, 1) == ["train", "train", "–", "val", "test"]
    assert column(dialog, 2) == ["yes", "yes", "–", "yes", "yes"]
    assert column(dialog, 3) == ["yes", "yes", "yes", "yes", "–"]
    assert [dialog.table.horizontalHeaderItem(index).text()
            for index in range(4)] == ["Flight", "Split", "Labelled",
                                       "Recordings"]


def test_a_flight_has_a_box_to_tick(dialog):
    for row in range(dialog.table.rowCount()):
        item = dialog.table.item(row, 0)
        assert item.flags() & Qt.ItemFlag.ItemIsUserCheckable
    # ... and nothing in the list can be typed over
    assert dialog.table.editTriggers() == \
        QAbstractItemView.EditTrigger.NoEditTriggers


def test_the_example_is_ticked_to_begin_with(dialog):
    assert bd.EXAMPLE == 306
    assert dialog.selected_ids() == [306]
    assert dialog.selected_flights() == [FLIGHTS[0]]
    assert dialog.summary_label.text() == "1 of 5 flight(s) ticked"
    assert dialog.download_btn.isEnabled()
    assert dialog.download_btn.text() == "Download"


def test_the_ticked_flight_is_in_view_when_the_list_opens():
    many = [flight(number) for number in range(400)]
    made = module.DatasetFlightsDialog(many, folder="/data")
    made.show()
    QgsApplication.instance().processEvents()

    shown = made.table.viewport().rect()
    assert shown.intersects(made.table.visualItemRect(
        made.table.item(306, 0)))
    made.close()
    made.deleteLater()


def test_select_all(dialog):
    dialog.select_all_btn.click()

    assert dialog.selected_ids() == [9, 10, 23, 306, 500]
    assert dialog.summary_label.text().startswith("5 of 5 flight(s) ticked")


def test_select_none(dialog):
    dialog.select_all_btn.click()
    dialog.select_none_btn.click()

    assert dialog.selected_ids() == []
    assert dialog.summary_label.text() == "0 of 5 flight(s) ticked"
    assert not dialog.download_btn.isEnabled()


def test_the_list_can_be_narrowed(dialog):
    dialog.filter_edit.setText("train")
    assert column(dialog, 0) == ["9", "10"]
    assert dialog.summary_label.text() == (
        "1 of 5 flight(s) ticked - 2 shown")

    dialog.filter_edit.setText("30")
    assert column(dialog, 0) == ["306"]
    dialog.filter_edit.setText("VAL 306")
    assert column(dialog, 0) == ["306"]
    dialog.filter_edit.setText("ibex")
    assert column(dialog, 0) == []
    dialog.filter_edit.setText("")
    assert len(column(dialog, 0)) == 5


def test_select_all_takes_the_flights_that_are_shown(dialog):
    dialog.select_none_btn.click()
    dialog.filter_edit.setText("train")

    dialog.select_all_btn.click()

    assert dialog.selected_ids() == [9, 10]
    dialog.filter_edit.setText("")
    assert dialog.selected_ids() == [9, 10]


def test_select_none_also_unticks_what_is_not_shown(dialog):
    dialog.filter_edit.setText("train")       # 306 is ticked, and hidden

    dialog.select_none_btn.click()

    assert dialog.selected_ids() == []


def test_the_list_is_put_in_order_by_what_a_column_means(dialog):
    dialog.table.sortItems(0, Qt.SortOrder.DescendingOrder)
    assert column(dialog, 0) == ["500", "306", "23", "10", "9"]
    assert dialog.selected_ids() == [306]            # the tick went along

    dialog.table.sortItems(2, Qt.SortOrder.AscendingOrder)
    assert column(dialog, 0)[-1] == "23"             # not labelled: last


def test_the_videos_are_fetched_unless_unticked(dialog):
    assert dialog.videos_check.text() == "Download video data"
    assert dialog.download_videos() is True
    assert "fast overview" in dialog.videos_check.toolTip()

    dialog.videos_check.setChecked(False)

    assert dialog.download_videos() is False


def test_what_is_done_with_the_annotations(dialog):
    # the route and the correction unless unticked; labels when asked for
    assert dialog.route_check.text() == "Import flight route"
    assert dialog.labels_check.text() == "Import labels"
    assert (dialog.import_route(), dialog.import_labels(),
            dialog.use_correction()) == (True, False, True)

    dialog.route_check.setChecked(False)
    dialog.labels_check.setChecked(True)
    dialog.correction_check.setChecked(False)

    assert (dialog.import_route(), dialog.import_labels(),
            dialog.use_correction()) == (False, True, False)
    assert "poses file" in dialog.route_check.toolTip()
    assert "between the key frames" in dialog.labels_check.toolTip()


def test_how_it_opens_can_be_said():
    made = module.DatasetFlightsDialog(
        FLIGHTS, ticked=(9, 10), route=False, labels=True, correction=False,
        videos=False)

    assert made.selected_ids() == [9, 10]
    assert (made.import_route(), made.import_labels(), made.use_correction(),
            made.download_videos()) == (False, True, False, False)
    made.deleteLater()


def test_a_ticked_flight_the_dataset_has_no_recordings_of_is_said(dialog):
    tick(dialog, 500)

    assert dialog.summary_label.text() == (
        "2 of 5 flight(s) ticked; 1 without recordings")
    assert "color" in dialog.summary_label.styleSheet()
    assert dialog.summary_label.toolTip() == (
        "The dataset has no recordings of flight(s) 500: they are left out.")

    dialog.videos_check.setChecked(False)       # its annotations are there

    assert dialog.summary_label.text() == "2 of 5 flight(s) ticked"
    assert dialog.summary_label.styleSheet() == ""


def test_without_the_videos_an_unlabelled_flight_has_nothing(dialog):
    tick(dialog, 23)
    assert dialog.summary_label.text() == "2 of 5 flight(s) ticked"

    dialog.videos_check.setChecked(False)

    assert dialog.summary_label.text() == (
        "2 of 5 flight(s) ticked; 1 without poses or labels")


def test_without_a_folder_nothing_can_be_downloaded():
    made = module.DatasetFlightsDialog(FLIGHTS)

    assert made.folder() == ""
    assert not made.download_btn.isEnabled()

    made.folder_edit.setText("  /data/flights ")
    assert made.folder() == "/data/flights"
    assert made.download_btn.isEnabled()
    made.deleteLater()


def test_the_folder_can_be_browsed_for(dialog, monkeypatch):
    asked = []

    def choose(_parent, title, start):
        asked.append((title, start))
        return "/media/disk/bambi"

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", choose)

    dialog.browse_btn.click()

    assert asked == [("Where to put the flights", "/data/flights")]
    assert dialog.folder() == "/media/disk/bambi"

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: "")
    dialog.browse_btn.click()
    assert dialog.folder() == "/media/disk/bambi"


def test_a_row_says_where_its_archives_are_published(dialog):
    tips = {dialog.table.item(row, 0).text():
            dialog.table.item(row, 1).toolTip()
            for row in range(dialog.table.rowCount())}

    assert tips["306"] == (
        "BAMBI flight 306\n"
        "Recordings: The BAMBI Raw Dataset (Part 1), "
        "https://doi.org/10.5281/zenodo.111\n"
        "Labelled flight: The BAMBI Dataset (Part 1), "
        "https://doi.org/10.5281/zenodo.222")
    assert "Labelled flight: not in the dataset" in tips["23"]
    assert "Recordings: not in the dataset" in tips["500"]

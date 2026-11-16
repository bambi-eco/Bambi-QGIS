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

MB = 1_000_000


def flight(number, size, date="", species=(), labelled=True, drone="",
           split="train", videos=True):
    def archive(record, length):
        return bd.Archive(record, f"flight_{number}.zip", length, "00" * 16)

    return bd.Flight(
        id=number, raw=archive(1, size),
        labelled=archive(2, 5 * MB) if labelled else None, date=date,
        species=tuple(species), animals=sum(n for _name, n in species),
        boxes=10 * sum(n for _name, n in species), drone=drone, split=split,
        videos=videos)


FLIGHTS = [
    flight(306, 122 * MB, "2024-11-19",
           [("Capreolus capreolus (Roe deer)", 2)], drone="Matrice 30 Thermal",
           split="val"),
    flight(9, 900 * MB, "2023-07-27", [("Sus scrofa (Wild boar)", 29),
                                       ("Unknown", 2)],
           drone="Mavic 3 Thermal"),
    flight(10, 4700 * MB, "2023-08-02", [], drone="Mavic 3 Thermal"),
    flight(23, 1300 * MB, labelled=False),
]


@pytest.fixture
def free(monkeypatch):
    room = {"bytes": 500_000 * MB}
    monkeypatch.setattr(module, "free_bytes", lambda folder: room["bytes"])
    return room


@pytest.fixture
def dialog(free):
    made = module.DatasetFlightsDialog(FLIGHTS, folder="/data/flights")
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
    assert column(dialog, 0) == ["9", "10", "23", "306"]      # as numbers
    assert column(dialog, 1) == ["2023-07-27", "2023-08-02", "–",
                                 "2024-11-19"]
    assert column(dialog, 2) == ["29 wild boar, 2 unknown", "none",
                                 "not labelled", "2 roe deer"]
    assert column(dialog, 3) == ["900 MB", "4.7 GB", "1.3 GB", "122 MB"]
    assert [dialog.table.horizontalHeaderItem(index).text()
            for index in range(4)] == ["Flight", "Flown", "Labelled animals",
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
    assert dialog.selected_size() == 122 * MB
    assert dialog.summary_label.text() == (
        "1 of 4 flight(s) ticked: 122 MB (500.0 GB free)")
    assert dialog.download_btn.isEnabled()
    assert dialog.download_btn.text() == "Download"


def test_the_ticked_flight_is_in_view_when_the_list_opens(free):
    made = module.DatasetFlightsDialog(bd.flights(), folder="/data")
    made.show()
    QgsApplication.instance().processEvents()

    ticked = [row for row in range(made.table.rowCount())
              if made.table.item(row, 0).text() == "306"][0]
    shown = made.table.viewport().rect()
    assert shown.intersects(made.table.visualItemRect(
        made.table.item(ticked, 0)))
    made.close()
    made.deleteLater()


def test_ticking_flights_adds_them_up(dialog):
    tick(dialog, 9)
    tick(dialog, 10)

    assert dialog.selected_ids() == [9, 10, 306]
    assert dialog.selected_size() == (900 + 4700 + 122) * MB
    assert dialog.summary_label.text().startswith(
        "3 of 4 flight(s) ticked: 5.7 GB")


def test_select_all(dialog):
    dialog.select_all_btn.click()

    assert dialog.selected_ids() == [9, 10, 23, 306]
    assert dialog.summary_label.text().startswith(
        "4 of 4 flight(s) ticked: 7.0 GB")


def test_select_none(dialog):
    dialog.select_all_btn.click()
    dialog.select_none_btn.click()

    assert dialog.selected_ids() == []
    assert dialog.summary_label.text() == "0 of 4 flight(s) ticked: 0 MB"
    assert not dialog.download_btn.isEnabled()


def test_the_list_can_be_narrowed(dialog):
    dialog.filter_edit.setText("mavic 2023")

    assert column(dialog, 0) == ["9", "10"]
    assert dialog.summary_label.text() == (
        "1 of 4 flight(s) ticked: 122 MB - 2 shown (500.0 GB free)")

    dialog.filter_edit.setText("roe deer")
    assert column(dialog, 0) == ["306"]
    dialog.filter_edit.setText("capreolus")          # either name
    assert column(dialog, 0) == ["306"]
    dialog.filter_edit.setText("not labelled")
    assert column(dialog, 0) == ["23"]
    dialog.filter_edit.setText("ibex")
    assert column(dialog, 0) == []
    dialog.filter_edit.setText("")
    assert column(dialog, 0) == ["9", "10", "23", "306"]


def test_select_all_takes_the_flights_that_are_shown(dialog):
    dialog.select_none_btn.click()
    dialog.filter_edit.setText("wild boar")

    dialog.select_all_btn.click()

    assert dialog.selected_ids() == [9]
    dialog.filter_edit.setText("")
    assert dialog.selected_ids() == [9]


def test_select_none_also_unticks_what_is_not_shown(dialog):
    dialog.filter_edit.setText("wild boar")       # 306 is ticked, and hidden

    dialog.select_none_btn.click()

    assert dialog.selected_ids() == []


def test_the_list_is_put_in_order_by_what_a_column_means(dialog):
    dialog.table.sortItems(3, Qt.SortOrder.AscendingOrder)
    assert column(dialog, 3) == ["122 MB", "900 MB", "1.3 GB", "4.7 GB"]

    dialog.table.sortItems(2, Qt.SortOrder.AscendingOrder)
    # the flights with most labelled animals first
    assert column(dialog, 0)[:2] == ["9", "306"]

    dialog.table.sortItems(0, Qt.SortOrder.DescendingOrder)
    assert column(dialog, 0) == ["306", "23", "10", "9"]
    assert dialog.selected_ids() == [306]            # the tick went along


def test_what_else_is_taken(dialog):
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


def test_how_it_opens_can_be_said(free):
    made = module.DatasetFlightsDialog(
        FLIGHTS, ticked=(9, 10), route=False, labels=True, correction=False)

    assert made.selected_ids() == [9, 10]
    assert (made.import_route(), made.import_labels(),
            made.use_correction()) == (False, True, False)
    made.deleteLater()


def test_without_a_folder_nothing_can_be_downloaded(free):
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


def test_flights_that_do_not_fit_are_said_before_they_are_fetched(dialog,
                                                                  free):
    dialog.select_all_btn.click()            # 7.0 GB, the largest 4.7 GB
    assert dialog.summary_label.styleSheet() == ""

    free["bytes"] = 11_000 * MB              # not with the largest unpacking
    dialog.folder_edit.setText("/small/disk")

    assert dialog.summary_label.text() == (
        "4 of 4 flight(s) ticked: 7.0 GB (11.0 GB free)")
    assert "color" in dialog.summary_label.styleSheet()
    assert "do not all fit" in dialog.summary_label.toolTip()
    assert dialog.download_btn.isEnabled()   # those that fit are fetched


def test_a_disk_nobody_can_measure(dialog, free):
    free["bytes"] = None
    dialog.folder_edit.setText("/somewhere/else")

    assert dialog.summary_label.text() == "1 of 4 flight(s) ticked: 122 MB"
    assert dialog.summary_label.styleSheet() == ""


def test_a_row_says_more_when_pointed_at(dialog):
    tips = {dialog.table.item(row, 0).text():
            dialog.table.item(row, 2).toolTip()
            for row in range(dialog.table.rowCount())}

    assert tips["306"] == (
        "BAMBI flight 306\n"
        "Flown on 2024-11-19 with a Matrice 30 Thermal\n"
        "2 labelled animal(s), 20 boxes on key frames (val split)\n"
        "Recordings: https://doi.org/10.5281/zenodo.1\n"
        "Labelled: https://doi.org/10.5281/zenodo.2")
    assert "Not in the labelled dataset: recordings only" in tips["23"]
    assert "without an animal on it" in tips["10"]


def test_a_flight_published_without_its_videos_cannot_be_ticked(free):
    made = module.DatasetFlightsDialog(
        FLIGHTS + [flight(397, 233594, "2024-11-22", videos=False)],
        folder="/data", ticked=(306, 397))

    assert column(made, 0) == ["9", "10", "23", "306", "397"]
    assert column(made, 3)[-1] == "no video"
    item = made.table.item(4, 0)
    assert not item.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert not item.flags() & Qt.ItemFlag.ItemIsEnabled
    assert "Published without its videos" in item.toolTip()
    assert made.selected_ids() == [306]

    made.select_all_btn.click()

    assert made.selected_ids() == [9, 10, 23, 306]
    assert made.summary_label.text().startswith("4 of 4 flight(s) ticked")
    made.deleteLater()


def test_sizes_to_read():
    assert module.size_text(121842908) == "122 MB"
    assert module.size_text(4_672_127_147) == "4.7 GB"
    assert module.size_text(2_154_000_000_000) == "2.15 TB"


def test_the_room_of_a_folder_that_is_not_there_yet(tmp_path):
    there = module.free_bytes(str(tmp_path))

    assert there > 0
    assert module.free_bytes(str(tmp_path / "not" / "yet")) == there
    assert module.free_bytes("") is None


def test_the_shipped_list_opens(free):
    made = module.DatasetFlightsDialog(bd.flights(), folder="/data")

    assert made.table.rowCount() == 400
    assert made.selected_ids() == [306]
    made.select_all_btn.click()
    # all but the seven that were published without their videos
    assert len(made.selected_ids()) == 393
    assert not set(range(397, 404)) & set(made.selected_ids())
    assert made.summary_label.text().startswith(
        "393 of 393 flight(s) ticked: 2.15 TB")
    made.filter_edit.setText("alpine ibex")
    assert 0 < len(column(made, 0)) < 400
    made.deleteLater()

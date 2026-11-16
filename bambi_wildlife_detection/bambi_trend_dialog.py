# -*- coding: utf-8 -*-
"""Compare Visits: the survey visit by visit, as a chart and a table.

A window over the survey's ``trend_{source}.json``
(:mod:`core.survey_trend`). For the species chosen at the top it draws the
density of every visit with its interval and the fitted rate of change,
lists the visits - when, which flights, how much ground, how many animals -
and says in words what changed from visit to visit and over all of them.

This module only shows; every number comes from the file.
"""

from datetime import datetime
from typing import List, Optional, Tuple

from qgis.PyQt.QtCore import QPointF, QRectF, Qt
from qgis.PyQt.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QDialogButtonBox,
    QHBoxLayout, QLabel, QSizePolicy, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .core import census, survey_trend

POINT_COLOR = QColor(31, 119, 180)
LINE_COLOR = QColor(214, 96, 28)
TEXT_COLOR = QColor(60, 60, 60)
GRID_COLOR = QColor(228, 228, 228)

_RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter


def _number(value, digits: int = 1) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _with_interval(value, low, high, digits: int = 1) -> str:
    if value is None:
        return "–"
    text = _number(value, digits)
    if low is not None and high is not None:
        text += f" ({_number(low, digits)}–{_number(high, digits)})"
    return text


def _breakdown(counts: dict) -> str:
    """``female 2, male 1`` - what is known, the most common first."""
    known = [(label, n) for label, n in counts.items()
             if label != census.UNKNOWN and n]
    known.sort(key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{label} {n}" for label, n in known)


def axis_step(top: float, most: int = 5) -> float:
    """The distance between the marks of an axis from nought to *top*: a
    round number that leaves no more than *most* marks above nought."""
    import math

    if top <= 0:
        return 1.0
    raw = top / most
    power = 10.0 ** math.floor(math.log10(raw))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        if factor * power >= raw:
            return factor * power
    return 10.0 * power


def species_title(name: str) -> str:
    return "All animals" if name == survey_trend.ALL else name


def chart_points(document: dict, species: str) -> dict:
    """What the chart of one species draws: per visit the day, the value
    with its interval and the fitted value, and what the values are.

    The values are densities per km² where the ground looked at is known,
    counts otherwise.
    """
    by_density = document.get("basis") == survey_trend.BASIS_GROUND
    keys = (("density_per_km2", "density_low", "density_high")
            if by_density else ("animals", "animals_low", "animals_high"))
    entries = document["series"].get(species, [])
    trend = document["trends"].get(species) or {}
    fitted = trend.get("fitted")
    return {
        "days": [visit["day"] for visit in document["visits"]],
        "labels": [visit["label"] for visit in document["visits"]],
        "values": [entry[keys[0]] for entry in entries],
        "lows": [entry[keys[1]] for entry in entries],
        "highs": [entry[keys[2]] for entry in entries],
        "fitted": list(fitted) if fitted else None,
        "unit": "animals / km²" if by_density else "animals counted",
    }


def visit_rows(document: dict, species: str) -> Tuple[List[str], List[list]]:
    """``(headers, rows)`` of the table of visits for one species, every
    cell a string."""
    by_density = document.get("basis") == survey_trend.BASIS_GROUND
    share = int(round(document.get("level", survey_trend.LEVEL) * 100))
    entries = document["series"].get(species, [])
    with_number = any(entry.get("number") is not None for entry in entries)
    with_sex = any(_breakdown(entry.get("sex") or {}) for entry in entries)
    with_age = any(_breakdown(entry.get("age") or {}) for entry in entries)
    temperature = "air_temperature_c"
    with_weather = any((visit.get("weather") or {}).get(temperature)
                       is not None for visit in document["visits"])
    previous = {change["to"]: change
                for change in document["changes"].get(species, [])
                if change["to"] - change["from"] == 1}

    headers = ["Visit", "Date", "Time", "Flights"]
    if by_density:
        headers += ["Looked at (ha)", "Animals",
                    f"Density /km² ({share} %)"]
    else:
        headers += [f"Animals ({share} %)"]
    if with_number:
        headers.append("In the survey area")
    headers += ["Groups", "Largest group", "Since the visit before"]
    if with_sex:
        headers.append("Sex")
    if with_age:
        headers.append("Age")
    if with_weather:
        headers.append("Air °C")

    rows = []
    for visit, entry in zip(document["visits"], entries):
        row = [str(visit["visit"]), visit["label"], visit["time_of_day"],
               ", ".join(str(name) for name in visit["flights"])]
        if by_density:
            row += [_number(visit.get("looked_at_ha"), 1),
                    str(entry["animals"]),
                    _with_interval(entry["density_per_km2"],
                                   entry["density_low"],
                                   entry["density_high"])]
        else:
            row += [_with_interval(entry["animals"], entry["animals_low"],
                                   entry["animals_high"])]
        if with_number:
            row.append(_with_interval(entry.get("number"),
                                      entry.get("number_low"),
                                      entry.get("number_high"), 0))
        change = previous.get(entry["visit"])
        if change is None:
            since = ""
        elif change.get("p_value") is None:
            since = "–"
        else:
            if change.get("ratio") is None:
                since = "from none"
            elif change["ratio"] == 0:
                since = "to none"
            else:
                since = survey_trend.signed(change["change_pct"])
            chance = change["p_value"]
            since += (" (p < 0.001)" if chance < 0.001
                      else f" (p = {chance:.3f})")
        row += [str(entry.get("groups") or 0),
                "–" if entry.get("largest_group") is None
                else str(entry["largest_group"]), since]
        if with_sex:
            row.append(_breakdown(entry.get("sex") or {}))
        if with_age:
            row.append(_breakdown(entry.get("age") or {}))
        if with_weather:
            row.append(_number((visit.get("weather") or {}).get(temperature)))
        rows.append(row)
    return headers, rows


def changes_text(document: dict, species: str) -> str:
    """The changes of one species from visit to visit, a line each - the
    last line is the change from the first visit to the last."""
    labels = {visit["visit"]: visit["label"] for visit in document["visits"]}
    return "\n".join(
        survey_trend.change_sentence(change, labels)
        for change in document["changes"].get(species, []))


def overall_text(document: dict, species: str) -> str:
    """The first visit against the last, in a sentence."""
    lines = changes_text(document, species).splitlines()
    return f"First against last visit - {lines[-1]}" if lines else ""


class TrendChart(QWidget):
    """One species over the visits: a point per visit with its interval,
    and the fitted rate of change as a line."""

    MARGIN_LEFT = 56
    MARGIN_RIGHT = 20
    MARGIN_TOP = 28
    MARGIN_BOTTOM = 34

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: Optional[dict] = None
        self._title = ""
        self.setMinimumHeight(230)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def set_points(self, points: dict, title: str = "") -> None:
        """Draw *points* (:func:`chart_points`)."""
        self._points = points
        self._title = title
        self.update()

    def points(self) -> Optional[dict]:
        return self._points

    def top_value(self) -> float:
        """The value the upper end of the axis stands for."""
        points = self._points or {}
        shown = [value for key in ("values", "fitted")
                 for value in (points.get(key) or []) if value is not None]
        highs = [value for value in (points.get("highs") or [])
                 if value is not None]
        if not shown:
            return 1.0
        top = max(shown)
        # an interval that reaches far above everything else is cut off (and
        # drawn with an open end) rather than flattening the points
        reach = min(max(highs), 3.0 * top) if highs else top
        return max(top, reach, 1e-9) * 1.08

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(252, 252, 252))
        rect = QRectF(
            self.MARGIN_LEFT, self.MARGIN_TOP,
            max(10, self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT),
            max(10, self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM))
        font = QFont()
        font.setPointSize(8)
        metrics = QFontMetrics(font)
        title_font = QFont()
        title_font.setPointSize(9)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(TEXT_COLOR)
        points = self._points
        unit = (points or {}).get("unit", "")
        painter.drawText(QRectF(8, 4, self.width() - 16, 18),
                         Qt.AlignmentFlag.AlignLeft,
                         f"{self._title} - {unit}" if unit else self._title)
        painter.setFont(font)
        if not points or not points.get("values"):
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No data")
            return

        days = points["days"]
        first, last = min(days), max(days)
        span = (last - first) or 1.0
        top = self.top_value()

        def x_of(day: float) -> float:
            return rect.left() + (0.06 + 0.88 * (day - first) / span) \
                * rect.width()

        def y_of(value: float) -> float:
            return rect.bottom() - min(value, top) / top * rect.height()

        step = axis_step(top)
        marks = [step * n for n in range(int(top / step + 1e-9) + 1)]
        painter.setPen(QPen(GRID_COLOR, 1))
        for value in marks[1:]:
            painter.drawLine(QPointF(rect.left(), y_of(value)),
                             QPointF(rect.right(), y_of(value)))
        painter.setPen(QPen(TEXT_COLOR, 1))
        painter.drawLine(QPointF(rect.left(), rect.bottom()),
                         QPointF(rect.right(), rect.bottom()))
        painter.drawLine(QPointF(rect.left(), rect.top()),
                         QPointF(rect.left(), rect.bottom()))
        whole = step >= 1 and float(step).is_integer()
        for value in marks:
            text = f"{value:,.0f}" if whole else f"{value:g}"
            painter.drawText(QRectF(0, y_of(value) - 7,
                                    self.MARGIN_LEFT - 6, 14), _RIGHT, text)

        # the dates, as many as fit side by side
        right_of_last = -1e9
        for day, label in zip(days, points["labels"]):
            x = x_of(day)
            painter.drawLine(QPointF(x, rect.bottom()),
                             QPointF(x, rect.bottom() + 3))
            text = label.split(" to ")[0]
            width = metrics.horizontalAdvance(text)
            left = x - width / 2.0
            if left < right_of_last + 8:
                continue
            painter.drawText(QPointF(left, rect.bottom() + 16), text)
            right_of_last = left + width

        fitted = points.get("fitted")
        if fitted:
            painter.setPen(QPen(LINE_COLOR, 1.6, Qt.PenStyle.DashLine))
            for index in range(len(days) - 1):
                painter.drawLine(
                    QPointF(x_of(days[index]), y_of(fitted[index])),
                    QPointF(x_of(days[index + 1]), y_of(fitted[index + 1])))

        for day, value, low, high in zip(days, points["values"],
                                         points["lows"], points["highs"]):
            if value is None:
                continue
            x = x_of(day)
            if low is not None and high is not None:
                painter.setPen(QPen(POINT_COLOR, 1.4))
                painter.drawLine(QPointF(x, y_of(low)), QPointF(x, y_of(high)))
                painter.drawLine(QPointF(x - 4, y_of(low)),
                                 QPointF(x + 4, y_of(low)))
                if high <= top:             # an interval cut off stays open
                    painter.drawLine(QPointF(x - 4, y_of(high)),
                                     QPointF(x + 4, y_of(high)))
            painter.setPen(QPen(POINT_COLOR.darker(130), 1))
            painter.setBrush(POINT_COLOR)
            painter.drawEllipse(QPointF(x, y_of(value)), 4.0, 4.0)


class TrendDialog(QDialog):
    """What the comparison of the visits found, species by species."""

    def __init__(self, document: dict, json_path: str = "", parent=None):
        super().__init__(parent)
        self._document = document
        self.setWindowTitle(
            f"Compare Visits ({document.get('camera', '')}, "
            f"{document.get('source', '')})")
        self.resize(980, 640)
        layout = QVBoxLayout(self)

        visits = document.get("visits") or []
        facts = [f"{len(visits)} visits",
                 self._span(visits),
                 "ground compared: " + survey_trend.GROUND_LABELS.get(
                     document.get("ground"), "").lower()]
        if document.get("common_ha") is not None:
            facts.append(f"{document['common_ha']:.1f} ha in common")
        if document.get("plot_ha") is not None:
            facts.append(f"survey area {document['plot_ha']:.1f} ha")
        gap = document.get("gap_days") or 0
        facts.append("one visit = the same day" if not gap else
                     f"one visit = flights up to {gap} day(s) apart")
        header = QLabel(" · ".join(fact for fact in facts if fact))
        header.setWordWrap(True)
        layout.addWidget(header)

        chooser = QHBoxLayout()
        chooser.addWidget(QLabel("Species:"))
        self.species_combo = QComboBox()
        names = list(document.get("species") or [])
        # all animals first: the question most people come with
        ordered = [name for name in names if name == survey_trend.ALL] + [
            name for name in names if name != survey_trend.ALL]
        for name in ordered:
            self.species_combo.addItem(species_title(name), name)
        self.species_combo.currentIndexChanged.connect(self._on_species)
        chooser.addWidget(self.species_combo)
        chooser.addStretch()
        layout.addLayout(chooser)

        self.trend_label = QLabel("")
        self.trend_label.setWordWrap(True)
        bold = self.trend_label.font()
        bold.setBold(True)
        self.trend_label.setFont(bold)
        layout.addWidget(self.trend_label)

        self.chart = TrendChart(self)
        layout.addWidget(self.chart, 2)

        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)   # "Visit" says it
        layout.addWidget(self.table, 2)

        self.changes_label = QLabel("")
        self.changes_label.setWordWrap(True)
        layout.addWidget(self.changes_label)

        notes = QLabel(document.get("notes", ""))
        notes.setWordWrap(True)
        notes.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(notes)
        if json_path:
            path_label = QLabel(
                f"Saved to: {json_path} (and _visits.csv beside it)")
            path_label.setStyleSheet("color: gray; font-size: 10px;")
            layout.addWidget(path_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy_btn = buttons.addButton(
            "Copy", QDialogButtonBox.ButtonRole.ActionRole)
        copy_btn.clicked.connect(self.copy_table)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_species()

    @staticmethod
    def _span(visits) -> str:
        if len(visits) < 2:
            return ""
        try:
            first = datetime.fromisoformat(visits[0]["mid"])
            last = datetime.fromisoformat(visits[-1]["mid"])
        except (KeyError, ValueError):
            return ""
        days = (last - first).total_seconds() / 86400.0
        return f"over {survey_trend.span_text(days)}"

    def species(self) -> str:
        """The species shown."""
        return self.species_combo.currentData() or survey_trend.ALL

    def _on_species(self, _index: int = 0) -> None:
        document, name = self._document, self.species()
        trend = (document.get("trends") or {}).get(name) or {}
        self.trend_label.setText(
            f"{species_title(name)}: " + survey_trend.trend_sentence(
                trend, document.get("level", survey_trend.LEVEL))
            if trend else "")
        self.chart.set_points(chart_points(document, name),
                              species_title(name))
        headers, rows = visit_rows(document, name)
        table = self.table
        table.clear()
        table.setColumnCount(len(headers))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(headers)
        text_columns = {"Date", "Time", "Flights", "Sex", "Age",
                        "Since the visit before"}
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                item = QTableWidgetItem(text)
                if headers[c] not in text_columns:
                    item.setTextAlignment(_RIGHT)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        # visit against visit is in the table; here the first against the
        # last, which no column holds
        self.changes_label.setText(overall_text(document, name))
        self.changes_label.setToolTip(changes_text(document, name))

    def copy_table(self):
        """Copy the table as tab-separated text (pastes into a
        spreadsheet)."""
        table = self.table
        lines = ["\t".join(table.horizontalHeaderItem(c).text()
                           for c in range(table.columnCount()))]
        for r in range(table.rowCount()):
            lines.append("\t".join(table.item(r, c).text()
                                   for c in range(table.columnCount())))
        QApplication.clipboard().setText("\n".join(lines))

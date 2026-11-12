# -*- coding: utf-8 -*-
"""Habitat charts: where the animals were against the ground that was flown.

A window over the survey's ``habitat.json`` and ``habitat_tracks.json``
(:mod:`core.habitat`). One chart per environment layer shows how its values
are distributed over the monitored ground (grey) and at the animals
(colour); a bar chart does the same for the land-cover classes; a table
lists each flight's satellite scene and weather. A species selector redraws
the animals' side - the ground stays what it is.

This module only draws; every number comes from the two files.
"""

from typing import List, Optional, Sequence

from qgis.PyQt.QtCore import QPointF, QRectF, Qt
from qgis.PyQt.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from qgis.PyQt.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .core import habitat

AVAILABLE_COLOR = QColor(170, 170, 170)
USED_COLOR = QColor(214, 96, 28)
TEXT_COLOR = QColor(60, 60, 60)
GRID_COLOR = QColor(228, 228, 228)

_MIDDLE = Qt.AlignmentFlag.AlignVCenter
ALIGN_LEFT_MIDDLE = Qt.AlignmentFlag.AlignLeft | _MIDDLE
ALIGN_RIGHT_MIDDLE = Qt.AlignmentFlag.AlignRight | _MIDDLE


def _shares(counts: Sequence[int]) -> List[float]:
    total = float(sum(counts))
    return [c / total if total else 0.0 for c in counts]


def _fmt(value, digits: int = 2, unit: str = "") -> str:
    if value is None:
        return "–"
    return f"{value:.{digits}f}{unit}"


class HistogramChart(QWidget):
    """One layer: its distribution over the ground and at the animals."""

    MARGIN_LEFT = 44
    MARGIN_RIGHT = 12
    MARGIN_TOP = 40
    MARGIN_BOTTOM = 30

    def __init__(self, layer: dict, parent=None):
        super().__init__(parent)
        self._layer = layer
        self._used_counts = list(layer.get("used_counts") or [])
        self._used_mean = (layer.get("used") or {}).get("mean")
        self._used_n = (layer.get("used") or {}).get("n", 0)
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setToolTip(layer.get("description", ""))

    def set_used(self, counts: Sequence[int], mean: Optional[float],
                 n: int) -> None:
        self._used_counts = list(counts)
        self._used_mean = mean
        self._used_n = n
        self.update()

    def _plot_rect(self) -> QRectF:
        margin_x = self.MARGIN_LEFT + self.MARGIN_RIGHT
        margin_y = self.MARGIN_TOP + self.MARGIN_BOTTOM
        return QRectF(self.MARGIN_LEFT, self.MARGIN_TOP,
                      max(10, self.width() - margin_x),
                      max(10, self.height() - margin_y))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(252, 252, 252))
        rect = self._plot_rect()
        layer = self._layer
        edges = layer.get("bin_edges") or []
        unit = layer.get("unit", "")

        title_font = QFont()
        title_font.setPointSize(9)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(TEXT_COLOR)
        painter.drawText(QRectF(8, 4, self.width() - 16, 16),
                         Qt.AlignmentFlag.AlignLeft, layer.get("title", ""))

        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        available = layer.get("available") or {}
        legend = (f"monitored ground: mean {_fmt(available.get('mean'), 2, unit)}"
                  f"      animals (n={self._used_n}): mean "
                  f"{_fmt(self._used_mean, 2, unit)}")
        painter.setPen(AVAILABLE_COLOR.darker(150))
        painter.drawText(QRectF(8, 20, self.width() - 16, 16),
                         Qt.AlignmentFlag.AlignLeft, legend)

        if len(edges) < 2:
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No data")
            return

        ground = _shares(layer.get("available_counts") or [])
        used = _shares(self._used_counts)
        top = max(ground + used + [0.01]) * 1.1
        low, high = edges[0], edges[-1]
        span = (high - low) or 1.0

        def x_of(value: float) -> float:
            return rect.left() + (value - low) / span * rect.width()

        def y_of(share: float) -> float:
            return rect.bottom() - share / top * rect.height()

        painter.setPen(QPen(GRID_COLOR, 1))
        for step in range(1, 4):
            y = rect.bottom() - rect.height() * step / 4.0
            painter.drawLine(QPointF(rect.left(), y),
                             QPointF(rect.right(), y))

        # The ground as filled bars, the animals as narrower bars inside
        # them, so both read even where they overlap.
        painter.setPen(Qt.PenStyle.NoPen)
        for index, share in enumerate(ground):
            x0, x1 = x_of(edges[index]), x_of(edges[index + 1])
            painter.setBrush(AVAILABLE_COLOR)
            painter.drawRect(QRectF(x0 + 0.5, y_of(share), x1 - x0 - 1,
                                    rect.bottom() - y_of(share)))
        for index, share in enumerate(used):
            if index + 1 >= len(edges) or share <= 0:
                continue
            x0, x1 = x_of(edges[index]), x_of(edges[index + 1])
            inset = (x1 - x0) * 0.28
            painter.setBrush(USED_COLOR)
            painter.drawRect(QRectF(x0 + inset, y_of(share),
                                    x1 - x0 - 2 * inset,
                                    rect.bottom() - y_of(share)))

        for mean, color in ((available.get("mean"), AVAILABLE_COLOR.darker(170)),
                            (self._used_mean, USED_COLOR.darker(130))):
            if mean is None:
                continue
            x = x_of(min(max(mean, low), high))
            painter.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))

        painter.setPen(QPen(TEXT_COLOR, 1))
        painter.drawLine(QPointF(rect.left(), rect.bottom()),
                         QPointF(rect.right(), rect.bottom()))
        painter.drawLine(QPointF(rect.left(), rect.top()),
                         QPointF(rect.left(), rect.bottom()))
        for step in range(5):
            value = low + span * step / 4.0
            text = f"{value:.2f}".rstrip("0").rstrip(".") or "0"
            x = x_of(value)
            painter.drawLine(QPointF(x, rect.bottom()),
                             QPointF(x, rect.bottom() + 3))
            width = metrics.horizontalAdvance(text)
            painter.drawText(QPointF(x - width / 2.0, rect.bottom() + 15),
                             text)
        for step in (0, 2, 4):
            share = top * step / 4.0
            text = f"{share * 100:.0f}%"
            painter.drawText(QRectF(0, y_of(share) - 7,
                                    self.MARGIN_LEFT - 5, 14),
                             ALIGN_RIGHT_MIDDLE, text)


class LandCoverChart(QWidget):
    """Land-cover classes: share of the ground against share of the animals."""

    ROW_HEIGHT = 30
    LABEL_WIDTH = 170
    NOTE_WIDTH = 170

    def __init__(self, classes: List[dict], parent=None):
        super().__init__(parent)
        self._classes = classes
        self._used = {c["name"]: (c.get("used_share"), c.get("n_tracks", 0))
                      for c in classes}
        self.setMinimumHeight(44 + self.ROW_HEIGHT * max(1, len(classes)))
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setToolTip(
            "Grey: share of the monitored ground a class covers. Colour: "
            "share of the animals found in it. The ratio of the two is the "
            "selection ratio - above 1 the class is used more than its area "
            "would suggest.")

    def set_used(self, used: dict) -> None:
        """``class name -> (share of the animals, number of animals)``."""
        self._used = used
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(252, 252, 252))
        title_font = QFont()
        title_font.setPointSize(9)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(TEXT_COLOR)
        painter.drawText(QRectF(8, 4, self.width() - 16, 16),
                         Qt.AlignmentFlag.AlignLeft, "Land cover")
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(AVAILABLE_COLOR.darker(150))
        painter.drawText(QRectF(8, 20, self.width() - 16, 16),
                         Qt.AlignmentFlag.AlignLeft,
                         "grey: monitored ground      colour: animals      "
                         "right: selection ratio")

        bar_left = self.LABEL_WIDTH
        bar_width = max(20, self.width() - self.LABEL_WIDTH - self.NOTE_WIDTH)
        shares = [c.get("available_share") or 0.0 for c in self._classes]
        shares += [share or 0.0 for share, _n in self._used.values()]
        top = max(shares + [0.01])

        for index, entry in enumerate(self._classes):
            y = 42 + index * self.ROW_HEIGHT
            name = entry["name"]
            painter.setPen(TEXT_COLOR)
            painter.drawText(QRectF(8, y, self.LABEL_WIDTH - 12,
                                    self.ROW_HEIGHT - 6),
                             ALIGN_LEFT_MIDDLE, name)
            ground = entry.get("available_share") or 0.0
            used_share, count = self._used.get(name, (None, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(AVAILABLE_COLOR)
            painter.drawRect(QRectF(bar_left, y + 2,
                                    bar_width * ground / top, 10))
            if used_share:
                colour = entry.get("colour") or [214, 96, 28]
                painter.setBrush(QColor(*colour).darker(115))
                painter.drawRect(QRectF(bar_left, y + 13,
                                        bar_width * used_share / top, 10))
            ratio = habitat.selection_ratio(used_share, ground)
            note = (f"{ground * 100:.0f}% | "
                    f"{_fmt(None if used_share is None else used_share * 100, 0, '%')}"
                    f" ({count})   ×{_fmt(ratio, 2)}")
            painter.setPen(TEXT_COLOR)
            painter.drawText(QRectF(bar_left + bar_width + 6, y,
                                    self.NOTE_WIDTH - 8,
                                    self.ROW_HEIGHT - 6),
                             ALIGN_LEFT_MIDDLE, note)


FLIGHT_COLUMNS = [
    ("Flight", lambda f: f.get("flight", "")),
    ("Tracks", lambda f: str(f.get("n_tracks", 0))),
    ("Scene (days from flight)", lambda f: _scene_text(f)),
    ("Clear", lambda f: _scene_value(f, "clear_fraction", 100, 0, "%")),
    ("Air °C", lambda f: _weather_text(f, "at_flight", "temperature_2m", 1)),
    ("Soil °C", lambda f: _weather_text(f, "at_flight", "soil_temperature", 1)),
    ("Humidity %", lambda f: _weather_text(
        f, "at_flight", "relative_humidity_2m", 0)),
    ("Wind m/s", lambda f: _weather_text(f, "at_flight", "wind_speed_10m", 1)),
    ("Cloud %", lambda f: _weather_text(f, "at_flight", "cloud_cover", 0)),
    ("Rain 24 h mm", lambda f: _weather_text(
        f, "previous_24h", "precipitation_sum", 1)),
    ("Sun 24 h h", lambda f: _weather_text(
        f, "previous_24h", "sunshine_hours", 1)),
    ("Not run", lambda f: ", ".join(f.get("missing") or [])),
]


def _scene(flight: dict) -> dict:
    return (flight.get("sources") or {}).get("sentinel2") or {}


def _scene_text(flight: dict) -> str:
    scene = _scene(flight)
    if not scene.get("scene_time"):
        return "–"
    offset = scene.get("offset_days")
    return (f"{str(scene['scene_time'])[:10]} "
            f"({'' if offset is None else format(offset, '+.1f')})")


def _scene_value(flight: dict, key: str, factor: float, digits: int,
                 unit: str) -> str:
    value = _scene(flight).get(key)
    return "–" if value is None else f"{value * factor:.{digits}f}{unit}"


def _weather_text(flight: dict, block: str, key: str, digits: int) -> str:
    weather = (flight.get("sources") or {}).get("weather") or {}
    return _fmt((weather.get(block) or {}).get(key), digits)


class BambiHabitatDialog(QDialog):
    """Charts and tables of a survey's habitat summary."""

    def __init__(self, summary: dict, tracks: List[dict], parent=None):
        super().__init__(parent)
        self._summary = summary
        self._tracks = tracks
        self._charts = {}
        self._landcover_chart = None
        self.setWindowTitle("BAMBI - Habitat Context")
        self.resize(820, 760)

        layout = QVBoxLayout(self)
        header = QLabel(
            f"<b>{summary.get('n_tracks', 0)} track(s)</b> over "
            f"{len(summary.get('flights') or [])} flight(s). Values at the "
            f"animals are averaged within {summary.get('radius_m', 0):g} m. "
            "Satellite cells are 10–20 m: they describe the patch an animal "
            "was in, not the spot it stood on.")
        header.setWordWrap(True)
        layout.addWidget(header)

        species_row = QHBoxLayout()
        species_row.addWidget(QLabel("Animals shown:"))
        self.species_combo = QComboBox()
        self.species_combo.addItem("All species", "")
        for name in sorted({row.get("species") or "" for row in tracks}):
            if name:
                self.species_combo.addItem(name, name)
        self.species_combo.currentIndexChanged.connect(self._on_species)
        species_row.addWidget(self.species_combo)
        self.species_note = QLabel("")
        species_row.addWidget(self.species_note)
        species_row.addStretch()
        layout.addLayout(species_row)

        self.flight_table = QTableWidget(
            len(summary.get("flights") or []), len(FLIGHT_COLUMNS))
        self.flight_table.setHorizontalHeaderLabels(
            [title for title, _fn in FLIGHT_COLUMNS])
        self.flight_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        for row, flight in enumerate(summary.get("flights") or []):
            for column, (_title, text_of) in enumerate(FLIGHT_COLUMNS):
                self.flight_table.setItem(
                    row, column, QTableWidgetItem(text_of(flight)))
        self.flight_table.resizeColumnsToContents()
        self.flight_table.setMaximumHeight(
            60 + 26 * min(6, self.flight_table.rowCount()))
        layout.addWidget(self.flight_table)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        charts_layout = QVBoxLayout(inner)
        for key, layer in (summary.get("layers") or {}).items():
            chart = HistogramChart(layer)
            self._charts[key] = chart
            charts_layout.addWidget(chart)
        cover = summary.get("landcover")
        if cover and cover.get("classes"):
            self._landcover_chart = LandCoverChart(cover["classes"])
            charts_layout.addWidget(self._landcover_chart)
        if not self._charts and self._landcover_chart is None:
            charts_layout.addWidget(QLabel(
                "No environment layers were available for these flights."))
        charts_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        self._on_species()

    def chart_keys(self) -> List[str]:
        return list(self._charts)

    def _on_species(self, _index: int = 0) -> None:
        """Redraw the animals' side for the chosen species."""
        species = self.species_combo.currentData() or ""
        rows = habitat.filter_tracks(self._tracks, species)
        self.species_note.setText(f"{len(rows)} track(s)")
        for key, chart in self._charts.items():
            layer = self._summary["layers"][key]
            values = [row[key] for row in rows if row.get(key) is not None]
            mean = sum(values) / len(values) if values else None
            chart.set_used(
                habitat.used_histogram(rows, key, layer.get("bin_edges") or []),
                mean, len(values))
        if self._landcover_chart is not None:
            named = [row["landcover"] for row in rows if row.get("landcover")]
            total = len(named)
            used = {}
            for name in set(named):
                count = named.count(name)
                used[name] = (count / total if total else None, count)
            self._landcover_chart.set_used(used)

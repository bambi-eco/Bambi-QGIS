# -*- coding: utf-8 -*-
"""The dialogs that ask for a range of frames.

One for the frames a correction applies to, one for the frames whose
results are added to the map.
"""

import math
from typing import Optional, Dict, Any

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QSpinBox, QDoubleSpinBox, QLabel,
    QDialog, QDialogButtonBox, QGridLayout, QRadioButton, QButtonGroup,
)


class CorrectionRangeDialog(QDialog):
    """Dialog for adding/editing frame-range specific corrections."""

    def __init__(self, parent=None, correction_data: Optional[Dict[str, Any]] = None, use_degrees: bool = False):
        """Initialize the dialog.

        :param parent: Parent widget
        :param correction_data: Existing correction data for editing (rotations in radians), or None for new
        :param use_degrees: If True, display and accept rotation values in degrees
        """
        super().__init__(parent)
        self.setWindowTitle("Frame-Range Correction")
        self.setMinimumWidth(400)
        self._use_degrees = use_degrees

        self._setup_ui()

        # Load existing data if editing
        if correction_data:
            self._load_data(correction_data)

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Frame range
        range_group = QGroupBox("Frame Range")
        range_layout = QHBoxLayout(range_group)

        range_layout.addWidget(QLabel("Start:"))
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, 999999)
        self.start_frame_spin.setValue(0)
        self.start_frame_spin.setToolTip("First frame index (inclusive)")
        range_layout.addWidget(self.start_frame_spin)

        range_layout.addWidget(QLabel("End:"))
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(0, 999999)
        self.end_frame_spin.setValue(999999)
        self.end_frame_spin.setToolTip("Last frame index (inclusive)")
        range_layout.addWidget(self.end_frame_spin)

        layout.addWidget(range_group)

        # Translation
        trans_group = QGroupBox("Translation (x, y, z)")
        trans_layout = QGridLayout(trans_group)

        trans_layout.addWidget(QLabel("X:"), 0, 0)
        self.trans_x_spin = QDoubleSpinBox()
        self.trans_x_spin.setRange(-100, 100)
        self.trans_x_spin.setValue(0.0)
        self.trans_x_spin.setDecimals(3)
        trans_layout.addWidget(self.trans_x_spin, 0, 1)

        trans_layout.addWidget(QLabel("Y:"), 0, 2)
        self.trans_y_spin = QDoubleSpinBox()
        self.trans_y_spin.setRange(-100, 100)
        self.trans_y_spin.setValue(0.0)
        self.trans_y_spin.setDecimals(3)
        trans_layout.addWidget(self.trans_y_spin, 0, 3)

        trans_layout.addWidget(QLabel("Z:"), 0, 4)
        self.trans_z_spin = QDoubleSpinBox()
        self.trans_z_spin.setRange(-100, 100)
        self.trans_z_spin.setValue(0.0)
        self.trans_z_spin.setDecimals(3)
        trans_layout.addWidget(self.trans_z_spin, 0, 5)

        layout.addWidget(trans_group)

        # Rotation - configure based on unit
        unit_str = "deg" if self._use_degrees else "rad"
        rot_group = QGroupBox(f"Rotation (pitch, roll, yaw) [{unit_str}]")
        rot_layout = QGridLayout(rot_group)

        if self._use_degrees:
            rot_range = (-180, 180)
            rot_decimals = 3
            rot_step = 1.0
        else:
            rot_range = (-6.28319, 6.28319)  # -2π to 2π
            rot_decimals = 5
            rot_step = 0.01

        rot_layout.addWidget(QLabel("Pitch:"), 0, 0)
        self.rot_x_spin = QDoubleSpinBox()
        self.rot_x_spin.setRange(*rot_range)
        self.rot_x_spin.setValue(0.0)
        self.rot_x_spin.setDecimals(rot_decimals)
        self.rot_x_spin.setSingleStep(rot_step)
        rot_layout.addWidget(self.rot_x_spin, 0, 1)

        rot_layout.addWidget(QLabel("Roll:"), 0, 2)
        self.rot_y_spin = QDoubleSpinBox()
        self.rot_y_spin.setRange(*rot_range)
        self.rot_y_spin.setValue(0.0)
        self.rot_y_spin.setDecimals(rot_decimals)
        self.rot_y_spin.setSingleStep(rot_step)
        rot_layout.addWidget(self.rot_y_spin, 0, 3)

        rot_layout.addWidget(QLabel("Yaw:"), 0, 4)
        self.rot_z_spin = QDoubleSpinBox()
        self.rot_z_spin.setRange(*rot_range)
        self.rot_z_spin.setValue(0.0)
        self.rot_z_spin.setDecimals(rot_decimals)
        self.rot_z_spin.setSingleStep(rot_step)
        rot_layout.addWidget(self.rot_z_spin, 0, 5)

        layout.addWidget(rot_group)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _load_data(self, data: Dict[str, Any]):
        """Load existing correction data into the dialog.

        :param data: Correction data dictionary (rotations in radians)
        """
        self.start_frame_spin.setValue(data.get('start', 0))
        self.end_frame_spin.setValue(data.get('end', 999999))

        trans = data.get('translation', {})
        self.trans_x_spin.setValue(trans.get('x', 0.0))
        self.trans_y_spin.setValue(trans.get('y', 0.0))
        self.trans_z_spin.setValue(trans.get('z', 0.0))

        rot = data.get('rotation', {})
        rot_x = rot.get('x', 0.0)
        rot_y = rot.get('y', 0.0)
        rot_z = rot.get('z', 0.0)

        # Convert from radians to degrees for display if needed
        if self._use_degrees:
            rot_x = math.degrees(rot_x)
            rot_y = math.degrees(rot_y)
            rot_z = math.degrees(rot_z)

        self.rot_x_spin.setValue(rot_x)
        self.rot_y_spin.setValue(rot_y)
        self.rot_z_spin.setValue(rot_z)

    def get_correction_data(self) -> Dict[str, Any]:
        """Get the correction data from the dialog.

        :return: Dictionary with correction data (rotations always in radians)
        """
        rot_x = self.rot_x_spin.value()
        rot_y = self.rot_y_spin.value()
        rot_z = self.rot_z_spin.value()

        # Convert from degrees to radians if needed
        if self._use_degrees:
            rot_x = math.radians(rot_x)
            rot_y = math.radians(rot_y)
            rot_z = math.radians(rot_z)

        return {
            "start": self.start_frame_spin.value(),
            "end": self.end_frame_spin.value(),
            "translation": {
                "x": self.trans_x_spin.value(),
                "y": self.trans_y_spin.value(),
                "z": self.trans_z_spin.value()
            },
            "rotation": {
                "x": rot_x,
                "y": rot_y,
                "z": rot_z
            }
        }


class FrameRangeDialog(QDialog):
    """Dialog for selecting a frame index range before importing per-frame QGIS layers."""

    def __init__(self, parent, min_idx: int, max_idx: int, total: int,
                 item_label: str = "frames"):
        super().__init__(parent)
        self.setWindowTitle("Select Frame Range")
        self.setMinimumWidth(350)

        self._min_idx = min_idx
        self._max_idx = max_idx

        layout = QVBoxLayout(self)

        info = QLabel(f"Available: {item_label} {min_idx}–{max_idx}  ({total} total)")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        self._all_radio = QRadioButton("All frames")
        self._all_radio.setChecked(True)
        self._range_radio = QRadioButton("Frame range:")

        self._btn_group = QButtonGroup(self)
        self._btn_group.addButton(self._all_radio)
        self._btn_group.addButton(self._range_radio)

        layout.addWidget(self._all_radio)

        range_row = QHBoxLayout()
        range_row.addWidget(self._range_radio)
        self._start_spin = QSpinBox()
        self._start_spin.setRange(min_idx, max_idx)
        self._start_spin.setValue(min_idx)
        self._start_spin.setEnabled(False)
        self._end_spin = QSpinBox()
        self._end_spin.setRange(min_idx, max_idx)
        self._end_spin.setValue(max_idx)
        self._end_spin.setEnabled(False)
        range_row.addWidget(QLabel("from"))
        range_row.addWidget(self._start_spin)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self._end_spin)
        layout.addLayout(range_row)

        self._all_radio.toggled.connect(
            lambda checked: (
                self._start_spin.setEnabled(not checked),
                self._end_spin.setEnabled(not checked),
            )
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_range(self):
        """Return ``(start, end)`` - clamped and always start <= end."""
        if self._all_radio.isChecked():
            return self._min_idx, self._max_idx
        start = self._start_spin.value()
        end = self._end_spin.value()
        if start > end:
            start, end = end, start
        return start, end

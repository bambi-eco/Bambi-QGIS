# -*- coding: utf-8 -*-
"""Configuration > Correction: where the camera really looked.

The translation and rotation that are added to every pose of a flight, the
corrections that apply to a range of frames only, and the file they are
read from and saved to.
"""

import json
import math
import os
from typing import Dict, Any, List

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QPushButton,
    QDoubleSpinBox, QFileDialog, QLabel, QComboBox, QMessageBox, QListWidget,
    QListWidgetItem, QDialog,
)

from .common import _cap_vertical_growth
from .range_dialogs import CorrectionRangeDialog


class CorrectionSettingsMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_correction_settings(self, config_sub_tabs):
        """Configuration > Position Correction."""
        # ----- Sub-Tab 2: Position Correction -----
        correction_tab = QWidget()
        correction_tab_layout = QVBoxLayout(correction_tab)
        config_sub_tabs.addTab(correction_tab, "Position Correction")

        # Rotation unit selection
        unit_group = QGroupBox("Rotation Unit")
        unit_layout = QHBoxLayout(unit_group)
        unit_layout.addWidget(QLabel("Display rotations in:"))
        self.rotation_unit_combo = QComboBox()
        self.rotation_unit_combo.addItems(["Radians", "Degrees"])
        self.rotation_unit_combo.setCurrentIndex(0)  # Default to radians
        self.rotation_unit_combo.currentIndexChanged.connect(self._on_rotation_unit_changed)
        self.rotation_unit_combo.setToolTip(
            "Select the unit for rotation values in the UI.\n"
            "Backend and saved files always use radians."
        )
        unit_layout.addWidget(self.rotation_unit_combo)
        unit_layout.addStretch()
        correction_tab_layout.addWidget(unit_group)

        correction_group = QGroupBox("Default Correction Factors")
        correction_layout = QFormLayout(correction_group)

        # Translation corrections
        trans_label = QLabel("Translation (x, y, z):")
        correction_layout.addRow(trans_label)

        trans_row = QHBoxLayout()
        self.trans_x_spin = QDoubleSpinBox()
        self.trans_x_spin.setRange(-100, 100)
        self.trans_x_spin.setValue(0.0)
        self.trans_x_spin.setDecimals(3)
        trans_row.addWidget(QLabel("X:"))
        trans_row.addWidget(self.trans_x_spin)

        self.trans_y_spin = QDoubleSpinBox()
        self.trans_y_spin.setRange(-100, 100)
        self.trans_y_spin.setValue(0.0)
        self.trans_y_spin.setDecimals(3)
        trans_row.addWidget(QLabel("Y:"))
        trans_row.addWidget(self.trans_y_spin)

        self.trans_z_spin = QDoubleSpinBox()
        self.trans_z_spin.setRange(-100, 100)
        self.trans_z_spin.setValue(0.0)
        self.trans_z_spin.setDecimals(3)
        trans_row.addWidget(QLabel("Z:"))
        trans_row.addWidget(self.trans_z_spin)
        correction_layout.addRow(trans_row)

        # Rotation corrections
        self.rot_label = QLabel("Rotation (pitch, roll, yaw) [rad]:")
        correction_layout.addRow(self.rot_label)

        rot_row = QHBoxLayout()
        self.rot_x_spin = QDoubleSpinBox()
        self.rot_x_spin.setRange(-6.28319, 6.28319)  # -2π to 2π
        self.rot_x_spin.setValue(0.0)
        self.rot_x_spin.setDecimals(5)
        self.rot_x_spin.setSingleStep(0.01)
        rot_row.addWidget(QLabel("Pitch:"))
        rot_row.addWidget(self.rot_x_spin)

        self.rot_y_spin = QDoubleSpinBox()
        self.rot_y_spin.setRange(-6.28319, 6.28319)  # -2π to 2π
        self.rot_y_spin.setValue(0.0)
        self.rot_y_spin.setDecimals(5)
        self.rot_y_spin.setSingleStep(0.01)
        rot_row.addWidget(QLabel("Roll:"))
        rot_row.addWidget(self.rot_y_spin)

        self.rot_z_spin = QDoubleSpinBox()
        self.rot_z_spin.setRange(-6.28319, 6.28319)  # -2π to 2π
        self.rot_z_spin.setValue(0.0)
        self.rot_z_spin.setDecimals(5)
        self.rot_z_spin.setSingleStep(0.01)
        rot_row.addWidget(QLabel("Yaw:"))
        rot_row.addWidget(self.rot_z_spin)
        correction_layout.addRow(rot_row)

        correction_tab_layout.addWidget(correction_group)

        # Frame-range specific corrections
        additional_group = QGroupBox("Frame-Range Specific Corrections")
        additional_layout = QVBoxLayout(additional_group)

        # Info label
        info_label = QLabel(
            "Define corrections for specific frame ranges. "
            "These override the default correction for frames within the specified range."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        additional_layout.addWidget(info_label)

        # List widget for additional corrections
        self.additional_corrections_list = QListWidget()
        self.additional_corrections_list.setMinimumHeight(100)
        self.additional_corrections_list.setMaximumHeight(150)
        _cap_vertical_growth(self.additional_corrections_list)
        self.additional_corrections_list.setToolTip(
            "Double-click to edit, or select and use buttons below"
        )
        self.additional_corrections_list.itemDoubleClicked.connect(
            self._edit_additional_correction
        )
        additional_layout.addWidget(self.additional_corrections_list)

        # Buttons for managing additional corrections
        additional_btn_row = QHBoxLayout()
        self.add_correction_btn = QPushButton("Add...")
        self.add_correction_btn.setToolTip("Add a new frame-range specific correction")
        self.add_correction_btn.clicked.connect(self._add_additional_correction)
        additional_btn_row.addWidget(self.add_correction_btn)

        self.edit_correction_btn = QPushButton("Edit...")
        self.edit_correction_btn.setToolTip("Edit the selected correction")
        self.edit_correction_btn.clicked.connect(self._edit_selected_correction)
        additional_btn_row.addWidget(self.edit_correction_btn)

        self.remove_correction_btn = QPushButton("Remove")
        self.remove_correction_btn.setToolTip("Remove the selected correction")
        self.remove_correction_btn.clicked.connect(self._remove_additional_correction)
        additional_btn_row.addWidget(self.remove_correction_btn)

        additional_layout.addLayout(additional_btn_row)

        correction_tab_layout.addWidget(additional_group)

        # Store additional corrections data (always in radians internally)
        self._additional_corrections: List[Dict[str, Any]] = []

        # Save correction button
        save_correction_btn = QPushButton("Save Correction to JSON...")
        save_correction_btn.setToolTip(
            "Save the current translation and rotation values to a JSON file (rotations saved in radians)")
        save_correction_btn.clicked.connect(self.save_correction_values)
        correction_tab_layout.addWidget(save_correction_btn)

        correction_tab_layout.addStretch()

    def browse_correction(self):
        """Browse for correction.json file and load its values."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Correction JSON", "", "JSON Files (*.json)")
        if file:
            self.correction_path_edit.setText(file)
            self.load_correction_values(file)

    def load_correction_values(self, correction_path: str):
        """Load correction values from JSON file and populate the spin boxes.

        :param correction_path: Path to the correction.json file (rotations stored in radians)
        """
        try:
            with open(correction_path, 'r') as f:
                correction = json.load(f)

            # Load translation values
            if 'translation' in correction:
                trans = correction['translation']
                if 'x' in trans:
                    self.trans_x_spin.setValue(float(trans['x']))
                if 'y' in trans:
                    self.trans_y_spin.setValue(float(trans['y']))
                if 'z' in trans:
                    self.trans_z_spin.setValue(float(trans['z']))

            # Load rotation values (stored in radians, convert to display unit if needed)
            if 'rotation' in correction:
                rot = correction['rotation']
                rot_x = float(rot.get('x', 0))
                rot_y = float(rot.get('y', 0))
                rot_z = float(rot.get('z', 0))

                # Convert from radians to degrees if UI is in degrees mode
                if self._is_using_degrees():
                    rot_x = math.degrees(rot_x)
                    rot_y = math.degrees(rot_y)
                    rot_z = math.degrees(rot_z)

                self.rot_x_spin.setValue(rot_x)
                self.rot_y_spin.setValue(rot_y)
                self.rot_z_spin.setValue(rot_z)

            # Load additional corrections (already in radians, stored as-is internally)
            self._additional_corrections = []
            self.additional_corrections_list.clear()
            if 'additional' in correction:
                for add_corr in correction['additional']:
                    self._add_correction_to_list(add_corr)

            self.log(f"Loaded correction values from: {os.path.basename(correction_path)}")

        except Exception as e:
            self.log(f"Warning: Could not load correction.json: {e}")

    def save_correction_values(self):
        """Save correction values to a JSON file.

        Saves the current translation and rotation values from the UI
        to a user-selected JSON file.
        """
        # Get save file path
        default_path = ""
        target_folder = self.target_folder_edit.text()
        if target_folder and os.path.exists(target_folder):
            default_path = os.path.join(target_folder, "correction.json")
        elif self.correction_path_edit.text():
            default_path = self.correction_path_edit.text()

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Correction JSON",
            default_path,
            "JSON Files (*.json)"
        )

        if not file_path:
            return  # User cancelled

        # Ensure .json extension
        if not file_path.lower().endswith('.json'):
            file_path += '.json'

        # Get rotation values and convert to radians if UI is in degrees mode
        rot_x = self.rot_x_spin.value()
        rot_y = self.rot_y_spin.value()
        rot_z = self.rot_z_spin.value()
        if self._is_using_degrees():
            rot_x = math.radians(rot_x)
            rot_y = math.radians(rot_y)
            rot_z = math.radians(rot_z)

        # Build correction data (rotations always saved in radians)
        correction_data = {
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

        # Add additional corrections if any (already in radians internally)
        if self._additional_corrections:
            correction_data["additional"] = self._additional_corrections

        try:
            with open(file_path, 'w') as f:
                json.dump(correction_data, f, indent=4)

            # Update the correction path field
            self.correction_path_edit.setText(file_path)

            self.log(f"Saved correction values to: {os.path.basename(file_path)}")
            QMessageBox.information(
                self,
                "Success",
                f"Correction values saved to:\n{file_path}\n\n(Rotations saved in radians)"
            )

        except Exception as e:
            self.log(f"Error saving correction.json: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save correction file:\n{str(e)}"
            )

    def _add_correction_to_list(self, correction_data: Dict[str, Any]):
        """Add a correction entry to the internal list and UI list.

        :param correction_data: Dictionary with correction data including
                               translation, rotation, start, and end
        """
        self._additional_corrections.append(correction_data)
        self._update_corrections_list_ui()

    def _is_using_degrees(self) -> bool:
        """Check if the UI is set to display rotations in degrees.

        :return: True if using degrees, False if using radians
        """
        return self.rotation_unit_combo.currentIndex() == 1

    def _on_rotation_unit_changed(self, index: int):
        """Handle rotation unit change - convert displayed values.

        :param index: 0 for radians, 1 for degrees
        """
        # Block signals to prevent recursion
        self.rot_x_spin.blockSignals(True)
        self.rot_y_spin.blockSignals(True)
        self.rot_z_spin.blockSignals(True)

        # Get current values
        rot_x = self.rot_x_spin.value()
        rot_y = self.rot_y_spin.value()
        rot_z = self.rot_z_spin.value()

        if index == 1:  # Changed to degrees
            # Convert from radians to degrees
            rot_x = math.degrees(rot_x)
            rot_y = math.degrees(rot_y)
            rot_z = math.degrees(rot_z)
            # Update spinbox ranges and precision
            self.rot_x_spin.setRange(-180, 180)
            self.rot_y_spin.setRange(-180, 180)
            self.rot_z_spin.setRange(-180, 180)
            self.rot_x_spin.setDecimals(3)
            self.rot_y_spin.setDecimals(3)
            self.rot_z_spin.setDecimals(3)
            self.rot_x_spin.setSingleStep(1.0)
            self.rot_y_spin.setSingleStep(1.0)
            self.rot_z_spin.setSingleStep(1.0)
            self.rot_label.setText("Rotation (pitch, roll, yaw) [deg]:")
        else:  # Changed to radians
            # Convert from degrees to radians
            rot_x = math.radians(rot_x)
            rot_y = math.radians(rot_y)
            rot_z = math.radians(rot_z)
            # Update spinbox ranges and precision
            self.rot_x_spin.setRange(-6.28319, 6.28319)
            self.rot_y_spin.setRange(-6.28319, 6.28319)
            self.rot_z_spin.setRange(-6.28319, 6.28319)
            self.rot_x_spin.setDecimals(5)
            self.rot_y_spin.setDecimals(5)
            self.rot_z_spin.setDecimals(5)
            self.rot_x_spin.setSingleStep(0.01)
            self.rot_y_spin.setSingleStep(0.01)
            self.rot_z_spin.setSingleStep(0.01)
            self.rot_label.setText("Rotation (pitch, roll, yaw) [rad]:")

        # Set converted values
        self.rot_x_spin.setValue(rot_x)
        self.rot_y_spin.setValue(rot_y)
        self.rot_z_spin.setValue(rot_z)

        # Re-enable signals
        self.rot_x_spin.blockSignals(False)
        self.rot_y_spin.blockSignals(False)
        self.rot_z_spin.blockSignals(False)

        # Update the list display
        self._update_corrections_list_ui()

    def _update_corrections_list_ui(self):
        """Update the list widget to reflect the current additional corrections."""
        self.additional_corrections_list.clear()
        use_degrees = self._is_using_degrees()
        unit_str = "deg" if use_degrees else "rad"

        for i, corr in enumerate(self._additional_corrections):
            start = corr.get('start', 0)
            end = corr.get('end', 999999)
            trans = corr.get('translation', {})
            rot = corr.get('rotation', {})

            # Get rotation values (stored in radians)
            rot_x = rot.get('x', 0)
            rot_y = rot.get('y', 0)
            rot_z = rot.get('z', 0)

            # Convert to degrees for display if needed
            if use_degrees:
                rot_x = math.degrees(rot_x)
                rot_y = math.degrees(rot_y)
                rot_z = math.degrees(rot_z)

            # Create summary text
            trans_str = f"T({trans.get('x', 0):.1f}, {trans.get('y', 0):.1f}, {trans.get('z', 0):.1f})"
            rot_str = f"R({rot_x:.2f}, {rot_y:.2f}, {rot_z:.2f}){unit_str}"
            item_text = f"Frames {start}-{end}: {trans_str} {rot_str}"

            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, i)  # Store index
            self.additional_corrections_list.addItem(item)

    def _add_additional_correction(self):
        """Show dialog to add a new frame-range specific correction."""
        dialog = CorrectionRangeDialog(self, use_degrees=self._is_using_degrees())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            correction_data = dialog.get_correction_data()
            self._add_correction_to_list(correction_data)
            self.log(f"Added correction for frames {correction_data['start']}-{correction_data['end']}")

    def _edit_additional_correction(self, item: QListWidgetItem):
        """Edit an additional correction when double-clicked.

        :param item: The list item that was double-clicked
        """
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is not None and 0 <= index < len(self._additional_corrections):
            correction_data = self._additional_corrections[index]
            dialog = CorrectionRangeDialog(self, correction_data, use_degrees=self._is_using_degrees())
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._additional_corrections[index] = dialog.get_correction_data()
                self._update_corrections_list_ui()
                self.log(f"Updated correction at index {index}")

    def _edit_selected_correction(self):
        """Edit the currently selected additional correction."""
        selected_items = self.additional_corrections_list.selectedItems()
        if selected_items:
            self._edit_additional_correction(selected_items[0])
        else:
            QMessageBox.information(
                self, "No Selection",
                "Please select a correction to edit."
            )

    def _remove_additional_correction(self):
        """Remove the currently selected additional correction."""
        selected_items = self.additional_corrections_list.selectedItems()
        if not selected_items:
            QMessageBox.information(
                self, "No Selection",
                "Please select a correction to remove."
            )
            return

        index = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if index is not None and 0 <= index < len(self._additional_corrections):
            removed = self._additional_corrections.pop(index)
            self._update_corrections_list_ui()
            self.log(f"Removed correction for frames {removed.get('start', 0)}-{removed.get('end', 0)}")

    @staticmethod
    def get_correction_for_frame(frame_idx: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get the appropriate correction factors for a given frame index.

        Single source: :func:`core.corrections.correction_for_frame_config`.
        """
        from ..core.corrections import correction_for_frame_config
        return correction_for_frame_config(frame_idx, config)

    def open_correction_wizard(self) -> None:
        """Open the correction calibration wizard as a modal dialog."""
        from ..bambi_correction_wizard import BambiCorrectionWizard
        config = self.get_config()
        wizard = BambiCorrectionWizard(self.iface, config, parent=self)
        wizard.correctionFileSaved.connect(self.load_correction_values)
        wizard.exec()
        # a child of this panel otherwise stays, with the DEM it loaded
        wizard.deleteLater()

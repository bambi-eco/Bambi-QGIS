# -*- coding: utf-8 -*-
"""Configuration > Classification: the models that name what was found.

The backbone that describes an animal, one model per question (hidden or
not, species, sex, age), the images each of them looks at, and the access
to the model repository: the token, the check that it works, the download.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QPushButton,
    QLineEdit, QSpinBox, QDoubleSpinBox, QFileDialog, QLabel, QComboBox,
    QCheckBox, QMessageBox, QDialog, QTableWidget, QTableWidgetItem,
    QHeaderView,
)

from ..bambi_processing import BambiProcessor
from ..core.hf_access import DEFAULT_BACKBONE as HF_DEFAULT_BACKBONE
from .. import gui_utils as gui_utils_hf
from .common import _CLASSIFICATION_INPUTS, _CLASSIFICATION_TASKS


class ClassificationSettingsMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_classification_settings(self, config_sub_tabs):
        """Configuration > Classification."""
        # ----- Sub-Tab: Classification -----
        classification_tab = QWidget()
        classification_tab_layout = QVBoxLayout(classification_tab)
        config_sub_tabs.addTab(classification_tab, "Classification")

        classification_info = QLabel(
            "Classifies tracked animals by occlusion, species and sex using "
            '<a href="https://huggingface.co/facebook/'
            'dinov3-vith16plus-pretrain-lvd1689m">DINOv3</a> features. '
            "The heads are small; the DINOv3 backbone they "
            "read is a <b>gated</b> Hugging Face model, so access has to be "
            "requested once and a token entered below."
        )
        classification_info.setWordWrap(True)
        classification_info.setTextFormat(Qt.TextFormat.RichText)
        classification_info.setOpenExternalLinks(True)
        classification_info.setStyleSheet("color: gray; font-size: 10px;")
        classification_tab_layout.addWidget(classification_info)

        # Per-task model choice, modality and label mapping. Bound to the
        # config schema's ``json`` role rather than to a value widget, because
        # it is a table; the table editing it arrives with the classifier
        # steps.
        self._classification_models = {}

        # --- Hugging Face access ---
        hf_group = QGroupBox("Hugging Face Access")
        hf_layout = QFormLayout(hf_group)

        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_edit.setPlaceholderText(
            "hf_… (read token; stored in the QGIS settings, not the project)")
        self.hf_token_edit.setToolTip(
            "A Hugging Face access token with read permission.\n\n"
            "Stored in your QGIS settings rather than in the project file, "
            "because a project gets shared and a token must not travel with "
            "it.\n\n"
            "Leave empty to use the HF_TOKEN environment variable or the "
            "token from 'hf auth login'."
        )
        self.hf_token_edit.editingFinished.connect(self._save_hf_token)
        hf_layout.addRow("Access Token:", self.hf_token_edit)

        self.hf_show_token_check = QCheckBox("Show token")
        self.hf_show_token_check.stateChanged.connect(self._toggle_hf_token_echo)
        hf_layout.addRow("", self.hf_show_token_check)

        self.classification_backbone_edit = QLineEdit()
        self.classification_backbone_edit.setPlaceholderText(HF_DEFAULT_BACKBONE)
        self.classification_backbone_edit.setToolTip(
            "The DINOv3 model producing the features the classifier heads "
            "consume. Leave empty for the model every published head was "
            "trained against.\n\n"
            "The default model is fetched at the commit this plugin "
            "release was tested with. For a model of your own, append '@' "
            "and a commit hash to pin it (repo@abc123…): without one, "
            "Hugging Face resolves a repository to whatever its main branch "
            "holds at the time, so one that changes would change your "
            "features. 'repo@main' follows the latest state on purpose."
        )
        hf_layout.addRow("Backbone:", self.classification_backbone_edit)

        check_row = QHBoxLayout()
        self.hf_check_access_btn = QPushButton("Check access")
        self.hf_check_access_btn.setToolTip(
            "Ask Hugging Face whether this token may download the backbone, "
            "before a long run finds out the hard way."
        )
        self.hf_check_access_btn.clicked.connect(self.check_hf_access)
        check_row.addWidget(self.hf_check_access_btn)
        self.hf_access_status = QLabel("⚪ Not checked")
        self.hf_access_status.setWordWrap(True)
        check_row.addWidget(self.hf_access_status, 1)
        hf_layout.addRow("", check_row)

        classification_tab_layout.addWidget(hf_group)

        # --- Source imagery ---
        source_group = QGroupBox("Source Imagery")
        source_layout = QFormLayout(source_group)

        self.classification_projection_combo = QComboBox()
        self.classification_projection_combo.addItems([
            "Perspective (camera frames)",
            "Orthorectified 1k (GeoTIFF)",
            "Orthorectified 2k (GeoTIFF)",
            "ALFS (light-field integrals)",
        ])
        self.classification_projection_combo.setToolTip(
            "Which imagery the crops come from - and, with it, which variant "
            "of each classifier is used, so a head always sees the kind of "
            "image it was trained on.\n\n"
            "Perspective crops the extracted frames. The orthorectified "
            "options crop the per-frame GeoTIFFs, so 'Export Frames as "
            "GeoTIFF' and geo-referencing both have to have run first - the "
            "boxes reach the rasters through their world coordinates.\n\n"
            "1k and 2k refer to the source width the two variants were fitted "
            "at; pick the one nearer your export. Perspective is the better "
            "view for RGB - orthorectification smears animals that are "
            "already small and low in contrast.\n\n"
            "ALFS crops the light-field integrals of ALFS sampling mode "
            "(one per central frame; a frame in between uses the nearest), "
            "in which occluding vegetation is blurred away. No default "
            "classifiers have been published for ALFS imagery yet - use a "
            "custom model or a custom implementation with it."
        )
        source_layout.addRow("Projection:", self.classification_projection_combo)

        classification_tab_layout.addWidget(source_group)

        # --- Classifier mapping ---
        models_group = QGroupBox("Classifiers")
        models_layout = QVBoxLayout(models_group)

        models_info = QLabel(
            "One row per task, choosing the model it runs. <b>Labels</b> "
            "connects what the model returns to this project's vocabulary. "
            "Which camera's features each classifier reads is picked beside "
            "its own step in the Processing tab, next to the other per-step "
            "camera choices."
        )
        models_info.setWordWrap(True)
        models_info.setTextFormat(Qt.TextFormat.RichText)
        models_info.setStyleSheet("color: gray; font-size: 10px;")
        models_layout.addWidget(models_info)

        self.classification_models_table = QTableWidget(
            len(_CLASSIFICATION_TASKS), 4)
        self.classification_models_table.setHorizontalHeaderLabels(
            ["Task", "Model", "File / repository", ""])
        header = self.classification_models_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.classification_models_table.verticalHeader().setVisible(False)
        self.classification_models_table.setMinimumHeight(120)
        models_layout.addWidget(self.classification_models_table)

        self._build_classification_models_table()

        download_row = QHBoxLayout()
        self.download_models_btn = QPushButton("Download Selected Default Models")
        self.download_models_btn.setToolTip(
            "Fetch every classifier above that is set to Default, for the "
            "chosen projection and input.\n\n"
            "Worth doing before anything else: with the files present, "
            "'Labels' can read each model's classes and the mapping can be "
            "set up without running the classifiers first. These are small - "
            "a few megabytes each. The DINOv3 model itself is not downloaded "
            "here; that happens on the first embedding run."
        )
        self.download_models_btn.clicked.connect(
            self.download_classification_models)
        download_row.addWidget(self.download_models_btn)

        self.download_models_status = QLabel("")
        self.download_models_status.setWordWrap(True)
        self.download_models_status.setStyleSheet("font-size: 10px;")
        download_row.addWidget(self.download_models_status, 1)
        models_layout.addLayout(download_row)

        classification_tab_layout.addWidget(models_group)

        # --- Crops ---
        crop_group = QGroupBox("Crops")
        crop_layout = QFormLayout(crop_group)

        crop_info = QLabel(
            "The classifiers never see a whole frame. Each animal is cut out "
            "of its frame as a small square image - a <i>crop</i> - and it is "
            "that picture, and only that picture, which decides what the "
            "animal is called.\n\n"
            "So these settings decide what the classifier gets to look at: "
            "how much ground around the animal comes with it, and whether its "
            "proportions are preserved. The defaults match how the published "
            "classifiers were trained; change them only for a model that "
            "expects something else."
        )
        crop_info.setWordWrap(True)
        crop_info.setTextFormat(Qt.TextFormat.RichText)
        crop_info.setStyleSheet("color: gray; font-size: 10px;")
        crop_layout.addRow(crop_info)

        self.classification_padding_spin = QDoubleSpinBox()
        self.classification_padding_spin.setRange(0.0, 2.0)
        self.classification_padding_spin.setDecimals(2)
        self.classification_padding_spin.setSingleStep(0.05)
        self.classification_padding_spin.setValue(0.25)
        self.classification_padding_spin.setToolTip(
            "Fraction of the box added on every side before cropping.\n\n"
            "0.25 is what the published classifiers were trained with: a "
            "square crop at 1.5x the longer box side, centred on the animal. "
            "Their accuracy depends on it, so change this only for a custom "
            "model trained differently."
        )
        crop_layout.addRow("Padding:", self.classification_padding_spin)

        self.classification_crop_size_spin = QSpinBox()
        self.classification_crop_size_spin.setRange(64, 1024)
        self.classification_crop_size_spin.setSingleStep(16)
        self.classification_crop_size_spin.setValue(224)
        self.classification_crop_size_spin.setSuffix(" px")
        self.classification_crop_size_spin.setToolTip(
            "Side length of the crop handed to the backbone. 224 is what "
            "DINOv3 expects; other sizes must be a multiple of 16."
        )
        crop_layout.addRow("Crop size:", self.classification_crop_size_spin)

        self.classification_letterbox_check = QCheckBox(
            "Keep the aspect ratio (pad to square)")
        self.classification_letterbox_check.setChecked(True)
        self.classification_letterbox_check.setToolTip(
            "A deer seen from above is elongated, and stretching it to a "
            "square changes its proportions - which is part of what the sex "
            "classifier reads. Turn this off only to reproduce a model "
            "trained on stretched crops."
        )
        crop_layout.addRow("", self.classification_letterbox_check)

        classification_tab_layout.addWidget(crop_group)

        # --- Voting ---
        vote_group = QGroupBox("Frame Selection and Voting")
        vote_layout = QFormLayout(vote_group)

        vote_info = QLabel(
            "Species and sex are decided per animal by a vote across its "
            "frames. Voting is what makes a noisy per-frame call safe: an "
            "antler only resolves from some angles, so many frames of a true "
            "male look female, and the majority still recovers him."
        )
        vote_info.setWordWrap(True)
        vote_info.setStyleSheet("color: gray; font-size: 10px;")
        vote_layout.addRow(vote_info)

        self.classification_frames_combo = QComboBox()
        self.classification_frames_combo.addItems(
            ["Visible frames only", "All frames"])
        self.classification_frames_combo.setToolTip(
            "Which frames species and sex may vote over.\n\n"
            "'Visible frames only' uses the occlusion classifier when it has "
            "run, otherwise occlusion values already annotated by hand, and "
            "otherwise every frame - the run log always says which.\n\n"
            "Occlusion classification is optional: species and sex work "
            "either way."
        )
        vote_layout.addRow("Frames used:", self.classification_frames_combo)

        self.classification_quorum_spin = QDoubleSpinBox()
        self.classification_quorum_spin.setRange(0.0, 1.0)
        self.classification_quorum_spin.setDecimals(2)
        self.classification_quorum_spin.setSingleStep(0.05)
        self.classification_quorum_spin.setValue(0.5)
        self.classification_quorum_spin.setToolTip(
            "Share of voting frames a class must exceed to be called.\n\n"
            "0.50 is a simple majority. Raising it makes the classifier "
            "abstain more often - an abstention leaves the animal in the "
            "census with the attribute unknown, it never discards it."
        )
        vote_layout.addRow("Quorum:", self.classification_quorum_spin)

        self.classification_min_frames_spin = QSpinBox()
        self.classification_min_frames_spin.setRange(1, 1000)
        self.classification_min_frames_spin.setValue(1)
        self.classification_min_frames_spin.setToolTip(
            "Fewest voting frames an animal needs before it is called at all. "
            "A vote over one or two frames is barely a vote."
        )
        vote_layout.addRow("Min voting frames:",
                           self.classification_min_frames_spin)

        self.classification_unmatched_combo = QComboBox()
        self.classification_unmatched_combo.addItems([
            "Skip the animal", "Fall back to RGB", "Fall back to thermal"])
        self.classification_unmatched_combo.setToolTip(
            "What to do with an animal that only one camera saw, when a "
            "classifier is set to 'matched'.\n\n"
            "The matched classifiers were trained on real pairs and cannot "
            "take a blank for the missing camera, so this has to be an "
            "explicit choice.\n\n"
            "Fall back: the animal is called by the published classifier of "
            "that one camera for the same task. A model file of your own "
            "has no such counterpart; its unmatched animals are skipped."
        )
        vote_layout.addRow("Unmatched animals:",
                           self.classification_unmatched_combo)

        self.classification_overwrite_check = QCheckBox(
            "Replace a species the detector already assigned")
        self.classification_overwrite_check.setChecked(False)
        self.classification_overwrite_check.setToolTip(
            "By default an animal the detector itself identified keeps that "
            "identification, and only unidentified ones are filled in - so a "
            "class mapping you configured is not quietly undone.\n\n"
            "Turn this on to let the classifier's answer win everywhere."
        )
        vote_layout.addRow("", self.classification_overwrite_check)

        classification_tab_layout.addWidget(vote_group)

        # --- Life stage ---
        life_group = QGroupBox("Life Stage from Body Size")
        life_layout = QFormLayout(life_group)

        life_info = QLabel(
            "A juvenile cannot be told from an adult female by appearance at "
            "survey resolution - which is why the sex classifier's second "
            "class is <i>female/juvenile</i>. Size settles it: a juvenile "
            "sits far below its cohort with a clear gap to the next animal.\n\n"
            "Sizes are only ever compared <b>within one flight</b>, because "
            "how tightly boxes fit varies between recordings. There is "
            "deliberately no absolute threshold.\n\n"
            "These settings apply to the species you set to <b>Size-based</b> "
            "under the life-stage classifier's <i>Species…</i> button."
        )
        life_info.setWordWrap(True)
        life_info.setTextFormat(Qt.TextFormat.RichText)
        life_info.setStyleSheet("color: gray; font-size: 10px;")
        life_layout.addRow(life_info)

        self.life_stage_z_spin = QDoubleSpinBox()
        self.life_stage_z_spin.setRange(-10.0, 0.0)
        self.life_stage_z_spin.setDecimals(1)
        self.life_stage_z_spin.setSingleStep(0.5)
        self.life_stage_z_spin.setValue(-2.0)
        self.life_stage_z_spin.setToolTip(
            "How far below the flight median an animal must sit, as a robust "
            "z-score. More negative is stricter."
        )
        life_layout.addRow("Size z-score below:", self.life_stage_z_spin)

        self.life_stage_iqr_spin = QDoubleSpinBox()
        self.life_stage_iqr_spin.setRange(0.0, 20.0)
        self.life_stage_iqr_spin.setDecimals(1)
        self.life_stage_iqr_spin.setSingleStep(0.5)
        self.life_stage_iqr_spin.setValue(0.5)
        self.life_stage_iqr_spin.setToolTip(
            "How wide the gap to the next-smallest animal must be, as a "
            "multiple of the flight's interquartile range.\n\n"
            "This is what stops the rule firing on the merely smallest adult "
            "of a herd - someone is always smallest, and that is not evidence "
            "of anything."
        )
        life_layout.addRow("Size gap over (× IQR):", self.life_stage_iqr_spin)

        self.life_stage_min_spin = QSpinBox()
        self.life_stage_min_spin.setRange(1, 100)
        self.life_stage_min_spin.setValue(4)
        self.life_stage_min_spin.setToolTip(
            "Fewest animals in a flight for the comparison to run at all. A "
            "robust score over three animals is not robust."
        )
        life_layout.addRow("Min individuals:", self.life_stage_min_spin)

        classification_tab_layout.addWidget(life_group)

        # --- Compute ---
        compute_group = QGroupBox("Compute")
        compute_layout = QFormLayout(compute_group)

        self.classification_device_combo = QComboBox()
        self.classification_device_combo.addItems(
            ["Auto", "CPU", "CUDA (GPU)"])
        self.classification_device_combo.setToolTip(
            "Auto uses the GPU when a CUDA build of PyTorch is installed.\n"
            "The backbone is large - on CPU expect minutes per hundred crops."
        )
        compute_layout.addRow("Device:", self.classification_device_combo)

        self.classification_batch_spin = QSpinBox()
        self.classification_batch_spin.setRange(1, 256)
        self.classification_batch_spin.setValue(16)
        self.classification_batch_spin.setToolTip(
            "Crops embedded per forward pass. Lower this if the GPU runs out "
            "of memory."
        )
        compute_layout.addRow("Batch Size:", self.classification_batch_spin)

        self.classification_fp16_check = QCheckBox(
            "Use half precision on the GPU")
        self.classification_fp16_check.setChecked(True)
        self.classification_fp16_check.setToolTip(
            "Halves the memory the backbone needs, with no measurable effect "
            "on the features. Ignored on CPU."
        )
        compute_layout.addRow("", self.classification_fp16_check)

        model_dir_label = QLabel(
            "Models and the backbone cache are stored in the shared "
            "<code>bambi_deps/models/</code> folder, alongside the detection "
            "weights - downloaded once, not once per flight."
        )
        model_dir_label.setWordWrap(True)
        model_dir_label.setTextFormat(Qt.TextFormat.RichText)
        model_dir_label.setStyleSheet("color: gray; font-size: 10px;")
        compute_layout.addRow("", model_dir_label)

        classification_tab_layout.addWidget(compute_group)
        classification_tab_layout.addStretch()

        # The token is per user, not per project, so it is restored here rather
        # than by load_config_from_project.
        self._load_hf_token()

    def set_hf_token(self, token: str):
        """Take over a token entered in the Segmentation tool.

        Both places download gated Meta models with the same credential, so
        typing it in one must be typing it in both.
        """
        if self.hf_token_edit.text().strip() != (token or "").strip():
            self.hf_token_edit.setText((token or "").strip())
        self._save_hf_token()

    # ------------------------------------------------------------------
    # Classification - Hugging Face access
    # ------------------------------------------------------------------
    def _load_hf_token(self):
        """Populate the token field from the QGIS settings.

        The token is a user credential, so it is kept per user rather than per
        project: a project file gets shared, and a token must not travel with
        it. Nothing is shown when the token comes from the environment or from
        ``hf auth login`` - the field stays empty and those are picked up at
        run time, which is why it is a placeholder rather than a default.
        """
        self.hf_token_edit.setText(gui_utils_hf.read_hf_token())

    # ------------------------------------------------------------------
    # Classifier mapping table
    # ------------------------------------------------------------------
    def _build_classification_models_table(self):
        """One row per task, bound to ``_classification_models``."""
        from ..core import hf_access
        from ..bambi_classification_model_dialog import (
            default_source_for as default_model_source,
            sources_for as model_sources,
        )

        table = self.classification_models_table
        # The Input combos live on the Processing tab but hold part of the
        # same per-task spec, and the table is built first.
        self._ensure_classification_input_combos()
        for row, task in enumerate(_CLASSIFICATION_TASKS):
            name = QTableWidgetItem(
                hf_access.TASK_LABELS.get(task, task.capitalize()))
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, name)

            source = QComboBox()
            for label, value in model_sources(task):
                source.addItem(label, value)
            index = source.findData("default")
            if index >= 0 and not hf_access.has_default_head(task):
                # Nothing published for this task yet, so "Default" would only
                # fail later. Say so here instead. Life stage has no such
                # entry: its default is the size estimate, which is real.
                source.model().item(index).setEnabled(False)
                source.setItemText(index, "Default (not released)")
            source.setCurrentIndex(
                max(0, source.findData(default_model_source(task))))
            source.currentIndexChanged.connect(
                lambda _i, t=task: self._on_classification_model_changed(t))
            table.setCellWidget(row, 1, source)

            path = QTableWidgetItem("")
            table.setItem(row, 2, path)

            actions = QWidget()
            actions_layout = QHBoxLayout(actions)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            actions_layout.setSpacing(2)

            browse = QPushButton("…")
            browse.setFixedWidth(28)
            browse.setToolTip("Choose a model file")
            browse.clicked.connect(
                lambda _c=False, t=task: self._browse_classification_model(t))
            actions_layout.addWidget(browse)

            labels = QPushButton("Labels")
            labels.setToolTip(
                "Connect the model's classes to this project's vocabulary")
            labels.clicked.connect(
                lambda _c=False, t=task: self._edit_classification_labels(t))
            actions_layout.addWidget(labels)

            if task in hf_access.PER_SPECIES_TASKS:
                species = QPushButton("Species…")
                species.setToolTip(
                    "Choose a classifier per species - the cue is "
                    "species-specific, so one model does not transfer")
                species.clicked.connect(
                    lambda _c=False, t=task:
                    self._edit_species_models(t))
                actions_layout.addWidget(species)

            table.setCellWidget(row, 3, actions)

        self._apply_classification_models_to_table()

    def _task_row(self, task: str) -> int:
        return _CLASSIFICATION_TASKS.index(task)

    def _ensure_classification_input_combos(self) -> dict:
        """One Input combo per task (C2-C5). Since 9.0.0 they are not shown:
        the Classification stage has one "Input" selector that sets them
        all, and these remain what the steps and the saved mapping read.

        Built on demand because the configuration tab and the processing tab
        both want them and are constructed in that order.
        """
        combos = getattr(self, '_classification_input_combos', None)
        if combos:
            return combos
        combos = {}
        for task in _CLASSIFICATION_TASKS:
            combo = QComboBox()
            for label, value in _CLASSIFICATION_INPUTS:
                combo.addItem(label, value)
            combo.setCurrentIndex(2)     # matched, where fusion helps most
            combo.setFixedWidth(100)
            combo.setToolTip(
                "Which camera's features this classifier reads.\n\n"
                "'Matched' reads both fused, which is where the two sensors "
                "complement each other most - it needs the tracks matched "
                "(A3) and both cameras embedded (C1)."
            )
            combo.currentIndexChanged.connect(
                lambda _i, t=task: self._on_classification_model_changed(t))
            combos[task] = combo
        self._classification_input_combos = combos
        return combos

    def _classification_input_combo(self, task: str) -> QComboBox:
        return self._ensure_classification_input_combos()[task]

    def _classification_spec(self, task: str) -> dict:
        models = getattr(self, '_classification_models', None) or {}
        return dict(models.get(task) or {})

    def _set_classification_spec(self, task: str, spec: dict):
        models = dict(getattr(self, '_classification_models', None) or {})
        models[task] = spec
        self._classification_models = models

    def _on_classification_model_changed(self, task: str):
        """Mirror a table edit into the saved mapping."""
        row = self._task_row(task)
        table = self.classification_models_table
        spec = self._classification_spec(task)
        spec["modality"] = self._classification_input_combo(task).currentData()
        spec["model"] = table.cellWidget(row, 1).currentData()
        item = table.item(row, 2)
        spec["path"] = item.text().strip() if item else ""
        self._set_classification_spec(task, spec)

    def _apply_classification_models_to_table(self):
        """Push the saved mapping back into the widgets after a config load."""
        from ..bambi_classification_model_dialog import default_source_for

        table = getattr(self, 'classification_models_table', None)
        if table is None:
            return
        for task in _CLASSIFICATION_TASKS:
            row = self._task_row(task)
            spec = self._classification_spec(task)

            modality = self._classification_input_combo(task)
            found = modality.findData(spec.get("modality", "matched"))
            modality.blockSignals(True)
            modality.setCurrentIndex(found if found >= 0 else 2)
            modality.blockSignals(False)

            default = default_source_for(task)
            source = table.cellWidget(row, 1)
            found = source.findData(spec.get("model", default))
            source.blockSignals(True)
            source.setCurrentIndex(found if found >= 0 else 0)
            source.blockSignals(False)

            table.item(row, 2).setText(spec.get("path", ""))
            self._on_classification_model_changed(task)

        self._sync_stage_cameras()

    def _default_head_specs(self, projection: str):
        """``(task, projection, modality)`` for every task set to Default.

        The per-species sex selections need no separate entry: they all use
        the one published sex model, so downloading it once covers them.
        """
        from ..core import hf_access

        wanted = []
        for task in _CLASSIFICATION_TASKS:
            spec = self._classification_spec(task)
            if spec.get("model") != "default":
                continue
            if not hf_access.has_default_head(task):
                continue      # nothing published for this task yet
            wanted.append((task, projection, spec.get("modality", "matched")))
        return wanted

    def download_classification_models(self):
        """Fetch the published classifiers the table is set to use.

        Deliberately separate from running anything: the class mapping needs
        the file present to read a model's classes from it, and asking a user
        to run a classifier once before they can configure it is backwards.
        """
        from ..core import hf_access

        config = self.get_config()
        models_dir = BambiProcessor._get_default_model_dir()
        projection = config.get("classification_projection", "non_geo")
        if projection == "alfs":
            self.download_models_status.setText(
                "No default classifiers have been published for ALFS "
                "imagery yet.")
            QMessageBox.information(
                self, "Download Models",
                "There are no default classifiers for the ALFS projection "
                "yet.\n\nUse a custom model or a custom implementation "
                "with it, or choose another projection.")
            return
        wanted = self._default_head_specs(projection)

        if not wanted:
            self.download_models_status.setText(
                "Nothing to download - no classifier is set to Default.")
            QMessageBox.information(
                self, "Download Models",
                "None of the classifiers is set to Default.\n\n"
                "Set one to Default in the table above, or choose a custom "
                "model file - those are used from where they are and need no "
                "download."
            )
            return

        missing = hf_access.missing_heads(models_dir, wanted)
        if not missing:
            self.download_models_status.setText(
                f"🟢 All {len(wanted)} model(s) are already downloaded.")
            self.log(f"Classification models already present in {models_dir}")
            return

        token, _source = hf_access.resolve_token(
            self.hf_token_edit.text() if hasattr(self, 'hf_token_edit') else "")

        self.download_models_btn.setEnabled(False)
        self.download_models_status.setText(
            f"🟡 Downloading {len(missing)} model(s)…")
        downloaded, failures = 0, []
        try:
            for task, projection_name, modality in missing:
                repo = hf_access.default_head_repo(task)
                destination = hf_access.head_local_path(
                    models_dir, task, projection_name, modality)
                try:
                    BambiProcessor._download_head(
                        repo, task, projection_name, modality, destination,
                        token, self.log)
                    downloaded += 1
                except Exception as exc:      # noqa: BLE001 - reported below
                    failures.append(f"{task}: {exc}")
                    self.log(f"Could not download the {task} classifier: {exc}")
        finally:
            self.download_models_btn.setEnabled(True)

        if failures:
            self.download_models_status.setText(
                f"🔴 {downloaded} downloaded, {len(failures)} failed.")
            detail = "\n\n".join(failures)
            QMessageBox.warning(
                self, "Download Models",
                f"Some classifiers could not be downloaded:\n\n{detail}"
            )
            return

        self.download_models_status.setText(
            f"🟢 Downloaded {downloaded} model(s) - 'Labels' can now read "
            "their classes.")
        self.log(f"Downloaded {downloaded} classification model(s) into "
                 f"{models_dir}")

    def _browse_classification_model(self, task: str):
        row = self._task_row(task)
        table = self.classification_models_table
        from ..core import custom_models
        if table.cellWidget(row, 1).currentData() == custom_models.SOURCE:
            # a custom implementation: file, class and options in a dialog
            from ..bambi_custom_impl_widget import CustomImplementationDialog
            spec = self._classification_spec(task)
            dialog = CustomImplementationDialog(
                spec, custom_models.CLASSIFIER,
                camera=spec.get("modality", "matched"), task=task,
                model_dir=BambiProcessor._get_default_model_dir(),
                parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                spec.update(dialog.result())
                self._set_classification_spec(task, spec)
                table.item(row, 2).setText(spec.get("path", ""))
                self._on_classification_model_changed(task)
            return
        path, _filter = QFileDialog.getOpenFileName(
            self, f"Choose a {task} classifier", "",
            "TorchScript model (*.pt *.pth);;All files (*)")
        if not path:
            return
        row = self._task_row(task)
        table = self.classification_models_table
        table.item(row, 2).setText(path)
        # Choosing a file means using it - otherwise the path sits there
        # looking configured while the default is what actually runs.
        source = table.cellWidget(row, 1)
        source.setCurrentIndex(max(0, source.findData("custom")))
        self._on_classification_model_changed(task)

    def _edit_classification_labels(self, task: str):
        from ..bambi_label_mapping_dialog import BambiLabelMappingDialog

        config = self.get_config()
        self._on_classification_model_changed(task)
        dialog = BambiLabelMappingDialog(
            task, self._classification_spec(task),
            target_folder=config.get("target_folder", ""),
            models_dir=BambiProcessor._get_default_model_dir(),
            projection=config.get("classification_projection", "non_geo"),
            hf_token=config.get("hf_token", ""),
            parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._set_classification_spec(task, dialog.result_spec())
            self.log(f"Updated the {task} class mapping")

    def _edit_species_models(self, task: str):
        """Per-species model selection for a demographic task."""
        from ..bambi_classification_model_dialog import BambiClassificationModelDialog
        from ..core import hf_access

        config = self.get_config()
        self._on_classification_model_changed(task)
        dialog = BambiClassificationModelDialog(
            self._classification_spec(task),
            target_folder=config.get("target_folder", ""), task=task,
            parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._set_classification_spec(task, dialog.result_spec())
            chosen = len(dialog.result_spec().get("species") or {})
            label = hf_access.TASK_LABELS.get(task, task).lower()
            self.log(f"{label.capitalize()} classifiers configured for "
                     f"{chosen} species")

    def _combo_choice(self, attr: str, values) -> str:
        """The value a combo's current index stands for.

        Kept beside the combo rather than derived from its label, so renaming
        a label in the UI cannot silently re-target a setting.
        """
        combo = getattr(self, attr, None)
        if combo is None:
            return values[0]
        index = combo.currentIndex()
        return values[index] if 0 <= index < len(values) else values[0]

    def _classification_backbone(self) -> str:
        """The configured backbone, falling back to the shipped default.

        An empty field means "whatever this plugin version considers standard",
        so the default can move without every saved project pinning the old one.
        """
        edit = getattr(self, 'classification_backbone_edit', None)
        text = edit.text().strip() if edit is not None else ""
        return text or HF_DEFAULT_BACKBONE

    def _resolve_hf_token(self) -> str:
        """The token the processing steps should use, or ``""`` if there is none."""
        from ..core import hf_access

        return hf_access.resolve_token(self.hf_token_edit.text())[0]

    def _save_hf_token(self):
        """Persist the token to the QGIS settings (never to the project)."""
        gui_utils_hf.write_hf_token(self.hf_token_edit.text())

    def _toggle_hf_token_echo(self, state):
        """Show or hide the token characters."""
        mode = (QLineEdit.EchoMode.Normal if state
                else QLineEdit.EchoMode.Password)
        self.hf_token_edit.setEchoMode(mode)

    def check_hf_access(self):
        """Report whether the configured token can read the backbone.

        A gated repository is the most likely thing to go wrong in the whole
        classification feature, and finding out three steps into a long run is
        the worst time to learn it - so it is answerable up front.
        """
        from ..core import hf_access

        self._save_hf_token()
        token, source = hf_access.resolve_token(self.hf_token_edit.text())
        repo = self._classification_backbone()

        self.hf_access_status.setText("🟡 Checking…")
        self.hf_check_access_btn.setEnabled(False)
        try:
            # Blocking, but it is a single metadata request and the user asked
            # for it - a worker thread here would cost more than it saves.
            result = hf_access.check_repo_access(repo, token)
        finally:
            self.hf_check_access_btn.setEnabled(True)

        icons = {
            hf_access.ACCESS_GRANTED: "🟢",
            hf_access.ACCESS_GATED: "🔴",
            hf_access.ACCESS_NO_TOKEN: "🔴",
            hf_access.ACCESS_MISSING: "🔴",
            hf_access.ACCESS_UNAVAILABLE: "🟠",
            hf_access.ACCESS_ERROR: "🟠",
        }
        icon = icons.get(result["status"], "⚪")

        if result["status"] == hf_access.ACCESS_GRANTED:
            using = hf_access.describe_token_source(source)
            self.hf_access_status.setText(f"{icon} Access granted (using {using})")
        else:
            self.hf_access_status.setText(f"{icon} {result['message']}")

        self.log(f"Hugging Face access check for {repo}: "
                 f"{result['status']} - {result['message']}")

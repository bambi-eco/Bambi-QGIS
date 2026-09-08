# -*- coding: utf-8 -*-
"""
BAMBI Dependency Manager
========================

Dialog for downloading and installing optional and required Python
dependencies for the BAMBI QGIS plugin.

For packages hosted only on GitHub the dialog will:
  - use  ``pip install git+<url>``  when git is available on PATH, or
  - download the repository archive ZIP, unpack it into the plugin's
    ``plugins/`` subdirectory and run ``pip install <local_path>`` otherwise.
"""

import os
import zipfile
import threading
import queue
import datetime

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QTextEdit, QScrollArea, QWidget, QFrame, QComboBox,
    QSizePolicy,
)
from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtGui import QFont


from .core import alfs_runtime
from .core.hf_access import DEFAULT_BACKBONE as _DEFAULT_BACKBONE
from .gui_utils import read_alfs_selection, write_alfs_selection
from .core.dependency_ops import (  # noqa: F401 - re-exported API
    ALFS_DIST,
    ALFS_LEGACY_DIST,
    ALFS_PY_TAG,
    ALFS_REPO,
    BAMBI_DETECTION_TAG,
    GEOREF_TRACKER_TAG,
    alfs_install_spec,
    _DJI_SDK_URL,
    _VERSION_RANGES,
    _find_python,
    _get_dji_sdk_version,
    _get_version_status,
    _git_available,
    _install_github_zip,
    _run_pip,
    _torch_build_variant,
)

# ---------------------------------------------------------------------------
# (no background-worker class needed - threading.Thread + Queue is used instead)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class DependencyManagerDialog(QDialog):
    """Non-modal dialog that manages BAMBI plugin dependencies."""

    def __init__(self, parent=None, plugin_dir=None):
        super().__init__(parent)
        self._plugin_dir = plugin_dir or os.path.dirname(__file__)
        # Store downloads outside the plugin folder so locked DLLs (e.g. the
        # DJI SDK) never prevent plugin reinstallation.
        try:
            from qgis.core import QgsApplication
            self._plugins_dir = os.path.join(
                QgsApplication.qgisSettingsDirPath(), 'bambi_deps'
            )
        except Exception:
            self._plugins_dir = os.path.join(self._plugin_dir, 'plugins')
        os.makedirs(self._plugins_dir, exist_ok=True)

        self._timers = []         # keep QTimer objects alive until worker finishes
        self._buttons = {}        # key -> QPushButton
        self._button_labels = {}  # key -> original button label text
        self._status_labels = {}  # key -> QLabel (status indicator)
        self._dist_names = {}     # key -> dist_name (for post-install status refresh)

        self.setWindowTitle('BAMBI Dependency Manager')
        self.setMinimumWidth(700)
        self.setMinimumHeight(580)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setSpacing(10)

        # ---- Required dependencies ----
        # alfspy 3.0 rasterises through one of three engines and casts rays
        # through one of two casters, all pip extras of the same release. The
        # dropdowns start from what the project asked for rather than from
        # what is installed: unlike the old two-repository split there is no
        # installed distribution to read the answer off, and the same choice
        # has to be exported to $ALFS_ENGINE / $ALFS_RAYCASTER at render time.
        engine, raycaster = read_alfs_selection()
        self._engine_combo = self._make_alfs_combo(
            alfs_runtime.ENGINES, engine,
            'Which rasteriser alfspy renders with. Installed as a pip extra '
            'and exported as $ALFS_ENGINE when a step renders, so the '
            'selection has to be installed to work.')
        self._raycaster_combo = self._make_alfs_combo(
            alfs_runtime.RAYCASTERS, raycaster,
            'Which ray caster alfspy intersects the terrain with. Installed '
            'as a pip extra and exported as $ALFS_RAYCASTER.')

        selector_box = QWidget()
        selector_layout = QVBoxLayout(selector_box)
        selector_layout.setContentsMargins(0, 2, 0, 0)
        selector_layout.setSpacing(2)
        for caption, combo in (('Engine:', self._engine_combo),
                               ('Ray caster:', self._raycaster_combo)):
            line = QHBoxLayout()
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(6)
            caption_lbl = QLabel(caption)
            caption_lbl.setFixedWidth(70)
            line.addWidget(caption_lbl, 0, Qt.AlignmentFlag.AlignVCenter)
            line.addWidget(combo, 0)
            line.addStretch(1)
            selector_layout.addLayout(line)

        self._alfs_hint = QLabel('')
        self._alfs_hint.setWordWrap(True)
        self._alfs_hint.setStyleSheet('color:#777;')
        # Ignored lets the hint shrink below its text width, so it re-wraps
        # with the dialog instead of widening it.
        self._alfs_hint.setSizePolicy(QSizePolicy.Policy.Ignored,
                                      QSizePolicy.Policy.Minimum)
        self._alfs_hint.setMinimumWidth(0)
        selector_layout.addWidget(self._alfs_hint)

        vbox.addWidget(self._build_group('Required Dependencies', [
            dict(
                key='alfs_py',
                label='ALFS-PY Framework',
                desc='Airborne light-field sampling framework for the actual geo-referencing processing.',
                callback=self._install_alfs_py,
                dist_name=ALFS_DIST,
                extra_widget=selector_box,
            ),
            dict(
                key='bambi_detection',
                label='BAMBI Detection Framework',
                desc='Utility library for geo-referencing drone flights.',
                callback=self._install_bambi_detection,
                dist_name='bambi-detection',
            )
        ]))

        # ---- Calibration (optional) ----
        vbox.addWidget(self._build_group('Calibration  (optional)', [
            dict(
                key='pycolmap',
                label='pycolmap  (v4.0.3)',
                desc='Structure-from-Motion via COLMAP for single-camera calibration.',
                callback=self._install_pycolmap,
                dist_name='pycolmap',
            ),
        ]))

        # ---- Extended Tracking (optional) ----
        vbox.addWidget(self._build_group('Extended Tracking  (optional)', [
            dict(
                key='boxmot',
                label='BoxMOT  (v17.0.0)',
                desc='Additional multi-object tracking backends.',
                callback=self._install_boxmot,
                dist_name='boxmot',
            ),
            dict(
                key='geo_ref_tracking',
                label=f'Geo-Referenced Tracking  (v{GEOREF_TRACKER_TAG})',
                desc='Geo-referenced tracker extensions – BoxMOT must be installed first.',
                callback=self._install_geo_ref_tracking,
                dist_name='georef-tracker',
            ),
        ]))

        # ---- Classification (optional) ----
        vbox.addWidget(self._build_group('Classification  (optional)', [
            dict(
                key='classification',
                label='Transformers + Hugging Face Hub',
                desc=(
                    'DINOv3 feature extraction for the occlusion, species and '
                    'sex classifiers, and local SAM3 segmentation (transformers '
                    '>= 5.0). Both models are gated on Hugging Face - request '
                    'access and enter a token in the Classification '
                    'configuration tab.'
                ),
                callback=self._install_classification,
                dist_names=[('transformers', 'transformers'),
                            ('hub', 'huggingface-hub')],
            ),
        ]))

        # ---- Flight route generation (optional) ----
        vbox.addWidget(self._build_group('Flight Route Generation  (optional)', [
            dict(
                key='fiona',
                label='Fiona  (v1.10.1)',
                desc='Geospatial file I/O – required for reading KML/GeoJSON area files in the flight planner.',
                callback=self._install_fiona,
                dist_name='fiona',
            ),
            dict(
                key='simplekml',
                label='simplekml  (v1.3.6)',
                desc='KML/KMZ export – required for writing flight routes in the flight planner.',
                callback=self._install_simplekml,
                dist_name='simplekml',
            ),
        ]))

        # ---- DJI Thermal SDK ----
        vbox.addWidget(self._build_group('DJI Thermal SDK  (optional)', [
            dict(
                key='dji_sdk',
                label='DJI Thermal SDK  (v1.8)',
                desc=(
                    'Downloads the DJI Thermal SDK ZIP into the plugin\'s plugins/ folder. '
                ),
                callback=self._download_dji_sdk,
                dist_name='dji-thermal-sdk',
                btn_label='Download',
            ),
        ]))

        # ---- GPU support – prominent warning ----
        gpu_group = QGroupBox('GPU Support – CUDA  (optional)')
        gpu_layout = QVBoxLayout(gpu_group)

        warn = QLabel(
            '<b style="color:#b00000;">⚠  Install GPU support LAST – '
            'after all other dependencies have been installed.</b><br>'
            'Installing torch/torchvision with the CUDA index <i>before</i> other packages '
            'may cause pip to downgrade them back to CPU-only builds.'
        )
        warn.setWordWrap(True)
        warn.setTextFormat(Qt.TextFormat.RichText)
        warn.setStyleSheet(
            'background:#fff3cd; border:1px solid #ffc107; '
            'border-radius:4px; padding:6px;'
        )
        gpu_layout.addWidget(warn)
        gpu_layout.addLayout(self._make_row(
            key='gpu_support',
            label='CUDA GPU Support  (PyTorch + torchvision cu121)',
            desc=(
                'Removes any existing CPU torch/torchvision build and installs '
                'the CUDA 12.1 variant from pytorch.org.'
            ),
            callback=self._install_gpu,
            dist_names=[('torch', 'torch'), ('torchvision', 'torchvision')],
        ))
        vbox.addWidget(gpu_group)

        vbox.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # The hint under the dropdowns describes the current pair, and warns
        # when the superseded fork is still installed over the same import
        # path.
        self._refresh_alfs_hint()

        # ---- Restart notice ----
        restart_label = QLabel(
            '<b>After pressing any install button, we recommend to restart QGIS '
            'to activate the newly installed packages.</b>'
        )
        restart_label.setWordWrap(True)
        restart_label.setTextFormat(Qt.TextFormat.RichText)
        restart_label.setStyleSheet(
            'background:#e8f0fe; border:1px solid #a8c7fa; '
            'border-radius:4px; padding:6px;'
        )
        root.addWidget(restart_label)

        # ---- Log output ----
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(150)
        self._log_view.setFont(QFont('Courier New', 8))
        self._log_view.setPlaceholderText('Installation log output will appear here …')
        root.addWidget(self._log_view)

    def _build_group(self, title, deps):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        for i, dep in enumerate(deps):
            layout.addLayout(self._make_row(**dep))
            if i < len(deps) - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setFrameShadow(QFrame.Shadow.Sunken)
                layout.addWidget(sep)
        return group

    def _make_row(self, key, label, desc, callback,
                  dist_name=None, dist_names=None, btn_label='Install',
                  extra_widget=None):
        """Build one dependency row.

        Pass ``dist_name`` (str) for a single package, or ``dist_names``
        (list of ``(display_label, dist_name)`` pairs) to show stacked
        per-package statuses (e.g. torch + torchvision).  ``extra_widget`` is
        appended under the description (e.g. the backend selector).
        """
        row = QHBoxLayout()
        row.setSpacing(8)

        btn = QPushButton(btn_label)
        btn.setFixedWidth(100)
        btn.clicked.connect(callback)
        self._buttons[key] = btn
        self._button_labels[key] = btn_label

        name_lbl = QLabel(f'<b>{label}</b>')
        name_lbl.setTextFormat(Qt.TextFormat.RichText)
        # Wrap rather than force the whole dialog wider than its 700px minimum;
        # without this the scroll area scrolls sideways and nothing else in the
        # row (descriptions, the backend selector) ever gets to re-wrap.
        name_lbl.setWordWrap(True)
        desc_lbl = QLabel(f'<span style="color:#555;">{desc}</span>')
        desc_lbl.setWordWrap(True)
        desc_lbl.setTextFormat(Qt.TextFormat.RichText)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        text_col.addWidget(name_lbl)
        text_col.addWidget(desc_lbl)
        if extra_widget is not None:
            extra_widget.setStyleSheet('color:#555;')
            text_col.addWidget(extra_widget)

        if dist_names:
            # ---- multi-package stacked status ----
            self._dist_names[key] = None  # handled via _status_labels list
            status_widget = QWidget()
            status_layout_v = QVBoxLayout(status_widget)
            status_layout_v.setContentsMargins(0, 0, 0, 0)
            status_layout_v.setSpacing(2)
            lbl_pairs = []
            for display_name, dn in dist_names:
                lbl = QLabel()
                # wide enough for e.g. "torchvision: ✔ v0.20.1+cu121"
                lbl.setFixedWidth(200)
                lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                lbl.setTextFormat(Qt.TextFormat.RichText)
                ver, status = _get_version_status(dn, self._plugins_dir)
                self._apply_status_label(lbl, dn, ver, status, prefix=display_name)
                status_layout_v.addWidget(lbl)
                lbl_pairs.append((dn, lbl))
            self._status_labels[key] = lbl_pairs  # list of (dist_name, QLabel)
            row.addWidget(btn)
            row.addLayout(text_col, 1)
            row.addWidget(status_widget)
        else:
            # ---- single-package status (existing behaviour) ----
            self._dist_names[key] = dist_name
            status_lbl = QLabel()
            status_lbl.setFixedWidth(130)
            status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_lbl.setTextFormat(Qt.TextFormat.RichText)
            if dist_name:
                ver, status = _get_version_status(dist_name, self._plugins_dir)
                self._apply_status_label(status_lbl, dist_name, ver, status)
            else:
                status_lbl.setText('<span style="color:#888;">-</span>')
            self._status_labels[key] = status_lbl  # single QLabel
            row.addWidget(btn)
            row.addLayout(text_col, 1)
            row.addWidget(status_lbl)

        return row

    def _apply_status_label(self, lbl, dist_name, ver, status, prefix=None):
        """Update a status QLabel based on the version-check result."""
        lbl.setToolTip('')
        pre = f'{prefix}: ' if prefix else ''

        # For torch/torchvision also surface the compute platform (CUDA vs CPU)
        variant = None
        if dist_name in ('torch', 'torchvision') and ver:
            variant = _torch_build_variant(ver)
        variant_note = {'cuda': 'CUDA build', 'rocm': 'ROCm build',
                        'cpu': 'CPU build'}.get(variant)

        if status == 'not_found':
            lbl.setText(f'<span style="color:#888;">{pre}not found</span>')
        elif status == 'ok':
            if variant == 'cpu':
                lbl.setText(
                    f'<span style="color:#e67e00;">{pre}⚠ v{ver}'
                    '<br><small>CPU build</small></span>'
                )
                lbl.setToolTip('CPU-only build installed - use Install to replace it '
                               'with the CUDA (cu121) variant for GPU support.')
            elif variant_note:
                lbl.setText(
                    f'<span style="color:green;">{pre}✔ v{ver}'
                    f'<br><small>{variant_note}</small></span>'
                )
            else:
                lbl.setText(f'<span style="color:green;">{pre}✔ v{ver}</span>')
        elif status == 'untested':
            min_ver, max_ver = _VERSION_RANGES.get(dist_name, (None, None))
            range_str = f'{min_ver or "any"} – {max_ver or "any"}'
            note = f'not tested · {variant_note}' if variant_note else 'not tested'
            lbl.setText(
                f'<span style="color:#e67e00;">{pre}⚠ v{ver}'
                f'<br><small>{note}</small></span>'
            )
            lbl.setToolTip(f'Installed version is outside the tested range: {range_str}')

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def _log_line(self, text):
        self._log_view.append(text)
        self._log_view.ensureCursorVisible()

    def _start_worker(self, btn_key, fn):
        btn = self._buttons.get(btn_key)
        if btn:
            btn.setEnabled(False)
            btn.setText('Working…')

        q = queue.Queue()
        t0 = datetime.datetime.now()

        def _run():
            def log_fn(msg):
                ts = datetime.datetime.now().strftime('%H:%M:%S')
                q.put(('log', f'[{ts}] {msg}'))
            try:
                fn(log_fn)
                elapsed = (datetime.datetime.now() - t0).total_seconds()
                q.put(('done', True, f'Completed in {elapsed:.1f}s'))
            except Exception as exc:  # noqa: BLE001
                elapsed = (datetime.datetime.now() - t0).total_seconds()
                log_fn(f'Failed after {elapsed:.1f}s: {exc}')
                q.put(('done', False, str(exc)))

        threading.Thread(target=_run, daemon=True).start()

        timer = QTimer(self)

        def _poll():
            try:
                while True:
                    item = q.get_nowait()
                    if item[0] == 'log':
                        self._log_line(item[1])
                    else:  # 'done'
                        timer.stop()
                        self._timers.remove(timer)
                        self._on_finished(btn_key, item[1], item[2])
                        return
            except queue.Empty:
                pass

        timer.timeout.connect(_poll)
        timer.start(100)
        self._timers.append(timer)

    def _on_finished(self, btn_key, success, message):
        btn = self._buttons.get(btn_key)
        status_info = self._status_labels.get(btn_key)
        original_label = self._button_labels.get(btn_key, 'Install')

        if btn:
            btn.setEnabled(True)
            btn.setText(original_label)

        # Determine whether this row has a single label or a list of (dn, lbl) pairs
        is_multi = isinstance(status_info, list)
        pairs = status_info if is_multi else [(self._dist_names.get(btn_key), status_info)]

        if success:
            self._log_line(f'✔ {message}')
            self._log_line('→ Please restart QGIS to activate the package.')
            for dn, lbl in pairs:
                if lbl is None:
                    continue
                if dn:
                    ver, status = _get_version_status(dn, self._plugins_dir)
                    prefix = dn if is_multi else None
                    if ver:
                        self._apply_status_label(lbl, dn, ver, status, prefix=prefix)
                    else:
                        lbl.setText('<span style="color:green;">✔ installed</span>')
                else:
                    lbl.setText('<span style="color:green;">✔ done</span>')
        else:
            self._log_line(f'✖ Error: {message}')
            for dn, lbl in pairs:
                if lbl is not None:
                    lbl.setText('<span style="color:red;">✖ error</span>')

        if btn_key == 'alfs_py':
            # An install removes the superseded fork, so the warning about it
            # has to go with it rather than waiting for the dialog to reopen.
            self._refresh_alfs_hint()

    # ------------------------------------------------------------------
    # GitHub package helper
    # ------------------------------------------------------------------

    def _install_github_pkg(self, pkg_key, btn_key, zip_url, git_url):
        def _do(log_fn):
            git_ver = _git_available()
            if git_ver:
                log_fn(f'{git_ver} detected')
                log_fn(f'Source: {git_url}')
                _run_pip(['install', '--force-reinstall', git_url], log_fn)
            else:
                log_fn('git not found on PATH – using ZIP download fallback')
                log_fn(f'Source: {zip_url}')
                _install_github_zip(zip_url, pkg_key, self._plugins_dir, log_fn)
        self._start_worker(btn_key, _do)

    # ------------------------------------------------------------------
    # Individual install callbacks
    # ------------------------------------------------------------------

    def _install_bambi_detection(self):
        self._log_line(f'─── BAMBI Detection Framework {BAMBI_DETECTION_TAG} ───')
        self._install_github_pkg(
            'bambi_detection', 'bambi_detection',
            'https://github.com/bambi-eco/bambi_detection/archive/refs/tags/'
            f'{BAMBI_DETECTION_TAG}.zip',
            'git+https://github.com/bambi-eco/bambi_detection.git@'
            f'{BAMBI_DETECTION_TAG}',
        )

    def _make_alfs_combo(self, options, selected, tooltip):
        """A dropdown over ``alfs_runtime`` options, on *selected*."""
        combo = QComboBox()
        for option in options:
            combo.addItem(option.label, option.name)
        combo.setCurrentIndex(max(0, combo.findData(selected)))
        combo.setToolTip(tooltip)
        combo.currentIndexChanged.connect(self._on_alfs_selection_changed)
        return combo

    def _selected_alfs_spec(self):
        """The alfspy install the two dropdowns currently describe."""
        return alfs_install_spec(self._engine_combo.currentData(),
                                 self._raycaster_combo.currentData())

    def _on_alfs_selection_changed(self):
        """Persist the choice and say what it will take to make it real."""
        spec = self._selected_alfs_spec()
        write_alfs_selection(spec['engine'], spec['raycaster'])
        self._refresh_alfs_hint()
        self._log_line(
            f"Rendering set to {spec['label']} - saved to the project. Press "
            f"Install to add what it needs "
            f"({spec['dist']}{spec['extras_suffix']})."
        )

    def _refresh_alfs_hint(self):
        """Describe the selected pair, and flag a leftover AlfsTorch install."""
        engine = self._engine_combo.currentData()
        raycaster = self._raycaster_combo.currentData()
        parts = [
            next(o.hint for o in alfs_runtime.ENGINES if o.name == engine),
            next(o.hint for o in alfs_runtime.RAYCASTERS
                 if o.name == raycaster),
        ]
        legacy = _get_version_status(ALFS_LEGACY_DIST, self._plugins_dir)[1]
        if legacy != 'not_found':
            # It owns the same ``alfspy`` import path, so whatever it provides
            # is what gets imported - the selection would be ignored.
            parts.append(
                f'<b>{ALFS_LEGACY_DIST} is still installed.</b> It provides '
                'the same "alfspy" package as this one, so it has to go '
                'before the engine choice means anything - pressing Install '
                'removes it.')
        self._alfs_hint.setTextFormat(Qt.TextFormat.RichText)
        self._alfs_hint.setText('<br>'.join(parts))

    def _install_alfs_py(self):
        spec = self._selected_alfs_spec()
        self._log_line(f"─── ALFS Framework {ALFS_REPO} "
                       f"{spec['tag']} ({spec['label']}) ───")

        def _do(log_fn):
            # Before 3.0 the PyTorch rasteriser was a separate distribution
            # that installed a package called ``alfspy`` too. Leaving it in
            # place would give two dists owning one import path, which pip
            # does not clean up on its own and which would quietly shadow the
            # engine selection.
            log_fn(f"Removing {spec['legacy_dist']} if present "
                   "(the superseded fork provides the 'alfspy' package) …")
            _run_pip(['uninstall', '-y', spec['legacy_dist']], log_fn)

            git_ver = _git_available()
            if git_ver:
                log_fn(f'{git_ver} detected')
                log_fn(f"Source: {spec['git_url']}")
                _run_pip(['install', '--force-reinstall', spec['git_url']], log_fn)
            else:
                log_fn('git not found on PATH – using ZIP download fallback')
                log_fn(f"Source: {spec['zip_url']}{spec['extras_suffix']}")
                _install_github_zip(spec['zip_url'], 'alfs_py',
                                    self._plugins_dir, log_fn,
                                    extras=spec['extras_suffix'])

        self._start_worker('alfs_py', _do)

    def _install_pycolmap(self):
        self._log_line('─── pycolmap 4.0.3 ───')

        def _do(log_fn):
            _run_pip(['install', '--force-reinstall', 'pycolmap==4.0.3'], log_fn)
        self._start_worker('pycolmap', _do)

    def _install_boxmot(self):
        self._log_line('─── BoxMOT 17.0.0 ───')

        def _do(log_fn):
            _run_pip(['install', '--force-reinstall', 'boxmot==17.0.0'], log_fn)
        self._start_worker('boxmot', _do)

    def _install_geo_ref_tracking(self):
        self._log_line(f'─── Geo-Referenced Tracking {GEOREF_TRACKER_TAG} ───')
        self._install_github_pkg(
            'geo_referenced_tracking', 'geo_ref_tracking',
            'https://github.com/bambi-eco/Geo-Referenced-Tracking/archive/'
            f'refs/tags/{GEOREF_TRACKER_TAG}.zip',
            'git+https://github.com/bambi-eco/Geo-Referenced-Tracking.git@'
            f'{GEOREF_TRACKER_TAG}',
        )

    def _install_fiona(self):
        self._log_line('─── Fiona 1.10.1 ───')

        def _do(log_fn):
            _run_pip(['install', '--force-reinstall', 'fiona==1.10.1'], log_fn)
        self._start_worker('fiona', _do)

    def _install_simplekml(self):
        self._log_line('─── simplekml 1.3.6 ───')

        def _do(log_fn):
            _run_pip(['install', '--force-reinstall', 'simplekml==1.3.6'], log_fn)
        self._start_worker('simplekml', _do)

    def _install_classification(self):
        self._log_line('─── Classification (transformers + huggingface-hub) ───')

        def _do(log_fn):
            # transformers pulls huggingface-hub itself, but naming it keeps the
            # status row honest when only one of the two is present.
            _run_pip(['install', '--upgrade',
                      'transformers>=4.56.0', 'huggingface-hub'], log_fn)
            log_fn('')
            log_fn('The DINOv3 backbone is a gated model: request access at')
            log_fn(f'  https://huggingface.co/{_DEFAULT_BACKBONE}')
            log_fn('then enter a read token in the Classification tab and press')
            log_fn('"Check access". The backbone (~3.3 GB) downloads on first use.')
            log_fn('')
            log_fn('The same token serves local SAM3 segmentation (facebook/sam3,')
            log_fn('also gated, ~3.4 GB): request access at')
            log_fn('  https://huggingface.co/facebook/sam3')
            log_fn('and tick "Run SAM3 locally" in the SAM3 Segmentation tab.')
        self._start_worker('classification', _do)

    def _install_gpu(self):
        self._log_line('─── GPU Support (CUDA 12.1) ───')

        def _do(log_fn):
            log_fn('Removing existing CPU torch/torchvision …')
            _run_pip(['uninstall', 'torch', 'torchvision', '-y'], log_fn)
            log_fn('Installing CUDA 12.1 builds from pytorch.org …')
            _run_pip(
                ['install', '--force-reinstall', 'torch', 'torchvision',
                 '--index-url', 'https://download.pytorch.org/whl/cu121'],
                log_fn,
            )
        self._start_worker('gpu_support', _do)

    def _download_dji_sdk(self):
        self._log_line('─── DJI Thermal SDK ───')
        plugins_dir = self._plugins_dir

        def _do(log_fn):
            import requests
            dest = os.path.join(plugins_dir, 'dji_thermal_sdk_v1.8_20250829.zip')
            log_fn(f'Destination: {dest}')
            log_fn(f'URL: {_DJI_SDK_URL}')
            try:
                response = requests.get(_DJI_SDK_URL, stream=True, timeout=120)
                log_fn(f'HTTP status: {response.status_code}')
                response.raise_for_status()
                total = int(response.headers.get('Content-Length', 0))
                if total:
                    log_fn(f'File size: {total / 1048576:.1f} MB')
                downloaded = 0
                last_pct = -10
                with open(dest, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = (downloaded * 100) // total
                                if pct >= last_pct + 10:
                                    log_fn(
                                        f'  {downloaded / 1048576:.1f} /'
                                        f' {total / 1048576:.1f} MB  ({pct}%)'
                                    )
                                    last_pct = pct
                            else:
                                log_fn(f'  {downloaded / 1024:.0f} KB …')
            except Exception as exc:
                log_fn(f'Error: {type(exc).__name__}: {exc}')
                raise RuntimeError(f'Download failed: {exc}') from exc
            log_fn(f'Download complete: {dest}')
            zip_name = os.path.splitext(os.path.basename(dest))[0]
            extract_dir = os.path.join(plugins_dir, zip_name)
            os.makedirs(extract_dir, exist_ok=True)
            log_fn(f'Extracting archive to: {extract_dir}')
            with zipfile.ZipFile(dest, 'r') as zf:
                num_files = len(zf.namelist())
                zf.extractall(extract_dir)
            os.remove(dest)
            log_fn(f'Extracted {num_files} files')
            log_fn('Follow the DJI Thermal SDK README for native library setup.')
        self._start_worker('dji_sdk', _do)

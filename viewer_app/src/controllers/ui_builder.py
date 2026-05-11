from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..app_settings import METADATA_FIELDS, TIME_SNAP
from ..widgets import MetadataInlineEditor, LabelBar


class UiBuilderMixin:
    """
    Builds the main application UI and connects Qt signals.

    This mixin should only contain:
    - widget creation
    - layout creation
    - plot initialization
    - signal wiring

    It should not contain business logic, file I/O, segment logic,
    playback logic, or keyboard shortcuts.
    """

    def build_ui(self: Any) -> None:
        """
        Build the complete main window UI.
        """
        # Window and menu
        self.setWindowTitle("Audio Annotation Tool")
        self.resize(1200, 820)

        m_file = self.menuBar().addMenu("File")
        self.act_open = m_file.addAction("Open folder…")

        # Central widget and main layout
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        main_layout = QtWidgets.QHBoxLayout(cw)

        # ------------------------------------------------------------------
        # Left panel: waveform, time slider, spectrogram
        # ------------------------------------------------------------------
        left = QtWidgets.QWidget()
        main_layout.addWidget(left, 3)

        left_grid = QtWidgets.QGridLayout(left)

        # Waveform plot
        self.p_wave = pg.PlotWidget()
        self.p_wave.setLabel("bottom", "Time (s)")
        self.p_wave.setLabel("left", "Amplitude")
        self.p_wave.showGrid(x=True, y=True, alpha=0.2)

        # Waveform overlay items
        self.playhead = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("#CC3333", width=1),
        )

        self.region = pg.LinearRegionItem(
            [0.0, 2.5],
            brush=(100, 180, 255, 60),
            movable=True,
        )

        self.curve = pg.PlotDataItem(
            pen=pg.mkPen("#1976D2", width=1.2),
            clipToView=True,
            autoDownsample=True,
            downsampleMethod="peak",
        )

        self.p_wave.addItem(self.curve)
        self.p_wave.addItem(self.playhead)
        self.p_wave.addItem(self.region)

        left_grid.addWidget(self.p_wave, 0, 0)

        # Safe waveform downsampling setup
        try:
            self.curve.setDownsampling(auto=True, method="peak")
        except Exception:
            pass

        # Time slider row
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 0)

        self.lbl_time = QtWidgets.QLabel("0.00 s")

        time_bar = QtWidgets.QHBoxLayout()
        time_bar.addWidget(QtWidgets.QLabel("Time:"))
        time_bar.addWidget(self.time_slider, 1)
        time_bar.addWidget(self.lbl_time)

        left_grid.addLayout(time_bar, 1, 0)

        # Spectrogram area
        self.init_spectrogram(left_grid)

        # ------------------------------------------------------------------
        # Right panel: file info, navigation, metadata, labels, segments, tools
        # ------------------------------------------------------------------
        right = QtWidgets.QWidget()
        main_layout.addWidget(right, 1)

        right_layout = QtWidgets.QVBoxLayout(right)

        # Current path + open folder
        self.lbl_path = QtWidgets.QLabel("—")
        self.lbl_path.setStyleSheet("font-weight:600;")
        right_layout.addWidget(self.lbl_path)

        self.btn_open_folder = QtWidgets.QPushButton("Open folder…")
        right_layout.addWidget(self.btn_open_folder)

        # Navigation row
        self.btn_prev = QtWidgets.QPushButton("◀ Prev")
        self.btn_next = QtWidgets.QPushButton("Next ▶")

        self.combo_jump = QtWidgets.QComboBox()
        self.combo_jump.setEnabled(False)
        self.combo_jump.setMinimumWidth(280)

        nav_row = QtWidgets.QHBoxLayout()
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.btn_next)
        nav_row.addWidget(QtWidgets.QLabel("Jump to:"))
        nav_row.addWidget(self.combo_jump)

        right_layout.addLayout(nav_row)

        # Selection row: selected start/end/delta
        selection_row = QtWidgets.QHBoxLayout()
        selection_row.addWidget(QtWidgets.QLabel("Selected:"))

        self.sel_start = QtWidgets.QDoubleSpinBox()
        self.sel_start.setDecimals(2)
        self.sel_start.setSingleStep(TIME_SNAP)
        self.sel_start.setRange(0, 1e6)

        self.sel_end = QtWidgets.QDoubleSpinBox()
        self.sel_end.setDecimals(2)
        self.sel_end.setSingleStep(TIME_SNAP)
        self.sel_end.setRange(0, 1e6)

        self.lbl_sel_delta = QtWidgets.QLabel("(Δ 0.00 s)")

        selection_row.addWidget(self.sel_start)
        selection_row.addWidget(QtWidgets.QLabel("–"))
        selection_row.addWidget(self.sel_end)
        selection_row.addWidget(self.lbl_sel_delta)

        right_layout.addLayout(selection_row)

        # Inline metadata editor
        self.meta_inline = MetadataInlineEditor(METADATA_FIELDS, parent=right)

        # Backward-compatible preference access.
        recent_environments = getattr(self.prefs, "recent_environments", [])

        self.meta_inline.set_recent_locations(recent_environments)

        right_layout.addWidget(self.meta_inline)

        # Label controls
        label_row = QtWidgets.QHBoxLayout()
        label_row.addWidget(QtWidgets.QLabel("Labels"))
        label_row.addStretch(1)

        self.btn_label_info = QtWidgets.QToolButton()
        self.btn_label_info.setText("Info")
        self.btn_label_info.setToolTip("How labels work")

        label_row.addWidget(self.btn_label_info)

        right_layout.addLayout(label_row)

        self.labelset_combo = QtWidgets.QComboBox()
        right_layout.addWidget(self.labelset_combo)

        self.labelbar = LabelBar({})
        right_layout.addWidget(self.labelbar)

        # Auto segmentation button
        self.btn_auto_seg = QtWidgets.QPushButton("Auto segment…")
        right_layout.addWidget(self.btn_auto_seg)

        # Segments list
        right_layout.addWidget(QtWidgets.QLabel("Segments"))

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )

        right_layout.addWidget(self.list, 1)

        # Edit segment group
        edit_group = QtWidgets.QGroupBox("Edit segment")
        edit_form = QtWidgets.QFormLayout(edit_group)

        self.spin_start = QtWidgets.QDoubleSpinBox()
        self.spin_start.setDecimals(2)
        self.spin_start.setSingleStep(TIME_SNAP)
        self.spin_start.setRange(0, 1e6)

        self.spin_end = QtWidgets.QDoubleSpinBox()
        self.spin_end.setDecimals(2)
        self.spin_end.setSingleStep(TIME_SNAP)
        self.spin_end.setRange(0, 1e6)

        self.list_labels = QtWidgets.QListWidget()

        self.btn_remove_label = QtWidgets.QPushButton("Delete selected label")

        edit_form.addRow("Start (s):", self.spin_start)
        edit_form.addRow("End (s):", self.spin_end)
        edit_form.addRow("Labels:", self.list_labels)
        edit_form.addRow("", self.btn_remove_label)

        right_layout.addWidget(edit_group)

        # Segment update/delete row
        segment_action_row = QtWidgets.QHBoxLayout()

        self.btn_update = QtWidgets.QPushButton("Update")
        self.btn_delete = QtWidgets.QPushButton("Delete")

        segment_action_row.addWidget(self.btn_update)
        segment_action_row.addWidget(self.btn_delete)

        right_layout.addLayout(segment_action_row)

        # Band-pass filter controls
        filter_group = QtWidgets.QGroupBox("Band-pass filter")
        filter_form = QtWidgets.QFormLayout(filter_group)

        filter_header = QtWidgets.QHBoxLayout()

        self.chk_bp = QtWidgets.QCheckBox("Filter on")
        filter_header.addWidget(self.chk_bp)
        filter_header.addStretch(1)

        self.btn_bp_info = QtWidgets.QToolButton()
        self.btn_bp_info.setText("Info")
        self.btn_bp_info.setToolTip("Explanation about filter settings")
        filter_header.addWidget(self.btn_bp_info)

        self.sp_low = QtWidgets.QDoubleSpinBox()
        self.sp_low.setRange(0.1, 20000.0)
        self.sp_low.setDecimals(1)
        self.sp_low.setSingleStep(10.0)
        self.sp_low.setValue(50.0)

        self.sp_high = QtWidgets.QDoubleSpinBox()
        self.sp_high.setRange(1.0, 20000.0)
        self.sp_high.setDecimals(1)
        self.sp_high.setSingleStep(10.0)
        self.sp_high.setValue(2000.0)

        self.sp_order = QtWidgets.QSpinBox()
        self.sp_order.setRange(1, 10)
        self.sp_order.setValue(2)

        self.chk_zero = QtWidgets.QCheckBox("Zero-phase")
        self.chk_zero.setChecked(True)

        filter_form.addRow(filter_header)
        filter_form.addRow("Low (Hz):", self.sp_low)
        filter_form.addRow("High (Hz):", self.sp_high)
        filter_form.addRow("Order:", self.sp_order)
        filter_form.addRow(self.chk_zero)

        right_layout.addWidget(filter_group)

        # Export tools
        self.btn_export_csv = QtWidgets.QPushButton("Export CSV")
        right_layout.addWidget(self.btn_export_csv)

        self.lbl_last_export = QtWidgets.QLabel("Last exported: —")
        self.lbl_last_export.setStyleSheet("color: gray; font-size: 10pt;")
        self.lbl_last_export.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        right_layout.addWidget(self.lbl_last_export)

        # Playback timer
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)  # ~33 fps

    def init_spectrogram(
        self: Any,
        grid_layout: QtWidgets.QGridLayout,
    ) -> None:
        """
        Build the spectrogram plot area.
        """
        self.p_spec = pg.PlotWidget()
        self.p_spec.setBackground("k")
        self.p_spec.setLabel("bottom", "Time (s)")
        self.p_spec.setLabel("left", "Frequency (Hz)")
        self.p_spec.setMouseEnabled(x=True, y=True)
        self.p_spec.setXLink(self.p_wave)

        self.img_spec = pg.ImageItem(axisOrder="row-major")
        self.img_spec.setImage(
            np.zeros((1, 1), dtype=np.float32),
            autoLevels=True,
        )
        self.img_spec.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
        self.p_spec.addItem(self.img_spec)

        self.colorbar = None

        try:
            cmap = pg.colormap.get("inferno")
            self.img_spec.setLookupTable(cmap.getLookupTable())  # type: ignore

            self.colorbar = pg.ColorBarItem(values=(-100, 0), colorMap=cmap)

            try:
                self.colorbar.setImageItem(self.img_spec)
            except Exception:
                self.colorbar = None

        except Exception:
            self.colorbar = None

        grid_layout.addWidget(self.p_spec, 2, 0)

        info_row = QtWidgets.QHBoxLayout()

        self.lbl_stft_params = QtWidgets.QLabel("")
        self.lbl_stft_params.setStyleSheet("color: gray; font-size: 10pt;")
        info_row.addWidget(self.lbl_stft_params)

        self.btn_stft_info = QtWidgets.QToolButton()
        self.btn_stft_info.setText("Info")
        info_row.addWidget(self.btn_stft_info)

        grid_layout.addLayout(info_row, 3, 0)

    def connect_signals(self: Any) -> None:
        """
        Connect UI signals to controller methods.
        """
        # File/navigation
        self.act_open.triggered.connect(self.open_folder_dialog)
        self.btn_open_folder.clicked.connect(self.open_folder_dialog)
        self.btn_prev.clicked.connect(lambda: self.advance(-1))
        self.btn_next.clicked.connect(lambda: self.advance(+1))
        self.combo_jump.currentIndexChanged.connect(self._on_jump_selected)

        # Metadata
        self.meta_inline.changed.connect(self._on_meta_inline_changed)

        # Selection region
        self.region.sigRegionChanged.connect(self.on_region_changed)
        self.sel_start.valueChanged.connect(lambda _: self.on_sel_spin_changed())
        self.sel_end.valueChanged.connect(lambda _: self.on_sel_spin_changed())

        # Segment list/editing
        self.list.currentRowChanged.connect(self.on_list_selection)
        self.list.itemSelectionChanged.connect(self.on_segment_selection_changed)
        self.list.itemDoubleClicked.connect(lambda *_: self._play_current_segment())

        self.btn_remove_label.clicked.connect(self.remove_selected_label)
        self.btn_update.clicked.connect(self.update_segment)
        self.btn_delete.clicked.connect(self.delete_selected)

        # Labels
        self.btn_label_info.clicked.connect(self._show_label_info)
        self.labelset_combo.currentTextChanged.connect(self._apply_labelset)
        self.labelbar.toggled.connect(self._on_labelbar_toggled)

        # Auto segmentation
        self.btn_auto_seg.clicked.connect(self.auto_segment_dialog)

        # Filter controls
        self.btn_bp_info.clicked.connect(self._show_bp_info)
        self.chk_bp.stateChanged.connect(self.on_filter_ui_changed)
        self.sp_low.valueChanged.connect(self.on_filter_ui_changed)
        self.sp_high.valueChanged.connect(self.on_filter_ui_changed)
        self.sp_order.valueChanged.connect(self.on_filter_ui_changed)
        self.chk_zero.stateChanged.connect(self.on_filter_ui_changed)

        # Spectrogram info
        self.btn_stft_info.clicked.connect(self._show_stft_info)

        # Export
        self.btn_export_csv.clicked.connect(self.export_csv)

        # Time slider
        self.time_slider.valueChanged.connect(self.on_slider_changed)

        # Player events
        self.player.started.connect(self.on_play_started)
        self.player.stopped.connect(self.on_play_stopped)
        self.timer.timeout.connect(self.tick_playhead)
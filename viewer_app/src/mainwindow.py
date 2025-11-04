import os, uuid, json, datetime
from typing import List, Dict, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import soundfile as sf
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder='row-major', antialias=False, useOpenGL=False, useNumba=False)

# extra failsafes
pg.setConfigOptions(useNumba=False)
try:
    pg.setConfigOptions(useCupy=False)
except Exception:
    pass

from .config import DEFAULT_SR, TIME_SNAP, DEBUG_STFT, DYNAMIC_SPECTRO_LEVELS, GRAYSCALE_DEBUG, METADATA_FIELDS, LABEL_SETS, DEFAULT_LABEL_SET, load_prefs, save_prefs, labels_list_to_dict, load_default_locations, UserPrefs
from .models import Segment, FileState, normalize_subject_id
from .utils import snap_t, human_relpath, json_sidecar_path, csv_path_for_root, labels_dataset_path, ensure_dir, LABEL_COLORS
from .audio import bandpass_filter, compute_stft_db, Player
from .dialogs import StartDialog, AutoSegmentDialog
from .widgets import ClickableRegion, MetadataInlineEditor, LabelBar

def _dbg(msg: str):
    if DEBUG_STFT:
        print(msg)

# Faster drawing options
pg.setConfigOptions(imageAxisOrder='row-major', antialias=False, useOpenGL=False)

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung Sound Viewer (PySide6 + PyQtGraph)")
        self.resize(1200, 820)

        m = self.menuBar().addMenu("Bestand")
        self.act_open = m.addAction("Map openen…")
        self.act_open.triggered.connect(self.open_folder_dialog)

        self._custom_labels: Optional[list[str]] = None
        self.player = Player()
        self.prefs: UserPrefs = load_prefs()
        self.root = ""
        self.files: List[str] = []
        self.idx = -1
        self.y_raw: Optional[np.ndarray] = None
        self.sr = DEFAULT_SR
        self.t: Optional[np.ndarray] = None
        self.state: Optional[FileState] = None
        self.overlay_regions: Dict[str, pg.LinearRegionItem] = {}
        self._blocking = False

        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        H = QtWidgets.QHBoxLayout(cw)

        # Left: waveform + slider + spectrogram
        left = QtWidgets.QWidget(); H.addWidget(left, 3)
        gl = QtWidgets.QGridLayout(left)

        self.p_wave = pg.PlotWidget()
        self.p_wave.setLabel("bottom", "Tijd (s)"); self.p_wave.setLabel("left", "Amplitude")
        self.p_wave.showGrid(x=True, y=True, alpha=0.2)

        self.playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#CC3333', width=1))
        self.region   = pg.LinearRegionItem([0.0, 2.5], brush=(100,180,255,60), movable=True)
        self.curve = pg.PlotDataItem(pen=pg.mkPen('#1976D2', width=1.2), clipToView=True, autoDownsample=True, downsampleMethod='peak')
        self.p_wave.addItem(self.curve); self.p_wave.addItem(self.playhead); self.p_wave.addItem(self.region)
        gl.addWidget(self.p_wave, 0, 0)

        try: self.curve.setDownsampling(auto=True, method='peak')
        except Exception: pass

        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.time_slider.setRange(0, 0)
        self.lbl_time = QtWidgets.QLabel("0.00 s")
        tbar = QtWidgets.QHBoxLayout(); tbar.addWidget(QtWidgets.QLabel("Tijd:")); tbar.addWidget(self.time_slider, 1); tbar.addWidget(self.lbl_time)
        gl.addLayout(tbar, 1, 0)

        self.init_spectrogram(gl)

        # Right panel
        right = QtWidgets.QWidget(); H.addWidget(right, 1)
        rv = QtWidgets.QVBoxLayout(right)

        self.lbl_path = QtWidgets.QLabel("—"); self.lbl_path.setStyleSheet("font-weight:600;"); rv.addWidget(self.lbl_path)
        self.btn_open_folder = QtWidgets.QPushButton("Open folder…"); rv.addWidget(self.btn_open_folder)

        self.btn_prev = QtWidgets.QPushButton("◀ Prev")
        self.btn_next = QtWidgets.QPushButton("Next ▶")
        self.combo_jump = QtWidgets.QComboBox(); self.combo_jump.setEnabled(False); self.combo_jump.setMinimumWidth(280); self.combo_jump.currentIndexChanged.connect(self._on_jump_selected)
        nav_row = QtWidgets.QHBoxLayout(); nav_row.addWidget(self.btn_prev); nav_row.addWidget(self.btn_next); nav_row.addWidget(QtWidgets.QLabel("Jump to:")); nav_row.addWidget(self.combo_jump); rv.addLayout(nav_row)

        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(QtWidgets.QLabel("Selected:"))
        self.sel_start = QtWidgets.QDoubleSpinBox(); self.sel_start.setDecimals(2); self.sel_start.setSingleStep(TIME_SNAP); self.sel_start.setRange(0, 1e6)
        self.sel_end   = QtWidgets.QDoubleSpinBox(); self.sel_end.setDecimals(2); self.sel_end.setSingleStep(TIME_SNAP); self.sel_end.setRange(0, 1e6)
        self.lbl_sel_delta = QtWidgets.QLabel("(Δ 0.00 s)")
        sel_row.addWidget(self.sel_start); sel_row.addWidget(QtWidgets.QLabel("–")); sel_row.addWidget(self.sel_end); sel_row.addWidget(self.lbl_sel_delta)

        rv.addLayout(sel_row)

        self.meta_inline = MetadataInlineEditor(METADATA_FIELDS, parent=right)
        self.meta_inline.set_recent_mics(self.prefs.recents_mic_types)
        self.meta_inline.set_recent_locations(self.prefs.recents_locations)
        rv.addWidget(self.meta_inline)
        self.meta_inline.changed.connect(self._on_meta_inline_changed)

        # --- Labelset + LabelBar -------------------------------------------------
        label_row = QtWidgets.QHBoxLayout()
        lbl_labelset = QtWidgets.QLabel("Label set")

        self.btn_label_info = QtWidgets.QToolButton()
        self.btn_label_info.setText("Info")
        self.btn_label_info.setToolTip("How to use the label sets")
        self.btn_label_info.clicked.connect(self._show_label_info)

        label_row.addWidget(lbl_labelset)
        label_row.addStretch(1)
        label_row.addWidget(self.btn_label_info)
        rv.addLayout(label_row)

        self.labelset_combo = QtWidgets.QComboBox()
        self.labelbar = LabelBar({})
        rv.addWidget(self.labelset_combo)
        rv.addWidget(self.labelbar)

        self.labelset_combo.currentTextChanged.connect(self._apply_labelset)
        self.labelbar.toggled.connect(self._on_labelbar_toggled)
        # -------------------------------------------------------------------------

        self.sel_start.valueChanged.connect(lambda _: self.on_sel_spin_changed())
        self.sel_end.valueChanged.connect(lambda _: self.on_sel_spin_changed())

        self.btn_auto_seg = QtWidgets.QPushButton("Auto segment…"); rv.addWidget(self.btn_auto_seg); self.btn_auto_seg.clicked.connect(self.auto_segment_dialog)

        rv.addWidget(QtWidgets.QLabel("Segments"))
        self.list = QtWidgets.QListWidget(); rv.addWidget(self.list, 1)

        edit = QtWidgets.QGroupBox("Edit Segment"); fe = QtWidgets.QFormLayout(edit)
        self.spin_start = QtWidgets.QDoubleSpinBox(); self.spin_start.setDecimals(2); self.spin_start.setSingleStep(TIME_SNAP); self.spin_start.setRange(0, 1e6)
        self.spin_end   = QtWidgets.QDoubleSpinBox(); self.spin_end.setDecimals(2); self.spin_end.setSingleStep(TIME_SNAP); self.spin_end.setRange(0, 1e6)
        self.list_labels = QtWidgets.QListWidget()
        self.btn_remove_label = QtWidgets.QPushButton("Delete selected label")
        fe.addRow("Start (s):", self.spin_start); fe.addRow("End (s):", self.spin_end); fe.addRow("Labels:", self.list_labels); fe.addRow("", self.btn_remove_label)
        rv.addWidget(edit)

        hb2 = QtWidgets.QHBoxLayout(); self.btn_update = QtWidgets.QPushButton("Update"); self.btn_delete = QtWidgets.QPushButton("Delete"); hb2.addWidget(self.btn_update); hb2.addWidget(self.btn_delete); rv.addLayout(hb2)

        grp_bp = QtWidgets.QGroupBox("Bandpass-filter"); fv = QtWidgets.QFormLayout(grp_bp)
        self.chk_bp = QtWidgets.QCheckBox("Filter On")
        self.sp_low = QtWidgets.QDoubleSpinBox(); self.sp_low.setRange(0.1, 20000.0); self.sp_low.setDecimals(1); self.sp_low.setSingleStep(10.0); self.sp_low.setValue(50.0)
        self.sp_high = QtWidgets.QDoubleSpinBox(); self.sp_high.setRange(1.0, 20000.0); self.sp_high.setDecimals(1); self.sp_high.setSingleStep(10.0); self.sp_high.setValue(2000.0)
        self.sp_order = QtWidgets.QSpinBox(); self.sp_order.setRange(1, 10); self.sp_order.setValue(2)
        self.chk_zero = QtWidgets.QCheckBox("Zero-phase"); self.chk_zero.setChecked(True)
        row_top = QtWidgets.QHBoxLayout(); row_top.addWidget(self.chk_bp); row_top.addStretch(1)
        self.btn_bp_info = QtWidgets.QToolButton(); self.btn_bp_info.setText("Info"); self.btn_bp_info.setToolTip("Explanation about filter settings"); self.btn_bp_info.clicked.connect(self._show_bp_info); row_top.addWidget(self.btn_bp_info)
        fv.addRow(row_top); fv.addRow("Low (Hz):", self.sp_low); fv.addRow("High (Hz):", self.sp_high); fv.addRow("Order:", self.sp_order); fv.addRow(self.chk_zero)
        rv.addWidget(grp_bp)

        self._filt_cache = None; self._filt_params = None

        self.chk_bp.stateChanged.connect(self.on_filter_ui_changed)
        self.sp_low.valueChanged.connect(self.on_filter_ui_changed)
        self.sp_high.valueChanged.connect(self.on_filter_ui_changed)
        self.sp_order.valueChanged.connect(self.on_filter_ui_changed)
        self.chk_zero.stateChanged.connect(self.on_filter_ui_changed)

        self.btn_export_csv = QtWidgets.QPushButton("Export CSV"); rv.addWidget(self.btn_export_csv)
        self.lbl_last_export = QtWidgets.QLabel("Last exported: —"); self.lbl_last_export.setStyleSheet("color: gray; font-size: 10pt;"); self.lbl_last_export.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse); rv.addWidget(self.lbl_last_export)

        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, self.toggle_play)
        QtGui.QShortcut(QtGui.QKeySequence("N"), self, lambda: self.advance(+1))
        QtGui.QShortcut(QtGui.QKeySequence("P"), self, lambda: self.advance(-1))
        QtGui.QShortcut(QtGui.QKeySequence("Return"), self, self.update_segment)
        QtGui.QShortcut(QtGui.QKeySequence("Enter"),  self, self.update_segment)
        QtGui.QShortcut(QtGui.QKeySequence("Delete"), self, self.delete_selected)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, self.reset_view)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self, self.undo)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self, self.redo)
        self._undo_stack: List[Tuple[Callable[[], None], Callable[[], None]]] = []
        self._redo_stack: List[Tuple[Callable[[], None], Callable[[], None]]] = []

        for seq, cb in [
            ("Left",        lambda: self.nudge_region(-TIME_SNAP, "move")),
            ("Right",       lambda: self.nudge_region(+TIME_SNAP, "move")),
            ("Shift+Left",  lambda: self.nudge_region(-TIME_SNAP, "start")),
            ("Shift+Right", lambda: self.nudge_region(+TIME_SNAP, "start")),
            ("Ctrl+Left",   lambda: self.nudge_region(-TIME_SNAP, "end")),
            ("Ctrl+Right",  lambda: self.nudge_region(+TIME_SNAP, "end")),
        ]:
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(cb)

        self.btn_prev.clicked.connect(lambda: self.advance(-1))
        self.btn_next.clicked.connect(lambda: self.advance(+1))
        self.btn_open_folder.clicked.connect(self.open_folder_dialog)
        self.btn_remove_label.clicked.connect(self.remove_selected_label)
        self.region.sigRegionChanged.connect(self.on_region_changed)
        self.list.currentRowChanged.connect(self.on_list_selection)
        self.btn_update.clicked.connect(self.update_segment)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.time_slider.valueChanged.connect(self.on_slider_changed)
        self.list.itemDoubleClicked.connect(lambda *_: self._play_current_segment())

        self.player.started.connect(self.on_play_started)
        self.player.stopped.connect(self.on_play_stopped)

        self.timer = QtCore.QTimer(self); self.timer.setInterval(30); self.timer.timeout.connect(self.tick_playhead)

        self.session_meta: Dict[str, object] = {}
        self.play_window: Tuple[float, float] = (0.0, 0.0)

        self.open_folder_dialog(first=True)

    def _show_stft_info(self):
        txt = (
            "Short explanation about STFT parameters:\n\n"
            "• nperseg: number of samples per FFT frame (large = better frequency resolution, but coarser in time).\n"
            "• hop: hop size between consecutive frames (small = smoother time resolution, but more data).\n"
            "• window: type of windowing function to reduce edge effects (common: 'hann', 'hamming').\n\n"
            "Note: these settings only affect how the spectrogram is displayed, not the segments or the audio itself."
        )
        QtWidgets.QMessageBox.information(self, "STFT parameters", txt)

    def _show_bp_info(self):
        txt = (
            "Short explanation about bandpass filtering:\n\n"
            "• Lung focus (80-3000 Hz): suppresses low hearttones and captures the higher energy of many Lung sounds.\n"
            "• Heart focus (20-250 Hz): focuses on S1/S2 and low-frequency components; murmurs may be slightly higher.\n\n"
            "• Order (4-6 recommended): higher order = steeper edges, but more risk of ringing/instability.\n"
            "  Order 4 is a safe and stable choice.\n"
            "• Zero-phase (on/off): ‘on’ uses forward–backward filtering (no phase delay, but non-causal).\n"
            "  Use ‘on’ for offline analysis/visualization; ‘off’ for real-time chains with minimal latency.\n"
        )
        QtWidgets.QMessageBox.information(self, "Bandpass – Info", txt)

    def _show_label_info(self):
        txt = (
            "<b>How the Label Set works:</b><br><br>"
            "• <b>Custom</b> — your own labels loaded from <code>labels_dataset.json</code>. "
            "Use this for experiments or non-clinical recordings (e.g., coughs, tracheal sounds).<br><br>"
            "• <b>Lung</b> — predefined clinical categories for respiratory sounds "
            "(crackles, wheezes, rhonchi, etc.).<br>"
            "• <b>Heart</b> — predefined cardiac sound types (S1, S2, murmurs, clicks, gallops).<br><br>"
            "Click a button below to add or remove that label on the current segment. "
            "The button stays highlighted while the segment contains that label. "
            "You can switch label sets at any time; it won't delete existing annotations."
        )
        QtWidgets.QMessageBox.information(self, "Label Set Info", txt)

    def _ensure_spec_imageitem(self):
        """Garandeer dat self.img_spec een geldige (minimaal 1x1) image heeft."""
        try:
            if getattr(self, "img_spec", None) is None:
                return  # wordt in init_spectrogram gemaakt
            # Als width/height None zijn, zet een 1x1 image
            if self.img_spec.image is None:
                self.img_spec.setImage(np.zeros((1, 1), dtype=np.float32), autoLevels=True)
                self.img_spec.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
        except Exception:
            # Worst case: still no image; force a tiny one
            try:
                self.img_spec.setImage(np.zeros((1, 1), dtype=np.float32), autoLevels=True)
                self.img_spec.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
            except Exception:
                pass

    def init_spectrogram(self, grid_layout: QtWidgets.QGridLayout):
        # Plot canvas
        self.p_spec = pg.PlotWidget()
        self.p_spec.setBackground('k')
        self.p_spec.setLabel("bottom", "Tijd (s)")
        self.p_spec.setLabel("left", "Frequentie (Hz)")
        self.p_spec.setMouseEnabled(x=True, y=True)
        self.p_spec.setXLink(self.p_wave)

        # ImageItem + veilige 1×1 start-image (voorkomt NoneType issues)
        self.img_spec = pg.ImageItem(axisOrder='row-major')
        self.img_spec.setImage(np.zeros((1, 1), dtype=np.float32), autoLevels=True)
        self.img_spec.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
        self.p_spec.addItem(self.img_spec)

        # Inferno colormap + ColorBar
        self.colorbar = None
        try:
            cmap = pg.colormap.get('inferno')                     # pyqtgraph >= 0.12
            self.img_spec.setLookupTable(cmap.getLookupTable())   # koppel LUT aan image # type: ignore
            self.colorbar = pg.ColorBarItem(values=(-100, 0), colorMap=cmap)
            try:
                # mooi in de plot integreren als jouw pg-versie het ondersteunt
                self.colorbar.setImageItem(self.img_spec, insertIn=self.p_spec.getPlotItem())  # type: ignore
            except TypeError:
                self.colorbar.setImageItem(self.img_spec)  # fallback
        except Exception:
            # Colormap is nice-to-have; zonder werkt het ook in grijswaarden
            self.colorbar = None

        grid_layout.addWidget(self.p_spec, 2, 0)

        # Info-balk onderaan
        row = QtWidgets.QHBoxLayout()
        self.lbl_stft_params = QtWidgets.QLabel("")
        self.lbl_stft_params.setStyleSheet("color: gray; font-size: 10pt;")
        row.addWidget(self.lbl_stft_params)
        self.btn_stft_info = QtWidgets.QToolButton()
        self.btn_stft_info.setText("Info")
        self.btn_stft_info.clicked.connect(self._show_stft_info)
        row.addWidget(self.btn_stft_info)
        grid_layout.addLayout(row, 3, 0)

    def update_spectrogram(self):
        try:
            y = self.current_signal()
            if y is None or len(y) == 0:
                return

            cfg = getattr(self, "_stft_cfg", {"nperseg":1024, "hop":256, "window":"hann"})
            f, t, S_db = compute_stft_db(
                y, self.sr,
                nperseg=int(cfg.get("nperseg", 1024)),
                hop=int(cfg.get("hop", 256)),
                window=str(cfg.get("window", "hann"))
            )
            if S_db.size == 0 or len(t) <= 1 or len(f) <= 1:
                return

            # Max te tonen frequentie (rekening houdend met bandpass UI)
            if getattr(self, "chk_bp", None) and self.chk_bp.isChecked():
                fmax_plot = min(float(self.sp_high.value()), float(self.sr)/2.0 - 1e-6)
            else:
                fmax_plot = float(f[-1])

            mask = f <= fmax_plot
            if not np.any(mask):
                mask = f <= f[-1]

            f_plot = f[mask]
            img = S_db[mask, :]  # (freq, time) verwacht

            # Sommige STFT helpers geven (time, freq)
            if img.shape == (len(t), len(f_plot)):
                img = img.T  # forceer (freq, time)

            # Sanitize & typify
            img = np.nan_to_num(img, neginf=-120.0, posinf=0.0).astype(np.float32, copy=False)

            # Robuuste levels via percentielen; val terug op [-100, 0] dB
            try:
                vmin = float(np.percentile(img, 2.0))
                vmax = float(np.percentile(img, 98.0))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                    raise ValueError
            except Exception:
                vmin, vmax = -100.0, 0.0

            # 1) Altijd EERST de image zetten (dan kent ImageItem breedte/hoogte)
            #    DYNAMIC_SPECTRO_LEVELS kan True/False zijn in je config
            auto_lv = bool(DYNAMIC_SPECTRO_LEVELS)
            if auto_lv:
                self.img_spec.setImage(img, autoLevels=True)
            else:
                self.img_spec.setImage(img, autoLevels=False, levels=(vmin, vmax))

            # 2) Dan de geometrie (wereldcoördinaten tijd×frequentie)
            t_max = max(float(t[-1]), 1e-6)
            f_max = max(float(f_plot[-1]) if len(f_plot) else float(f[-1]), 1e-6)
            self.img_spec.setRect(QtCore.QRectF(0.0, 0.0, t_max, f_max))

            # 3) As-limieten / ranges
            self.p_spec.setLimits(xMin=0.0, xMax=t_max, yMin=0.0, yMax=f_max)
            self.p_spec.setXRange(0.0, t_max)
            self.p_spec.setYRange(0.0, f_max)

            # 4) UI label
            self.lbl_stft_params.setText(
                f"STFT: nperseg={cfg.get('nperseg')} | hop={cfg.get('hop')} | window={cfg.get('window')}"
            )
        except Exception as e:
            import traceback; traceback.print_exc()

    def on_filter_ui_changed(self, *args):
        """Herbereken/teken wanneer filter-instellingen wijzigen."""
        # cache ongeldig maken zodat nieuwe parameters effect hebben
        self._filt_cache = None
        self._filt_params = None

        # zachte validatie: high > low afdwingen
        try:
            low = float(self.sp_low.value())
            high = float(self.sp_high.value())
        except Exception:
            low, high = 50.0, 2000.0
        if high <= low:
            self.sp_high.blockSignals(True)
            self.sp_high.setValue(low + 1.0)
            self.sp_high.blockSignals(False)

        # UI updaten (veilig als nog niet alles klaar is)
        try:
            self.draw_waveform()
            self.update_spectrogram()
        except Exception:
            pass

    def reset_view(self):
        """
        Reset waveform and spectrogram views to their initial, sensible state.

        - Waveform (self.p_wave): re-enable auto-ranging and show the full duration.
        - Spectrogram (self.p_spec): recompute/update and restore full time/frequency view.
        - Selection and segments remain unchanged.
        """
        # Reset waveform view
        try:
            vb_wave = self.p_wave.getViewBox()
            # Re-enable autorange on both axes
            vb_wave.enableAutoRange(x=True, y=True)

            # If we know the duration, show the full time window explicitly
            if self.t is not None and len(self.t) > 1:
                xmax = float(self.t[-1])
                # Slight padding to avoid clipping extremes
                vb_wave.setXRange(0.0, xmax, padding=0.02)
        except Exception:
            pass

        # Reset spectrogram view: re-render and autorange
        try:
            # Recompute / update spectrogram image + axes ranges
            self.update_spectrogram()

            # Also ensure autorange on spectrogram's viewbox
            vb_spec = self.p_spec.getViewBox()
            vb_spec.enableAutoRange(x=True, y=True)
        except Exception:
            pass

    def on_region_changed(self):
        if self._blocking: return
        a, b = self.region.getRegion()
        a = max(0.0, snap_t(a)); b = max(a + TIME_SNAP, snap_t(b)) # type: ignore
        self._blocking = True
        self.region.setRegion((a, b)); self.sel_start.setValue(a); self.sel_end.setValue(b); self.lbl_sel_delta.setText(f"(Δ {(b - a):.2f} s)")
        self._blocking = False

    def on_slider_changed(self, val: int):
        t = val/100.0
        self.playhead.setPos(t); self.lbl_time.setText(f"{t:.2f} s")

    def on_sel_spin_changed(self):
        if self._blocking: return
        a = float(self.sel_start.value()); b = float(self.sel_end.value())
        a = max(0.0, snap_t(a)); b = max(a + TIME_SNAP, snap_t(b))
        self._blocking = True
        self.sel_start.setValue(a); self.sel_end.setValue(b); self.region.setRegion((a, b)); self.lbl_sel_delta.setText(f"(Δ {(b - a):.2f} s)")
        self._blocking = False

    def nudge_region(self, dt: float, mode: str = "move"):
        if self.t is None or len(self.t) == 0: return
        dur = float(self.t[-1]); a, b = self.region.getRegion(); a = float(a); b = float(b); step = float(dt) # type: ignore
        if mode == "move":
            width = b - a; new_a = max(0.0, min(a + step, dur - width)); new_b = new_a + width
        elif mode == "start":
            new_a = max(0.0, min(a + step, b - TIME_SNAP)); new_b = b
        elif mode == "end":
            new_a = a; new_b = min(dur, max(b + step, a + TIME_SNAP))
        else:
            return
        new_a = snap_t(new_a); new_b = snap_t(new_b)
        if new_b <= new_a:
            new_b = min(dur, new_a + TIME_SNAP)
        self._blocking = True
        self.region.setRegion((new_a, new_b)); self.sel_start.setValue(new_a); self.sel_end.setValue(new_b); self.lbl_sel_delta.setText(f"(Δ {(new_b - new_a):.2f} s)")
        self._blocking = False

    def toggle_play(self):
        from .audio import HAVE_SD
        if not HAVE_SD or self.y_raw is None:
            QtWidgets.QMessageBox.information(self, "Playback", "Playback niet beschikbaar.")
            return
        if self.player.playing:
            self.player.stop()
        else:
            t0 = self.time_slider.value()/100.0
            t1 = len(self.current_signal())/self.sr
            if t0 >= t1:
                t0 = max(0.0, t1-0.01)
            self.player.play(self.current_signal(), self.sr, t0, t1)

    def on_play_started(self, a: float, b: float):
        self.play_window = (a, b)
        self._elapsed = QtCore.QElapsedTimer(); self._elapsed.start(); self.timer.start()

    def on_play_stopped(self):
        self.timer.stop()

    def tick_playhead(self):
        a, b = self.play_window
        t_now = a + self._elapsed.elapsed()/1000.0
        if t_now >= b:
            self.player.stop(); t_now = b
        self.playhead.setPos(t_now)
        self.time_slider.blockSignals(True); self.time_slider.setValue(int(t_now*100)); self.time_slider.blockSignals(False)
        self.lbl_time.setText(f"{t_now:.2f} s")

    def _brush_for_labels(self, labels: List[str]):
        return LABEL_COLORS.color_for(labels)

    def _on_meta_inline_changed(self, values: dict):
        if not self.state:
            return
        vals = dict(values or {})
        # Normaliseer subject_id indien aanwezig
        sid = vals.get("subject_id", "")
        if sid:
            try:
                vals["subject_id"] = normalize_subject_id(str(sid))
            except Exception:
                # laat ongewijzigd in UI; validatie gebeurt al in widget
                pass
        self.state.meta = dict(self.state.meta or {})
        self.state.meta.update(vals)
        self.save_json()
        # recents
        self.bump_recents(vals.get("microphone_type"), vals.get("location"))

    def refresh_segment_list(self):
        self.list.clear()
        if not self.state:
            return
        for reg in getattr(self, "overlay_regions", {}).values():
            try: self.p_wave.removeItem(reg)
            except Exception: pass
        self.overlay_regions.clear()
        for s in self.state.segments:
            self.list.addItem(f"{s.t_start:.2f}-{s.t_end:.2f}s | {'; '.join(s.labels) or '(geen labels)'}")
            reg = ClickableRegion([s.t_start, s.t_end], brush=self._brush_for_labels(s.labels), seg_id=s.id)
            self.p_wave.addItem(reg); reg.clicked.connect(self.on_overlay_clicked); self.overlay_regions[s.id] = reg
        self._reflect_labelbar()

    def _find_segment_by_bounds(self, a: float, b: float) -> Optional[Segment]:
        if not self.state: return None
        for s in self.state.segments:
            if abs(s.t_start - a) < 1e-6 and abs(s.t_end - b) < 1e-6:
                return s
        return None

    def on_list_selection(self, row: int):
        if not self.state or row < 0 or row >= len(self.state.segments):
            self.list_labels.clear(); return
        s = self.state.segments[row]
        self.spin_start.setValue(s.t_start); self.spin_end.setValue(s.t_end)
        self.rebuild_label_list()
        self._blocking = True
        self.region.setRegion((s.t_start, s.t_end))
        self.sel_start.setValue(s.t_start); self.sel_end.setValue(s.t_end); self.lbl_sel_delta.setText(f"(Δ {(s.t_end - s.t_start):.2f} s)")
        self._blocking = False
        self._reflect_labelbar()

    def remove_selected_label(self):
        row = self.list.currentRow()
        if not self.state or row < 0: return
        s = self.state.segments[row]
        lab_row = self.list_labels.currentRow()
        if lab_row < 0 or lab_row >= len(s.labels): return
        del s.labels[lab_row]
        self.on_list_selection(row); self.refresh_segment_list(); self.save_json(); self.rebuild_label_list()

    def update_segment(self):
        row = self.list.currentRow()
        if not self.state or row < 0: return
        s = self.state.segments[row]
        new_a = snap_t(self.spin_start.value()); new_b = max(new_a + TIME_SNAP, snap_t(self.spin_end.value()))
        new_labels = [self.list_labels.item(i).text() for i in range(self.list_labels.count())]
        s.t_start, s.t_end, s.labels = new_a, new_b, new_labels
        self.refresh_segment_list(); self.list.setCurrentRow(row); self.save_json(); self.rebuild_label_list()

    def delete_selected(self):
        row = self.list.currentRow()
        if not self.state or row < 0: return
        ans = QtWidgets.QMessageBox.question(self, "Delete segment", "Verwijder geselecteerd segment?")
        if ans != QtWidgets.QMessageBox.Yes: # type: ignore
            return
        del self.state.segments[row]
        self.refresh_segment_list(); self.save_json()

    def _rel_display_name(self, abspath: str) -> str:
        try: rel = os.path.relpath(abspath, self.root)
        except Exception: rel = os.path.basename(abspath)
        return rel.replace("\\", "/")

    def _populate_jump_list(self):
        self.combo_jump.blockSignals(True); self.combo_jump.clear()
        display_names = []
        for p in self.files:
            abspath = p if isinstance(p, str) else getattr(p, "path", "")
            display_names.append(self._rel_display_name(abspath))
        self.combo_jump.addItems(display_names); self.combo_jump.setEnabled(len(display_names) > 0)
        if 0 <= self.idx < len(display_names):
            self.combo_jump.setCurrentIndex(self.idx)
        self.combo_jump.blockSignals(False)

    def _on_jump_selected(self, i: int):
        if not (0 <= i < len(self.files)): return
        if i == self.idx: return
        self.idx = i; self.load_current()

    def _after_navigation_changed(self):
        if self.combo_jump.isEnabled() and 0 <= self.idx < self.combo_jump.count():
            self.combo_jump.blockSignals(True); self.combo_jump.setCurrentIndex(self.idx); self.combo_jump.blockSignals(False)

    def export_csv(self):
        if not self.state: return
        rows = []
        today = datetime.date.today().isoformat()
        for fp in self.files:
            js = json_sidecar_path(fp)
            if not os.path.isfile(js): continue
            try:
                with open(js, "r", encoding="utf-8") as fh:
                    st = FileState.from_json(json.load(fh))
            except Exception:
                continue
            for s in st.segments:
                    rows.append({
                        "date": today,
                        "filename": self._rel_display_name(fp),
                        "t_start": s.t_start, "t_end": s.t_end,
                        "label": ";".join(s.labels),
                        "subject_id": st.meta.get("subject_id", ""),
                        "microphone_type": st.meta.get("microphone_type", ""),
                        "sample_rate": st.meta.get("sample_rate", ""),
                        "location": st.meta.get("location", ""),
                    })
        if not rows:
            QtWidgets.QMessageBox.information(self, "Export", "Geen segmenten om te exporteren."); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", csv_path_for_root(self.root), "CSV (*.csv)")
        if not path: return
        pd.DataFrame(rows).to_csv(path, index=False)
        QtWidgets.QMessageBox.information(self, "Export", f"Opgeslagen {len(rows)} rijen naar:\n{path}")

    def bump_recents(self, mic: str | None = None, loc: str | None = None):
        changed = False
        if mic:
            if mic in self.prefs.recents_mic_types:
                self.prefs.recents_mic_types.remove(mic)
            self.prefs.recents_mic_types.append(mic)
            self.prefs.recents_mic_types = self.prefs.recents_mic_types[-8:]
            changed = True
        if loc:
            if loc in self.prefs.recents_locations:
                self.prefs.recents_locations.remove(loc)
            self.prefs.recents_locations.append(loc)
            self.prefs.recents_locations = self.prefs.recents_locations[-8:]
            changed = True
        if changed:
            save_prefs(self.prefs)
            # Mic: gewoon recents
            self.meta_inline.set_recent_mics(self.prefs.recents_mic_types)
            defaults = load_default_locations(self.labelset_combo.currentText())
            merged = list(dict.fromkeys(defaults + self.prefs.recents_locations))
            self.meta_inline.set_recent_locations(merged)

    def _refresh_location_choices(self):
        # haal defaults voor huidige labelset
        defaults = load_default_locations(self.labelset_combo.currentText())
        # voeg je 'recents' toe, zonder dubbels (defaults eerst)
        merged = list(dict.fromkeys(defaults + self.prefs.recents_locations))
        # zet ze in de MetadataInlineEditor combobox
        self.meta_inline.set_recent_locations(merged)

    def load_labels_json(self):
        """
        Leest labels_dataset.json en zet:
        - self._custom_labels : Optional[List[str]]
        - STFT/filter/auto-segment defaults (zoals voorheen)
        Laat de LabelBar invullen door _refresh_labelset_combo().
        """
        path = labels_dataset_path()
        default_cfg = {
            "version": 1,
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
            "labels": [],  # <— standaard leeg: 'custom' wordt dan niet getoond
            "meta_defaults": {"location":""},
            "filter_defaults": {"lowcut": 50, "highcut": 3000, "order": 4, "zero_phase": True},
            "stft_params": {"nperseg": 1024, "hop": 256, "window": "hann"},
            "auto_segment_defaults": {"length_s": 3.00, "overlap_s": 0.00, "label": ""}
        }
        from pathlib import Path
        if Path(path).is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = default_cfg
        else:
            cfg = default_cfg
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default_cfg, f, ensure_ascii=False, indent=2)

        # 1) labels.json → custom labels
        labels = cfg.get("labels", []) or []
        self._custom_labels = list(labels) if labels else None

        # 2) kleuren opbouwen (optioneel; blijft werken)
        LABEL_COLORS.build(labels if labels else [])

        # 3) meta/filter/stft/auto-seg defaults (zoals je had)
        defaults = cfg.get("meta_defaults", {})
        for k, v in defaults.items():
            self.session_meta.setdefault(k, v)

        fdef = cfg.get("filter_defaults", {})
        if "lowcut" in fdef:     self.sp_low.setValue(float(fdef["lowcut"]))
        if "highcut" in fdef:    self.sp_high.setValue(float(fdef["highcut"]))
        if "order" in fdef:      self.sp_order.setValue(int(fdef["order"]))
        if "zero_phase" in fdef: self.chk_zero.setChecked(bool(fdef["zero_phase"]))

        self._stft_cfg     = cfg.get("stft_params", {"nperseg":1024,"hop":256,"window":"hann"})
        self._auto_seg_cfg = cfg.get("auto_segment_defaults", {"length_s":3.0,"overlap_s":0.0,"label": (labels[0] if labels else "")})

        # 4) combobox + labelbar updaten
        self._refresh_labelset_combo()
        self._refresh_location_choices()

    def _refresh_labelset_combo(self):
        """
        Bouw de lijst: [Custom] + ['Lung','Heart'].
        Selecteer 'custom' als die bestaat, anders DEFAULT_LABEL_SET.
        """

        items = []
        if self._custom_labels:
            items.append("Custom")
        items.extend(LABEL_SETS.keys())

        self.labelset_combo.blockSignals(True)
        self.labelset_combo.clear()
        self.labelset_combo.addItems(items)
        # kies default
        if "Custom" in items:
            self.labelset_combo.setCurrentText("Custom")
            self.labelbar.set_labels(labels_list_to_dict(self._custom_labels))  # type: ignore
        else:
            self.labelset_combo.setCurrentText(DEFAULT_LABEL_SET)
            self.labelbar.set_labels(LABEL_SETS[DEFAULT_LABEL_SET])
        self.labelset_combo.blockSignals(False)


    def add_label_to_dataset(self, label: str):
        path = labels_dataset_path()
        d = {"labels": [], "meta_defaults": self.session_meta}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                pass
        if label not in d.get("labels", []):
            d["labels"] = d.get("labels", []) + [label]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)

    def rebuild_label_list(self):
        self.list_labels.clear()
        row = self.list.currentRow()
        if not self.state or row < 0 or row >= len(self.state.segments):
            return
        seg = self.state.segments[row]
        for L in seg.labels:
            item = QtWidgets.QListWidgetItem(self.list_labels); item.setSizeHint(QtCore.QSize(0, 26))
            w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(6, 2, 6, 2); h.setSpacing(8)
            lbl = QtWidgets.QLabel(L); btn = QtWidgets.QToolButton(); btn.setText("×"); btn.setToolTip(f"Verwijder label: {L}"); btn.setFixedSize(22, 22); btn.setStyleSheet("QToolButton { font-weight: bold; }")
            btn.setProperty("label_text", L); btn.clicked.connect(self._on_remove_label_btn)
            h.addWidget(lbl); h.addStretch(1); h.addWidget(btn)
            self.list_labels.addItem(item); self.list_labels.setItemWidget(item, w)

    def _on_remove_label_btn(self):
        if not self.state: return
        sender = self.sender()
        if not isinstance(sender, QtWidgets.QToolButton): return
        L = sender.property("label_text"); row = self.list.currentRow()
        if row < 0 or row >= len(self.state.segments): return
        seg = self.state.segments[row]
        try: seg.labels.remove(L)
        except ValueError: return
        self.rebuild_label_list(); self.refresh_segment_list(); self.list.setCurrentRow(row); self.save_json()

    def reload_labels_json(self):
        self.load_labels_json()

    def save_json(self):
        if not self.state: return
        js = json_sidecar_path(self.files[self.idx]); from .utils import ensure_dir; ensure_dir(js)
        with open(js, "w", encoding="utf-8") as fh:
            json.dump(self.state.to_json(), fh, ensure_ascii=False, indent=2)

    def advance(self, step: int):
        self.player.stop()
        new = self.idx + step
        if new < 0 or new >= len(self.files): return
        self.idx = new; self.load_current(); self._after_navigation_changed()

    # -------- Undo/Redo helpers ----------------------------------------------
    def _push_edit(self, do: Callable[[], None], undo: Callable[[], None]):
        do()
        self._undo_stack.append((do, undo))
        self._redo_stack.clear()
        self._post_edit_refresh()

    def undo(self):
        if not self._undo_stack:
            return
        do, undo = self._undo_stack.pop()
        undo()
        self._redo_stack.append((do, undo))
        self._post_edit_refresh()

    def redo(self):
        if not self._redo_stack:
            return
        do, undo = self._redo_stack.pop()
        do()
        self._undo_stack.append((do, undo))
        self._post_edit_refresh()

    def _post_edit_refresh(self):
        self.refresh_segment_list()
        self._reflect_labelbar()
        self.save_json()
    # -------------------------------------------------------------------------


    # -------- Labelbar / segment helpers -------------------------------------
    def _current_segment_or_none(self) -> Optional[Segment]:
        if not self.state:
            return None
        row = self.list.currentRow()
        if 0 <= row < len(self.state.segments):
            return self.state.segments[row]
        return None

    def _create_segment(self, t_start: float, t_end: float) -> Segment:
        seg = Segment(id=str(uuid.uuid4()), t_start=t_start, t_end=t_end, labels=[])
        if self.state:
            self.state.segments.append(seg)
        return seg

    def _reflect_labelbar(self):
        # update knoppen-checked toestand op basis van huidig segment
        seg = self._current_segment_or_none()
        self.labelbar.reflect_segment(seg.labels if seg else [])

    def _on_labelbar_toggled(self, label: str, checked: bool):
        if not self.state:
            return

        seg = self._current_segment_or_none()
        if seg is None:
            # geen huidig segment -> maak er een uit de huidige selectie
            a, b = self.region.getRegion()
            a = snap_t(a); b = max(a + TIME_SNAP, snap_t(b))  # type: ignore
            seg = self._create_segment(a, b)
            self.list.setCurrentRow(len(self.state.segments) - 1)

        def do_add():
            if label not in seg.labels:
                seg.labels.append(label)

        def undo_add():
            try:
                seg.labels.remove(label)
            except ValueError:
                pass

        def do_remove():
            try:
                seg.labels.remove(label)
            except ValueError:
                pass

        def undo_remove():
            if label not in seg.labels:
                seg.labels.append(label)

        if checked:
            self._push_edit(do_add, undo_add)
        else:
            self._push_edit(do_remove, undo_remove)
    # -------------------------------------------------------------------------


    # -------- Play-per-segment ------------------------------------------------
    def _play_current_segment(self):
        if not self.state or self.y_raw is None:
            return
        seg = self._current_segment_or_none()
        if not seg:
            return
        self.player.stop()
        self.player.play(self.current_signal(), self.sr, float(seg.t_start), float(seg.t_end))
    # -------------------------------------------------------------------------

    # -------- Labelbar / segment helpers -------------------------------------
    def _apply_labelset(self, name: str):
        from .config import LABEL_SETS, labels_list_to_dict

        # Zorg dat custom altijd geladen is
        if name.lower().startswith("custom"):
            if not self._custom_labels:
                # probeer opnieuw labels_dataset.json te laden
                self.load_labels_json()

            if self._custom_labels:
                self.labelbar.set_labels(labels_list_to_dict(self._custom_labels))
            else:
                # fallback: toon iets i.p.v. leeg veld
                self.labelbar.set_labels({"No labels loaded": "Edit labels_dataset.json and reload"})
        else:
            labels = LABEL_SETS.get(name, {})
            self.labelbar.set_labels(labels if labels else {"No labels": "Empty label set"})

        # altijd even de metadata dropdown updaten
        self._refresh_location_choices()
        # knoppen up-to-date met huidig segment
        self._reflect_labelbar()

    def open_folder_dialog(self, first=False):
        self.player.stop()
        dlg = StartDialog(self)
        if not dlg.exec():
            return
        try:
            self.root = dlg.root or ""
            self.session_meta = dlg.get_meta()
            mic = self.session_meta.get("microphone_type", "")
            loc = self.session_meta.get("location", "")
            if mic: self.bump_recents(mic=mic) # type: ignore
            if loc: self.bump_recents(loc=loc) # type: ignore

            self.load_labels_json()
            self.build_file_queue(self.root)
            print(f"[DEBUG] open_folder_dialog: root={self.root} files={len(self.files)}")

            if not self.files:
                QtWidgets.QMessageBox.information(self, "Info", "Geen .wav-bestanden gevonden.")
                return

            self._populate_jump_list()
            self.idx = 0
            self.load_current()
            self._after_navigation_changed()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Fout bij openen", f"{e}")

    def build_file_queue(self, root: str):
        files: list[str] = []
        try:
            root = os.path.normpath(root)
            if not os.path.isdir(root):
                print(f"[DEBUG] build_file_queue: not a directory -> {root}")
                self.files = []
                return

            # 1) .wav direct in root
            for name in sorted(os.listdir(root)):
                p = os.path.join(root, name)
                if os.path.isfile(p) and name.lower().endswith(".wav"):
                    files.append(p)

            # 2) .wav in submappen (recursief)
            for dirpath, dirnames, filenames in os.walk(root):
                # sla het root-niveau over; dat hebben we net gedaan
                if os.path.abspath(dirpath) == os.path.abspath(root):
                    continue
                for f in sorted(filenames):
                    if f.lower().endswith(".wav"):
                        files.append(os.path.join(dirpath, f))

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Map lezen", f"Kon map niet lezen:\n{root}\n\n{e}")
            print(f"[DEBUG] build_file_queue error: {e!r}")
            files = []

        # Log resultaat en bewaar
        self.files = files
        print(f"[DEBUG] build_file_queue: root={root} -> {len(self.files)} wavs")
        if self.files[:3]:
            print("[DEBUG] examples:", *self.files[:3], sep="\n  - ")

    def _safe_read_wav(self, path: str):
        try:
            y, sr = sf.read(path, dtype="float32", always_2d=False)
            if isinstance(y, np.ndarray) and y.ndim == 2:
                y = y.mean(axis=1)
            if y is None or (isinstance(y, np.ndarray) and y.size == 0):
                raise ValueError("Empty audio")
            return y.astype(np.float32, copy=False), int(sr)
        except Exception as e:
            print(f"[WARN] Skip unreadable WAV: {path} -> {e}")
            return None, None

    def load_current(self):
        try:
            print(f"[DEBUG] load_current: idx={self.idx} total={len(self.files)}")
            if not self.files:
                QtWidgets.QMessageBox.information(self, "Info", "Geen .wav-bestanden in de gekozen map.")
                return
            if not (0 <= self.idx < len(self.files)):
                self.idx = 0

            # probeer door totdat een leesbare wav gevonden is
            tried = 0
            y = None; sr = None
            while tried < len(self.files):
                f = self.files[self.idx]
                y, sr = self._safe_read_wav(f)
                if y is not None:
                    break
                self.idx = (self.idx + 1) % len(self.files)
                tried += 1

            if y is None:
                QtWidgets.QMessageBox.warning(self, "Openen mislukt", "Geen enkele .wav kon worden gelezen.")
                return

            # UI + state
            f = self.files[self.idx]
            self.lbl_path.setText(f"{human_relpath(self.root, os.path.dirname(f))}/{os.path.basename(f)}")
            self.y_raw = y
            self._filt_cache = None; self._filt_params = None
            self.sr = int(sr) # type: ignore
            assert self.y_raw is not None
            self.t = np.arange(len(self.y_raw), dtype=float) / self.sr

            dur = float(len(self.y_raw)) / float(self.sr) if len(self.y_raw) else 0.0
            self.time_slider.blockSignals(True); self.time_slider.setRange(0, int(dur*100)); self.time_slider.setValue(0); self.time_slider.blockSignals(False)
            self.lbl_time.setText("0.00 s"); self.playhead.setPos(0.0)

            # JSON
            js_path = json_sidecar_path(f)
            if os.path.isfile(js_path):
                try:
                    with open(js_path, "r", encoding="utf-8") as fh:
                        self.state = FileState.from_json(json.load(fh))
                except Exception:
                    self.state = FileState(file=os.path.basename(f), sr=self.sr, meta=dict(self.session_meta), segments=[])
            else:
                self.state = FileState(file=os.path.basename(f), sr=self.sr, meta=dict(self.session_meta), segments=[])

            if not isinstance(self.state.meta, dict):
                self.state.meta = {}
            for k, v in (self.session_meta or {}).items():
                self.state.meta.setdefault(k, v)

            meta_for_editor = {k: self.state.meta.get(k, "") for k in METADATA_FIELDS}
            self.meta_inline.set_values(meta_for_editor)
            self._refresh_location_choices()
            self._apply_labelset(self.labelset_combo.currentText())

            self.draw_waveform()
            self.update_spectrogram()

            self._blocking = True
            init_len = min(3.0, dur) if dur > 0.0 else 0.0
            self.region.setRegion((0.0, init_len)); self.sel_start.setValue(0.0); self.sel_end.setValue(init_len); self.lbl_sel_delta.setText(f"(Δ {init_len:.2f} s)")
            self._blocking = False

            self.refresh_segment_list()
            self.save_json()
            print(f"[DEBUG] loaded: {f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Fout tijdens laden", f"{e}")


    def draw_waveform(self):
        y = self.current_signal(); x = self.t
        if y is None or x is None: return
        self.curve.setData(x=x, y=y, connect='finite')
        self.p_wave.setLabel("bottom", "Tijd (s)"); self.p_wave.setLabel("left", "Amplitude"); self.p_wave.showGrid(x=True, y=True, alpha=0.2)
        if len(x) > 1:
            xmax = float(x[-1]); self.p_wave.setLimits(xMin=0.0, xMax=xmax); vb = self.p_wave.getViewBox(); vb.setXRange(0.0, xmax, padding=0.02)
        for reg in getattr(self, "overlay_regions", {}).values():
            if reg.scene() is None:
                self.p_wave.addItem(reg)

    def on_overlay_clicked(self, reg: ClickableRegion):
        if self.state is None: return
        try_id = getattr(reg, "seg_id", None)
        if try_id is None: return
        for i, s in enumerate(self.state.segments):
            if s.id == try_id:
                self.list.setCurrentRow(i)
                self._blocking = True
                self.region.setRegion((s.t_start, s.t_end))
                self.sel_start.setValue(s.t_start); self.sel_end.setValue(s.t_end); self.lbl_sel_delta.setText(f"(Δ {(s.t_end - s.t_start):.2f} s)")
                self._blocking = False
                break

    def current_signal(self) -> np.ndarray:
        if self.y_raw is None:
            return self.y_raw  # type: ignore
        if getattr(self, "chk_bp", None) and self.chk_bp.isChecked():
            return self.get_filtered_signal()
        return self.y_raw

    def get_filtered_signal(self) -> np.ndarray:
        if self.y_raw is None:
            return self.y_raw  # type: ignore
        sr = float(self.sr)
        params = (float(self.sp_low.value()), float(self.sp_high.value()), int(self.sp_order.value()), bool(self.chk_zero.isChecked()), len(self.y_raw), sr)
        if self._filt_cache is not None and self._filt_params == params:
            return self._filt_cache  # type: ignore
        try:
            y_f = bandpass_filter(self.y_raw, fs=sr, fc=(params[0], params[1]), order=params[2], zero_phase=params[3], axis=-1)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Bandpass-filter", f"Kon filter niet toepassen:\n{e}\nVal terug op raw.")
            self.chk_bp.setChecked(False)
            self._filt_cache = None; self._filt_params = None
            return self.y_raw
        self._filt_cache = y_f.astype(np.float32, copy=False); self._filt_params = params
        return self._filt_cache

    def _current_label_options(self) -> list[str]:
        # Huidige bron van de LabelBar
        name = self.labelset_combo.currentText()
        if name == "Custom" and self._custom_labels:
            return list(self._custom_labels)
        from .config import LABEL_SETS
        return list(LABEL_SETS.get(name, {}).keys())


    def auto_segment_dialog(self):
        if self.t is None or len(self.t) == 0:
            QtWidgets.QMessageBox.information(self, "Auto segmenteren", "Geen audio geladen."); return
        label_options = self._current_label_options()
        default_label = getattr(self, "_auto_seg_cfg", {}).get("label", (label_options[0] if label_options else None))
        default_len   = float(getattr(self, "_auto_seg_cfg", {}).get("length_s", 3.0))
        default_ovl   = float(getattr(self, "_auto_seg_cfg", {}).get("overlap_s", 0.0))
        dlg = AutoSegmentDialog(self, default_len=default_len, default_overlap=default_ovl, default_replace=False, label_options=label_options, default_label=default_label)
        if not dlg.exec():
            return
        seg_len, seg_ovl, replace, auto_label = dlg.values()
        self.apply_auto_segments(seg_len, seg_ovl, replace, auto_label=auto_label)

    def apply_auto_segments(self, seg_len: float, seg_ovl: float, replace: bool, auto_label: Optional[str] = None):
        if self.state is None or self.t is None or len(self.t) == 0:
            return
        dur = float(self.t[-1]); snap = float(TIME_SNAP)
        len_ticks = max(1, int(round(seg_len / snap)))
        ovl_ticks = max(0, int(round(seg_ovl / snap)))
        if ovl_ticks >= len_ticks:
            QtWidgets.QMessageBox.warning(self, "Ongeldige parameters", "Zorg dat: 0 ≤ overlap < lengte."); return
        stride_ticks = len_ticks - ovl_ticks; total_ticks  = int(round(dur / snap))
        new_segments: List[Segment] = []
        start_tick = 0
        while start_tick < total_ticks:
            end_tick = min(start_tick + len_ticks, total_ticks)
            a = round(start_tick * snap, 2); b = round(end_tick * snap, 2)
            labels = [auto_label] if auto_label else []
            new_segments.append(Segment(id=str(uuid.uuid4()), t_start=a, t_end=b, labels=labels))
            start_tick += stride_ticks
        if replace:
            self.state.segments = new_segments
        else:
            self.state.segments.extend(new_segments)
        self.refresh_segment_list(); self.save_json()
        if self.state.segments:
            idx = len(self.state.segments) - len(new_segments)
            self.list.setCurrentRow(max(0, idx))

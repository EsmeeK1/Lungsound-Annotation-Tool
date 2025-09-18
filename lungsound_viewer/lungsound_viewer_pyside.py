# lungsound_viewer_pyside.py
# pip install PySide6 pyqtgraph numpy pandas soundfile scipy sounddevice
from __future__ import annotations

import json, sys, os, uuid, datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import stft, butter, filtfilt

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# -------------------------
# Config & constants
# -------------------------
EXPECTED_DURATION_S = 30.0
DEFAULT_SR = 16000
TIME_SNAP = 0.01

# Default SUK filter (kept constant unless user changes UI controls)
DEFAULT_FILTER = dict(lowcut=120, highcut=1800, order=12)

LABEL_PALETTE = [
    ("Inademing", (80, 180, 255, 60)),
    ("Uitademing", (255, 200, 80, 60)),
    ("Piep (Wheeze)", (255, 120, 120, 60)),
    ("Knetter (Crackle)", (120, 255, 160, 60)),
    ("Rhonchi", (200, 160, 255, 60)),
    ("Stridor", (255, 140, 220, 60)),
    ("Pleura-wrijving", (160, 200, 200, 60)),
    ("Hoest", (255, 170, 120, 60)),
    ("Artefact", (180, 180, 180, 60)),
    ("Normaal", (140, 220, 140, 60)),
]

pg.setConfigOptions(imageAxisOrder='row-major')

# -------------------------
# Data models
# -------------------------
@dataclass
class Segment:
    id: str
    t_start: float
    t_end: float
    labels: List[str] = field(default_factory=list)

@dataclass
class FileState:
    file: str
    sr: int
    meta: Dict[str, object] = field(default_factory=dict)
    segments: List[Segment] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "file": self.file,
            "sr": self.sr,
            "meta": self.meta,
            "segments": [asdict(s) for s in self.segments],
        }

    @staticmethod
    def from_json(d: dict) -> "FileState":
        fs = FileState(file=d.get("file",""), sr=int(d.get("sr", DEFAULT_SR)),
                       meta=d.get("meta", {}), segments=[])
        for s in d.get("segments", []):
            fs.segments.append(Segment(**s))
        return fs

# -------------------------
# Helpers
# -------------------------
def snap_t(x: float) -> float:
    return round(float(x) / TIME_SNAP) * TIME_SNAP

def human_relpath(root: str, path: str) -> str:
    try: rel = os.path.relpath(path, root)
    except ValueError: rel = os.path.basename(path)
    return rel.replace("\\", "/")

def json_sidecar_path(wav_path: str) -> str:
    base, _ = os.path.splitext(wav_path)
    return base + ".json"

def csv_path_for_root(root: str) -> str:
    return os.path.join(root, "labels_export.csv")

def labels_dataset_path(root: str) -> str:
    return os.path.join(root, "labels_dataset.json")

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def apply_suk_filter(audio, sr, lowcut=120, highcut=1800, order=12):
    """Zero-phase Butterworth band-pass as provided by the SUK team."""
    nyq = 0.5 * sr
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")  # type: ignore
    return filtfilt(b, a, audio)

# -------------------------
# Optional audio playback (sounddevice)
# -------------------------
try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:
    HAVE_SD = False

class Player(QtCore.QObject):
    """Small helper for audio playback from a time window."""
    started = QtCore.Signal(float, float)  # emits (t0, t1)
    stopped = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self._stream = None
        self.playing = False

    def play(self, y: np.ndarray, sr: int, t0: float, t1: float):
        """Play y[t0:t1] once."""
        if not HAVE_SD:
            return
        self.stop()
        start = int(max(0, t0) * sr)
        end = int(min(len(y), t1 * sr))
        data = y[start:end].astype(np.float32)
        pos = 0

        def _cb(outdata, frames, timeinfo, status):
            nonlocal pos
            n = min(frames, len(data) - pos)
            if n > 0:
                outdata[:n, 0] = data[pos:pos+n]
            if frames > n:
                outdata[n:, 0] = 0
            pos += n
            if pos >= len(data):
                raise sd.CallbackStop

        self._stream = sd.OutputStream(channels=1, samplerate=sr, dtype="float32", callback=_cb)
        self._stream.start()
        self.playing = True
        self.started.emit(t0, t1)

    def stop(self):
        if not HAVE_SD:
            return
        if self._stream is not None:
            try: self._stream.stop()
            except Exception: pass
            try: self._stream.close()
            except Exception: pass
            self._stream = None
        if self.playing:
            self.playing = False
            self.stopped.emit()

# -------------------------
# Start dialog
# -------------------------
class StartDialog(QtWidgets.QDialog):
    """Start dialog to capture session metadata and pick a root folder."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start – Lung Sound Annotator")
        self.setModal(True)
        self.resize(520, 280)
        v = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.gender = QtWidgets.QComboBox()
        self.gender.addItems(["", "Vrouw", "Man", "Overig", "Onbekend"])
        self.age = QtWidgets.QSpinBox()
        self.age.setRange(0, 120); self.age.setSpecialValueText("")
        self.location = QtWidgets.QComboBox()
        self.location.addItems([""] + [f"Opnamelocatie {i}" for i in range(1, 12+1)])
        form.addRow("Geslacht (optioneel):", self.gender)
        form.addRow("Leeftijd (optioneel):", self.age)
        form.addRow("Opnamelocatie:", self.location)
        v.addLayout(form)

        self.btn_pick = QtWidgets.QPushButton("Kies map met .wav bestanden…")
        v.addWidget(self.btn_pick)

        self.lbl_folder = QtWidgets.QLabel("Geen map gekozen")
        self.lbl_folder.setStyleSheet("color:#777;")
        v.addWidget(self.lbl_folder)

        h = QtWidgets.QHBoxLayout()
        self.btn_cancel = QtWidgets.QPushButton("Annuleren")
        self.btn_ok = QtWidgets.QPushButton("Start")
        self.btn_ok.setEnabled(False)
        h.addStretch(1); h.addWidget(self.btn_cancel); h.addWidget(self.btn_ok)
        v.addLayout(h)

        self.btn_pick.clicked.connect(self.pick_folder)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)

        self.root = None

    def pick_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Selecteer root-map")
        if not d: return
        self.root = d
        self.lbl_folder.setText(d)
        self.btn_ok.setEnabled(True)

    def get_meta(self) -> Dict[str, object]:
        meta = {}
        g = self.gender.currentText().strip()
        a = self.age.value()
        loc = self.location.currentText().strip()
        if g: meta["gender"] = g
        if a > 0: meta["age"] = int(a)
        if loc: meta["location"] = loc
        return meta

# -------------------------
# Main window
# -------------------------
class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung Sound Viewer (PySide6 + PyQtGraph)")
        self.resize(1500, 950)

        # Menu: reopen a new folder (re-shows start dialog)
        m = self.menuBar().addMenu("Bestand")
        act_open = m.addAction("Map openen…")
        act_open.triggered.connect(self.open_folder_dialog)

        # Runtime state
        self.player = Player()
        self.root = ""
        self.files: List[str] = []
        self.idx = -1

        self.y_raw: Optional[np.ndarray] = None   # always keep the raw signal
        self.y_filt: Optional[np.ndarray] = None  # cached filtered signal
        self.use_filter = False
        self.filter_params = DEFAULT_FILTER.copy()

        self.sr = DEFAULT_SR
        self.t: Optional[np.ndarray] = None
        self.state: Optional[FileState] = None
        self.overlay_regions: Dict[str, pg.LinearRegionItem] = {}
        self._blocking = False

        # Central UI
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        H = QtWidgets.QHBoxLayout(cw)

        # ----- Left: plots -----
        left = QtWidgets.QWidget(); H.addWidget(left, 3)
        gl = QtWidgets.QGridLayout(left)

        # Waveform (filled) – NOT pink; playhead line is red.
        self.p_wave = pg.PlotWidget()
        self.p_wave.setLabel("bottom", "Time (s)")
        self.p_wave.setLabel("left", "Amplitude")
        self.p_wave.showGrid(x=True, y=True, alpha=0.2)
        self.playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#CC3333', width=1))
        self.region = pg.LinearRegionItem([2.0, 3.0], brush=(100,180,255,60), movable=True)
        self.p_wave.addItem(self.playhead); self.p_wave.addItem(self.region)
        gl.addWidget(self.p_wave, 0, 0)

        # Time slider + time label (free playback position)
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, int(EXPECTED_DURATION_S*100))
        self.lbl_time = QtWidgets.QLabel("0.00 s")
        tbar = QtWidgets.QHBoxLayout()
        tbar.addWidget(QtWidgets.QLabel("Tijd:")); tbar.addWidget(self.time_slider, 1); tbar.addWidget(self.lbl_time)
        gl.addLayout(tbar, 1, 0)

        # Spectrogram (interactive via x-link with waveform)
        self.p_spec = pg.PlotWidget()
        self.p_spec.setLabel("bottom", "Time (s)")
        self.p_spec.setLabel("left", "Frequency (Hz)")
        self.p_spec.setMouseEnabled(x=True, y=True)     # interactive
        self.p_spec.setXLink(self.p_wave)               # link x-axes
        self.img_spec = pg.ImageItem(axisOrder='row-major')
        self.p_spec.addItem(self.img_spec)
        gl.addWidget(self.p_spec, 2, 0)

        # Colorbar (if available in your pyqtgraph version)
        self.colorbar = None
        try:
            self.colorbar = pg.ColorBarItem(values=(-100, 0), colorMap=pg.colormap.get('inferno'))
            self.colorbar.setImageItem(self.img_spec, insertIn=self.p_spec.getPlotItem())
        except Exception:
            pass  # older pyqtgraph: skip

        # ----- Right panel -----
        right = QtWidgets.QWidget(); H.addWidget(right, 1)
        rv = QtWidgets.QVBoxLayout(right)

        self.lbl_path = QtWidgets.QLabel("—"); self.lbl_path.setStyleSheet("font-weight:600;")
        rv.addWidget(self.lbl_path)
        nav = QtWidgets.QHBoxLayout(); self.btn_prev = QtWidgets.QPushButton("◀ Prev"); self.btn_next = QtWidgets.QPushButton("Next ▶")
        nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next); rv.addLayout(nav)

        self.lbl_sel = QtWidgets.QLabel("Selected: –"); rv.addWidget(self.lbl_sel)

        # Filter controls
        grp_f = QtWidgets.QGroupBox("SUK-filter")
        f_l = QtWidgets.QFormLayout(grp_f)
        self.chk_filter = QtWidgets.QCheckBox("Filter aan/uit")
        self.spin_low = QtWidgets.QSpinBox(); self.spin_low.setRange(10, 5000); self.spin_low.setValue(DEFAULT_FILTER["lowcut"])
        self.spin_high = QtWidgets.QSpinBox(); self.spin_high.setRange(100, 8000); self.spin_high.setValue(DEFAULT_FILTER["highcut"])
        self.spin_order = QtWidgets.QSpinBox(); self.spin_order.setRange(2, 20); self.spin_order.setValue(DEFAULT_FILTER["order"])
        self.btn_apply_filter = QtWidgets.QPushButton("Toepassen")
        f_l.addRow(self.chk_filter); f_l.addRow("Lowcut (Hz):", self.spin_low); f_l.addRow("Highcut (Hz):", self.spin_high); f_l.addRow("Orde:", self.spin_order); f_l.addRow(self.btn_apply_filter)
        rv.addWidget(grp_f)

        # Labels management
        box = QtWidgets.QGroupBox("Labels"); vb = QtWidgets.QVBoxLayout(box)
        self.combo_labels = QtWidgets.QComboBox(); self.combo_labels.addItems([name for name,_ in LABEL_PALETTE])
        self.btn_reload_labels = QtWidgets.QPushButton("Herlaad labels.json")
        vb.addWidget(self.combo_labels); vb.addWidget(self.btn_reload_labels)
        hb = QtWidgets.QHBoxLayout(); self.txt_new_label = QtWidgets.QLineEdit(); self.txt_new_label.setPlaceholderText("Nieuw label…")
        self.btn_add_label = QtWidgets.QPushButton("Voeg label toe aan selectie")
        hb.addWidget(self.txt_new_label); vb.addLayout(hb); vb.addWidget(self.btn_add_label)
        rv.addWidget(box)

        # Segments list + editor (remove via selection)
        rv.addWidget(QtWidgets.QLabel("Segments"))
        self.list = QtWidgets.QListWidget(); rv.addWidget(self.list, 1)
        edit = QtWidgets.QGroupBox("Segment bewerken"); fe = QtWidgets.QFormLayout(edit)
        self.spin_start = QtWidgets.QDoubleSpinBox(); self.spin_start.setDecimals(2); self.spin_start.setSingleStep(TIME_SNAP); self.spin_start.setRange(0, 1e6)
        self.spin_end   = QtWidgets.QDoubleSpinBox(); self.spin_end.setDecimals(2); self.spin_end.setSingleStep(TIME_SNAP); self.spin_end.setRange(0, 1e6)
        self.list_labels = QtWidgets.QListWidget(); self.btn_remove_label = QtWidgets.QPushButton("Verwijder geselecteerd label")
        fe.addRow("Start (s):", self.spin_start); fe.addRow("End (s):", self.spin_end); fe.addRow("Labels:", self.list_labels); fe.addRow("", self.btn_remove_label)
        rv.addWidget(edit)
        hb2 = QtWidgets.QHBoxLayout(); self.btn_update = QtWidgets.QPushButton("Update"); self.btn_delete = QtWidgets.QPushButton("Delete")
        hb2.addWidget(self.btn_update); hb2.addWidget(self.btn_delete); rv.addLayout(hb2)

        # Export
        self.btn_export_csv = QtWidgets.QPushButton("Export CSV"); rv.addWidget(self.btn_export_csv)

        # Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, self.toggle_play)
        QtGui.QShortcut(QtGui.QKeySequence("N"), self, lambda: self.advance(+1))
        QtGui.QShortcut(QtGui.QKeySequence("P"), self, lambda: self.advance(-1))
        QtGui.QShortcut(QtGui.QKeySequence("Delete"), self, self.delete_selected)

        # Signals
        self.btn_prev.clicked.connect(lambda: self.advance(-1))
        self.btn_next.clicked.connect(lambda: self.advance(+1))
        self.btn_add_label.clicked.connect(self.add_label_to_selection)
        self.txt_new_label.returnPressed.connect(self.add_label_to_selection)
        self.btn_remove_label.clicked.connect(self.remove_selected_label)
        self.region.sigRegionChanged.connect(self.on_region_changed)
        self.list.currentRowChanged.connect(self.on_list_selection)
        self.btn_update.clicked.connect(self.update_segment)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_export_csv.clicked.connect(self.export_csv)

        self.chk_filter.toggled.connect(self.on_filter_toggle)
        self.btn_apply_filter.clicked.connect(self.on_apply_filter)
        self.time_slider.valueChanged.connect(self.on_slider_changed)
        self.btn_reload_labels.clicked.connect(self.reload_labels_json)

        self.player.started.connect(self.on_play_started)
        self.player.stopped.connect(self.on_play_stopped)

        # Timer to animate playhead during playback
        self.timer = QtCore.QTimer(self); self.timer.setInterval(30); self.timer.timeout.connect(self.tick_playhead)

        # Session meta stored from start dialog
        self.session_meta: Dict[str, object] = {}
        self.play_window: Tuple[float, float] = (0.0, 0.0)

        # Kick off
        self.open_folder_dialog(first=True)

    # --------- Folder flow ----------
    def open_folder_dialog(self, first=False):
        """Show the start dialog and (re)build the file queue."""
        dlg = StartDialog(self)
        if not dlg.exec():
            if first: sys.exit(0)
            return
        self.root = dlg.root or ""
        self.session_meta = dlg.get_meta()
        self.load_labels_json()
        self.build_file_queue(self.root)
        if not self.files:
            QtWidgets.QMessageBox.information(self, "Info", "Geen .wav-bestanden gevonden.")
            if first: sys.exit(0)
            return
        self.idx = 0
        self.load_current()

    def build_file_queue(self, root: str):
        """Collect .wav files: top-level first, then subfolders."""
        files = []
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isfile(p) and p.lower().endswith(".wav"):
                files.append(p)
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.abspath(dirpath) == os.path.abspath(root):
                continue
            for f in sorted(filenames):
                if f.lower().endswith(".wav"):
                    files.append(os.path.join(dirpath, f))
        self.files = files

    # --------- Load & render ----------
    def load_current(self):
        """Load current file, render waveform and spectrogram, reset UI."""
        f = self.files[self.idx]
        self.lbl_path.setText(f"{human_relpath(self.root, os.path.dirname(f))}/{os.path.basename(f)}")

        y, sr = sf.read(f, dtype="float32", always_2d=False)
        if y.ndim == 2: y = y.mean(axis=1)
        self.y_raw = y.astype(np.float32)
        self.y_filt = None
        self.sr = int(sr)
        self.t = np.arange(len(self.y_raw), dtype=float) / self.sr

        dur = len(self.y_raw)/self.sr
        self.time_slider.blockSignals(True); self.time_slider.setRange(0, int(dur*100)); self.time_slider.setValue(0); self.time_slider.blockSignals(False)
        self.lbl_time.setText("0.00 s"); self.playhead.setPos(0.0)

        # Load or create sidecar JSON
        js_path = json_sidecar_path(f)
        if os.path.isfile(js_path):
            with open(js_path, "r", encoding="utf-8") as fh:
                self.state = FileState.from_json(json.load(fh))
        else:
            self.state = FileState(file=os.path.basename(f), sr=self.sr, meta=dict(self.session_meta), segments=[])

        # Render
        self.draw_waveform()
        self.draw_spectrogram()

        # Selection defaults
        self.region.blockSignals(True); self.region.setRegion([2.0, 3.0]); self.region.blockSignals(False)
        self.lbl_sel.setText("Selected: 2.00–3.00 s (Δ 1.00 s)")

        self.refresh_segment_list()
        self.save_json()

    # --------- Signal selection ----------
    def current_signal(self) -> np.ndarray:
        """Return raw or SUK-filtered signal depending on toggle; cache filtered output."""
        if not getattr(self, "use_filter", False):
            return self.y_raw
        if getattr(self, "y_filt", None) is None:
            p = getattr(self, "filter_params", {"lowcut": 120, "highcut": 1800, "order": 12})
            y = np.nan_to_num(self.y_raw, nan=0.0, posinf=0.0, neginf=0.0)
            self.y_filt = np.nan_to_num(apply_suk_filter(y, self.sr, **p)).astype(np.float32)
        return self.y_filt


    # --------- Drawing ----------
    def draw_waveform(self):
        y = self.current_signal()
        x = self.t
        self.p_wave.clear()
        self.p_wave.setLabel("bottom", "Time (s)")
        self.p_wave.setLabel("left", "Amplitude")
        self.p_wave.showGrid(x=True, y=True, alpha=0.2)

        curve = pg.PlotCurveItem(
            x, y,
            pen=pg.mkPen(120, width=1),
            fillLevel=0.0,
            brush=pg.mkBrush(100, 180, 255, 60),
        )
        self.p_wave.addItem(curve)
        self.p_wave.addItem(self.playhead)
        self.p_wave.addItem(self.region)
        for reg in getattr(self, "overlay_regions", {}).values():
            self.p_wave.addItem(reg)

    def draw_spectrogram(self):
        y = self.current_signal()
        if y is None or len(y) == 0:
            return

        # STFT: nperseg=1024, hop=256, padding tot eind zodat de tijd-as doorloopt
        f, t, Z = stft(
            y, fs=self.sr,
            window="hann",
            nperseg=1024,
            noverlap=1024-256,
            boundary="zeros",
            padded=True,
        )
        S = 20 * np.log10(np.maximum(1e-10, np.abs(Z)))
        img = S[::-1, :]  # 0 Hz onderaan

        # Kleur & levels
        lut = pg.colormap.get("inferno").getLookupTable(256)
        self.img_spec.setLookupTable(lut)
        self.img_spec.setLevels((-100, 0))
        self.img_spec.setImage(img, autoLevels=False)  # niet laten autoscalen

        # Pixels -> (tijd, frequentie)
        if len(t) > 1 and len(f) > 1:
            self.img_spec.setRect(QtCore.QRectF(0, 0, float(t[-1]), float(f[-1])))
            self.p_spec.setLimits(xMin=0, xMax=float(t[-1]), yMin=0, yMax=float(f[-1]))
            self.p_spec.setXRange(0, float(t[-1]))
            self.p_spec.setYRange(0, float(f[-1]))

    # --------- Interactions ----------
    def on_region_changed(self):
        if self._blocking: return
        a, b = self.region.getRegion()
        a = max(0.0, snap_t(a)); b = max(a + TIME_SNAP, snap_t(b))
        self._blocking = True; self.region.setRegion((a,b)); self._blocking = False
        self.lbl_sel.setText(f"Selected: {a:.2f}–{b:.2f} s (Δ {(b-a):.2f} s)")

    def on_slider_changed(self, val: int):
        """Slider is the free playback position; does not follow the selection."""
        t = val/100.0
        self.playhead.setPos(t)
        self.lbl_time.setText(f"{t:.2f} s")

    # --------- Playback (free: from slider to end) ----------
    def toggle_play(self):
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
        self._elapsed = QtCore.QElapsedTimer(); self._elapsed.start()
        self.timer.start()

    def on_play_stopped(self):
        self.timer.stop()

    def tick_playhead(self):
        a, b = self.play_window
        t_now = a + self._elapsed.elapsed()/1000.0
        if t_now >= b:
            self.player.stop()
            t_now = b
        self.playhead.setPos(t_now)
        self.time_slider.blockSignals(True); self.time_slider.setValue(int(t_now*100)); self.time_slider.blockSignals(False)
        self.lbl_time.setText(f"{t_now:.2f} s")

    # --------- Segments CRUD ----------
    def _brush_for_labels(self, labels: List[str]):
        for L, color in LABEL_PALETTE:
            if L in labels:
                return color
        return (120,120,120,40)

    def refresh_segment_list(self):
        self.list.clear()
        if not self.state: return
        # remove old overlays
        for reg in self.overlay_regions.values():
            try: self.p_wave.removeItem(reg)
            except Exception: pass
        self.overlay_regions.clear()
        # repopulate
        for s in self.state.segments:
            self.list.addItem(f"{s.t_start:.2f}-{s.t_end:.2f}s | {'; '.join(s.labels) or '(geen labels)'}")
            reg = pg.LinearRegionItem([s.t_start, s.t_end], brush=self._brush_for_labels(s.labels), movable=False)
            self.p_wave.addItem(reg)
            self.overlay_regions[s.id] = reg

    def _find_segment_by_bounds(self, a: float, b: float) -> Optional[Segment]:
        if not self.state: return None
        for s in self.state.segments:
            if abs(s.t_start - a) < 1e-6 and abs(s.t_end - b) < 1e-6:
                return s
        return None

    def add_label_to_selection(self):
        if not self.state: return
        entered = self.txt_new_label.text().strip()
        if entered:
            if self.combo_labels.findText(entered) < 0:
                self.combo_labels.addItem(entered)
                self.add_label_to_dataset(entered)
            label = entered
            self.txt_new_label.clear()
        else:
            label = self.combo_labels.currentText()

        a, b = self.region.getRegion()
        a = snap_t(a); b = max(a+TIME_SNAP, snap_t(b))

        seg = self._find_segment_by_bounds(a, b)
        if seg is None:
            seg = Segment(id=str(uuid.uuid4()), t_start=a, t_end=b, labels=[])
            self.state.segments.append(seg)
        if label not in seg.labels:
            seg.labels.append(label)

        self.refresh_segment_list()
        self.list.setCurrentRow(self.state.segments.index(seg))
        self.save_json()

    def on_list_selection(self, row: int):
        if not self.state or row < 0 or row >= len(self.state.segments):
            self.list_labels.clear(); return
        s = self.state.segments[row]
        self.spin_start.setValue(s.t_start)
        self.spin_end.setValue(s.t_end)
        self.list_labels.clear()
        for L in s.labels:
            self.list_labels.addItem(L)
        self._blocking = True; self.region.setRegion((s.t_start, s.t_end)); self._blocking = False

    def remove_selected_label(self):
        row = self.list.currentRow()
        if not self.state or row < 0: return
        s = self.state.segments[row]
        lab_row = self.list_labels.currentRow()
        if lab_row < 0 or lab_row >= len(s.labels): return
        del s.labels[lab_row]
        self.on_list_selection(row)
        self.refresh_segment_list()
        self.save_json()

    def update_segment(self):
        row = self.list.currentRow()
        if not self.state or row < 0: return
        s = self.state.segments[row]
        new_a = snap_t(self.spin_start.value())
        new_b = max(new_a + TIME_SNAP, snap_t(self.spin_end.value()))
        new_labels = [self.list_labels.item(i).text() for i in range(self.list_labels.count())]
        s.t_start, s.t_end, s.labels = new_a, new_b, new_labels
        self.refresh_segment_list()
        self.list.setCurrentRow(row)
        self.save_json()

    def delete_selected(self):
        row = self.list.currentRow()
        if not self.state or row < 0: return
        ans = QtWidgets.QMessageBox.question(self, "Delete segment", "Verwijder geselecteerd segment?")
        if ans != QtWidgets.QMessageBox.Yes: return
        del self.state.segments[row]
        self.refresh_segment_list()
        self.save_json()

    # --------- Export ----------
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
                    "filename": human_relpath(self.root, fp),
                    "t_start": s.t_start,
                    "t_end": s.t_end,
                    "label": ";".join(s.labels),
                    "gender": st.meta.get("gender", ""),
                    "age": st.meta.get("age", ""),
                    "location": st.meta.get("location", ""),
                })
        if not rows:
            QtWidgets.QMessageBox.information(self, "Export", "Geen segmenten om te exporteren.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", csv_path_for_root(self.root), "CSV (*.csv)")
        if not path: return
        pd.DataFrame(rows).to_csv(path, index=False)
        QtWidgets.QMessageBox.information(self, "Export", f"Opgeslagen {len(rows)} rijen naar:\n{path}")

    # --------- Filter handling ----------
    def on_filter_toggle(self, checked: bool):
        self.use_filter = bool(checked)
        if not self.use_filter:
            self.y_filt = None
        self.draw_waveform()
        self.draw_spectrogram()

    def on_apply_filter(self):
        self.filter_params = dict(
            lowcut=int(self.spin_low.value()),
            highcut=int(self.spin_high.value()),
            order=int(self.spin_order.value()),
        )
        self.y_filt = None
        if self.use_filter:
            self.draw_waveform()
            self.draw_spectrogram()

    # --------- Labels dataset JSON ----------
    def load_labels_json(self):
        path = labels_dataset_path(self.root)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                labels = d.get("labels", [])
                if labels:
                    self.combo_labels.clear()
                    self.combo_labels.addItems(labels)
                defaults = d.get("meta_defaults", {})
                for k, v in defaults.items():
                    self.session_meta.setdefault(k, v)
            except Exception:
                pass
        else:
            d = {"labels": [name for name,_ in LABEL_PALETTE], "meta_defaults": self.session_meta}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)

    def add_label_to_dataset(self, label: str):
        path = labels_dataset_path(self.root)
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

    def reload_labels_json(self):
        self.load_labels_json()

    # --------- Persistence ----------
    def save_json(self):
        if not self.state: return
        js = json_sidecar_path(self.files[self.idx])
        ensure_dir(js)
        with open(js, "w", encoding="utf-8") as fh:
            json.dump(self.state.to_json(), fh, ensure_ascii=False, indent=2)

    # --------- Navigation ----------
    def advance(self, step: int):
        self.player.stop()
        new = self.idx + step
        if new < 0 or new >= len(self.files): return
        self.idx = new
        self.load_current()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())

# lungsound_viewer_pyside.py
# pip install PySide6 pyqtgraph numpy pandas soundfile
from __future__ import annotations

import json, sys, os, uuid, datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from scipy.signal import butter, sosfiltfilt, sosfilt
from pathlib import Path

# Map van dit script (viewer.py)
APP_DIR = Path(__file__).resolve().parent
# Vaste locatie voor labels_dataset.json
LABELS_JSON_PATH = APP_DIR / "labels_dataset.json"


# -------------------------
# Config & constants
# -------------------------
DEFAULT_SR = 16000
TIME_SNAP = 0.01

# ---------- Label-kleuren (deterministisch, bibliotheekpalet) ----------
try:
    from matplotlib import cm, colors as mpl_colors
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

def _to_qcolor_tuple(c, alpha=60):
    # accepteer matplotlib RGBA of hex; retourneer (r,g,b,a) 0-255
    if isinstance(c, str) and c.startswith("#") and len(c) in (7, 9):
        q = QtGui.QColor(c)
        return (q.red(), q.green(), q.blue(), alpha)
    if isinstance(c, (list, tuple)) and len(c) in (3, 4):
        r, g, b = c[:3]
        if any(v <= 1.0 for v in (r, g, b)):  # mpl 0..1
            r, g, b = int(r*255), int(g*255), int(b*255)
        return (r, g, b, alpha)
    return (120, 120, 120, alpha)

def _qualitative_palette(n: int):
    """
    Geef n kleuren uit een vast palet.
    - Eerst probeer 'tab20' (stabiel, 20 duidelijke categorieën).
    - Zo niet, val terug op een handgemaakte reeks (Set3-achtig).
    """
    if _HAVE_MPL:
        cmap = cm.get_cmap('tab20', max(2, n))
        return [_to_qcolor_tuple(cmap(i)) for i in range(n)]
    # Fallback zonder matplotlib (hexes van Set3-achtig palet)
    base = [
        "#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3",
        "#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd",
        "#ccebc5","#ffed6f","#1b9e77","#d95f02","#7570b3",
        "#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
    ]
    if n <= len(base):
        return [_to_qcolor_tuple(h) for h in base[:n]]
    # repeat (deterministisch)
    return [_to_qcolor_tuple(base[i % len(base)]) for i in range(n)]

class LabelColorMap:
    """Houdt een stabiele mapping label -> kleur bij, o.b.v. labels_dataset.json"""
    def __init__(self):
        self.labels: list[str] = []
        self.colors: list[tuple[int,int,int,int]] = []
        self.map: dict[str, tuple[int,int,int,int]] = {}

    def build(self, labels: list[str]):
        self.labels = list(labels)
        self.colors = _qualitative_palette(len(self.labels))
        self.map = {lab: col for lab, col in zip(self.labels, self.colors)}

    def color_for(self, labels_in_segment: list[str]):
        # Kies de eerste bekende labelkleur; anders grijs
        for L in labels_in_segment:
            if L in self.map:
                return self.map[L]
        return (120,120,120,40)

LABEL_COLORS = LabelColorMap()

# Sneller tekenen: geen antialias (lijnen), en laat OpenGL uit (kan wisselend presteren)
pg.setConfigOptions(imageAxisOrder='row-major', antialias=False, useOpenGL=False)

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

def labels_dataset_path() -> str:
    """Altijd het centrale JSON-bestand naast viewer.py."""
    return str(LABELS_JSON_PATH)

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def bandpass_filter(x: np.ndarray, fs: float, fc=(50.0, 2000.0), order=2, zero_phase=True, axis=-1) -> np.ndarray:
    """
    Butterworth band-pass filter. Zero-phase (sosfiltfilt) standaard.
    """
    x = np.asarray(x)
    low, high = float(fc[0]), float(fc[1])
    nyq = fs / 2.0
    # Veiligheidsclamp: zorg dat 0 < low < high < Nyquist
    high = min(high, nyq - 1e-6)
    if not (0.0 < low < high < nyq):
        raise ValueError(f"Cutoffs ongeldig: 0 < {low=} < {high=} < {nyq=}")
    wn = (low/nyq, high/nyq)
    sos = butter(int(order), wn, btype='band', output='sos')
    if zero_phase:
        return sosfiltfilt(sos, x, axis=axis)
    else:
        return sosfilt(sos, x, axis=axis) # type: ignore


# -------------------------
# Debug helpers
# -------------------------
DEBUG_STFT = False                 # zet op False om prints uit te zetten
DYNAMIC_SPECTRO_LEVELS = False    # True = levels uit percentielen (handig bij debug)
GRAYSCALE_DEBUG = True            # tijdelijk op True zetten om LUT/levels te omzeilen

def _dbg(msg: str):
    if DEBUG_STFT:
        print(msg); sys.stdout.flush()

def _arr_stats(name: str, a: np.ndarray):
    if not DEBUG_STFT: return
    a = np.asarray(a)
    total = a.size
    finite = np.isfinite(a)
    n_finite = int(finite.sum())
    if n_finite == 0:
        _dbg(f"{name}: shape={a.shape} dtype={a.dtype} NO FINITE VALUES"); return
    avals = a[finite]
    p = lambda q: float(np.percentile(avals, q))
    _dbg(f"{name}: shape={a.shape} dtype={a.dtype} "
         f"finite={n_finite}/{total} "
         f"min={avals.min():.3f} max={avals.max():.3f} mean={avals.mean():.3f} "
         f"p1={p(1):.3f} p50={p(50):.3f} p99={p(99):.3f}")


def compute_stft_db(
    y: np.ndarray,
    sr: int,
    nperseg: int = 1024,
    hop: int = 256,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STFT magnitude in dB, genormaliseerd zodat max == 0 dB (dus overal <= 0).
    Retourneert f(Hz), t(sec), S_db met shape (F, T).
    """
    y = np.asarray(y, dtype=np.float32)
    N = int(len(y))
    _dbg(f"[STFT] N={N} sr={sr} nperseg={nperseg} hop={hop}")
    if N == 0:
        return (np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros((0, 0), np.float32))

    win = np.hanning(nperseg).astype(np.float32) if window == "hann" else np.ones(nperseg, np.float32)

    n_frames = 1 + int(np.ceil(max(0, N - nperseg) / float(hop)))
    total_len = (n_frames - 1) * hop + nperseg
    if total_len > N:
            y = np.pad(y, (0, total_len - N), mode="constant")

    shape = (n_frames, nperseg)
    strides = (y.strides[0] * hop, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    _arr_stats("[STFT] frames (pre-window)", frames[: min(3, len(frames))])

    frames = frames * win[None, :]
    spec = np.fft.rfft(frames, axis=1)
    mag  = np.abs(spec).astype(np.float32)

    mmax = float(mag.max()) if mag.size else 1.0
    _dbg(f"[STFT] mag.max={mmax:.6e}")
    mag_norm = mag / (mmax + 1e-12)
    mag_norm = np.minimum(mag_norm, 1.0)          # clamp: voorkomt >0 dB

    S_db = (20.0 * np.log10(np.maximum(mag_norm, 1e-12))).astype(np.float32)

    f = np.fft.rfftfreq(nperseg, d=1.0/float(sr)).astype(np.float32)
    t = (np.arange(n_frames, dtype=np.float32) * (hop / float(sr)))
    S_db_FT = S_db.T
    _arr_stats("[STFT] S_db (F,T)", S_db_FT)
    return f, t, S_db_FT

# -------------------------
# Optional audio playback (sounddevice)
# -------------------------
try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:
    HAVE_SD = False

class Player(QtCore.QObject):
    """Kleine helper voor audioweergave vanaf een tijdvenster."""
    started = QtCore.Signal(float, float)  # emits (t0, t1)
    stopped = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self._stream = None
        self.playing = False

    def play(self, y: np.ndarray, sr: int, t0: float, t1: float):
        """Speel y[t0:t1] eenmalig af."""
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
    """Startdialoog om sessiemetadata te kiezen en een root-map te selecteren."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Selecteer dataset en meta")
        self.resize(480, 300)

        v = QtWidgets.QVBoxLayout(self)

        # --- Boven: dataset-map kiezen ---
        form_top = QtWidgets.QFormLayout()
        self.btn_choose = QtWidgets.QPushButton("Kies map…")
        self.le_root    = QtWidgets.QLineEdit()
        self.le_root.setReadOnly(True)
        form_top.addRow("Datasetmap:", self.btn_choose)
        form_top.addRow("Gekozen pad:", self.le_root)
        v.addLayout(form_top)

        # --- Metadata-groep ---
        grp = QtWidgets.QGroupBox()
        grp_layout = QtWidgets.QVBoxLayout(grp)

        # Kopregel in de groupbox met titel + kleine info-knop
        header = QtWidgets.QHBoxLayout()
        lbl_hdr = QtWidgets.QLabel("Metadata (optioneel)")
        lbl_hdr.setStyleSheet("font-weight: 600;")
        self.btn_info = QtWidgets.QToolButton()
        self.btn_info.setText("?")
        self.btn_info.setToolTip("Korte uitleg over de velden")
        self.btn_info.setFixedWidth(24)
        self.btn_info.clicked.connect(self._show_meta_info)
        header.addWidget(lbl_hdr)
        header.addStretch(1)
        header.addWidget(self.btn_info)
        grp_layout.addLayout(header)

        # Form in de groupbox
        form_meta = QtWidgets.QFormLayout()

        # Geslacht
        self.gender = QtWidgets.QComboBox()
        self.gender.addItems(["", "Man", "Vrouw", "Onbekend"])
        self.gender.setToolTip("Niet per se nodig; alleen invullen als je analyses op geslacht wilt doen.")

        # Leeftijd
        self.age = QtWidgets.QSpinBox()
        self.age.setRange(0, 120)
        self.age.setSpecialValueText("")      # toont leeg bij minimum
        self.age.setValue(self.age.minimum()) # start leeg
        self.age.setToolTip("Niet per se nodig; laat leeg als onbekend of niet relevant voor je analyse.")

        # Opnamelocatie
        self.location = QtWidgets.QLineEdit()
        self.location.setToolTip("Aanbevolen bij longgeluid: het type/intensiteit kan per locatie verschillen.")

        form_meta.addRow("Geslacht:", self.gender)
        form_meta.addRow("Leeftijd:", self.age)
        form_meta.addRow("Opnamelocatie:", self.location)

        grp_layout.addLayout(form_meta)
        v.addWidget(grp)

        # --- Knoppen (OK standaard uit tot er een map is gekozen) ---
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        v.addWidget(btns)
        self._btn_ok = btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self._btn_ok.setEnabled(False)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        # Map kiezen
        self.root = ""  # start leeg
        self.btn_choose.clicked.connect(self.pick_folder)

    def _show_meta_info(self):
        txt = (
            "Korte toelichting velden:\n\n"
            "• Geslacht: niet per se nodig; alleen invullen als je analyses wilt doen met deze variabele.\n"
            "• Leeftijd: niet per se nodig; laat leeg als onbekend of niet relevant voor je analyse.\n"
            "• Opnamelocatie: aanbevolen bij longgeluid. Het type en de intensiteit van geluid kunnen per locatie verschillen."
        )
        QtWidgets.QMessageBox.information(self, "Uitleg metadata", txt)

    def pick_folder(self):
        # Directory-dialoog (robust, non-native fallback)
        dlg = QtWidgets.QFileDialog(self, "Selecteer root-map")
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        dlg.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly, True)
        try:
            dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        except Exception:
            pass
        if getattr(self, "root", ""):
            dlg.setDirectory(self.root)

        if dlg.exec():
            sel = dlg.selectedFiles()
            if sel:
                self.root = sel[0]
                self.le_root.setText(self.root)
                self._btn_ok.setEnabled(True)

    def get_meta(self) -> Dict[str, object]:
        meta = {}
        g = self.gender.currentText().strip()
        a = self.age.value()
        loc = self.location.text().strip()
        if g:
            meta["gender"] = g
        if a > 0:
            meta["age"] = int(a)
        if loc:
            meta["location"] = loc
        return meta

class ClickableRegion(pg.LinearRegionItem):
    """LinearRegionItem dat clicks doorgeeft."""
    clicked = QtCore.Signal(object)  # emit self

    def __init__(self, *args, seg_id: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seg_id = seg_id
        self.setMovable(False)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(self)
            ev.accept()
        else:
            ev.ignore()

class AutoSegmentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, default_len=3.00, default_overlap=0.00, default_replace=False, label_options=None, default_label=None):
        super().__init__(parent)
        self.setWindowTitle("Auto segmenteren")
        self.setModal(True)
        v = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.len_s = QtWidgets.QDoubleSpinBox()
        self.len_s.setDecimals(2); self.len_s.setSingleStep(TIME_SNAP)
        self.len_s.setRange(TIME_SNAP, 600.0); self.len_s.setValue(default_len)

        self.ovl_s = QtWidgets.QDoubleSpinBox()
        self.ovl_s.setDecimals(2); self.ovl_s.setSingleStep(TIME_SNAP)
        self.ovl_s.setRange(0.0, 600.0); self.ovl_s.setValue(default_overlap)

        self.chk_replace = QtWidgets.QCheckBox("Vervang bestaande segmenten")
        self.chk_replace.setChecked(default_replace)

        # Label-keuze: altijd op basis van doorgegeven opties (uit labels_dataset.json)
        self.combo_label = QtWidgets.QComboBox()
        if label_options:
            self.combo_label.addItems(label_options)
        if default_label and default_label in (label_options or []):
            self.combo_label.setCurrentText(default_label)

        form.addRow("Segmentlengte (s):", self.len_s)
        form.addRow("Overlap tussen segmenten (s):", self.ovl_s)
        form.addRow("Label voor alle segmenten:", self.combo_label)
        v.addLayout(form)
        v.addWidget(self.chk_replace)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        v.addWidget(btns)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.on_accept)

        self._ok = False

    def on_accept(self):
        L = float(self.len_s.value()); O = float(self.ovl_s.value())
        if L <= 0 or O < 0 or O >= L:
            QtWidgets.QMessageBox.warning(self, "Ongeldige parameters",
                "Zorg dat: lengte > 0 en 0 ≤ overlap < lengte.")
            return
        self._ok = True
        self.accept()

    def values(self):
        return (
            float(self.len_s.value()),
            float(self.ovl_s.value()),
            bool(self.chk_replace.isChecked()),
            self.combo_label.currentText()  # NIEUW: gekozen label
        )

# -------------------------
# Main window
# -------------------------
class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung Sound Viewer (PySide6 + PyQtGraph)")
        self.resize(1200, 820)

        # --- Menu ---
        m = self.menuBar().addMenu("Bestand")
        self.act_open = m.addAction("Map openen…")
        self.act_open.triggered.connect(self.open_folder_dialog)

        # Runtime state
        self.player = Player()
        self.root = ""
        self.files: List[str] = []
        self.idx = -1

        self.y_raw: Optional[np.ndarray] = None
        self.sr = DEFAULT_SR
        self.t: Optional[np.ndarray] = None
        self.state: Optional[FileState] = None
        self.overlay_regions: Dict[str, pg.LinearRegionItem] = {}
        self._blocking = False

        # Central UI
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        H = QtWidgets.QHBoxLayout(cw)

        # ----- Links: waveform -----
        left = QtWidgets.QWidget(); H.addWidget(left, 3)
        gl = QtWidgets.QGridLayout(left)

        # PlotWidget
        self.p_wave = pg.PlotWidget()
        self.p_wave.setLabel("bottom", "Tijd (s)")
        self.p_wave.setLabel("left", "Amplitude")
        self.p_wave.showGrid(x=True, y=True, alpha=0.2)

        # Playhead en selectie (eerst aanmaken)
        self.playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#CC3333', width=1))
        self.region   = pg.LinearRegionItem([0.0, 2.5], brush=(100,180,255,60), movable=True)

        # Sneller tekenen: één cached PlotDataItem met downsampling
        # Let op: sommige pyqtgraph-versies accepteren de keyword-args direct;
        # anders kun je de setDownsampling(...) methode gebruiken (zie try/except hieronder).
        self.curve = pg.PlotDataItem(
            pen=pg.mkPen('#1976D2', width=1.2),
            clipToView=True,
            autoDownsample=True,
            downsampleMethod='peak',
        )

        # Items in de juiste volgorde toevoegen (elk slechts één keer!)
        self.p_wave.addItem(self.curve)
        self.p_wave.addItem(self.playhead)
        self.p_wave.addItem(self.region)

        gl.addWidget(self.p_wave, 0, 0)

        # (Compat-laagje voor oudere pyqtgraph-versies)
        try:
            # Als de kwargs niet bestaan in jouw versie, forceer via methodes:
            self.curve.setDownsampling(auto=True, method='peak')
        except Exception:
            pass

        # Tijdslider + label
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 0)   # dynamisch, bij laden bestand gezet
        self.lbl_time = QtWidgets.QLabel("0.00 s")
        tbar = QtWidgets.QHBoxLayout()
        tbar.addWidget(QtWidgets.QLabel("Tijd:")); tbar.addWidget(self.time_slider, 1); tbar.addWidget(self.lbl_time)
        gl.addLayout(tbar, 1, 0)
        self.init_spectrogram(gl)

        # ----- Rechts: panel -----
        right = QtWidgets.QWidget(); H.addWidget(right, 1)
        rv = QtWidgets.QVBoxLayout(right)

        self.lbl_path = QtWidgets.QLabel("—"); self.lbl_path.setStyleSheet("font-weight:600;")
        rv.addWidget(self.lbl_path)

        self.btn_open_folder = QtWidgets.QPushButton("Open folder…")
        rv.addWidget(self.btn_open_folder)

        self.btn_prev = QtWidgets.QPushButton("◀ Prev")
        self.btn_next = QtWidgets.QPushButton("Next ▶")

        self.combo_jump = QtWidgets.QComboBox()
        self.combo_jump.setEnabled(False)
        self.combo_jump.setMinimumWidth(280)
        self.combo_jump.currentIndexChanged.connect(self._on_jump_selected)

        nav_row = QtWidgets.QHBoxLayout()
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.btn_next)
        nav_row.addWidget(QtWidgets.QLabel("Jump to:"))
        nav_row.addWidget(self.combo_jump)
        rv.addLayout(nav_row)

        # "Selected:" met invoervelden die gekoppeld zijn aan de sleep-regio
        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(QtWidgets.QLabel("Selected:"))
        self.sel_start = QtWidgets.QDoubleSpinBox()
        self.sel_start.setDecimals(2); self.sel_start.setSingleStep(TIME_SNAP); self.sel_start.setRange(0, 1e6)
        self.sel_end   = QtWidgets.QDoubleSpinBox()
        self.sel_end.setDecimals(2); self.sel_end.setSingleStep(TIME_SNAP); self.sel_end.setRange(0, 1e6)
        self.lbl_sel_delta = QtWidgets.QLabel("(Δ 0.00 s)")
        sel_row.addWidget(self.sel_start); sel_row.addWidget(QtWidgets.QLabel("–")); sel_row.addWidget(self.sel_end)
        sel_row.addWidget(self.lbl_sel_delta)
        rv.addLayout(sel_row)

        # sync: typen → regio
        self.sel_start.valueChanged.connect(lambda _: self.on_sel_spin_changed())
        self.sel_end.valueChanged.connect(lambda _: self.on_sel_spin_changed())

        # Labels beheer
        box = QtWidgets.QGroupBox("Labels")
        vb = QtWidgets.QVBoxLayout(box)

        self.combo_labels = QtWidgets.QComboBox()

        hb = QtWidgets.QHBoxLayout()
        self.txt_new_label = QtWidgets.QLineEdit()
        self.txt_new_label.setPlaceholderText("Nieuw label…")
        self.btn_add_label = QtWidgets.QPushButton("Voeg label toe aan selectie")
        self.btn_reload_labels = QtWidgets.QPushButton("Herlaad labels.json")

        # Eerst aanmaken, dán toevoegen
        vb.addWidget(self.combo_labels)
        hb.addWidget(self.txt_new_label)
        vb.addLayout(hb)
        vb.addWidget(self.btn_add_label)
        vb.addWidget(self.btn_reload_labels)

        rv.addWidget(box)

        # Auto-segmenteren
        self.btn_auto_seg = QtWidgets.QPushButton("Auto segment…")
        rv.addWidget(self.btn_auto_seg)
        self.btn_auto_seg.clicked.connect(self.auto_segment_dialog)

        # Segmentlijst + editor
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

        # --- Bandpass filter (rechts) ---
        grp_bp = QtWidgets.QGroupBox("Bandpass-filter")
        fv = QtWidgets.QFormLayout(grp_bp)

        self.chk_bp = QtWidgets.QCheckBox("Filter aan")
        self.sp_low = QtWidgets.QDoubleSpinBox()
        self.sp_low.setRange(0.1, 20000.0); self.sp_low.setDecimals(1); self.sp_low.setSingleStep(10.0); self.sp_low.setValue(50.0)
        self.sp_high = QtWidgets.QDoubleSpinBox()
        self.sp_high.setRange(1.0, 20000.0); self.sp_high.setDecimals(1); self.sp_high.setSingleStep(10.0); self.sp_high.setValue(2000.0)
        self.sp_order = QtWidgets.QSpinBox()
        self.sp_order.setRange(1, 10); self.sp_order.setValue(2)
        self.chk_zero = QtWidgets.QCheckBox("Zero-phase"); self.chk_zero.setChecked(True)

        row_top = QtWidgets.QHBoxLayout()
        row_top.addWidget(self.chk_bp)
        row_top.addStretch(1)

        self.btn_bp_info = QtWidgets.QToolButton()
        self.btn_bp_info.setText("Info")
        self.btn_bp_info.setToolTip("Uitleg over filterinstellingen")
        self.btn_bp_info.clicked.connect(self._show_bp_info)
        row_top.addWidget(self.btn_bp_info)

        fv.addRow(row_top)

        fv.addRow("Low (Hz):", self.sp_low)
        fv.addRow("High (Hz):", self.sp_high)
        fv.addRow("Order:", self.sp_order)
        fv.addRow(self.chk_zero)

        rv.addWidget(grp_bp)

        # Cache voor gefilterd signaal
        self._filt_cache = None
        self._filt_params = None

        # UI events → hertekenen
        self.chk_bp.stateChanged.connect(self.on_filter_ui_changed)
        self.sp_low.valueChanged.connect(self.on_filter_ui_changed)
        self.sp_high.valueChanged.connect(self.on_filter_ui_changed)
        self.sp_order.valueChanged.connect(self.on_filter_ui_changed)
        self.chk_zero.stateChanged.connect(self.on_filter_ui_changed)

        # Export
        self.btn_export_csv = QtWidgets.QPushButton("Export CSV"); rv.addWidget(self.btn_export_csv)
        # Klein statuslabel met laatst geëxporteerde locatie
        self.lbl_last_export = QtWidgets.QLabel("Laatst geëxporteerd: —")
        self.lbl_last_export.setStyleSheet("color: gray; font-size: 10pt;")
        self.lbl_last_export.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        rv.addWidget(self.lbl_last_export)

        # Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, self.toggle_play)
        QtGui.QShortcut(QtGui.QKeySequence("N"), self, lambda: self.advance(+1))
        QtGui.QShortcut(QtGui.QKeySequence("P"), self, lambda: self.advance(-1))
        QtGui.QShortcut(QtGui.QKeySequence("Return"), self, self.update_segment)
        QtGui.QShortcut(QtGui.QKeySequence("Enter"),  self, self.update_segment)
        QtGui.QShortcut(QtGui.QKeySequence("Delete"), self, self.delete_selected)


        # Arrow-key nudges voor selectie-regio (0.01 s per stap)
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

        # Signals
        self.btn_prev.clicked.connect(lambda: self.advance(-1))
        self.btn_next.clicked.connect(lambda: self.advance(+1))
        self.btn_open_folder.clicked.connect(self.open_folder_dialog)
        self.btn_add_label.clicked.connect(self.add_label_to_selection)
        self.txt_new_label.returnPressed.connect(self.add_label_to_selection)
        self.btn_remove_label.clicked.connect(self.remove_selected_label)
        self.region.sigRegionChanged.connect(self.on_region_changed)
        self.list.currentRowChanged.connect(self.on_list_selection)
        self.btn_update.clicked.connect(self.update_segment)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.time_slider.valueChanged.connect(self.on_slider_changed)
        self.btn_reload_labels.clicked.connect(self.reload_labels_json)

        self.player.started.connect(self.on_play_started)
        self.player.stopped.connect(self.on_play_stopped)

        # Timer voor playhead-animatie
        self.timer = QtCore.QTimer(self); self.timer.setInterval(30); self.timer.timeout.connect(self.tick_playhead)

        # Sessie-meta uit startdialoog
        self.session_meta: Dict[str, object] = {}
        self.play_window: Tuple[float, float] = (0.0, 0.0)

        # Start
        self.open_folder_dialog(first=True)

    # --------- Auto-segmenteren ----------
    def auto_segment_dialog(self):
        if self.t is None or len(self.t) == 0:
            QtWidgets.QMessageBox.information(self, "Auto segmenteren", "Geen audio geladen.")
            return

        # opties voor dropdown: huidige combobox + defaults uit labels_dataset.json
        label_options = [self.combo_labels.itemText(i) for i in range(self.combo_labels.count())]
        default_label = getattr(self, "_auto_seg_cfg", {}).get("label", self.combo_labels.currentText() if self.combo_labels.count() else None)
        default_len   = float(getattr(self, "_auto_seg_cfg", {}).get("length_s", 3.0))
        default_ovl   = float(getattr(self, "_auto_seg_cfg", {}).get("overlap_s", 0.0))

        dlg = AutoSegmentDialog(self,
            default_len=default_len,
            default_overlap=default_ovl,
            default_replace=False,
            label_options=label_options,
            default_label=default_label
        )

        if not dlg.exec():
            return
        seg_len, seg_ovl, replace, auto_label = dlg.values()
        self.apply_auto_segments(seg_len, seg_ovl, replace, auto_label=auto_label)

    def apply_auto_segments(self, seg_len: float, seg_ovl: float, replace: bool, auto_label: Optional[str] = None):
        """
        Maak segmenten van seg_len met 'seg_ovl' seconden overlap tussen opeenvolgende segmenten.
        Elk nieuw segment krijgt (optioneel) het label 'auto_label'.
        """
        if self.state is None or self.t is None or len(self.t) == 0:
            return

        dur = float(self.t[-1])
        snap = float(TIME_SNAP)

        len_ticks = max(1, int(round(seg_len / snap)))
        ovl_ticks = max(0, int(round(seg_ovl / snap)))
        if ovl_ticks >= len_ticks:
            QtWidgets.QMessageBox.warning(self, "Ongeldige parameters", "Zorg dat: 0 ≤ overlap < lengte.")
            return

        stride_ticks = len_ticks - ovl_ticks
        total_ticks  = int(round(dur / snap))

        new_segments: List[Segment] = []
        start_tick = 0
        while start_tick < total_ticks:
            end_tick = min(start_tick + len_ticks, total_ticks)
            a = round(start_tick * snap, 2)
            b = round(end_tick   * snap, 2)
            labels = [auto_label] if auto_label else []
            new_segments.append(Segment(id=str(uuid.uuid4()), t_start=a, t_end=b, labels=labels))
            start_tick += stride_ticks

        if replace:
            self.state.segments = new_segments
        else:
            self.state.segments.extend(new_segments)

        self.refresh_segment_list()
        self.save_json()

        if self.state.segments:
            idx = len(self.state.segments) - len(new_segments)
            self.list.setCurrentRow(max(0, idx))

    # --------- Folder flow ----------
    def open_folder_dialog(self, first=False):
        """Toon startdialoog en (opnieuw) bouw de filequeue."""
        self.player.stop()

        dlg = StartDialog(self)
        if not dlg.exec():
            # if first: sys.exit(0)   # weg
            return

        self.root = dlg.root or ""
        self.session_meta = dlg.get_meta()
        self.load_labels_json()
        self.build_file_queue(self.root)
        if not self.files:
            QtWidgets.QMessageBox.information(self, "Info", "Geen .wav-bestanden gevonden.")
            # if first: sys.exit(0)   # weg
            return

        self._populate_jump_list()
        self.idx = 0
        self.load_current()


    def build_file_queue(self, root: str):
        """Verzamel .wav-bestanden: eerst top-level, daarna submappen."""
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
        """Laad huidig bestand, render waveform en reset UI."""
        f = self.files[self.idx]
        self.lbl_path.setText(f"{human_relpath(self.root, os.path.dirname(f))}/{os.path.basename(f)}")

        y, sr = sf.read(f, dtype="float32", always_2d=False)
        if y.ndim == 2: y = y.mean(axis=1)
        self.y_raw = y.astype(np.float32)

        # Reset filter-cache bij nieuw bestand
        self._filt_cache = None
        self._filt_params = None
        self.sr = int(sr)
        self.t = np.arange(len(self.y_raw), dtype=float) / self.sr # type: ignore

        dur = len(self.y_raw)/self.sr # type: ignore
        self.time_slider.blockSignals(True); self.time_slider.setRange(0, int(dur*100)); self.time_slider.setValue(0); self.time_slider.blockSignals(False)
        self.lbl_time.setText("0.00 s"); self.playhead.setPos(0.0)

        # Sidecar JSON laden of aanmaken
        js_path = json_sidecar_path(f)
        if os.path.isfile(js_path):
            with open(js_path, "r", encoding="utf-8") as fh:
                self.state = FileState.from_json(json.load(fh))
        else:
            self.state = FileState(file=os.path.basename(f), sr=self.sr, meta=dict(self.session_meta), segments=[])

        # Render waveform
        self.draw_waveform()
        self.update_spectrogram()

        # Default selectie
        self._blocking = True
        init_len = min(3.0, float(len(self.y_raw))/self.sr)  # 3 s of minder als opname kort is # type: ignore
        self.region.setRegion((0.0, init_len))
        self.sel_start.setValue(0.0)
        self.sel_end.setValue(init_len)
        self.lbl_sel_delta.setText(f"(Δ {init_len:.2f} s)")
        self._blocking = False

        self.refresh_segment_list()
        self.save_json()

    # --------- Segmentlijst & overlays ----------
    def on_overlay_clicked(self, reg: ClickableRegion):
        # Zoek de segment-index bij deze overlay en selecteer die rij
        if self.state is None:
            return
        try_id = getattr(reg, "seg_id", None)
        if try_id is None:
            return
        for i, s in enumerate(self.state.segments):
            if s.id == try_id:
                self.list.setCurrentRow(i)
                # ook de selectie-regio (boven) gelijk zetten
                self._blocking = True
                self.region.setRegion((s.t_start, s.t_end))
                self.sel_start.setValue(s.t_start)
                self.sel_end.setValue(s.t_end)
                self.lbl_sel_delta.setText(f"(Δ {(s.t_end - s.t_start):.2f} s)")
                self._blocking = False
                break

    # --------- Signaalkeuze ----------
    def current_signal(self) -> np.ndarray:
        """
        Signaal zoals getoond/afgespeeld: raw of bandpass-gefilterd afhankelijk van toggle.
        """
        if self.y_raw is None:
            return self.y_raw  # type: ignore
        if getattr(self, "chk_bp", None) and self.chk_bp.isChecked():
            return self.get_filtered_signal()
        return self.y_raw

    def get_filtered_signal(self) -> np.ndarray:
        """
        Lazy filter + eenvoudige cache die invalideert als params wijzigen.
        """
        if self.y_raw is None:
            return self.y_raw  # type: ignore
        sr = float(self.sr)
        params = (
            float(self.sp_low.value()),
            float(self.sp_high.value()),
            int(self.sp_order.value()),
            bool(self.chk_zero.isChecked()),
            len(self.y_raw),  # lengte hoort gelijk te blijven
            sr,
        )
        if self._filt_cache is not None and self._filt_params == params:
            return self._filt_cache  # type: ignore

        try:
            y_f = bandpass_filter(self.y_raw, fs=sr, fc=(params[0], params[1]),
                                  order=params[2], zero_phase=params[3], axis=-1)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Bandpass-filter", f"Kon filter niet toepassen:\n{e}\nVal terug op raw.")
            self.chk_bp.setChecked(False)
            self._filt_cache = None
            self._filt_params = None
            return self.y_raw

        # Cache
        self._filt_cache = y_f.astype(np.float32, copy=False)
        self._filt_params = params
        return self._filt_cache

    def on_filter_ui_changed(self, *args):
        """
        Filter-UI gewijzigd: cache ongeldig maken en alles hertekenen.
        """
        self._filt_cache = None
        self._filt_params = None
        self.draw_waveform()
        self.update_spectrogram()

    # --------- Tekenen ----------
    def draw_waveform(self):
        y = self.current_signal()
        x = self.t
        if y is None or x is None:
            return

        # Hergebruik dezelfde curve; geen fill onder de curve (scheelt veel)
        self.curve.setData(x=x, y=y, connect='finite')

        # Aslabels & grid (éénmalig zetten is voldoende, maar dit is goedkoop)
        self.p_wave.setLabel("bottom", "Tijd (s)")
        self.p_wave.setLabel("left", "Amplitude")
        self.p_wave.showGrid(x=True, y=True, alpha=0.2)

        # Houd de x-limieten stabiel zodat er geen auto-range kost plaatsvindt
        if len(x) > 1:
            xmax = float(x[-1])
            self.p_wave.setLimits(xMin=0.0, xMax=xmax)
            # Laat de volledige opname zien bij laden
            vb = self.p_wave.getViewBox()
            vb.setXRange(0.0, xmax, padding=0.02)

        # Zorg dat playhead/region aanwezig blijven
        # (ze zijn al toegevoegd in __init__, dus niets te doen)
        # Overlays opnieuw toevoegen als je die gebruikt:
        for reg in getattr(self, "overlay_regions", {}).values():
            if reg.scene() is None:  # nog niet toegevoegd
                self.p_wave.addItem(reg)

    # --------- Spectrogram ----------
    # STFT instellingen
    def _show_stft_info(self):
        txt = (
            "Korte uitleg STFT parameters:\n\n"
            "• nperseg: aantal samples per FFT-frame (groot = betere frequentieresolutie, maar grover in tijd).\n"
            "• hop: stapgrootte tussen opeenvolgende frames (klein = vloeiendere tijdsresolutie, maar meer data).\n"
            "• window: vensterfunctie (Hann vermindert randartefacten, standaard voor audio).\n\n"
            "Let op: deze instellingen beïnvloeden alleen hoe het spectrogram getoond wordt, "
            "niet de segmenten of het audio zelf."
        )
        QtWidgets.QMessageBox.information(self, "STFT parameters", txt)

    def _show_bp_info(self):
        txt = (
            "Bandpass-filter – korte uitleg:\n\n"
            "• Long focus (80–3000 Hz): dempt lage harttonen en pakt de hogere energie van veel longgeluiden mee.\n"
            "• Hart focus (20–250 Hz): richt zich op S1/S2 en lage-frequentiecomponenten; murmurs kunnen iets hoger liggen.\n\n"
            "• Order (4–6 aanbevolen): hogere orde = steilere randen, maar meer kans op ringing/instabiliteit.\n"
            "  Orde 4 is een veilige, stabiele keuze.\n"
            "• Zero-phase (aan/uit): ‘aan’ gebruikt forward–backward filtering (geen fasevertraging, wel niet-causaal).\n"
            "  Gebruik ‘aan’ voor offline analyse/visualisatie; ‘uit’ voor echte realtime-ketens met minimale latentie.\n"
        )
        QtWidgets.QMessageBox.information(self, "Bandpass – Info", txt)

    def init_spectrogram(self, grid_layout: QtWidgets.QGridLayout):
        """Create the spectrogram plot below the waveform."""
        self.p_spec = pg.PlotWidget()
        self.p_spec.setBackground('k')   # belangrijk: expliciet zwarte achtergrond
        self.p_spec.setLabel("bottom", "Tijd (s)")
        self.p_spec.setLabel("left", "Frequentie (Hz)")
        self.p_spec.setMouseEnabled(x=True, y=True)   # mag interactief zijn
        self.p_spec.setXLink(self.p_wave)             # koppel X-zoom/pan met waveform

        # init_spectrogram(...)
        self.img_spec = pg.ImageItem(axisOrder='row-major')
        self.p_spec.addItem(self.img_spec)

        self.colorbar = None
        if not GRAYSCALE_DEBUG:
            try:
                cmap = pg.colormap.get('inferno')
                self.colorbar = pg.ColorBarItem(values=(-100, 0), colorMap=cmap)
                # sommige versies hebben geen insertIn:
                try:
                    self.colorbar.setImageItem(self.img_spec, insertIn=self.p_spec.getPlotItem()) # type: ignore
                except TypeError:
                    self.colorbar.setImageItem(self.img_spec)
            except Exception:
                pass

        # Onder de tijdslider plaatsen (rij 2, kolom 0)
        grid_layout.addWidget(self.p_spec, 2, 0)

        # Info-blok onder het spectrogram
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
        """Compute STFT en teken als image."""
        y = self.current_signal()
        if y is None or len(y) == 0:
            _dbg("[STFT] geen data"); return

        cfg = getattr(self, "_stft_cfg", {"nperseg":1024,"hop":256,"window":"hann"})
        f, t, S_db = compute_stft_db(y, self.sr,
                                    nperseg=int(cfg.get("nperseg",1024)),
                                    hop=int(cfg.get("hop",256)),
                                    window=str(cfg.get("window","hann")))
        if S_db.size == 0:
            _dbg("[STFT] lege spectrogram array"); return

        # Bepaal y-as eindpunt: filter aan => tot 'high', anders tot Nyquist
        if getattr(self, "chk_bp", None) and self.chk_bp.isChecked():
            fmax_plot = min(float(self.sp_high.value()), float(self.sr)/2.0 - 1e-6)
        else:
            fmax_plot = float(f[-1])

        # Slice freq-as tot fmax_plot (y-as begint altijd bij 0 Hz)
        mask = f <= fmax_plot
        if not np.any(mask):
            mask = f <= f[-1]
        f_plot = f[mask]
        img = S_db[mask, :]  # (F_plot, T)

        _arr_stats("[UI] S_db (voor beeld)", img)

        if GRAYSCALE_DEBUG:
            finite = np.isfinite(img)
            if finite.any():
                vmin = float(np.percentile(img[finite], 5))
                vmax = float(np.percentile(img[finite], 99))
                if vmin >= vmax: vmin, vmax = -100.0, 0.0
            else:
                vmin, vmax = -100.0, 0.0
            im8 = (255.0 * np.clip((img - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)).astype(np.uint8)
            _arr_stats("[UI] im8", im8)
            self.img_spec.setLookupTable(None)
            self.img_spec.setLevels((0, 255))
            self.img_spec.setImage(im8, autoLevels=False)
        else:
            try:
                lut = pg.colormap.get("inferno").getLookupTable(256) # type: ignore
                self.img_spec.setLookupTable(lut)
            except Exception:
                self.img_spec.setLookupTable(None)
            if DYNAMIC_SPECTRO_LEVELS:
                finite = np.isfinite(img)
                if finite.any():
                    vmin = float(np.percentile(img[finite], 5))
                    vmax = float(np.percentile(img[finite], 99))
                    if vmin >= vmax: vmin, vmax = -100.0, 0.0
                else:
                    vmin, vmax = -100.0, 0.0
                self.img_spec.setLevels((vmin, vmax))
                _dbg(f"[UI] levels dynamic: vmin={vmin:.2f} vmax={vmax:.2f}")
            else:
                self.img_spec.setLevels((-100.0, 0.0))
            self.img_spec.setImage(img, autoLevels=False)

        # Geometrie & assen
        if len(t) > 1 and len(f_plot) > 1:
            t_max = float(t[-1]); f_max = float(f_plot[-1])
            self.img_spec.setRect(QtCore.QRectF(0.0, 0.0, t_max, f_max))
            self.p_spec.setLimits(xMin=0.0, xMax=t_max, yMin=0.0, yMax=f_max)
            self.p_spec.setXRange(0.0, t_max)
            self.p_spec.setYRange(0.0, f_max)
            _dbg(f"[UI] rect set: t_max={t_max:.3f}s f_max={f_max:.1f}Hz img.shape={img.shape}")

        # Toon de huidige STFT-settings
        cfg = getattr(self, "_stft_cfg", {"nperseg":1024,"hop":256,"window":"hann"})
        self.lbl_stft_params.setText(
            f"STFT: nperseg={cfg.get('nperseg')} | hop={cfg.get('hop')} | window={cfg.get('window')}"
        )

    # --------- Interactions ----------
    def on_region_changed(self):
        if self._blocking: return
        a, b = self.region.getRegion()
        a = max(0.0, snap_t(a)); b = max(a + TIME_SNAP, snap_t(b)) # type: ignore
        self._blocking = True
        self.region.setRegion((a, b))
        self.sel_start.setValue(a)
        self.sel_end.setValue(b)
        self.lbl_sel_delta.setText(f"(Δ {(b - a):.2f} s)")
        self._blocking = False

    def on_slider_changed(self, val: int):
        """Slider is vrije afspeelpositie; volgt de selectie niet."""
        t = val/100.0
        self.playhead.setPos(t)
        self.lbl_time.setText(f"{t:.2f} s")

    def on_sel_spin_changed(self):
        if self._blocking: return
        a = float(self.sel_start.value())
        b = float(self.sel_end.value())
        a = max(0.0, snap_t(a))
        b = max(a + TIME_SNAP, snap_t(b))
        self._blocking = True
        self.sel_start.setValue(a)
        self.sel_end.setValue(b)
        self.region.setRegion((a, b))
        self.lbl_sel_delta.setText(f"(Δ {(b - a):.2f} s)")
        self._blocking = False

    def nudge_region(self, dt: float, mode: str = "move"):
        """
        Verplaats of resize de selectie-regio in stapjes van dt seconden.
        mode: "move" | "start" | "end"
        """
        if self.t is None or len(self.t) == 0:
            return
        dur = float(self.t[-1])
        a, b = self.region.getRegion()
        a = float(a); b = float(b) # type: ignore
        step = float(dt)

        if mode == "move":
            width = b - a
            new_a = max(0.0, min(a + step, dur - width))
            new_b = new_a + width
        elif mode == "start":
            new_a = max(0.0, min(a + step, b - TIME_SNAP))
            new_b = b
        elif mode == "end":
            new_a = a
            new_b = min(dur, max(b + step, a + TIME_SNAP))
        else:
            return

        new_a = snap_t(new_a); new_b = snap_t(new_b)
        if new_b <= new_a:
            new_b = min(dur, new_a + TIME_SNAP)

        self._blocking = True
        self.region.setRegion((new_a, new_b))
        self.sel_start.setValue(new_a)
        self.sel_end.setValue(new_b)
        self.lbl_sel_delta.setText(f"(Δ {(new_b - new_a):.2f} s)")
        self._blocking = False

    # --------- Playback (vrij: van slider tot einde) ----------
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
        return LABEL_COLORS.color_for(labels)

    def refresh_segment_list(self):
        self.list.clear()
        if not self.state: return
        # oude overlays verwijderen
        for reg in self.overlay_regions.values():
            try: self.p_wave.removeItem(reg)
            except Exception: pass
        self.overlay_regions.clear()
        # opnieuw vullen
        for s in self.state.segments:
            self.list.addItem(f"{s.t_start:.2f}-{s.t_end:.2f}s | {'; '.join(s.labels) or '(geen labels)'}")
            reg = ClickableRegion([s.t_start, s.t_end],
                                brush=self._brush_for_labels(s.labels),
                                seg_id=s.id)
            self.p_wave.addItem(reg)
            reg.clicked.connect(self.on_overlay_clicked)
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
        a = snap_t(a); b = max(a+TIME_SNAP, snap_t(b)) # type: ignore

        seg = self._find_segment_by_bounds(a, b)
        if seg is None:
            seg = Segment(id=str(uuid.uuid4()), t_start=a, t_end=b, labels=[])
            self.state.segments.append(seg)
        if label not in seg.labels:
            seg.labels.append(label)

        self.refresh_segment_list()
        self.list.setCurrentRow(self.state.segments.index(seg))
        self.save_json()
        self.rebuild_label_list()

    def on_list_selection(self, row: int):
        if not self.state or row < 0 or row >= len(self.state.segments):
            self.list_labels.clear(); return
        s = self.state.segments[row]
        self.spin_start.setValue(s.t_start)
        self.spin_end.setValue(s.t_end)
        self.rebuild_label_list()
        self._blocking = True
        self.region.setRegion((s.t_start, s.t_end))
        self.sel_start.setValue(s.t_start)
        self.sel_end.setValue(s.t_end)
        self.lbl_sel_delta.setText(f"(Δ {(s.t_end - s.t_start):.2f} s)")
        self._blocking = False

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
        self.rebuild_label_list()

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
        self.rebuild_label_list()

    def delete_selected(self):
        row = self.list.currentRow()
        if not self.state or row < 0: return
        ans = QtWidgets.QMessageBox.question(self, "Delete segment", "Verwijder geselecteerd segment?")
        if ans != QtWidgets.QMessageBox.Yes: return # type: ignore
        del self.state.segments[row]
        self.refresh_segment_list()
        self.save_json()

    # --------- Padnamen ----------
    def _rel_display_name(self, abspath: str) -> str:
        """
        Toon een leesbare, relatieve padnaam vanaf self.root.
        """
        try:
            rel = os.path.relpath(abspath, self.root)
        except Exception:
            rel = os.path.basename(abspath)
        return rel.replace("\\", "/")

    # --------- Jump list ----------
    def _populate_jump_list(self):
        self.combo_jump.blockSignals(True)
        self.combo_jump.clear()

        display_names = []
        for p in self.files:
            abspath = p if isinstance(p, str) else getattr(p, "path", "")
            display_names.append(self._rel_display_name(abspath))

        self.combo_jump.addItems(display_names)
        self.combo_jump.setEnabled(len(display_names) > 0)

        if 0 <= self.idx < len(display_names):
            self.combo_jump.setCurrentIndex(self.idx)

        self.combo_jump.blockSignals(False)

    def _on_jump_selected(self, i: int):
        if not (0 <= i < len(self.files)):
            return
        if i == self.idx:
            return
        self.idx = i
        self.load_current()

    def _after_navigation_changed(self):
        if self.combo_jump.isEnabled() and 0 <= self.idx < self.combo_jump.count():
            self.combo_jump.blockSignals(True)
            self.combo_jump.setCurrentIndex(self.idx)
            self.combo_jump.blockSignals(False)

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
        pd.DataFrame(rows).to_csv(path, index=False) # type: ignore
        QtWidgets.QMessageBox.information(self, "Export", f"Opgeslagen {len(rows)} rijen naar:\n{path}")

    def _set_last_export_path(self, path: str):
        """Update het statuslabel met de laatst geëxporteerde locatie."""
        # Nettere, korte weergave (elide in het midden als het lang is)
        shown = path
        max_chars = 80
        if len(shown) > max_chars:
            shown = shown[:40] + " … " + shown[-35:]
        self.lbl_last_export.setText(f"Laatst geëxporteerd: {shown}")
        self.lbl_last_export.setToolTip(path)  # volledige pad als tooltip

    # --------- Labels dataset JSON ----------
    def load_labels_json(self):
        path = labels_dataset_path()

        # Fallback-config als bestand (nog) niet bestaat
        default_cfg = {
            "version": 1,
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
            "labels": [
                "Hoest","Normaal","Hartslag","Piep (Wheeze)","Knetter (Crackle)"
            ],
            "meta_defaults": {"gender":"","age":"","location":""},
            "filter_defaults": {"lowcut": 50, "highcut": 3000, "order": 4, "zero_phase": True},
            "stft_params": {"nperseg": 1024, "hop": 256, "window": "hann"},
            "auto_segment_defaults": {"length_s": 3.00, "overlap_s": 0.00, "label": "Hoest"}
        }

        # Lees centrale JSON of maak ‘m daar aan (niet in de datamap)
        if LABELS_JSON_PATH.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = default_cfg
        else:
            cfg = default_cfg
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default_cfg, f, ensure_ascii=False, indent=2)

        # Labels → combobox
        labels = cfg.get("labels", [])
        self.combo_labels.clear()
        if labels:
            self.combo_labels.addItems(labels)

        # Label-kleurenmapping opbouwen (zoals we eerder hebben toegevoegd)
        LABEL_COLORS.build(labels)

        # Meta defaults
        defaults = cfg.get("meta_defaults", {})
        for k, v in defaults.items():
            self.session_meta.setdefault(k, v)

        # Filter defaults
        fdef = cfg.get("filter_defaults", {})
        if "lowcut" in fdef:     self.sp_low.setValue(float(fdef["lowcut"]))
        if "highcut" in fdef:    self.sp_high.setValue(float(fdef["highcut"]))
        if "order" in fdef:      self.sp_order.setValue(int(fdef["order"]))
        if "zero_phase" in fdef: self.chk_zero.setChecked(bool(fdef["zero_phase"]))

        # STFT + Auto-segment defaults in geheugen
        self._stft_cfg     = cfg.get("stft_params", {"nperseg":1024,"hop":256,"window":"hann"})
        self._auto_seg_cfg = cfg.get("auto_segment_defaults", {"length_s":3.0,"overlap_s":0.0,"label":labels[0] if labels else ""})


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
        """Vul self.list_labels met per label een widget: 'naam  [×]'."""
        self.list_labels.clear()
        row = self.list.currentRow()
        if not self.state or row < 0 or row >= len(self.state.segments):
            return
        seg = self.state.segments[row]

        for L in seg.labels:
            item = QtWidgets.QListWidgetItem(self.list_labels)
            item.setSizeHint(QtCore.QSize(0, 26))

            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w)
            h.setContentsMargins(6, 2, 6, 2)
            h.setSpacing(8)

            lbl = QtWidgets.QLabel(L)
            btn = QtWidgets.QToolButton()
            btn.setText("×")
            btn.setToolTip(f"Verwijder label: {L}")
            btn.setFixedSize(22, 22)
            btn.setStyleSheet("QToolButton { font-weight: bold; }")
            # Koppel de labelwaarde aan de knop
            btn.setProperty("label_text", L)
            btn.clicked.connect(self._on_remove_label_btn)

            h.addWidget(lbl)
            h.addStretch(1)
            h.addWidget(btn)

            self.list_labels.addItem(item)
            self.list_labels.setItemWidget(item, w)

    def _on_remove_label_btn(self):
        """Slot voor de ‘×’-knoppen in de labellijst."""
        if not self.state:
            return
        sender = self.sender()
        if not isinstance(sender, QtWidgets.QToolButton):
            return
        L = sender.property("label_text")
        row = self.list.currentRow()
        if row < 0 or row >= len(self.state.segments):
            return
        seg = self.state.segments[row]
        try:
            seg.labels.remove(L)
        except ValueError:
            return
        # UI bijwerken
        self.rebuild_label_list()
        self.refresh_segment_list()   # overlays/tekst bijwerken
        self.list.setCurrentRow(row)
        self.save_json()


    def reload_labels_json(self):
        self.load_labels_json()

    # --------- Persistence ----------
    def save_json(self):
        if not self.state: return
        js = json_sidecar_path(self.files[self.idx])
        ensure_dir(js)
        with open(js, "w", encoding="utf-8") as fh:
            json.dump(self.state.to_json(), fh, ensure_ascii=False, indent=2)

    # --------- Navigatie ----------
    def advance(self, step: int):
        self.player.stop()
        new = self.idx + step
        if new < 0 or new >= len(self.files):
            return
        self.idx = new
        self.load_current()
        self._after_navigation_changed()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())

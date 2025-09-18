"""
ecg_viewer_pyqt.py

Snelle ECG-viewer/annotator met PySide6 (Qt) + PyQtGraph.

Features:
- Weergave van 6 limb leads (I, II, III, aVR, aVL, aVF) + een brede "V1-avg" (gemiddelde van alle 6 leads).
- Interactieve cursor (verticale lijn + marker) op V1-avg.
- Intervalselectie (LinearRegion) met 0,01 s snapping.
- Mirror-highlight: het geselecteerde interval is zichtbaar in V1-avg en alle 6 leads.
- Interval is ook vanuit de 6 leads te slepen (alles blijft gesynchroniseerd).
- Labels toevoegen aan "V1-avg" óf aan een specifieke lead; export naar CSV met kolom 'lead'.

Benodigd:
    pip install PySide6 pyqtgraph numpy pandas
"""

from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import pandas as pd
import sys, uuid

# Volgorde/naam van de 6 limb leads zoals intern gebruikt
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"]


# ---------- Loader: zelfde format als jouw .txt notebook ----------
def load_ecg_txt(
    path: str,
    first_seconds: float = 10.0,
    sep: str = r"\t",
    decimal: str = ",",
    engine: str = "python",
):
    """
    Lees ECG-data in uit een .txt-bestand en leid de 6 limb leads af.

    Verwacht bestandsindeling:
        - Eerste kolom: datum (YYYY-MM-DD)
        - Tweede kolom: tijd (HH:MM:SS.ffffff)
        - Ten minste kolommen 'Lead I (mV)' en 'Lead II (mV)'
        - Decimaalteken: standaard ',' (aanpasbaar)
        - Scheiding: standaard tab ('\\t', aanpasbaar)

    Parameters
    ----------
    path : str
        Pad naar het .txt-bestand.
    first_seconds : float, optional
        Aantal seconden dat wordt geselecteerd vanaf het begin (default: 10.0).
    sep : str, optional
        Scheidingsteken (default: tab).
    decimal : str, optional
        Decimaalteken (default: ',').
    engine : str, optional
        Pandas read_csv engine (default: 'python').

    Returns
    -------
    t : np.ndarray (float)
        Tijd-as in seconden vanaf het eerste sample.
    sigs : dict[str, np.ndarray]
        Dictionary met de 6 afgeleide limb leads (sleutels: 'I','II','III','aVR','aVL','aVF').
    fs : float
        Geschatte samplingfrequentie (Hz).

    Raises
    ------
    ValueError
        Als verplichte kolommen ontbreken of timestamps niet te parsen zijn.
    """
    # Lees ruw in zonder parse_dates (vermijdt pandas FutureWarning)
    df = pd.read_csv(path, sep=sep, decimal=decimal, engine=engine)  # type: ignore

    # Kolomnamen opschonen
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    # Verwacht: eerste twee kolommen zijn datum + tijd (zoals jouw .txt)
    date_col = df.columns[0]
    time_col = df.columns[1]

    # Combineer naar timestamp (formaat: 'YYYY-MM-DD HH:MM:SS.ffffff')
    ts = pd.to_datetime(
        df[date_col].astype(str) + " " + df[time_col].astype(str),
        format="%Y-%m-%d %H:%M:%S.%f",  # expliciet format → geen dayfirst-warnings
        errors="coerce",
    )
    if ts.isna().any():
        raise ValueError("Kon timestamps niet parsen; controleer datum/tijd kolommen.")

    # Timestamp vooraan zetten en indexeren
    df.insert(0, "timestamp", ts)
    df = df.drop(columns=[date_col, time_col])
    df.set_index("timestamp", inplace=True)

    # Minimaal Lead I/II vereist
    req_src = ["Lead I (mV)", "Lead II (mV)"]
    missing = [c for c in req_src if c not in df.columns]
    if missing:
        raise ValueError(f"Kolommen ontbreken: {missing}. Beschikbaar: {list(df.columns)}")

    # 6 leads afleiden uit I en II
    df["Lead III (mV)"] = df["Lead II (mV)"] - df["Lead I (mV)"]
    df["aVR (mV)"] = -0.5 * (df["Lead I (mV)"] + df["Lead II (mV)"])
    df["aVL (mV)"] = df["Lead I (mV)"] - 0.5 * df["Lead II (mV)"]
    df["aVF (mV)"] = df["Lead II (mV)"] - 0.5 * df["Lead I (mV)"]

    # Map bronkolommen naar interne leadnamen
    colmap = {
        "Lead I (mV)": "I",
        "Lead II (mV)": "II",
        "Lead III (mV)": "III",
        "aVR (mV)": "aVR",
        "aVL (mV)": "aVL",
        "aVF (mV)": "aVF",
    }
    src_cols = list(colmap.keys())

    # Samplingrate schatten uit timestamp-diffs (mediaan tegen outliers)
    dts = df.index.to_series().diff().dropna().dt.total_seconds().values
    if len(dts) == 0:
        raise ValueError("Onvoldoende rijen om sampling-interval te bepalen.")
    dt = float(np.median(dts))  # type: ignore
    fs = 1.0 / dt
    expected = int(round(first_seconds * fs))

    # Precies first_seconds selecteren vanaf de start
    t0 = df.index[0]
    m = (df.index >= t0) & (df.index < t0 + pd.Timedelta(seconds=first_seconds))
    cut = df.loc[m, src_cols]
    if len(cut) > expected:
        cut = cut.iloc[:expected]

    # Tijd-as (s) en signals dict met juiste sleutels
    t = (cut.index - cut.index[0]).total_seconds().astype(float)
    sigs = {colmap[src]: cut[src].to_numpy(dtype=float) for src in src_cols}
    return t, sigs, fs


# ---------- Eenvoudige label store ----------
class LabelStore:
    """
    Simpele container voor interval-labels.

    Elk item:
        {
            "id":   str (uuid),
            "t0":   float (starttijd, s),
            "t1":   float (eindtijd, s),
            "lead": str   ("V1-avg" of één van de LEADS),
            "labels": list[str]
        }
    """

    def __init__(self):
        """Initialiseer een lege labelverzameling."""
        self.items = []

    def add(self, t0, t1, lead, labels):
        """
        Voeg een label-interval toe.

        Parameters
        ----------
        t0, t1 : float
            Begin- en eindtijd van het interval in seconden.
        lead : str
            Target lead ("V1-avg" of één van de LEADS).
        labels : list[str] | tuple[str] | set[str]
            Eén of meerdere labels die bij het interval horen.
        """
        self.items.append(
            dict(id=str(uuid.uuid4()), t0=t0, t1=t1, lead=lead, labels=list(labels))
        )

    def to_df(self) -> pd.DataFrame:
        """
        Exporteer de labels als pandas DataFrame (voor CSV-export).

        Returns
        -------
        pd.DataFrame met kolommen:
            - t_start
            - t_end
            - lead
            - labels (samengevoegd met ';')
        """
        return pd.DataFrame(
            [
                dict(
                    t_start=x["t0"],
                    t_end=x["t1"],
                    lead=x["lead"],
                    labels=";".join(x["labels"]),
                )
                for x in self.items
            ]
        )


LABELS = [
    "Normal Rhythm",
    "Atrial Fibrillation",
    "Atrial Flutter",
    "SVT",
    "VT",
    "ST Depression",
    "ST Elevation",
    "QRS Widening",
]


# ---------- Hoofdvenster ----------
class App(QtWidgets.QMainWindow):
    """
    Hoofdvenster van de ECG-viewer.

    Bestaat uit:
      - Links: 6 lead-plots (2x3 grid) + onderin V1-avg (gemiddelde van alle leads).
      - Rechts: bestandsknop, live 'Selected' readout, label-selector, lead-target, lijst en CSV-export.

    Interacties:
      - Cursor: mouse move over V1-avg.
      - Interval: LinearRegion op V1-avg én per lead (allemaal gesynchroniseerd).
      - Verslepen van interval kan op V1-avg of op een willekeurige lead.
    """

    def __init__(self):
        """Initialiseer UI, state, en verbind eventhandlers."""
        super().__init__()
        self.setWindowTitle("ECG – Fast Viewer (PySide6 + PyQtGraph)")
        self.resize(1400, 900)

        # Data/state
        self.t = None               # np.ndarray of None
        self.sigs = None            # dict[str, np.ndarray] of None
        self.avg = None             # np.ndarray of None
        self.labels = LabelStore()  # opslag van toegevoegde labels
        self._region_syncing = False  # guard tegen signaal-lussen bij synchroniseren

        # --- UI skeleton ---
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        h = QtWidgets.QHBoxLayout(cw)

        # Links: grid met 6 leads + V1-avg
        left = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(left)
        h.addWidget(left, 3)

        self.plots = {}
        for i, name in enumerate(LEADS):
            pw = pg.PlotWidget(title=name)
            pw.showGrid(x=True, y=True, alpha=0.15)
            pw.setMenuEnabled(False)
            # Snelle rendering: min/max per pixel + alleen zichtbare data
            pw.getPlotItem().setDownsampling(mode="peak")  # type: ignore
            pw.getPlotItem().setClipToView(True)  # type: ignore
            grid.addWidget(pw, i // 2, i % 2)
            self.plots[name] = pw

        # Onderin: V1-avg
        self.p_avg = pg.PlotWidget(title="Extended V1 (Average of All Leads)")
        self.p_avg.showGrid(x=True, y=True, alpha=0.15)
        self.p_avg.getPlotItem().setDownsampling(mode="peak")  # type: ignore
        self.p_avg.getPlotItem().setClipToView(True)  # type: ignore
        grid.addWidget(self.p_avg, 3, 0, 1, 2)

        # Cursor + selectie op V1-avg
        self.cursor = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#CC3333", width=1)
        )  # type: ignore
        self.p_avg.addItem(self.cursor)
        self.dot = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush("#CC3333"))
        self.p_avg.addItem(self.dot)
        # Intervalselectie (startwaarde 2.0–2.5 s; wordt later bij laden gezet)
        self.region = pg.LinearRegionItem(
            [2.0, 2.5], brush=(100, 180, 255, 60), movable=True
        )
        self.p_avg.addItem(self.region)

        # Mirror regions in de 6 lead-plots (movable + eigen events)
        self.lead_regions = {}  # name -> LinearRegionItem
        for name, pw in self.plots.items():
            r = pg.LinearRegionItem(
                self.region.getRegion(), brush=(100, 180, 255, 40), movable=True
            )
            pw.addItem(r)
            self.lead_regions[name] = r
            # Verbind per lead met gezamenlijke handler (naam meegeven via lambda)
            r.sigRegionChanged.connect(
                lambda _=None, nm=name: self._on_any_region_changed(nm)
            )

        # Rechts: besturing/labels/export
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        h.addWidget(right, 1)

        open_btn = QtWidgets.QPushButton("Open .txt")
        rv.addWidget(open_btn)

        # Live readout van de huidige selectie
        self.lbl_selected = QtWidgets.QLabel("Selected: –")
        self.lbl_selected.setStyleSheet("color: #ddd;")
        rv.addWidget(self.lbl_selected)

        # Labelkeuze
        rv.addWidget(QtWidgets.QLabel("Add label to current interval"))
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(LABELS)
        rv.addWidget(self.combo)

        # Target lead kiezen: V1-avg of een specifieke lead
        rv.addWidget(QtWidgets.QLabel("Label target"))
        self.combo_target = QtWidgets.QComboBox()
        self.combo_target.addItems(["V1-avg"] + LEADS)  # eerst V1-avg
        rv.addWidget(self.combo_target)

        add_btn = QtWidgets.QPushButton("Add")
        rv.addWidget(add_btn)

        rv.addWidget(QtWidgets.QLabel("Annotated Intervals"))
        self.list = QtWidgets.QListWidget()
        rv.addWidget(self.list, 1)

        export_btn = QtWidgets.QPushButton("Export CSV")
        rv.addWidget(export_btn)

        # Events koppelen
        open_btn.clicked.connect(self.open_txt)
        add_btn.clicked.connect(self.add_label)
        export_btn.clicked.connect(self.export_csv)
        # Cursor volgt muis over V1-avg
        self.p_avg.scene().sigMouseMoved.connect(self.on_move)  # type: ignore
        # Eén gezamenlijke handler voor alle regio’s
        self.region.sigRegionChanged.connect(self.on_region_changed)

        # Init-tekst voor live readout
        self.lbl_selected.setText("Selected: – s – – s (Δ – s)")

        # X-as ticks: major 1.0 s en minor 0.5 s (rustig maar toch informatief)
        for pw in list(self.plots.values()) + [self.p_avg]:
            ax = pw.getAxis("bottom")
            ax.setTickSpacing(1.0, 0.5)

    # ------- Data laden + tekenen -------
    def open_txt(self):
        """
        Open een ECG .txt-bestand, laad de data en teken alle grafieken opnieuw.

        - Laadt t, sigs en berekent V1-avg.
        - Tekent 6 leads + V1-avg opnieuw.
        - Herplaatst cursor, dot en intervalregion.
        - Bouwt mirror-regions per lead opnieuw op (na clear()).
        - Reset zoom in V1-avg.
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ECG .txt", filter="Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            t, sigs, fs = load_ecg_txt(path, first_seconds=10.0)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            return

        # State zetten
        self.t = t.astype(float)
        self.sigs = sigs
        self.avg = np.mean([self.sigs[k] for k in LEADS], axis=0)

        # Nieuwe file → labels resetten
        self.labels = LabelStore()
        self.list.clear()

        # --- 6 leads tekenen ---
        for name in LEADS:
            pw = self.plots[name]
            pw.clear()
            pw.plot(
                self.t,
                self.sigs[name],
                pen=pg.mkPen(120, width=1),
                antialias=False,
                clear=True,
            )

        # --- V1 (avg) tekenen + interactieve items terugplaatsen ---
        self.p_avg.clear()
        self.c_avg = self.p_avg.plot(
            self.t, self.avg, pen=pg.mkPen(180, 0, 0, width=2), antialias=False, clear=True
        )
        self.p_avg.addItem(self.cursor)
        self.p_avg.addItem(self.dot)
        self.p_avg.addItem(self.region)
        self.region.setRegion([2.0, 2.5])  # startselectie

        # --- mirror-regio’s opnieuw opbouwen in alle lead-plots ---
        ra, rb = self.region.getRegion()
        for name, pw in self.plots.items():
            # Verwijder bestaande mirror-items (clear() heeft ze al verwijderd)
            if hasattr(self, "lead_regions") and name in self.lead_regions:
                try:
                    pw.removeItem(self.lead_regions[name])
                except Exception:
                    pass
            # Nieuwe (movable) mirror-region toevoegen en koppelen
            r = pg.LinearRegionItem([ra, rb], brush=(100, 180, 255, 40), movable=True)
            pw.addItem(r)
            self.lead_regions[name] = r
            r.sigRegionChanged.connect(
                lambda _=None, nm=name: self._on_any_region_changed(nm)
            )

        # Live readout initialiseren
        self.lbl_selected.setText(f"Selected: {ra:.2f}s – {rb:.2f}s  (Δ {rb-ra:.2f}s)") # type: ignore

        # --- zoom reset (V1-avg) ---
        self.p_avg.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.p_avg.getViewBox().setXRange(self.t[0], self.t[-1], padding=0)

    # ------- Interacties -------
    def on_move(self, pos):
        """
        Cursor- en dot-update op basis van muispositie over V1-avg.

        Parameters
        ----------
        pos : QPointF (scene coordinates)
            Door PyQtGraph aangeleverde muispositie in scenespace.
        """
        if self.t is None or self.avg is None:
            return
        vb = self.p_avg.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return
        mp = vb.mapSceneToView(pos)
        # Snapping op 0,01 s
        x = round(mp.x() * 100) / 100.0
        self.cursor.setPos(x)  # type: ignore
        # Dot op dichtstbijzijnde sample van V1-avg
        idx = int(np.clip(np.searchsorted(self.t, x), 0, len(self.t) - 1))
        self.dot.setData([self.t[idx]], [self.avg[idx]])

    def on_region_changed(self):
        """
        Doorverwijzer: wanneer V1-avg-regio wijzigt, verwerk dat via de uniforme handler.
        """
        self._on_any_region_changed("V1-avg")

    def _on_any_region_changed(self, origin: str):
        """
        Uniforme handler voor regio-wijzigingen in V1-avg en in alle leads.

        - Bepaalt de bron (V1-avg of lead-naam).
        - Past 0,01 s snapping en minimale duur toe.
        - Synchroniseert alle regio's zonder signal loops (blockSignals + guard).
        - Werkt de live 'Selected' readout bij.

        Parameters
        ----------
        origin : str
            "V1-avg" of één van de lead-namen uit LEADS (de regio die je versleept).
        """
        if self._region_syncing or self.t is None:
            return
        try:
            self._region_syncing = True

            # 1) Lees de 'bron' regio uit
            if origin == "V1-avg":
                a, b = self.region.getRegion()
            else:
                a, b = self.lead_regions[origin].getRegion()

            # 2) Snap op 0,01 s en minimale duur van 0,01 s
            a = round(a * 100) / 100.0 # type: ignore
            b = round(b * 100) / 100.0 # type: ignore
            if b <= a:
                b = a + 0.01

            # 3) Schrijf terug naar alle regio’s (signalen tijdelijk blokkeren)
            # V1-avg
            self.region.blockSignals(True)
            self.region.setRegion((a, b))
            self.region.blockSignals(False)

            # Alle leads mirroren
            for nm, r in self.lead_regions.items():
                r.blockSignals(True)
                r.setRegion((a, b))
                r.blockSignals(False)

            # 4) Live readout bijwerken
            self.lbl_selected.setText(f"Selected: {a:.2f}s – {b:.2f}s  (Δ {b-a:.2f}s)")

        finally:
            self._region_syncing = False

    def add_label(self):
        """
        Voeg een label toe aan het huidige interval, voor V1-avg of een specifieke lead.

        - Leest het interval uit de V1-avg-regio (die is leidend).
        - Past 0,01 s snapping toe.
        - Slaat het interval op in de LabelStore, voegt een visuele 'fix' toe op de gekozen plot.
        - Werkt de rechterlijst bij.
        """
        if self.t is None:
            QtWidgets.QMessageBox.information(self, "Info", "Open eerst een .txt bestand.")
            return
        a, b = self.region.getRegion()
        a, b = round(a * 100) / 100.0, round(b * 100) / 100.0  # type: ignore
        if b <= a:
            b = a + 0.01

        label = self.combo.currentText()
        target_lead = self.combo_target.currentText()  # "V1-avg" of specifieke lead
        self.labels.add(a, b, target_lead, [label])
        self.list.addItem(f"{a:.2f}s–{b:.2f}s | {target_lead} | {label}")

        # Visueel: interval vastleggen als niet-movable overlay op de gekozen plot
        region = pg.LinearRegionItem([a, b], brush=(100, 180, 255, 60), movable=False)
        if target_lead == "V1-avg":
            self.p_avg.addItem(region)
        else:
            self.plots[target_lead].addItem(region)

    def export_csv(self):
        """
        Exporteer alle toegevoegde labels naar CSV.

        - Vraagt via dialog om een pad/naam (default: labels_export.csv).
        - CSV bevat kolommen: t_start, t_end, lead, labels.
        """
        if not self.labels.items:
            QtWidgets.QMessageBox.information(
                self, "Export", "Geen intervallen om te exporteren."
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV", "labels_export.csv", "CSV (*.csv)"
        )
        if not path:
            return
        self.labels.to_df().to_csv(path, index=False)
        QtWidgets.QMessageBox.information(
            self, "Export", f"Saved {len(self.labels.items)} intervals to:\n{path}"
        )


if __name__ == "__main__":
    # Start Qt-app en toon hoofdvenster
    app = QtWidgets.QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())

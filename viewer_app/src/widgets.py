from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from typing import Optional
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression
import typing as _t

class ClickableRegion(pg.LinearRegionItem):
    """
    A clickable region on a PyQtGraph plot that emits a signal when clicked.

    This subclass of LinearRegionItem is used to represent a labeled region
    (for example, an audio segment) that can respond to mouse clicks.
    """

    # Define a signal that emits the region instance when clicked
    clicked = QtCore.Signal(object)

    def __init__(self, *args, seg_id: Optional[str] = None, **kwargs):
        """
        Initialize a non-movable clickable region.
        """
        super().__init__(*args, **kwargs)

        # Store optional segment ID
        self.seg_id = seg_id

        # Disable dragging; region is static (clickable only)
        self.setMovable(False)

    def mouseClickEvent(self, ev):
        """
        Handle mouse click events on the region.

        Emits the `clicked` signal when the left mouse button is pressed.
        """
        # Check if the left mouse button was clicked
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            # Emit the clicked signal with this region as the argument
            self.clicked.emit(self)

            # Accept the event to stop further propagation
            ev.accept()
        else:
            # Ignore other mouse buttons
            ev.ignore()

class MetadataInlineEditor(QtWidgets.QWidget):
    """
    Inline, bewerkbare opname-metadata met 4 velden (subject_id, microphone_type, sample_rate, location).
    """
    changed = QtCore.Signal(dict)

    def __init__(self, fields: list[str], parent=None):
        super().__init__(parent)
        self._fields = fields
        self._widgets: dict[str, QtWidgets.QWidget] = {}

        box = QtWidgets.QGroupBox("Recording metadata")
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)

        # Subject ID (validator: 001 of P001)
        w_subject = QtWidgets.QLineEdit()
        w_subject.setPlaceholderText("001 of P001")
        w_subject.setValidator(QRegularExpressionValidator(QRegularExpression(r"^(P?\d{3})$")))
        w_subject.editingFinished.connect(self._emit)
        self._widgets["subject_id"] = w_subject
        form.addRow("Subject ID:", w_subject)

        # Microphone type (editable ComboBox met recents)
        w_mic = QtWidgets.QComboBox(); w_mic.setEditable(True)
        w_mic.lineEdit().editingFinished.connect(self._emit)  # type: ignore
        self._widgets["microphone_type"] = w_mic
        form.addRow("Microphone type:", w_mic)

        # Sample Rate
        w_sr = QtWidgets.QSpinBox()
        w_sr.setRange(0, 384000); w_sr.setSingleStep(100)
        w_sr.valueChanged.connect(self._emit)
        self._widgets["sample_rate"] = w_sr
        form.addRow("Sample Rate:", w_sr)

        # Recording Location (editable ComboBox met recents)
        w_loc = QtWidgets.QComboBox(); w_loc.setEditable(True)
        w_loc.lineEdit().editingFinished.connect(self._emit)  # type: ignore
        self._widgets["location"] = w_loc
        form.addRow("Recording location:", w_loc)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(box)

    # -------- public helpers voor recents --------
    def set_recent_mics(self, items: _t.Sequence[str]):
        cb: QtWidgets.QComboBox = _t.cast(QtWidgets.QComboBox, self._widgets["microphone_type"])
        txt = cb.currentText()
        cb.blockSignals(True); cb.clear(); cb.addItems(list(items)); cb.setCurrentText(txt); cb.blockSignals(False)

    def set_recent_locations(self, items: _t.Sequence[str]):
        cb: QtWidgets.QComboBox = _t.cast(QtWidgets.QComboBox, self._widgets["location"])
        txt = cb.currentText()
        cb.blockSignals(True); cb.clear(); cb.addItems(list(items)); cb.setCurrentText(txt); cb.blockSignals(False)

    # -------- bestaande API --------
    def set_values(self, meta: dict | None):
        meta = meta or {}
        # Subject
        self._widgets["subject_id"].blockSignals(True)
        self._widgets["subject_id"].setText(str(meta.get("subject_id", "")))  # type: ignore
        self._widgets["subject_id"].blockSignals(False)

        # Mic (ComboBox)
        cb_m: QtWidgets.QComboBox = _t.cast(QtWidgets.QComboBox, self._widgets["microphone_type"])
        cb_m.blockSignals(True); cb_m.setCurrentText(str(meta.get("microphone_type", ""))); cb_m.blockSignals(False)

        # SR
        sr = meta.get("sample_rate", "")
        try: sr = int(sr) if str(sr).strip() != "" else 0
        except Exception: sr = 0
        self._widgets["sample_rate"].blockSignals(True)
        self._widgets["sample_rate"].setValue(sr)  # type: ignore
        self._widgets["sample_rate"].blockSignals(False)

       # Loc (ComboBox)
        cb_l: QtWidgets.QComboBox = _t.cast(QtWidgets.QComboBox, self._widgets["location"])
        current_items = [cb_l.itemText(i) for i in range(cb_l.count())]
        current_value = str(meta.get("location", ""))

        # als de huidige locatie in de lijst staat, selecteer die;
        # zo niet, voeg alleen dat item toe (zonder de lijst te wissen)
        cb_l.blockSignals(True)
        if current_value and current_value not in current_items:
            cb_l.addItem(current_value)
        cb_l.setCurrentText(current_value)
        cb_l.blockSignals(False)

    def values(self) -> dict:
        out: dict = {}
        t = self._widgets["subject_id"].text().strip()  # type: ignore
        if t: out["subject_id"] = t
        cb_m: QtWidgets.QComboBox = _t.cast(QtWidgets.QComboBox, self._widgets["microphone_type"])
        m = cb_m.currentText().strip()
        if m: out["microphone_type"] = m
        sr = int(self._widgets["sample_rate"].value())  # type: ignore
        if sr > 0: out["sample_rate"] = sr
        cb_l: QtWidgets.QComboBox = _t.cast(QtWidgets.QComboBox, self._widgets["location"])
        loc = cb_l.currentText().strip()
        if loc: out["location"] = loc
        return out

    def _emit(self):
        self.changed.emit(self.values())

class LabelBar(QtWidgets.QWidget):
    toggled = QtCore.Signal(str, bool)  # (label, checked)

    def __init__(self, labels: dict[str, str], parent=None):
        super().__init__(parent)
        self._labels: dict[str, str] = dict(labels)
        self._buttons: dict[str, QtWidgets.QPushButton] = {}
        self._lay = QtWidgets.QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(6)
        self._build_buttons()

    def _build_buttons(self):
        # layout leegmaken
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self._buttons.clear()

        # knoppen opnieuw opbouwen
        for i, (lbl, tip) in enumerate(self._labels.items(), start=1):
            btn = QtWidgets.QPushButton(lbl)
            btn.setCheckable(True)
            if tip:
                btn.setToolTip(tip)
            # let op: bind lbl als default arg om late-binding te vermijden
            btn.toggled.connect(lambda checked, L=lbl: self.toggled.emit(L, checked))
            if i <= 9:
                btn.setShortcut(str(i))  # 1..9
            self._lay.addWidget(btn)
            self._buttons[lbl] = btn

        self._lay.addStretch(1)

    def set_labels(self, labels: dict[str, str]):
        """Update de labelset zonder de widget opnieuw te initialiseren."""
        self._labels = dict(labels)
        self._build_buttons()

    def reflect_segment(self, labels_on_segment: list[str] | None):
        labels_on_segment = labels_on_segment or []
        # zet checked-states zonder signals te triggeren
        for lbl, btn in self._buttons.items():
            btn.blockSignals(True)
            btn.setChecked(lbl in labels_on_segment)
            btn.blockSignals(False)

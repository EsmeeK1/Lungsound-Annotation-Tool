from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression
import typing as _t
import pyqtgraph as pg
from typing import Optional


class ClickableRegion(pg.LinearRegionItem):
    """
    Clickable region on a plot. Emits a signal when the user clicks it.

    UI map:
      region: visual span on the time axis
      clicked: signal that passes the region instance

    Notes:
      - Region is not movable, only clickable.
    """

    clicked = QtCore.Signal(object)  # emits the region instance

    def __init__(self, *args, seg_id: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seg_id = seg_id
        self.setMovable(False)  # keep static so clicks are unambiguous

    def mouseClickEvent(self, ev):
        """
        Emit 'clicked' when the left mouse button is pressed.
        """
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(self)
            ev.accept()   # stop further handling
        else:
            ev.ignore()


class MetadataInlineEditor(QtWidgets.QWidget):
    """
    Inline editor for recording metadata.

    UI map:
      group box: "Recording metadata"
        - Subject ID: QLineEdit, accepts "001" or "P001"
        - Microphone type: editable QComboBox (supports recent items)
        - Sample Rate: QSpinBox in Hz
        - Recording location: editable QComboBox (supports recent items)

    Signals:
      - changed(dict): emitted whenever a field is edited, returns current values
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

        # subject id, accepts "001" or "P001"
        w_subject = QtWidgets.QLineEdit()
        w_subject.setPlaceholderText("001 or P001")
        w_subject.setValidator(QRegularExpressionValidator(QRegularExpression(r"^(P?\d{3})$")))
        w_subject.editingFinished.connect(self._emit)
        self._widgets["subject_id"] = w_subject
        form.addRow("Subject ID:", w_subject)

        # microphone type, editable combobox so users can type or pick
        w_mic = QtWidgets.QComboBox()
        w_mic.setEditable(True)
        w_mic.lineEdit().editingFinished.connect(self._emit)  # type: ignore
        self._widgets["microphone_type"] = w_mic
        form.addRow("Microphone type:", w_mic)

        # sample rate in Hz
        w_sr = QtWidgets.QSpinBox()
        w_sr.setRange(0, 384000)
        w_sr.setSingleStep(100)
        w_sr.valueChanged.connect(self._emit)
        self._widgets["sample_rate"] = w_sr
        form.addRow("Sample Rate:", w_sr)

        # recording location, editable to allow custom entries
        w_loc = QtWidgets.QComboBox()
        w_loc.setEditable(True)
        w_loc.lineEdit().editingFinished.connect(self._emit)  # type: ignore
        self._widgets["location"] = w_loc
        form.addRow("Recording location:", w_loc)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(box)

    # recent-items helpers

    def set_recent_mics(self, items: _t.Sequence[str]):
        """
        Replace microphone recent items while keeping the current text.
        """
        cb: QtWidgets.QComboBox = _t.cast(QtWidgets.QComboBox, self._widgets["microphone_type"])
        txt = cb.currentText()
        cb.blockSignals(True)
        cb.clear()
        cb.addItems(list(items))
        cb.setCurrentText(txt)
        cb.blockSignals(False)

    def set_recent_locations(self, items: _t.Sequence[str]):
        """
        Replace location recent items while keeping the current text.
        """
        cb: QtWidgets.QComboBox = _t.cast(QtWidgets.QComboBox, self._widgets["location"])
        txt = cb.currentText()
        cb.blockSignals(True)
        cb.clear()
        cb.addItems(list(items))
        cb.setCurrentText(txt)
        cb.blockSignals(False)

    # existing api

    def set_values(self, meta: dict | None):
        """
        Fill the widgets from a metadata dict.
        Only sets fields that are present.
        """
        meta = meta or {}

        # subject id
        self._widgets["subject_id"].blockSignals(True)
        self._widgets["subject_id"].setText(str(meta.get("subject_id", "")))  # type: ignore
        self._widgets["subject_id"].blockSignals(False)

        # microphone type
        cb_m: QtWidgets.QComboBox = _t.cast(QtWidgets.QComboBox, self._widgets["microphone_type"])
        cb_m.blockSignals(True)
        cb_m.setCurrentText(str(meta.get("microphone_type", "")))
        cb_m.blockSignals(False)

        # sample rate, coerce to int and use 0 for empty/invalid
        sr = meta.get("sample_rate", "")
        try:
            sr = int(sr) if str(sr).strip() != "" else 0
        except Exception:
            sr = 0
        self._widgets["sample_rate"].blockSignals(True)
        self._widgets["sample_rate"].setValue(sr)  # type: ignore
        self._widgets["sample_rate"].blockSignals(False)

        # recording location
        cb_l: QtWidgets.QComboBox = _t.cast(QtWidgets.QComboBox, self._widgets["location"])
        current_items = [cb_l.itemText(i) for i in range(cb_l.count())]
        current_value = str(meta.get("location", ""))

        # if current value is not in the list, add it without clearing the list
        cb_l.blockSignals(True)
        if current_value and current_value not in current_items:
            cb_l.addItem(current_value)
        cb_l.setCurrentText(current_value)
        cb_l.blockSignals(False)

    def values(self) -> dict:
        """
        Collect current values from the editor.
        Empty fields are omitted.
        """
        out: dict = {}

        # subject id
        t = self._widgets["subject_id"].text().strip()  # type: ignore
        if t:
            out["subject_id"] = t

        # microphone type
        cb_m: QtWidgets.QComboBox = _t.cast(QtWidgets.QComboBox, self._widgets["microphone_type"])
        m = cb_m.currentText().strip()
        if m:
            out["microphone_type"] = m

        # sample rate
        sr = int(self._widgets["sample_rate"].value())  # type: ignore
        if sr > 0:
            out["sample_rate"] = sr

        # recording location
        cb_l: QtWidgets.QComboBox = _t.cast(QtWidgets.QComboBox, self._widgets["location"])
        loc = cb_l.currentText().strip()
        if loc:
            out["location"] = loc

        return out

    def _emit(self):
        """
        Emit 'changed' with the current values.
        """
        self.changed.emit(self.values())


class LabelBar(QtWidgets.QWidget):
    """
    Horizontal row of toggle buttons for labels.

    UI map:
      buttons: one per label, togglable
      shortcut: digits 1..9 for the first nine labels
      toggled(label, checked): emitted when a button is toggled
    """

    toggled = QtCore.Signal(str, bool)  # (label, checked)

    def __init__(self, labels: dict[str, str], parent=None):
        super().__init__(parent)
        self._labels: dict[str, str] = dict(labels)  # label -> tooltip
        self._buttons: dict[str, QtWidgets.QPushButton] = {}
        self._lay = QtWidgets.QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(6)
        self._build_buttons()

    def _build_buttons(self):
        """
        Rebuild all buttons from the current label set.
        Keeps layout clean, then adds buttons in order.
        """
        # clear layout
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self._buttons.clear()

        # add one button per label
        for i, (lbl, tip) in enumerate(self._labels.items(), start=1):
            btn = QtWidgets.QPushButton(lbl)
            btn.setCheckable(True)
            if tip:
                btn.setToolTip(tip)

            # capture lbl as default argument to avoid late binding
            btn.toggled.connect(lambda checked, L=lbl: self.toggled.emit(L, checked))

            # numeric shortcuts for quick access
            if i <= 9:
                btn.setShortcut(str(i))  # 1..9

            self._lay.addWidget(btn)
            self._buttons[lbl] = btn

        self._lay.addStretch(1)

    def set_labels(self, labels: dict[str, str]):
        """
        Update the label set at runtime.
        """
        self._labels = dict(labels)
        self._build_buttons()

    def reflect_segment(self, labels_on_segment: list[str] | None):
        """
        Update button checked states to reflect the given segment labels.
        Signals are blocked to avoid feedback loops.
        """
        labels_on_segment = labels_on_segment or []
        for lbl, btn in self._buttons.items():
            btn.blockSignals(True)
            btn.setChecked(lbl in labels_on_segment)
            btn.blockSignals(False)

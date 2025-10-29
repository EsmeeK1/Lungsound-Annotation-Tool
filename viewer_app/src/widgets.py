from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from typing import Optional, Dict, Any
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
    Inline, bewerkbare opname-metadata met precies 6 velden.
    """
    changed = QtCore.Signal(dict)

    def __init__(self, fields: list[str], parent=None):
        super().__init__(parent)
        self._fields = fields
        self._widgets: Dict[str, QtWidgets.QWidget] = {}

        box = QtWidgets.QGroupBox("Recording metadata")
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)

        # Subject ID
        w_subject = QtWidgets.QLineEdit()
        w_subject.setPlaceholderText("e.g. P0123")
        w_subject.editingFinished.connect(self._emit)
        self._widgets["subject_id"] = w_subject
        form.addRow("Subject ID:", w_subject)

        # Microphone type
        w_mic = QtWidgets.QLineEdit()
        w_mic.setPlaceholderText("e.g. Plastic Membrane")
        w_mic.editingFinished.connect(self._emit)
        self._widgets["microphone_type"] = w_mic
        form.addRow("Microphone type:", w_mic)

        # Sample Rate
        w_sr = QtWidgets.QSpinBox()
        w_sr.setRange(0, 384000); w_sr.setSingleStep(100)
        w_sr.valueChanged.connect(self._emit)
        self._widgets["sample_rate"] = w_sr
        form.addRow("Sample Rate:", w_sr)

        # Recording Location
        w_loc = QtWidgets.QLineEdit()
        w_loc.setPlaceholderText("e.g. LLL, RUL")
        w_loc.editingFinished.connect(self._emit)
        self._widgets["location"] = w_loc
        form.addRow("Recording Location:", w_loc)

        # Gender
        w_gender = QtWidgets.QComboBox()
        w_gender.addItems(["", "Male", "Female", "Unknown"])
        w_gender.currentIndexChanged.connect(self._emit)
        self._widgets["gender"] = w_gender
        form.addRow("Gender:", w_gender)

        # Age
        w_age = QtWidgets.QSpinBox()
        w_age.setRange(0, 120)
        w_age.setSpecialValueText("")  # toont leeg bij 0
        w_age.valueChanged.connect(self._emit)
        self._widgets["age"] = w_age
        form.addRow("Age:", w_age)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(box)

    def set_values(self, meta: Dict[str, Any] | None):
        meta = meta or {}
        # Subject ID
        self._widgets["subject_id"].blockSignals(True)
        self._widgets["subject_id"].setText(str(meta.get("subject_id", ""))) # type: ignore
        self._widgets["subject_id"].blockSignals(False)

        # Mic
        self._widgets["microphone_type"].blockSignals(True)
        self._widgets["microphone_type"].setText(str(meta.get("microphone_type", ""))) # type: ignore
        self._widgets["microphone_type"].blockSignals(False)

        # Sample Rate
        sr = meta.get("sample_rate", "")
        try: sr = int(sr) if str(sr).strip() != "" else 0
        except Exception: sr = 0
        self._widgets["sample_rate"].blockSignals(True)
        self._widgets["sample_rate"].setValue(sr) # type: ignore
        self._widgets["sample_rate"].blockSignals(False)

        # Location
        self._widgets["location"].blockSignals(True)
        self._widgets["location"].setText(str(meta.get("location", ""))) # type: ignore
        self._widgets["location"].blockSignals(False)

        # Gender
        g = str(meta.get("gender", "")).strip()
        cb: QtWidgets.QComboBox = self._widgets["gender"]  # type: ignore
        self._widgets["gender"].blockSignals(True)
        idx = cb.findText(g) if g else 0
        cb.setCurrentIndex(idx if idx >= 0 else 0)
        self._widgets["gender"].blockSignals(False)

        # Age
        a = meta.get("age", 0)
        try: a = int(a) if str(a).strip() != "" else 0
        except Exception: a = 0
        self._widgets["age"].blockSignals(True)
        self._widgets["age"].setValue(a) # type: ignore
        self._widgets["age"].blockSignals(False)

    def values(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # Subject ID
        t = self._widgets["subject_id"].text().strip()  # type: ignore
        if t: out["subject_id"] = t
        # Mic
        m = self._widgets["microphone_type"].text().strip()  # type: ignore
        if m: out["microphone_type"] = m
        # Sample Rate
        sr = int(self._widgets["sample_rate"].value())       # type: ignore
        if sr > 0: out["sample_rate"] = sr
        # Location
        loc = self._widgets["location"].text().strip()       # type: ignore
        if loc: out["location"] = loc
        # Gender
        g = self._widgets["gender"].currentText().strip()    # type: ignore
        if g: out["gender"] = g
        # Age
        a = int(self._widgets["age"].value())                # type: ignore
        if a > 0: out["age"] = a
        return out

    def _emit(self):
        self.changed.emit(self.values())
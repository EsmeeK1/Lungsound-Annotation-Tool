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
    Inline editor for generic recording metadata.

    Fields:
      - environment
      - notes
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

        # Environment
        w_env = QtWidgets.QComboBox()
        w_env.setEditable(True)
        w_env.addItems([
            "",
            "indoor",
            "outdoor",
            "quiet",
            "noisy",
            "traffic",
            "home",
            "lab",
            "clinical",
            "other",
        ])
        w_env.lineEdit().editingFinished.connect(self._emit)  # type: ignore
        w_env.currentTextChanged.connect(lambda *_: self._emit())
        self._widgets["environment"] = w_env
        form.addRow("Environment:", w_env)

        # Notes
        w_notes = QtWidgets.QLineEdit()
        w_notes.setPlaceholderText("Optional notes")
        w_notes.editingFinished.connect(self._emit)
        self._widgets["notes"] = w_notes
        form.addRow("Notes:", w_notes)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(box)

    def set_recent_mics(self, items: _t.Sequence[str]):
        """
        Kept for backward compatibility with mainwindow.py.
        No longer used.
        """
        return

    def set_recent_locations(self, items: _t.Sequence[str]):
        """
        Reuse the existing location-recents pipeline for environment suggestions.
        """
        if "environment" not in self._widgets:
            return

        cb: QtWidgets.QComboBox = _t.cast(QtWidgets.QComboBox, self._widgets["environment"])
        txt = cb.currentText()

        cb.blockSignals(True)

        base_items = [
            "",
            "indoor",
            "outdoor",
            "quiet",
            "noisy",
            "traffic",
            "home",
            "lab",
            "clinical",
            "other",
        ]

        merged = list(dict.fromkeys(base_items + list(items or [])))

        cb.clear()
        cb.addItems(merged)
        cb.setCurrentText(txt)

        cb.blockSignals(False)

    def set_values(self, meta: dict | None):
        """
        Fill widgets from metadata.

        Supports old keys for migration:
        - location -> environment
        """
        meta = dict(meta or {})

        # Backward compatibility: old location can become environment.
        if "environment" not in meta and meta.get("location"):
            meta["environment"] = meta.get("location")

        for key, widget in self._widgets.items():
            value = meta.get(key, "")

            widget.blockSignals(True)

            if isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(str(value or ""))

            elif isinstance(widget, QtWidgets.QComboBox):
                text = str(value or "")
                current_items = [widget.itemText(i) for i in range(widget.count())]
                if text and text not in current_items:
                    widget.addItem(text)
                widget.setCurrentText(text)

            widget.blockSignals(False)

    def values(self) -> dict:
        """
        Collect current values.

        Empty fields are omitted.
        """
        out: dict = {}

        for key, widget in self._widgets.items():
            if isinstance(widget, QtWidgets.QLineEdit):
                value = widget.text().strip()
                if value:
                    out[key] = value

            elif isinstance(widget, QtWidgets.QComboBox):
                value = widget.currentText().strip()
                if value:
                    out[key] = value

        return out

    def _emit(self):
        """
        Emit changed metadata.
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

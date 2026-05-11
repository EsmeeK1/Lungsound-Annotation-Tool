from PySide6 import QtWidgets
from .config import TIME_SNAP
from typing import Tuple


class StartDialog(QtWidgets.QDialog):
    """
    Start dialog for choosing a dataset folder.

    UI Map:
      Dataset Selection
        - btn_choose: opens folder picker
        - le_root: shows selected folder path (read-only)

      Dialog Buttons
        - _btn_ok: OK, enabled after folder is chosen
        - Cancel: closes without loading
    """

    def __init__(self, parent=None):
        """Build the layout and wire folder selection."""
        super().__init__(parent)
        self.setWindowTitle("Select dataset folder")
        self.resize(480, 140)

        v = QtWidgets.QVBoxLayout(self)

        form_top = QtWidgets.QFormLayout()

        self.btn_choose = QtWidgets.QPushButton("Choose folder…")
        self.le_root = QtWidgets.QLineEdit()
        self.le_root.setReadOnly(True)

        form_top.addRow("Dataset folder:", self.btn_choose)
        form_top.addRow("Selected path:", self.le_root)

        v.addLayout(form_top)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        v.addWidget(btns)

        self._btn_ok = btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self._btn_ok.setEnabled(False)

        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        self.root = ""
        self.btn_choose.clicked.connect(self.pick_folder)

    def pick_folder(self):
        """
        Let the user select a dataset root folder and update the path field.
        """
        dlg = QtWidgets.QFileDialog(self, "Select root folder")
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


class AutoSegmentDialog(QtWidgets.QDialog):
    """
    Dialog to configure automatic segmentation.

    UI Map:
      Segment Parameters
        - len_s: segment length in seconds
        - ovl_s: overlap in seconds
        - chk_replace: replace existing segments

      Label Choice
        - combo_label: label applied to all created segments

      Dialog Buttons
        - OK / Cancel

    Methods:
      - on_accept(): validates values and accepts dialog
      - values(): returns tuple (length_s, overlap_s, replace, label)
    """

    def __init__(
        self,
        parent=None,
        default_len=3.00,
        default_overlap=0.00,
        default_replace=False,
        label_options=None,
        default_label=None,
    ):
        """Build the layout and set default values."""
        super().__init__(parent)
        self.setWindowTitle("Auto segmentation")
        self.setModal(True)

        v = QtWidgets.QVBoxLayout(self)

        # Group: Segment Parameters
        form = QtWidgets.QFormLayout()

        # Segment length (seconds)
        self.len_s = QtWidgets.QDoubleSpinBox()
        self.len_s.setDecimals(2)
        self.len_s.setSingleStep(TIME_SNAP)
        self.len_s.setRange(TIME_SNAP, 600.0)
        self.len_s.setValue(default_len)

        # Overlap (seconds)
        self.ovl_s = QtWidgets.QDoubleSpinBox()
        self.ovl_s.setDecimals(2)
        self.ovl_s.setSingleStep(TIME_SNAP)
        self.ovl_s.setRange(0.0, 600.0)
        self.ovl_s.setValue(default_overlap)

        # Replace toggle
        self.chk_replace = QtWidgets.QCheckBox("Replace existing segments")
        self.chk_replace.setChecked(default_replace)

        form.addRow("Segment length (s):", self.len_s)
        form.addRow("Overlap between segments (s):", self.ovl_s)

        # Group: Label Choice
        self.combo_label = QtWidgets.QComboBox()
        if label_options:
            self.combo_label.addItems(label_options)
        if default_label and default_label in (label_options or []):
            self.combo_label.setCurrentText(default_label)
        form.addRow("Label for all segments:", self.combo_label)

        v.addLayout(form)
        v.addWidget(self.chk_replace)

        # Group: Dialog Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        v.addWidget(btns)

        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.on_accept)

        self._ok = False  # Tracks if valid settings were confirmed

    def on_accept(self):
        """
        Validate inputs and accept if they are logical.

        Rules:
          - length > 0
          - 0 ≤ overlap < length
        """
        L = float(self.len_s.value())
        O = float(self.ovl_s.value())

        if L <= 0 or O < 0 or O >= L:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid parameters",
                "Ensure that: length > 0 and 0 ≤ overlap < length.",
            )
            return

        self._ok = True
        self.accept()

    def values(self) -> Tuple[float, float, bool, str]:
        """
        Get the chosen segmentation settings.

        Returns:
            tuple: (length_seconds, overlap_seconds, replace_existing, label_text)
        """
        return (
            float(self.len_s.value()),
            float(self.ovl_s.value()),
            bool(self.chk_replace.isChecked()),
            self.combo_label.currentText(),
        )

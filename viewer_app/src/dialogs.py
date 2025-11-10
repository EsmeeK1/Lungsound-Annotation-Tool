from PySide6 import QtWidgets, QtCore, QtGui
from .config import TIME_SNAP
from typing import Dict, Tuple


class StartDialog(QtWidgets.QDialog):
    """
    Start dialog for choosing a dataset folder and optional metadata.

    UI Map:
      Dataset Selection
        - btn_choose: opens folder picker
        - le_root: shows selected folder path (read-only)

      Metadata (optional)
        - subject_id: subject identifier
        - mic_type: microphone type (editable combobox)
        - sr_spin: sample rate in Hz
        - location: recording location (editable combobox)
        - btn_info: small help button ("?")

      Dialog Buttons
        - _btn_ok: OK, enabled after folder is chosen
        - Cancel: closes without saving

    Returns:
      - get_meta(): dict with any filled metadata fields
      - root: string path to selected folder
    """

    def __init__(self, parent=None):
        """Build the layout and wire basic interactions."""
        super().__init__(parent)
        self.setWindowTitle("Select dataset and metadata")
        self.resize(480, 300)

        # Root layout for the dialog
        v = QtWidgets.QVBoxLayout(self)

        # Group: Dataset Selection
        form_top = QtWidgets.QFormLayout()
        self.btn_choose = QtWidgets.QPushButton("Choose folder…")     # Click to pick a folder
        self.le_root = QtWidgets.QLineEdit()                           # Displays the chosen path
        self.le_root.setReadOnly(True)
        form_top.addRow("Dataset folder:", self.btn_choose)
        form_top.addRow("Selected path:", self.le_root)
        v.addLayout(form_top)

        # Group: Metadata (optional)
        grp = QtWidgets.QGroupBox()
        grp_layout = QtWidgets.QVBoxLayout(grp)

        # Metadata header with inline help
        header = QtWidgets.QHBoxLayout()
        lbl_hdr = QtWidgets.QLabel("Metadata (optional)")
        lbl_hdr.setStyleSheet("font-weight: 600;")

        self.btn_info = QtWidgets.QToolButton()                        # Shows short help text
        self.btn_info.setText("?")
        self.btn_info.setToolTip("Show help for metadata fields")
        self.btn_info.setFixedWidth(24)
        self.btn_info.clicked.connect(self._show_meta_info)

        header.addWidget(lbl_hdr)
        header.addStretch(1)
        header.addWidget(self.btn_info)
        grp_layout.addLayout(header)

        # Metadata form (fixed order)
        form_meta = QtWidgets.QFormLayout()

        # Subject ID
        self.subject_id = QtWidgets.QLineEdit()
        self.subject_id.setPlaceholderText("001 or P001")
        self.subject_id.setValidator(
            QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression(r"^(P?\d{3})$")
            )
        )
        form_meta.addRow("Subject ID:", self.subject_id)

        # Microphone type (editable combobox)
        self.mic_type = QtWidgets.QComboBox()
        self.mic_type.setEditable(True)
        self.mic_type.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.InsertAlphabetically)
        self.mic_type.setPlaceholderText("e.g. Plastic Membrane")
        form_meta.addRow("Microphone type:", self.mic_type)

        # Sample Rate
        self.sr_spin = QtWidgets.QSpinBox()
        self.sr_spin.setRange(0, 384000)
        self.sr_spin.setSingleStep(100)
        self.sr_spin.setSpecialValueText("")  # empty text when value is 0
        form_meta.addRow("Sample Rate:", self.sr_spin)

        # Recording location (editable combobox)
        self.location = QtWidgets.QComboBox()
        self.location.setEditable(True)
        self.location.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.InsertAlphabetically)
        self.location.setPlaceholderText("e.g. LLL, RUL")
        form_meta.addRow("Recording location:", self.location)

        # Attach metadata form
        grp_layout.addLayout(form_meta)
        v.addWidget(grp)

        # Group: Dialog Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        v.addWidget(btns)

        # OK button is disabled until a folder is chosen
        self._btn_ok = btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self._btn_ok.setEnabled(False)

        # Wire buttons
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        # Initial state
        self.root = ""  # folder path
        self.btn_choose.clicked.connect(self.pick_folder)

    def _show_meta_info(self):
        """
        Show a short explanation of each metadata field.
        """
        txt = (
            "Metadata are stored with each recording and used as defaults:\n\n"
            "• Subject ID – unique participant identifier\n"
            "• Microphone type – sensor or stethoscope used\n"
            "• Sample Rate – sampling frequency (Hz)\n"
            "• Recording location – chest area or body position\n"
        )
        QtWidgets.QMessageBox.information(self, "Metadata Info", txt)

    def pick_folder(self):
        """
        Let the user select a dataset root folder and update the path field.
        """
        dlg = QtWidgets.QFileDialog(self, "Select root folder")
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        dlg.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly, True)

        # Use a non-native dialog where needed to avoid platform issues
        try:
            dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        except Exception:
            pass

        # Start in the last chosen directory if available
        if getattr(self, "root", ""):
            dlg.setDirectory(self.root)

        # Update selection and enable OK if a folder is chosen
        if dlg.exec():
            sel = dlg.selectedFiles()
            if sel:
                self.root = sel[0]
                self.le_root.setText(self.root)
                self._btn_ok.setEnabled(True)

    def get_meta(self) -> Dict[str, object]:
        """
        Collect metadata from the fields.

        Returns:
            dict: Keys may include 'subject_id', 'microphone_type',
                  'location', and 'sample_rate' if filled.
        """
        meta: Dict[str, object] = {}
        sid = self.subject_id.text().strip()
        mic = self.mic_type.currentText().strip()
        loc = self.location.currentText().strip()
        sr = int(self.sr_spin.value())

        if sid:
            meta["subject_id"] = sid
        if mic:
            meta["microphone_type"] = mic
        if loc:
            meta["location"] = loc
        if sr > 0:
            meta["sample_rate"] = sr
        return meta


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

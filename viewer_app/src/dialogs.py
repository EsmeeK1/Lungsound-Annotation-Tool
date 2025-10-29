from PySide6 import QtWidgets, QtCore
from .config import TIME_SNAP
from .utils import labels_dataset_path
from typing import List, Dict, Optional, Tuple

class StartDialog(QtWidgets.QDialog):
    """
    Start dialog for selecting a dataset folder and optional session metadata.

    This dialog allows the user to:
      - Select the root folder containing the dataset.
      - Optionally enter metadata such as gender, age, and recording location.

    Attributes:
        btn_choose (QtWidgets.QPushButton): Button to open folder picker.
        le_root (QtWidgets.QLineEdit): Read-only field showing chosen folder path.
        gender (QtWidgets.QComboBox): Dropdown for selecting gender.
        age (QtWidgets.QSpinBox): Field for entering age (0–120).
        location (QtWidgets.QLineEdit): Field for entering recording location.
        btn_info (QtWidgets.QToolButton): Info button that shows metadata field explanations.
        _btn_ok (QtWidgets.QPushButton): OK button (enabled only after a folder is chosen).
        root (str): Path of the selected dataset folder.
    """

    def __init__(self, parent=None):
        """Initialize the dialog UI layout and components."""
        super().__init__(parent)
        self.setWindowTitle("Select dataset and metadata")
        self.resize(480, 300)

        # Create the main vertical layout for the dialog
        v = QtWidgets.QVBoxLayout(self)

        # ------- Top: Dataset folder selection -------
        form_top = QtWidgets.QFormLayout()
        self.btn_choose = QtWidgets.QPushButton("Choose folder…")  # Button to open folder picker
        self.le_root = QtWidgets.QLineEdit()                        # Display chosen folder path
        self.le_root.setReadOnly(True)                              # Prevent user editing directly
        form_top.addRow("Dataset folder:", self.btn_choose)
        form_top.addRow("Selected path:", self.le_root)
        v.addLayout(form_top)

        # ------- Metadata group section -------
        grp = QtWidgets.QGroupBox()
        grp_layout = QtWidgets.QVBoxLayout(grp)

        # Header in the group box with title and info button
        header = QtWidgets.QHBoxLayout()
        lbl_hdr = QtWidgets.QLabel("Metadata (optional)")
        lbl_hdr.setStyleSheet("font-weight: 600;")

        self.btn_info = QtWidgets.QToolButton()      # Small '?' button for help
        self.btn_info.setText("?")
        self.btn_info.setToolTip("Show help for metadata fields")
        self.btn_info.setFixedWidth(24)
        self.btn_info.clicked.connect(self._show_meta_info)

        header.addWidget(lbl_hdr)
        header.addStretch(1)
        header.addWidget(self.btn_info)
        grp_layout.addLayout(header)

        # Form layout for metadata fields
        form_meta = QtWidgets.QFormLayout()

        # Form layout voor metadata (alleen 6 velden in vaste volgorde)
        form_meta = QtWidgets.QFormLayout()

        # 1) Subject ID
        self.subject_id = QtWidgets.QLineEdit()
        self.subject_id.setPlaceholderText("e.g. P0123")
        form_meta.addRow("Subject ID:", self.subject_id)

        # 2) Microphone type
        self.mic_type = QtWidgets.QLineEdit()
        self.mic_type.setPlaceholderText("e.g. Plastic Membrane")
        form_meta.addRow("Microphone type:", self.mic_type)

        # 3) Sample Rate
        self.sr_spin = QtWidgets.QSpinBox()
        self.sr_spin.setRange(0, 384000)
        self.sr_spin.setSingleStep(100)
        self.sr_spin.setSpecialValueText("")  # leeg bij 0
        form_meta.addRow("Sample Rate:", self.sr_spin)

        # 4) Recording location
        self.location = QtWidgets.QLineEdit()
        self.location.setPlaceholderText("e.g. LLL, RUL")
        form_meta.addRow("Recording location:", self.location)

        # 5) Gender
        self.gender = QtWidgets.QComboBox()
        self.gender.addItems(["", "Male", "Female", "Unknown"])
        form_meta.addRow("Gender:", self.gender)

        # 6) Age
        self.age = QtWidgets.QSpinBox()
        self.age.setRange(0, 120)
        self.age.setSpecialValueText("")  # leeg bij 0
        form_meta.addRow("Age:", self.age)

        # Voeg het formulier toe
        grp_layout.addLayout(form_meta)
        v.addWidget(grp)

        # ------- Dialog buttons (OK and Cancel) -------
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        v.addWidget(btns)

        # Get reference to OK button and disable it until a folder is selected
        self._btn_ok = btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self._btn_ok.setEnabled(False)

        # Connect dialog actions
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        # ------- Connect folder picker button -------
        self.root = ""  # Start with no selected folder
        self.btn_choose.clicked.connect(self.pick_folder)

    def _show_meta_info(self):
        """
        Show a short popup message explaining the metadata fields.

        This helps users understand which metadata are optional
        and how they can be used in analysis.
        """
        txt = (
            "Metadata fields are stored with each recording and used as defaults for new files:\n\n"
            "• Subject ID – unique identifier for the participant\n"
            "• Microphone type – sensor or stethoscope used\n"
            "• Sample Rate – recording sampling frequency (Hz)\n"
            "• Recording location – chest area or body position\n"
            "• Gender – patient gender\n"
            "• Age – patient age in years"
        )
        QtWidgets.QMessageBox.information(self, "Metadata Info", txt)

    def pick_folder(self):
        """
        Open a directory selection dialog and update the chosen path.

        This method opens a folder picker dialog (non-native fallback),
        updates the path display field, and enables the OK button when a valid
        folder is selected.
        """
        dlg = QtWidgets.QFileDialog(self, "Select root folder")
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        dlg.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly, True)

        # Try to use a robust fallback dialog (avoids native bugs on some systems)
        try:
            dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        except Exception:
            pass

        # Start in the last selected directory if available
        if getattr(self, "root", ""):
            dlg.setDirectory(self.root)

        # Execute the dialog and get the selected folder
        if dlg.exec():
            sel = dlg.selectedFiles()
            if sel:
                self.root = sel[0]
                self.le_root.setText(self.root)
                self._btn_ok.setEnabled(True)  # Enable OK once a valid folder is chosen

    def get_meta(self) -> Dict[str, object]:
        """
        Collect and return metadata entered by the user.

        Returns:
            dict: A dictionary with any filled metadata fields.
                  Example: {"gender": "Male", "age": 32, "location": "Basal left"}
        """
        meta = {}

        # Get current field values
        g = self.gender.currentText().strip()
        a = self.age.value()
        loc = self.location.text().strip()
        sid = self.subject_id.text().strip()
        mic = self.mic_type.text().strip()
        sr  = int(self.sr_spin.value())

        # Add non-empty fields to metadata dictionary
        if g:
            meta["gender"] = g
        if a > 0:
            meta["age"] = int(a)
        if loc:
            meta["location"] = loc

        if sid: meta["subject_id"] = sid
        if mic: meta["microphone_type"] = mic
        if sr > 0: meta["sample_rate"] = sr
        return meta

class AutoSegmentDialog(QtWidgets.QDialog):
    """
    Dialog for automatic segmentation settings.

    This dialog allows the user to configure how audio segments should be
    automatically divided — specifying segment length, overlap, replacement
    behavior, and a label for all created segments.

    Attributes:
        len_s (QtWidgets.QDoubleSpinBox): Input for segment length in seconds.
        ovl_s (QtWidgets.QDoubleSpinBox): Input for overlap between segments in seconds.
        chk_replace (QtWidgets.QCheckBox): Option to replace existing segments.
        combo_label (QtWidgets.QComboBox): Dropdown for selecting a label for new segments.
        _ok (bool): Indicates whether valid settings were confirmed.
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
        """
        Initialize the auto-segmentation configuration dialog.

        Args:
            parent (Optional[QWidget]): Parent widget.
            default_len (float): Default segment length in seconds.
            default_overlap (float): Default overlap in seconds.
            default_replace (bool): Whether to replace existing segments by default.
            label_options (Optional[List[str]]): List of available label names.
            default_label (Optional[str]): Default selected label, if present in options.
        """
        super().__init__(parent)
        self.setWindowTitle("Auto segmentation")
        self.setModal(True)  # Prevent interaction with other windows while open

        # Create the main layout for the dialog
        v = QtWidgets.QVBoxLayout(self)

        # ------- Segment length and overlap form -------
        form = QtWidgets.QFormLayout()

        # Segment length (in seconds)
        self.len_s = QtWidgets.QDoubleSpinBox()
        self.len_s.setDecimals(2)              # Display two decimals
        self.len_s.setSingleStep(TIME_SNAP)    # Step size for fine control
        self.len_s.setRange(TIME_SNAP, 600.0)  # Valid range for length
        self.len_s.setValue(default_len)       # Default length value

        # Overlap between segments (in seconds)
        self.ovl_s = QtWidgets.QDoubleSpinBox()
        self.ovl_s.setDecimals(2)
        self.ovl_s.setSingleStep(TIME_SNAP)
        self.ovl_s.setRange(0.0, 600.0)
        self.ovl_s.setValue(default_overlap)

        # Checkbox for replacing existing segments
        self.chk_replace = QtWidgets.QCheckBox("Replace existing segments")
        self.chk_replace.setChecked(default_replace)

        # ------- Label choice dropdown -------
        self.combo_label = QtWidgets.QComboBox()
        if label_options:
            # Populate label options (usually from labels_dataset.json)
            self.combo_label.addItems(label_options)
        if default_label and default_label in (label_options or []):
            # Set default label if provided and valid
            self.combo_label.setCurrentText(default_label)

        # Add all input fields to the form
        form.addRow("Segment length (s):", self.len_s)
        form.addRow("Overlap between segments (s):", self.ovl_s)
        form.addRow("Label for all segments:", self.combo_label)

        # Add form and checkbox to the layout
        v.addLayout(form)
        v.addWidget(self.chk_replace)

        # ------- Dialog buttons (OK / Cancel) -------
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        v.addWidget(btns)

        # Connect dialog buttons
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.on_accept)

        # Track whether valid settings were accepted
        self._ok = False

    def on_accept(self):
        """
        Validate inputs and accept the dialog if parameters are valid.

        Checks that:
          - Segment length > 0
          - Overlap ≥ 0 and < segment length

        If parameters are invalid, shows a warning message instead.
        """
        L = float(self.len_s.value())
        O = float(self.ovl_s.value())

        # Validate logical relationship between length and overlap
        if L <= 0 or O < 0 or O >= L:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid parameters",
                "Ensure that: length > 0 and 0 ≤ overlap < length.",
            )
            return

        # Mark dialog as accepted and close it
        self._ok = True
        self.accept()

    def values(self):
        """
        Retrieve the user-selected segmentation settings.
        """
        return (
            float(self.len_s.value()),             # Segment length in seconds
            float(self.ovl_s.value()),             # Overlap in seconds
            bool(self.chk_replace.isChecked()),    # Replace existing segments?
            self.combo_label.currentText(),        # Label applied to all segments
        )

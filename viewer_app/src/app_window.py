from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from .app_settings import DEFAULT_SR, load_prefs, UserPrefs
from .audio_playback import Player
from .data_models import FileState
from .controllers.ui_builder import UiBuilderMixin
from .controllers.shortcuts import ShortcutsMixin
from .controllers.audio_view import AudioViewMixin
from .controllers.file_io import FileIOMixin
from .controllers.segments import SegmentsMixin
from .controllers.metadata import MetadataMixin
from .controllers.labels import LabelsMixin


class App(
    UiBuilderMixin,
    ShortcutsMixin,
    AudioViewMixin,
    FileIOMixin,
    SegmentsMixin,
    MetadataMixin,
    LabelsMixin,
    QtWidgets.QMainWindow,
):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Annotation Tool")
        self.resize(1200, 820)

        self._custom_labels: Optional[list[str]] = None
        self.player = Player()
        self.prefs: UserPrefs = load_prefs()

        self.root = ""
        self.files: List[str] = []
        self.idx = -1

        self.y_raw: Optional[np.ndarray] = None
        self.sr = DEFAULT_SR
        self.t: Optional[np.ndarray] = None
        self.state: Optional[FileState] = None

        self.overlay_regions: Dict[str, pg.LinearRegionItem] = {}
        self._blocking = False
        self._undo_stack: List[Tuple[Callable[[], None], Callable[[], None]]] = []
        self._redo_stack: List[Tuple[Callable[[], None], Callable[[], None]]] = []

        self._filt_cache = None
        self._filt_params = None
        self.session_meta: Dict[str, object] = {}
        self.play_window: Tuple[float, float] = (0.0, 0.0)

        self.build_ui()
        self.install_shortcuts()
        self.connect_signals()

        self.open_folder_dialog(first=True)
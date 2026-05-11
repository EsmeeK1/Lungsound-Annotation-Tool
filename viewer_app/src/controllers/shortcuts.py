from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtGui

from ..app_settings import TIME_SNAP


class ShortcutsMixin:
    """
    Installs keyboard shortcuts for the main application window.

    This mixin only registers shortcuts. The actual behavior lives in
    the other controllers.
    """

    def install_shortcuts(self: Any) -> None:
        """
        Install all keyboard shortcuts.
        """
        # Playback
        sc_play = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        sc_play.activated.connect(self.toggle_play)

        # File navigation
        sc_next = QtGui.QShortcut(QtGui.QKeySequence("N"), self)
        sc_next.activated.connect(lambda: self.advance(+1))

        sc_prev = QtGui.QShortcut(QtGui.QKeySequence("P"), self)
        sc_prev.activated.connect(lambda: self.advance(-1))

        # Segment editing
        sc_return = QtGui.QShortcut(QtGui.QKeySequence("Return"), self)
        sc_return.activated.connect(self.update_segment)

        sc_enter = QtGui.QShortcut(QtGui.QKeySequence("Enter"), self)
        sc_enter.activated.connect(self.update_segment)

        sc_delete = QtGui.QShortcut(QtGui.QKeySequence("Delete"), self)
        sc_delete.activated.connect(self.delete_selected)

        # View
        sc_reset = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        sc_reset.activated.connect(self.reset_view)

        # Global undo
        sc_undo = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        sc_undo.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        sc_undo.activated.connect(self.undo)

        # Global redo
        sc_redo = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self)
        sc_redo.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        sc_redo.activated.connect(self.redo)

        # Selection region nudging
        for sequence, callback in [
            ("Left", lambda: self.nudge_region(-TIME_SNAP, "move")),
            ("Right", lambda: self.nudge_region(+TIME_SNAP, "move")),
            ("Shift+Left", lambda: self.nudge_region(-TIME_SNAP, "start")),
            ("Shift+Right", lambda: self.nudge_region(+TIME_SNAP, "start")),
            ("Ctrl+Left", lambda: self.nudge_region(-TIME_SNAP, "end")),
            ("Ctrl+Right", lambda: self.nudge_region(+TIME_SNAP, "end")),
        ]:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback)
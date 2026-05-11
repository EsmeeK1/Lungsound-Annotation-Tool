from __future__ import annotations

from typing import Any
import datetime
import json
import os

import numpy as np
import pandas as pd
import soundfile as sf
from PySide6 import QtWidgets

from ..app_settings import METADATA_FIELDS
from ..dialogs import StartDialog
from ..data_models import FileState
from ..file_paths import (
    csv_path_for_root,
    ensure_dir,
    human_relpath,
    json_sidecar_path,
)


class FileIOMixin:
    """
    Handles folder opening, WAV loading, JSON sidecars, file navigation,
    and CSV export.
    """

    # ------------------------------------------------------------------
    # Folder opening / file queue
    # ------------------------------------------------------------------

    def open_folder_dialog(self: Any, first: bool = False) -> None:
        """
        Show the start dialog, collect the dataset root, then load files.

        Metadata is edited inline in the main window.
        """
        self.player.stop()

        dlg = StartDialog(self)
        if not dlg.exec():
            return

        try:
            self.root = dlg.root or ""

            # Metadata is handled inline per file.
            # Session metadata is only used for carry-over values such as
            # the last used environment.
            self.session_meta = {}

            self.load_labels_json()
            self.build_file_queue(self.root)

            #print(f"[DEBUG] open_folder_dialog: root={self.root} files={len(self.files)}")

            if not self.files:
                QtWidgets.QMessageBox.information(
                    self,
                    "Info",
                    "No .wav files found.",
                )
                return

            self._populate_jump_list()
            self.idx = 0
            self.load_current()
            self._after_navigation_changed()

        except Exception as exc:
            import traceback

            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Open error", f"{exc}")

    def build_file_queue(self: Any, root: str) -> None:
        """
        Build a list of .wav files from the root and its subfolders.
        """
        files: list[str] = []

        try:
            root = os.path.normpath(root)

            if not os.path.isdir(root):
                #print(f"[DEBUG] build_file_queue: not a directory -> {root}")
                self.files = []
                return

            # WAV files directly in root.
            for name in sorted(os.listdir(root)):
                path = os.path.join(root, name)
                if os.path.isfile(path) and name.lower().endswith(".wav"):
                    files.append(path)

            # WAV files in subfolders.
            for dirpath, _dirnames, filenames in os.walk(root):
                if os.path.abspath(dirpath) == os.path.abspath(root):
                    continue

                for filename in sorted(filenames):
                    if filename.lower().endswith(".wav"):
                        files.append(os.path.join(dirpath, filename))

        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Read folder",
                f"Could not read folder:\n{root}\n\n{exc}",
            )
            #print(f"[DEBUG] build_file_queue error: {exc!r}")
            files = []

        self.files = files

        #print(f"[DEBUG] build_file_queue: root={root} -> {len(self.files)} wavs")
        #if self.files[:3]:
        #    print("[DEBUG] examples:", *self.files[:3], sep="\n  - ")

    # ------------------------------------------------------------------
    # WAV loading / sidecar state
    # ------------------------------------------------------------------

    def _safe_read_wav(self: Any, path: str):
        """
        Read a WAV file as mono float32.

        Returns:
            (audio, sample_rate) or (None, None) on failure.
        """
        try:
            y, sr = sf.read(path, dtype="float32", always_2d=False)

            if isinstance(y, np.ndarray) and y.ndim == 2:
                y = y.mean(axis=1)

            if y is None or (isinstance(y, np.ndarray) and y.size == 0):
                raise ValueError("Empty audio")

            return y.astype(np.float32, copy=False), int(sr)

        except Exception as exc:
            print(f"[WARN] Skip unreadable WAV: {path} -> {exc}")
            return None, None

    def load_current(self: Any) -> None:
        """
        Load the current WAV file, update UI, and draw plots.
        """
        try:
            #print(f"[DEBUG] load_current: idx={self.idx} total={len(self.files)}")

            if not self.files:
                QtWidgets.QMessageBox.information(
                    self,
                    "Info",
                    "No .wav files in the selected folder.",
                )
                return

            if not (0 <= self.idx < len(self.files)):
                self.idx = 0

            # Keep advancing until we find a readable WAV.
            tried = 0
            y = None
            sr = None

            while tried < len(self.files):
                file_path = self.files[self.idx]
                y, sr = self._safe_read_wav(file_path)

                if y is not None:
                    break

                self.idx = (self.idx + 1) % len(self.files)
                tried += 1

            if y is None or sr is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Open failed",
                    "No .wav file could be read.",
                )
                return

            file_path = self.files[self.idx]

            self.lbl_path.setText(
                f"{human_relpath(self.root, os.path.dirname(file_path))}/"
                f"{os.path.basename(file_path)}"
            )

            self.y_raw = y
            self._filt_cache = None
            self._filt_params = None
            self.sr = int(sr)

            self.t = np.arange(len(self.y_raw), dtype=float) / self.sr

            duration = (
                float(len(self.y_raw)) / float(self.sr)
                if len(self.y_raw)
                else 0.0
            )

            self.time_slider.blockSignals(True)
            self.time_slider.setRange(0, int(duration * 100))
            self.time_slider.setValue(0)
            self.time_slider.blockSignals(False)

            self.lbl_time.setText("0.00 s")
            self.playhead.setPos(0.0)

            # Load or create JSON sidecar.
            json_path = json_sidecar_path(file_path)

            if os.path.isfile(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as fh:
                        self.state = FileState.from_json(json.load(fh))
                except Exception:
                    self.state = FileState(
                        file=os.path.basename(file_path),
                        sr=self.sr,
                        meta=dict(self.session_meta),
                        segments=[],
                    )
            else:
                self.state = FileState(
                    file=os.path.basename(file_path),
                    sr=self.sr,
                    meta=dict(self.session_meta),
                    segments=[],
                )

            # Ensure metadata is a dict.
            if not isinstance(self.state.meta, dict):
                self.state.meta = {}

            # Remove old metadata keys from previous versions.
            for old_key in (
                "subject_id",
                "microphone_type",
                "sample_rate",
                "location",
                "recording_id",
                "source_type",
                "device_type",
                "recording_location",
            ):
                self.state.meta.pop(old_key, None)

            # Apply supported session defaults.
            # This makes the last used environment carry over to the next file.
            for key, value in (self.session_meta or {}).items():
                if key in METADATA_FIELDS and str(value).strip() != "":
                    self.state.meta.setdefault(key, value)

            # Reflect metadata into UI.
            meta_for_editor = {
                key: self.state.meta.get(key, "")
                for key in METADATA_FIELDS
            }
            self.meta_inline.set_values(meta_for_editor)

            self._refresh_location_choices()
            self._apply_labelset(self.labelset_combo.currentText())

            # Draw audio views.
            self.draw_waveform()
            self.update_spectrogram()

            # Initialize selection region.
            self._blocking = True

            init_len = min(3.0, duration) if duration > 0.0 else 0.0
            self.region.setRegion((0.0, init_len))
            self.sel_start.setValue(0.0)
            self.sel_end.setValue(init_len)
            self.lbl_sel_delta.setText(f"(Δ {init_len:.2f} s)")

            self._blocking = False

            self.refresh_segment_list()
            self.save_json()

            #print(f"[DEBUG] loaded: {file_path}")

        except Exception as exc:
            import traceback

            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Load error", f"{exc}")

    def save_json(self: Any) -> None:
        """
        Write the current FileState to its sidecar JSON.
        """
        if not self.state:
            return

        if not (0 <= self.idx < len(self.files)):
            return

        json_path = json_sidecar_path(self.files[self.idx])
        ensure_dir(json_path)

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(
                self.state.to_json(),
                fh,
                ensure_ascii=False,
                indent=2,
            )

    # ------------------------------------------------------------------
    # File navigation
    # ------------------------------------------------------------------

    def advance(self: Any, step: int) -> None:
        """
        Move to the next or previous file and refresh the UI.
        """
        self.player.stop()

        new_idx = self.idx + step

        if new_idx < 0 or new_idx >= len(self.files):
            return

        self.idx = new_idx
        self.load_current()
        self._after_navigation_changed()

    def _rel_display_name(self: Any, abspath: str) -> str:
        """
        Return a path relative to the current root, with forward slashes.
        """
        try:
            rel = os.path.relpath(abspath, self.root)
        except Exception:
            rel = os.path.basename(abspath)

        return rel.replace("\\", "/")

    def _populate_jump_list(self: Any) -> None:
        """
        Fill the Jump-to combo with the current file list and select the active one.
        """
        self.combo_jump.blockSignals(True)
        self.combo_jump.clear()

        display_names: list[str] = []

        for path in self.files:
            abspath = path if isinstance(path, str) else getattr(path, "path", "")
            display_names.append(self._rel_display_name(abspath))

        self.combo_jump.addItems(display_names)
        self.combo_jump.setEnabled(len(display_names) > 0)

        if 0 <= self.idx < len(display_names):
            self.combo_jump.setCurrentIndex(self.idx)

        self.combo_jump.blockSignals(False)

    def _on_jump_selected(self: Any, index: int) -> None:
        """
        Load the file corresponding to the selected Jump-to item.
        """
        if not (0 <= index < len(self.files)):
            return

        if index == self.idx:
            return

        self.idx = index
        self.load_current()
        self._after_navigation_changed()

    def _after_navigation_changed(self: Any) -> None:
        """
        Keep the Jump-to combo in sync with the current file index.
        """
        if self.combo_jump.isEnabled() and 0 <= self.idx < self.combo_jump.count():
            self.combo_jump.blockSignals(True)
            self.combo_jump.setCurrentIndex(self.idx)
            self.combo_jump.blockSignals(False)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_csv(self: Any) -> None:
        """
        Export all segments for all files to a CSV file.

        Metadata columns are included only when values are filled in at least
        one exported row.
        """
        if not self.state:
            return

        rows: list[dict[str, object]] = []
        today = datetime.date.today().isoformat()

        for file_path in self.files:
            json_path = json_sidecar_path(file_path)

            if not os.path.isfile(json_path):
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    state = FileState.from_json(json.load(fh))
            except Exception:
                continue

            sorted_segments = sorted(
                state.segments,
                key=lambda segment: (
                    float(segment.t_start),
                    float(segment.t_end),
                    str(segment.id),
                ),
            )

            for segment in sorted_segments:
                row: dict[str, object] = {
                    "date": today,
                    "filename": self._rel_display_name(file_path),
                    "t_start": segment.t_start,
                    "t_end": segment.t_end,
                    "label": ";".join(segment.labels),
                }

                meta = dict(state.meta or {})

                # Backward compatibility: old location can become environment.
                if "environment" not in meta and meta.get("location"):
                    meta["environment"] = meta.get("location")

                # Only include filled supported metadata fields.
                for key in METADATA_FIELDS:
                    value = meta.get(key, "")
                    if str(value).strip() != "":
                        row[key] = value

                rows.append(row)

        if not rows:
            QtWidgets.QMessageBox.information(
                self,
                "Export",
                "No segments to export.",
            )
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            csv_path_for_root(self.root),
            "CSV (*.csv)",
        )

        if not path:
            return

        pd.DataFrame(rows).to_csv(path, index=False)

        self.lbl_last_export.setText(f"Last exported: {path}")

        QtWidgets.QMessageBox.information(
            self,
            "Export",
            f"Saved {len(rows)} rows to:\n{path}",
        )
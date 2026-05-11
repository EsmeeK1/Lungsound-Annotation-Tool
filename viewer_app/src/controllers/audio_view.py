from __future__ import annotations

from typing import Any

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..app_settings import DYNAMIC_SPECTRO_LEVELS, TIME_SNAP
from ..audio_processing import bandpass_filter, compute_stft_db
from ..audio_playback import HAVE_SD
from ..file_paths import snap_t


class AudioViewMixin:
    """
    Handles waveform display, spectrogram display, playback, filter controls,
    the active selection region, and clicking segment overlays in the waveform.
    """

    # ------------------------------------------------------------------
    # Spectrogram
    # ------------------------------------------------------------------

    def _ensure_spec_imageitem(self: Any) -> None:
        """
        Ensure self.img_spec always has a valid image and geometry.
        """
        try:
            if getattr(self, "img_spec", None) is None:
                return

            if self.img_spec.image is None:
                self.img_spec.setImage(
                    np.zeros((1, 1), dtype=np.float32),
                    autoLevels=True,
                )
                self.img_spec.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))

        except Exception:
            try:
                self.img_spec.setImage(
                    np.zeros((1, 1), dtype=np.float32),
                    autoLevels=True,
                )
                self.img_spec.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
            except Exception:
                pass

    def update_spectrogram(self: Any) -> None:
        """
        Recompute and redraw the spectrogram.
        """
        try:
            y = self.current_signal()
            if y is None or len(y) == 0:
                return

            self._ensure_spec_imageitem()

            cfg = getattr(
                self,
                "_stft_cfg",
                {
                    "nperseg": 1024,
                    "hop": 256,
                    "window": "hann",
                },
            )

            freqs, times, spectrum_db = compute_stft_db(
                y,
                self.sr,
                nperseg=int(cfg.get("nperseg", 1024)),
                hop=int(cfg.get("hop", 256)),
                window=str(cfg.get("window", "hann")),
            )

            if spectrum_db.size == 0 or len(times) <= 1 or len(freqs) <= 1:
                return

            # Limit displayed frequency range based on the band-pass UI.
            if getattr(self, "chk_bp", None) and self.chk_bp.isChecked():
                fmax_plot = min(
                    float(self.sp_high.value()),
                    float(self.sr) / 2.0 - 1e-6,
                )
            else:
                fmax_plot = float(freqs[-1])

            mask = freqs <= fmax_plot
            if not np.any(mask):
                mask = freqs <= freqs[-1]

            freqs_plot = freqs[mask]
            image = spectrum_db[mask, :]  # expected shape: frequency x time

            # Some STFT helpers may return time x frequency.
            if image.shape == (len(times), len(freqs_plot)):
                image = image.T

            image = np.nan_to_num(
                image,
                neginf=-120.0,
                posinf=0.0,
            ).astype(np.float32, copy=False)

            try:
                vmin = float(np.percentile(image, 2.0))
                vmax = float(np.percentile(image, 98.0))

                if (
                    not np.isfinite(vmin)
                    or not np.isfinite(vmax)
                    or vmin >= vmax
                ):
                    raise ValueError

            except Exception:
                vmin, vmax = -100.0, 0.0

            if bool(DYNAMIC_SPECTRO_LEVELS):
                self.img_spec.setImage(image, autoLevels=True)
            else:
                self.img_spec.setImage(
                    image,
                    autoLevels=False,
                    levels=(vmin, vmax),
                )

            t_max = max(float(times[-1]), 1e-6)
            f_max = max(
                float(freqs_plot[-1]) if len(freqs_plot) else float(freqs[-1]),
                1e-6,
            )

            self.img_spec.setRect(QtCore.QRectF(0.0, 0.0, t_max, f_max))

            self.p_spec.setLimits(
                xMin=0.0,
                xMax=t_max,
                yMin=0.0,
                yMax=f_max,
            )
            self.p_spec.setXRange(0.0, t_max)
            self.p_spec.setYRange(0.0, f_max)

            self.lbl_stft_params.setText(
                "STFT: "
                f"nperseg={cfg.get('nperseg')} | "
                f"hop={cfg.get('hop')} | "
                f"window={cfg.get('window')}"
            )

        except Exception:
            import traceback

            traceback.print_exc()

    def _show_stft_info(self: Any) -> None:
        """
        Show a short explanation of STFT parameters.
        """
        text = (
            "Short explanation of STFT parameters:\n\n"
            "• nperseg: samples per FFT frame. Larger improves frequency detail, "
            "but time detail becomes coarser.\n"
            "• hop: step between frames. Smaller makes time smoother, but increases "
            "computation.\n"
            "• window: function to reduce edge effects. Common choices are 'hann' "
            "and 'hamming'.\n\n"
            "These settings affect only the spectrogram display, not the audio data "
            "or segment times."
        )

        QtWidgets.QMessageBox.information(
            self,
            "STFT parameters",
            text,
        )

    # ------------------------------------------------------------------
    # Band-pass filter
    # ------------------------------------------------------------------

    def _show_bp_info(self: Any) -> None:
        """
        Show a short explanation of band-pass filter settings.
        """
        text = (
            "Short explanation of band-pass filtering:\n\n"
            "• Low / High: keep only frequencies between these cutoff values.\n"
            "• Use filtering to reduce unwanted low-frequency rumble or high-frequency "
            "noise while annotating.\n"
            "• Order: higher values create steeper cutoff edges, but can introduce "
            "ringing or artifacts.\n"
            "• Zero-phase: applies forward-backward filtering to avoid phase delay. "
            "This is useful for offline annotation and visualization."
        )

        QtWidgets.QMessageBox.information(
            self,
            "Band-pass info",
            text,
        )

    def on_filter_ui_changed(self: Any, *args: object) -> None:
        """
        React to changes in band-pass UI controls.
        """
        # Invalidate cached filtered signal.
        self._filt_cache = None
        self._filt_params = None

        # Soft validation: enforce high > low.
        try:
            low = float(self.sp_low.value())
            high = float(self.sp_high.value())
        except Exception:
            low, high = 50.0, 2000.0

        if high <= low:
            self.sp_high.blockSignals(True)
            self.sp_high.setValue(low + 1.0)
            self.sp_high.blockSignals(False)

        try:
            self.draw_waveform()
            self.update_spectrogram()
        except Exception:
            pass

    def current_signal(self: Any) -> np.ndarray | None:
        """
        Return the signal that should be displayed and played.
        """
        if self.y_raw is None:
            return None

        if getattr(self, "chk_bp", None) and self.chk_bp.isChecked():
            return self.get_filtered_signal()

        return self.y_raw

    def get_filtered_signal(self: Any) -> np.ndarray | None:
        """
        Compute or retrieve the cached band-pass filtered signal.
        """
        if self.y_raw is None:
            return None

        sr = float(self.sr)

        params = (
            float(self.sp_low.value()),
            float(self.sp_high.value()),
            int(self.sp_order.value()),
            bool(self.chk_zero.isChecked()),
            len(self.y_raw),
            sr,
        )

        if self._filt_cache is not None and self._filt_params == params:
            return self._filt_cache

        try:
            y_filtered = bandpass_filter(
                self.y_raw,
                fs=sr,
                fc=(params[0], params[1]),
                order=params[2],
                zero_phase=params[3],
                axis=-1,
            )

        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Band-pass filter",
                f"Could not apply filter:\n{exc}\nFalling back to raw signal.",
            )

            self.chk_bp.setChecked(False)
            self._filt_cache = None
            self._filt_params = None

            return self.y_raw

        self._filt_cache = y_filtered.astype(np.float32, copy=False)
        self._filt_params = params

        return self._filt_cache

    # ------------------------------------------------------------------
    # View reset / waveform drawing
    # ------------------------------------------------------------------

    def reset_view(self: Any) -> None:
        """
        Reset waveform and spectrogram views to their full range.
        """
        try:
            vb_wave = self.p_wave.getViewBox()
            vb_wave.enableAutoRange(x=True, y=True)

            if self.t is not None and len(self.t) > 1:
                xmax = float(self.t[-1])
                vb_wave.setXRange(0.0, xmax, padding=0.02)

        except Exception:
            pass

        try:
            self.update_spectrogram()
            vb_spec = self.p_spec.getViewBox()
            vb_spec.enableAutoRange(x=True, y=True)

        except Exception:
            pass

    def draw_waveform(self: Any) -> None:
        """
        Draw the waveform and keep overlay regions attached.
        """
        y = self.current_signal()
        x = self.t

        if y is None or x is None:
            return

        self.curve.setData(x=x, y=y, connect="finite")

        self.p_wave.setLabel("bottom", "Time (s)")
        self.p_wave.setLabel("left", "Amplitude")
        self.p_wave.showGrid(x=True, y=True, alpha=0.2)

        if len(x) > 1:
            xmax = float(x[-1])
            self.p_wave.setLimits(xMin=0.0, xMax=xmax)

            view_box = self.p_wave.getViewBox()
            view_box.setXRange(0.0, xmax, padding=0.02)

        # Re-attach regions if they were removed by view changes.
        for region in getattr(self, "overlay_regions", {}).values():
            if region.scene() is None:
                self.p_wave.addItem(region)

    # ------------------------------------------------------------------
    # Selection region
    # ------------------------------------------------------------------

    def on_region_changed(self: Any) -> None:
        """
        Sync the selection region with the spin boxes and delta label.
        """
        if self._blocking:
            return

        a, b = self.region.getRegion()

        a = max(0.0, snap_t(a))  # type: ignore[arg-type]
        b = max(a + TIME_SNAP, snap_t(b))  # type: ignore[arg-type]

        self._blocking = True

        self.region.setRegion((a, b))
        self.sel_start.setValue(a)
        self.sel_end.setValue(b)
        self.lbl_sel_delta.setText(f"(Δ {(b - a):.2f} s)")

        self._blocking = False

    def on_sel_spin_changed(self: Any) -> None:
        """
        Sync the selection spin boxes with the selection region.
        """
        if self._blocking:
            return

        a = float(self.sel_start.value())
        b = float(self.sel_end.value())

        a = max(0.0, snap_t(a))
        b = max(a + TIME_SNAP, snap_t(b))

        self._blocking = True

        self.sel_start.setValue(a)
        self.sel_end.setValue(b)
        self.region.setRegion((a, b))
        self.lbl_sel_delta.setText(f"(Δ {(b - a):.2f} s)")

        self._blocking = False

    def nudge_region(
        self: Any,
        dt: float,
        mode: str = "move",
    ) -> None:
        """
        Move or resize the selection region by a small step.
        """
        if self.t is None or len(self.t) == 0:
            return

        duration = float(self.t[-1])

        a, b = self.region.getRegion()
        a = float(a)
        b = float(b)
        step = float(dt)

        if mode == "move":
            width = b - a
            new_a = max(0.0, min(a + step, duration - width))
            new_b = new_a + width

        elif mode == "start":
            new_a = max(0.0, min(a + step, b - TIME_SNAP))
            new_b = b

        elif mode == "end":
            new_a = a
            new_b = min(duration, max(b + step, a + TIME_SNAP))

        else:
            return

        new_a = snap_t(new_a)
        new_b = snap_t(new_b)

        if new_b <= new_a:
            new_b = min(duration, new_a + TIME_SNAP)

        self._blocking = True

        self.region.setRegion((new_a, new_b))
        self.sel_start.setValue(new_a)
        self.sel_end.setValue(new_b)
        self.lbl_sel_delta.setText(f"(Δ {(new_b - new_a):.2f} s)")

        self._blocking = False

    # ------------------------------------------------------------------
    # Time slider / playback
    # ------------------------------------------------------------------

    def on_slider_changed(self: Any, value: int) -> None:
        """
        Update playhead and time label when the slider moves.
        """
        t = value / 100.0

        self.playhead.setPos(t)
        self.lbl_time.setText(f"{t:.2f} s")

    def toggle_play(self: Any) -> None:
        """
        Start or stop playback from the current slider time to the end.
        """
        if not HAVE_SD or self.y_raw is None:
            QtWidgets.QMessageBox.information(
                self,
                "Playback",
                "Playback is not available.",
            )
            return

        if self.player.playing:
            self.player.stop()
            return

        y = self.current_signal()
        if y is None:
            return

        t0 = self.time_slider.value() / 100.0
        t1 = len(y) / self.sr

        if t0 >= t1:
            t0 = max(0.0, t1 - 0.01)

        self.player.play(y, self.sr, t0, t1)

    def on_play_started(
        self: Any,
        a: float,
        b: float,
    ) -> None:
        """
        Record playback window and start playhead timer.
        """
        self.play_window = (a, b)
        self._elapsed = QtCore.QElapsedTimer()
        self._elapsed.start()
        self.timer.start()

    def on_play_stopped(self: Any) -> None:
        """
        Stop the playhead timer when playback ends.
        """
        self.timer.stop()

    def tick_playhead(self: Any) -> None:
        """
        Advance playhead while audio is playing.
        """
        a, b = self.play_window
        t_now = a + self._elapsed.elapsed() / 1000.0

        if t_now >= b:
            self.player.stop()
            t_now = b

        self.playhead.setPos(t_now)

        self.time_slider.blockSignals(True)
        self.time_slider.setValue(int(t_now * 100))
        self.time_slider.blockSignals(False)

        self.lbl_time.setText(f"{t_now:.2f} s")

    def _play_current_segment(self: Any) -> None:
        """
        Play only the currently selected segment.
        """
        if not self.state or self.y_raw is None:
            return

        segment = self._current_segment_or_none()
        if not segment:
            return

        y = self.current_signal()
        if y is None:
            return

        self.player.stop()
        self.player.play(
            y,
            self.sr,
            float(segment.t_start),
            float(segment.t_end),
        )

    # ------------------------------------------------------------------
    # Overlay region click handling
    # ------------------------------------------------------------------

    def on_overlay_clicked(self: Any, region: object) -> None:
        """
        Select a segment by clicking its overlay region on the waveform.

        Behavior:
        - Normal click: select only this segment.
        - Shift-click: select range from the last clicked segment to this segment.
        - Ctrl-click: toggle this segment in the current selection.

        Important:
        - Do not use self.list.setCurrentRow(row) after manually selecting items,
          because that can clear the multi-selection.
        """
        if self.state is None:
            return

        segment_id = getattr(region, "seg_id", None)
        if segment_id is None:
            return

        clicked_row = -1

        for i, segment in enumerate(self.state.segments):
            if segment.id == segment_id:
                clicked_row = i
                break

        if clicked_row < 0:
            return

        def set_current_without_changing_selection(row: int) -> None:
            """
            Make row the current row without replacing the current selection.
            """
            model_index = self.list.model().index(row, 0)
            selection_model = self.list.selectionModel()

            if selection_model is not None:
                selection_model.setCurrentIndex(
                    model_index,
                    QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
                )

        modifiers = QtWidgets.QApplication.keyboardModifiers()

        shift = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)
        ctrl = bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)

        if not hasattr(self, "_selection_anchor_row"):
            self._selection_anchor_row = clicked_row

        self.list.blockSignals(True)

        if shift:
            anchor = getattr(self, "_selection_anchor_row", clicked_row)
            start_row, end_row = sorted((anchor, clicked_row))

            self.list.clearSelection()

            for row in range(start_row, end_row + 1):
                item = self.list.item(row)
                if item is not None:
                    item.setSelected(True)

            set_current_without_changing_selection(clicked_row)

        elif ctrl:
            item = self.list.item(clicked_row)
            if item is not None:
                item.setSelected(not item.isSelected())

            set_current_without_changing_selection(clicked_row)
            self._selection_anchor_row = clicked_row

        else:
            self.list.clearSelection()

            item = self.list.item(clicked_row)
            if item is not None:
                item.setSelected(True)

            set_current_without_changing_selection(clicked_row)
            self._selection_anchor_row = clicked_row

        self.list.blockSignals(False)

        segment = self.state.segments[clicked_row]

        self._blocking = True

        self.region.setRegion((segment.t_start, segment.t_end))
        self.sel_start.setValue(segment.t_start)
        self.sel_end.setValue(segment.t_end)
        self.lbl_sel_delta.setText(
            f"(Δ {(segment.t_end - segment.t_start):.2f} s)"
        )

        self._blocking = False

        # Because signals were blocked while changing selection, refresh manually.
        self.rebuild_label_list()
        self._update_overlay_selection_styles()
        self._reflect_labelbar()
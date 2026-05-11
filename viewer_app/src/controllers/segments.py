from __future__ import annotations

from typing import Any, Callable, List, Optional
import uuid

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from ..app_settings import TIME_SNAP
from ..dialogs import AutoSegmentDialog
from ..data_models import Segment
from ..label_colors import LABEL_COLORS
from ..file_paths import snap_t
from ..widgets import ClickableRegion


class SegmentsMixin:
    """
    Handles segment creation, editing, selection, label assignment,
    auto-segmentation, and undo/redo for segment edits.
    """

    # ------------------------------------------------------------------
    # Segment display / overlays
    # ------------------------------------------------------------------

    def _brush_for_labels(self: Any, labels: List[str]):
        """
        Pick a stable color for a segment based on its labels.
        """
        return LABEL_COLORS.color_for(labels)

    def _overlay_brush_for_labels(
    self: Any,
    labels: List[str],
    selected: bool = False,
    ):
        """
        Return a waveform overlay brush.

        Selected overlays are drawn with higher opacity so multi-selection is
        visible directly in the waveform view.
        """
        r, g, b, _a = LABEL_COLORS.color_for(labels)
        alpha = 185 if selected else 85
        return pg.mkBrush(r, g, b, alpha)

    def _style_overlay_region(
    self: Any,
    region: ClickableRegion,
    labels: List[str],
    selected: bool = False,
    ) -> None:
        """
        Apply visual styling to one waveform segment overlay.

        Selected:
        - stronger fill
        - thicker boundary lines
        - higher z-index

        Unselected:
        - lighter fill
        - thin boundary lines
        - normal z-index
        """
        region.setBrush(self._overlay_brush_for_labels(labels, selected=selected))
        region.setZValue(20 if selected else 5)

        # LinearRegionItem has two internal boundary lines.
        # Styling them makes selected regions obvious even when colors overlap.
        try:
            # Keep borders subtle. Selection is shown through stronger fill
            # opacity and higher z-index, not a white outline.
            pen = pg.mkPen("#333333", width=0.6)
            hover_pen = pg.mkPen("#333333", width=0.9)

            for line in getattr(region, "lines", []):
                line.setPen(pen)
                line.setHoverPen(hover_pen)
        except Exception:
            pass

    def _update_overlay_selection_styles(self: Any) -> None:
        """
        Refresh overlay styles so selected visible segments are visible in the waveform.
        """
        if not self.state:
            return

        selected_ids = set(self._selected_segment_ids())

        for segment_id in getattr(self, "visible_segment_ids", []):
            segment = self._segment_by_id(segment_id)
            region = self.overlay_regions.get(segment_id)

            if segment is None or region is None:
                continue

            self._style_overlay_region(
                region,
                segment.labels,
                selected=segment.id in selected_ids,
            )

    def _sort_segments_chronologically(self: Any) -> None:
        """
        Keep segments ordered from low to high time.

        Sort order:
        1. start time
        2. end time
        3. id as stable fallback
        """
        if not self.state:
            return

        self.state.segments.sort(
            key=lambda s: (
                float(s.t_start),
                float(s.t_end),
                str(s.id),
            )
        )

    def _chunk_start(self: Any) -> float:
        item = getattr(self, "current_item", None)
        return float(getattr(item, "chunk_start", 0.0) or 0.0)

    def _chunk_end(self: Any) -> float:
        item = getattr(self, "current_item", None)

        if item is not None:
            return float(getattr(item, "chunk_end", 0.0) or 0.0)

        if self.t is not None and len(self.t) > 0:
            return float(self.t[-1])

        return 0.0

    def _chunk_duration(self: Any) -> float:
        return max(0.0, self._chunk_end() - self._chunk_start())

    def _local_to_absolute(self: Any, t_local: float) -> float:
        return snap_t(self._chunk_start() + float(t_local))

    def _absolute_to_local(self: Any, t_abs: float) -> float:
        return snap_t(float(t_abs) - self._chunk_start())

    def _segment_overlaps_current_chunk(self: Any, segment: Segment) -> bool:
        chunk_start = self._chunk_start()
        chunk_end = self._chunk_end()

        return (
            float(segment.t_end) > chunk_start
            and float(segment.t_start) < chunk_end
        )

    def _visible_segments(self: Any) -> list[Segment]:
        if not self.state:
            return []

        return [
            segment
            for segment in self.state.segments
            if self._segment_overlaps_current_chunk(segment)
        ]

    def _segment_by_id(self: Any, segment_id: str) -> Optional[Segment]:
        if not self.state:
            return None

        for segment in self.state.segments:
            if segment.id == segment_id:
                return segment

        return None

    def _segment_for_visible_row(self: Any, row: int) -> Optional[Segment]:
        visible_ids = getattr(self, "visible_segment_ids", [])

        if not (0 <= row < len(visible_ids)):
            return None

        return self._segment_by_id(visible_ids[row])

    def _visible_row_for_segment_id(self: Any, segment_id: str) -> int:
        visible_ids = getattr(self, "visible_segment_ids", [])

        try:
            return visible_ids.index(segment_id)
        except ValueError:
            return -1

    def refresh_segment_list(self: Any) -> None:
        """
        Rebuild the segment list and overlay regions on the waveform.

        Only segments that overlap the current virtual chunk are displayed.
        Segment times in state remain absolute; displayed times are local to
        the current chunk.
        """
        self.list.clear()
        self.visible_segment_ids = []

        if not self.state:
            return

        self._sort_segments_chronologically()

        # Remove old overlays safely.
        for reg in getattr(self, "overlay_regions", {}).values():
            try:
                self.p_wave.removeItem(reg)
            except Exception:
                pass

        self.overlay_regions.clear()

        chunk_start = self._chunk_start()
        chunk_end = self._chunk_end()

        for segment in self._visible_segments():
            self.visible_segment_ids.append(segment.id)

            local_start = max(float(segment.t_start), chunk_start) - chunk_start
            local_end = min(float(segment.t_end), chunk_end) - chunk_start

            label_text = "; ".join(segment.labels) or "(no labels)"
            self.list.addItem(
                f"{local_start:.2f}-{local_end:.2f}s | {label_text}"
            )

            reg = ClickableRegion(
                [local_start, local_end],
                brush=self._overlay_brush_for_labels(segment.labels, selected=False),
                seg_id=segment.id,
            )

            self._style_overlay_region(
                reg,
                segment.labels,
                selected=False,
            )

            self.p_wave.addItem(reg)
            reg.clicked.connect(self.on_overlay_clicked)
            self.overlay_regions[segment.id] = reg

        self._update_overlay_selection_styles()
        self._reflect_labelbar()

    def _find_segment_by_bounds(
        self: Any,
        a: float,
        b: float,
    ) -> Optional[Segment]:
        """
        Find a segment that exactly matches the given start and end times.
        """
        if not self.state:
            return None

        for segment in self.state.segments:
            if abs(segment.t_start - a) < 1e-6 and abs(segment.t_end - b) < 1e-6:
                return segment

        return None

    # ------------------------------------------------------------------
    # Segment selection
    # ------------------------------------------------------------------

    def on_list_selection(self: Any, row: int) -> None:
        """
        When a visible segment is selected in the list, reflect it in the
        editors and selection region.

        The UI shows local chunk times. The stored segment uses absolute times.
        """
        segment = self._segment_for_visible_row(row)

        if segment is None:
            self.list_labels.clear()
            return

        chunk_start = self._chunk_start()
        chunk_end = self._chunk_end()

        local_start = max(float(segment.t_start), chunk_start) - chunk_start
        local_end = min(float(segment.t_end), chunk_end) - chunk_start

        self.spin_start.setValue(local_start)
        self.spin_end.setValue(local_end)
        self.rebuild_label_list()

        self._blocking = True
        self.region.setRegion((local_start, local_end))
        self.sel_start.setValue(local_start)
        self.sel_end.setValue(local_end)
        self.lbl_sel_delta.setText(
            f"(Δ {(local_end - local_start):.2f} s)"
        )
        self._blocking = False

        self._selection_anchor_row = row
        self._update_overlay_selection_styles()
        self._reflect_labelbar()

    def on_segment_selection_changed(self: Any) -> None:
        """
        Refresh label UI and waveform overlay styling when multi-selection changes.
        """
        self.rebuild_label_list()
        self._update_overlay_selection_styles()
        self._reflect_labelbar()

    def clear_segment_selection_for_free_region(self: Any) -> None:
        """
        Clear the selected segment(s) when the user manually moves the free
        selection region.

        This makes the next label click create a new segment instead of editing
        the previously selected segment.
        """
        self.list.blockSignals(True)
        self.list.clearSelection()
        self.list.setCurrentRow(-1)
        self.list.blockSignals(False)

        self.list_labels.clear()
        self._update_overlay_selection_styles()
        self._reflect_labelbar()

    def _selected_segment_rows(self: Any) -> list[int]:
        """
        Return all selected segment rows, sorted.

        Falls back to the current row when nothing is explicitly selected.
        """
        rows = sorted({
            idx.row()
            for idx in self.list.selectedIndexes()
            if idx.isValid()
        })

        if rows:
            return rows

        row = self.list.currentRow()
        if self.state and 0 <= row < len(self.state.segments):
            return [row]

        return []

    def _selected_segment_ids(self: Any) -> list[str]:
        """
        Return IDs of selected visible segments.
        """
        ids: list[str] = []

        for row in self._selected_segment_rows():
            segment = self._segment_for_visible_row(row)
            if segment is not None:
                ids.append(segment.id)

        return ids

    def _set_selected_segment_ids(
        self: Any,
        ids: list[str],
    ) -> None:
        """
        Restore list selection by segment IDs within the current visible chunk.
        """
        wanted = set(ids or [])

        self.list.blockSignals(True)
        self.list.clearSelection()

        first_row = -1

        for row, segment_id in enumerate(getattr(self, "visible_segment_ids", [])):
            if segment_id in wanted:
                item = self.list.item(row)
                if item is not None:
                    item.setSelected(True)
                    if first_row < 0:
                        first_row = row

        if first_row >= 0:
            model_index = self.list.model().index(first_row, 0)
            selection_model = self.list.selectionModel()
            if selection_model is not None:
                selection_model.setCurrentIndex(
                    model_index,
                    QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
                )

        self.list.blockSignals(False)

        if first_row >= 0:
            self.on_list_selection(first_row)
        else:
            self._update_overlay_selection_styles()
            self._reflect_labelbar()

    # ------------------------------------------------------------------
    # Segment editing
    # ------------------------------------------------------------------

    def remove_selected_label(self: Any) -> None:
        """
        Remove the currently selected label from all selected visible segments.
        Undoable.
        """
        if not self.state:
            return

        lab_row = self.list_labels.currentRow()
        if lab_row < 0 or lab_row >= self.list_labels.count():
            return

        item = self.list_labels.item(lab_row)
        if item is None:
            return

        label_to_remove = item.text().strip()
        if not label_to_remove:
            return

        selected_ids = self._selected_segment_ids()
        if not selected_ids:
            return

        before = self._segments_snapshot()
        before_selection = selected_ids

        changed = False

        for segment_id in selected_ids:
            segment = self._segment_by_id(segment_id)
            if segment is None:
                continue

            while label_to_remove in segment.labels:
                segment.labels.remove(label_to_remove)
                changed = True

        if not changed:
            return

        self._commit_segments_edit(
            before,
            before_selection=before_selection,
            after_selection=selected_ids,
        )
        self.rebuild_label_list()

    def update_segment(self: Any) -> None:
        """
        Save changes to the current visible segment.
        Undoable.

        The UI uses local chunk times; stored times are converted to absolute
        source-file times.
        """
        row = self.list.currentRow()
        segment = self._segment_for_visible_row(row)

        if not self.state or segment is None:
            return

        before = self._segments_snapshot()
        before_selection = self._selected_segment_ids()

        chunk_duration = self._chunk_duration()

        new_a_local = max(0.0, snap_t(self.spin_start.value()))
        new_b_local = max(new_a_local + TIME_SNAP, snap_t(self.spin_end.value()))
        new_b_local = min(chunk_duration, new_b_local)

        new_a_abs = self._local_to_absolute(new_a_local)
        new_b_abs = self._local_to_absolute(new_b_local)

        new_labels = [
            self.list_labels.item(i).text()
            for i in range(self.list_labels.count())
        ]

        segment.t_start = new_a_abs
        segment.t_end = new_b_abs
        segment.labels = new_labels

        self._commit_segments_edit(
            before,
            before_selection=before_selection,
            after_selection=[segment.id],
        )
        self.rebuild_label_list()

    def delete_selected(self: Any) -> None:
        """
        Delete selected visible segment(s) after confirmation.
        Supports multi-selection.
        Undoable.
        """
        if not self.state:
            return

        selected_ids = self._selected_segment_ids()
        if not selected_ids:
            return

        count = len(selected_ids)
        msg = (
            "Delete the selected segment?"
            if count == 1
            else f"Delete {count} selected segments?"
        )

        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete segment",
            msg,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        before = self._segments_snapshot()
        before_selection = selected_ids

        selected_id_set = set(selected_ids)

        self.state.segments = [
            segment
            for segment in self.state.segments
            if segment.id not in selected_id_set
        ]

        self._commit_segments_edit(
            before,
            before_selection=before_selection,
            after_selection=[],
        )

    # ------------------------------------------------------------------
    # Label list inside "Edit segment"
    # ------------------------------------------------------------------

    def rebuild_label_list(self: Any) -> None:
        """
        Rebuild the small label list for selected visible segment(s).
        """
        self.list_labels.clear()

        if not self.state:
            return

        selected_ids = self._selected_segment_ids()
        if not selected_ids:
            return

        labels: list[str] = []
        seen: set[str] = set()

        for segment_id in selected_ids:
            segment = self._segment_by_id(segment_id)
            if segment is None:
                continue

            for label in segment.labels:
                if label not in seen:
                    labels.append(label)
                    seen.add(label)

        for label in labels:
            item = QtWidgets.QListWidgetItem(self.list_labels)
            item.setSizeHint(QtCore.QSize(0, 26))

            widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(widget)
            layout.setContentsMargins(6, 2, 6, 2)
            layout.setSpacing(8)

            lbl = QtWidgets.QLabel(label)

            btn = QtWidgets.QToolButton()
            btn.setText("×")
            btn.setToolTip(f"Remove label from selected segment(s): {label}")
            btn.setFixedSize(22, 22)
            btn.setStyleSheet("QToolButton { font-weight: bold; }")
            btn.setProperty("label_text", label)
            btn.clicked.connect(self._on_remove_label_btn)

            layout.addWidget(lbl)
            layout.addStretch(1)
            layout.addWidget(btn)

            self.list_labels.addItem(item)
            self.list_labels.setItemWidget(item, widget)

    def _on_remove_label_btn(self: Any) -> None:
        """
        Remove the label associated with the clicked '×' button from all
        selected visible segments.
        Undoable.
        """
        if not self.state:
            return

        sender = self.sender()
        if not isinstance(sender, QtWidgets.QToolButton):
            return

        label_to_remove = str(sender.property("label_text") or "").strip()
        if not label_to_remove:
            return

        selected_ids = self._selected_segment_ids()
        if not selected_ids:
            return

        before = self._segments_snapshot()
        before_selection = selected_ids

        changed = False

        for segment_id in selected_ids:
            segment = self._segment_by_id(segment_id)
            if segment is None:
                continue

            while label_to_remove in segment.labels:
                segment.labels.remove(label_to_remove)
                changed = True

        if not changed:
            return

        self._commit_segments_edit(
            before,
            before_selection=before_selection,
            after_selection=selected_ids,
        )
        self.rebuild_label_list()

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def _segments_snapshot(self: Any) -> list[Segment]:
        """
        Deep-copy all segments so undo/redo can restore exact state.
        """
        if not self.state:
            return []

        return [
            Segment(
                id=segment.id,
                t_start=segment.t_start,
                t_end=segment.t_end,
                labels=list(segment.labels),
            )
            for segment in self.state.segments
        ]

    def _restore_segments_snapshot(
        self: Any,
        snapshot: list[Segment],
        selected_ids: list[str] | None = None,
    ) -> None:
        """
        Restore segments from a snapshot and refresh UI/persistence.
        """
        if not self.state:
            return

        self.state.segments = [
            Segment(
                id=segment.id,
                t_start=segment.t_start,
                t_end=segment.t_end,
                labels=list(segment.labels),
            )
            for segment in snapshot
        ]

        self.refresh_segment_list()
        self._set_selected_segment_ids(selected_ids or [])
        self._reflect_labelbar()
        self.save_json()

    def _commit_segments_edit(
        self: Any,
        before: list[Segment],
        before_selection: list[str] | None = None,
        after_selection: list[str] | None = None,
    ) -> None:
        """
        Commit the current segment state as one undoable edit.
        Call this after mutating self.state.segments.
        """
        if not self.state:
            return

        after = self._segments_snapshot()

        if before == after:
            self.refresh_segment_list()
            self._reflect_labelbar()
            self.save_json()
            return

        before_selection = before_selection or []
        after_selection = after_selection or self._selected_segment_ids()

        def do() -> None:
            self._restore_segments_snapshot(after, after_selection)

        def undo() -> None:
            self._restore_segments_snapshot(before, before_selection)

        self._undo_stack.append((do, undo))
        self._redo_stack.clear()

        self.refresh_segment_list()
        self._set_selected_segment_ids(after_selection)
        self._reflect_labelbar()
        self.save_json()

    def _push_edit(
        self: Any,
        do: Callable[[], None],
        undo: Callable[[], None],
    ) -> None:
        """
        Push an edit action onto the undo stack and execute it.

        Kept for compatibility with older edit actions. New segment edits
        should prefer _commit_segments_edit().
        """
        do()
        self._undo_stack.append((do, undo))
        self._redo_stack.clear()
        self._post_edit_refresh()

    def undo(self: Any) -> None:
        """
        Undo the last edit if available.
        """
        if not self._undo_stack:
            return

        do, undo = self._undo_stack.pop()
        undo()
        self._redo_stack.append((do, undo))
        self._post_edit_refresh()

    def redo(self: Any) -> None:
        """
        Redo the last undone edit if available.
        """
        if not self._redo_stack:
            return

        do, undo = self._redo_stack.pop()
        do()
        self._undo_stack.append((do, undo))
        self._post_edit_refresh()

    def _post_edit_refresh(self: Any) -> None:
        """
        Refresh lists and persist after an edit operation.
        """
        self.refresh_segment_list()
        self._reflect_labelbar()
        self.save_json()

    # ------------------------------------------------------------------
    # Segment labels / labelbar
    # ------------------------------------------------------------------

    def _current_segment_or_none(self: Any) -> Optional[Segment]:
        """
        Return the currently selected visible Segment object, or None.
        """
        row = self.list.currentRow()
        return self._segment_for_visible_row(row)

    def _create_segment(
        self: Any,
        t_start: float,
        t_end: float,
    ) -> Segment:
        """
        Create and append a new segment for the current file.

        Input times are local chunk times.
        Stored times are absolute source-file times.
        """
        segment = Segment(
            id=str(uuid.uuid4()),
            t_start=self._local_to_absolute(t_start),
            t_end=self._local_to_absolute(t_end),
            labels=[],
        )

        if self.state:
            self.state.segments.append(segment)

        return segment

    def _reflect_labelbar(self: Any) -> None:
        """
        Update the LabelBar toggle state.

        For multiple selected visible segments, only labels present on all
        selected segments are shown as active.
        """
        if not self.state:
            self.labelbar.reflect_segment([])
            return

        selected_ids = self._selected_segment_ids()

        if not selected_ids:
            self.labelbar.reflect_segment([])
            return

        selected_segments = [
            self._segment_by_id(segment_id)
            for segment_id in selected_ids
        ]
        selected_segments = [
            segment
            for segment in selected_segments
            if segment is not None
        ]

        if not selected_segments:
            self.labelbar.reflect_segment([])
            return

        common = set(selected_segments[0].labels)

        for segment in selected_segments[1:]:
            common &= set(segment.labels)

        self.labelbar.reflect_segment(list(common))

    def _on_labelbar_toggled(
        self: Any,
        label: str,
        checked: bool,
    ) -> None:
        """
        Add or remove a label.

        If visible segments are selected, apply to all selected visible segments.
        If no segment is selected, create a new segment from the current local
        chunk selection.
        """
        if not self.state:
            return

        before = self._segments_snapshot()
        before_selection = self._selected_segment_ids()

        selected_ids = self._selected_segment_ids()
        affected_ids: list[str] = []

        if selected_ids:
            for segment_id in selected_ids:
                segment = self._segment_by_id(segment_id)
                if segment is None:
                    continue

                affected_ids.append(segment.id)

                if checked:
                    if label not in segment.labels:
                        segment.labels.append(label)
                else:
                    try:
                        segment.labels.remove(label)
                    except ValueError:
                        pass

        else:
            # No selected segment: create one from the current local region.
            a, b = self.region.getRegion()
            a = snap_t(a)  # type: ignore[arg-type]
            b = max(a + TIME_SNAP, snap_t(b))  # type: ignore[arg-type]

            segment = self._create_segment(a, b)
            affected_ids.append(segment.id)

            if checked and label not in segment.labels:
                segment.labels.append(label)

        self._commit_segments_edit(
            before,
            before_selection=before_selection,
            after_selection=affected_ids,
        )
    # ------------------------------------------------------------------
    # Auto segmentation
    # ------------------------------------------------------------------

    def auto_segment_dialog(self: Any) -> None:
        """
        Open the auto-segmentation dialog and apply the chosen settings.
        """
        if self.t is None or len(self.t) == 0:
            QtWidgets.QMessageBox.information(
                self,
                "Auto segment",
                "No audio loaded.",
            )
            return

        label_options = self._current_label_options()

        default_label = getattr(self, "_auto_seg_cfg", {}).get(
            "label",
            label_options[0] if label_options else None,
        )
        default_len = float(
            getattr(self, "_auto_seg_cfg", {}).get("length_s", 1.0)
        )
        default_ovl = float(
            getattr(self, "_auto_seg_cfg", {}).get("overlap_s", 0.0)
        )

        dlg = AutoSegmentDialog(
            self,
            default_len=default_len,
            default_overlap=default_ovl,
            default_replace=False,
            label_options=label_options,
            default_label=default_label,
        )

        if not dlg.exec():
            return

        seg_len, seg_ovl, replace, auto_label = dlg.values()

        self.apply_auto_segments(
            seg_len,
            seg_ovl,
            replace,
            auto_label=auto_label,
        )

    def apply_auto_segments(
        self: Any,
        seg_len: float,
        seg_ovl: float,
        replace: bool,
        auto_label: Optional[str] = None,
    ) -> None:
        """
        Create fixed-length segments across the current virtual chunk.
        Stored times are absolute source-file times.
        Undoable.
        """
        if self.state is None or self.t is None or len(self.t) == 0:
            return

        dur = self._chunk_duration()
        snap = float(TIME_SNAP)

        len_ticks = max(1, int(round(seg_len / snap)))
        ovl_ticks = max(0, int(round(seg_ovl / snap)))

        if ovl_ticks >= len_ticks:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid parameters",
                "Ensure that 0 ≤ overlap < length.",
            )
            return

        stride_ticks = len_ticks - ovl_ticks
        total_ticks = int(round(dur / snap))

        new_segments: List[Segment] = []
        start_tick = 0

        while start_tick < total_ticks:
            end_tick = min(start_tick + len_ticks, total_ticks)

            local_a = round(start_tick * snap, 2)
            local_b = round(end_tick * snap, 2)

            labels = [auto_label] if auto_label else []

            new_segments.append(
                Segment(
                    id=str(uuid.uuid4()),
                    t_start=self._local_to_absolute(local_a),
                    t_end=self._local_to_absolute(local_b),
                    labels=labels,
                )
            )

            start_tick += stride_ticks

        before = self._segments_snapshot()
        before_selection = self._selected_segment_ids()

        if replace:
            # Replace only segments that overlap the current virtual chunk.
            self.state.segments = [
                segment
                for segment in self.state.segments
                if not self._segment_overlaps_current_chunk(segment)
            ]
            self.state.segments.extend(new_segments)
        else:
            self.state.segments.extend(new_segments)

        after_selection = [new_segments[0].id] if new_segments else []

        self._commit_segments_edit(
            before,
            before_selection=before_selection,
            after_selection=after_selection,
        )
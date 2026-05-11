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
        Refresh overlay styles so selected segments are visible in the waveform.
        """
        if not self.state:
            return

        selected_ids = set(self._selected_segment_ids())

        for segment in self.state.segments:
            region = self.overlay_regions.get(segment.id)
            if region is None:
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

    def refresh_segment_list(self: Any) -> None:
        """
        Rebuild the segment list and overlay regions on the waveform.
        """
        self.list.clear()

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

        # Add list items and waveform overlay regions.
        for segment in self.state.segments:
            label_text = "; ".join(segment.labels) or "(no labels)"
            self.list.addItem(
                f"{segment.t_start:.2f}-{segment.t_end:.2f}s | {label_text}"
            )

            reg = ClickableRegion(
                [segment.t_start, segment.t_end],
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
        When a segment is selected in the list, reflect it in the editors
        and selection region.
        """
        if not self.state or row < 0 or row >= len(self.state.segments):
            self.list_labels.clear()
            return

        segment = self.state.segments[row]

        # Update edit fields.
        self.spin_start.setValue(segment.t_start)
        self.spin_end.setValue(segment.t_end)
        self.rebuild_label_list()

        # Update visible selection region.
        self._blocking = True
        self.region.setRegion((segment.t_start, segment.t_end))
        self.sel_start.setValue(segment.t_start)
        self.sel_end.setValue(segment.t_end)
        self.lbl_sel_delta.setText(
            f"(Δ {(segment.t_end - segment.t_start):.2f} s)"
        )
        self._blocking = False

        # Keep shift-click anchor in sync.
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
        Return IDs of selected segments.
        """
        if not self.state:
            return []

        ids: list[str] = []

        for row in self._selected_segment_rows():
            if 0 <= row < len(self.state.segments):
                ids.append(self.state.segments[row].id)

        return ids

    def _set_selected_segment_ids(
        self: Any,
        ids: list[str],
    ) -> None:
        """
        Restore list selection by segment IDs.
        """
        if not self.state:
            return

        wanted = set(ids or [])

        self.list.blockSignals(True)
        self.list.clearSelection()

        first_row = -1

        for i, segment in enumerate(self.state.segments):
            if segment.id in wanted:
                item = self.list.item(i)
                if item is not None:
                    item.setSelected(True)
                    if first_row < 0:
                        first_row = i

        if first_row >= 0:
            self.list.setCurrentRow(first_row)

        self.list.blockSignals(False)

        if first_row >= 0:
            self.on_list_selection(first_row)
        else:
            self._reflect_labelbar()

    # ------------------------------------------------------------------
    # Segment editing
    # ------------------------------------------------------------------

    def remove_selected_label(self: Any) -> None:
        """
        Remove the currently selected label from all selected segments.
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

        rows = self._selected_segment_rows()
        if not rows:
            return

        before = self._segments_snapshot()
        before_selection = self._selected_segment_ids()

        changed = False

        for row in rows:
            if not (0 <= row < len(self.state.segments)):
                continue

            segment = self.state.segments[row]

            while label_to_remove in segment.labels:
                segment.labels.remove(label_to_remove)
                changed = True

        if not changed:
            return

        after_selection = [
            self.state.segments[row].id
            for row in rows
            if 0 <= row < len(self.state.segments)
        ]

        self._commit_segments_edit(
            before,
            before_selection=before_selection,
            after_selection=after_selection,
        )
        self.rebuild_label_list()

    def update_segment(self: Any) -> None:
        """
        Save changes to the current segment.
        Undoable.

        This remains a single-segment edit. Multi-selected segments are not
        batch-resized because that would make all selected segments overlap.
        """
        row = self.list.currentRow()
        if not self.state or row < 0:
            return

        segment = self.state.segments[row]

        before = self._segments_snapshot()
        before_selection = self._selected_segment_ids()

        new_a = snap_t(self.spin_start.value())
        new_b = max(new_a + TIME_SNAP, snap_t(self.spin_end.value()))

        new_labels = [
            self.list_labels.item(i).text()
            for i in range(self.list_labels.count())
        ]

        segment.t_start = new_a
        segment.t_end = new_b
        segment.labels = new_labels

        self._commit_segments_edit(
            before,
            before_selection=before_selection,
            after_selection=[segment.id],
        )
        self.rebuild_label_list()

    def delete_selected(self: Any) -> None:
        """
        Delete selected segment(s) after confirmation.
        Supports multi-selection.
        Undoable.
        """
        if not self.state:
            return

        rows = self._selected_segment_rows()
        if not rows:
            return

        count = len(rows)
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
        before_selection = self._selected_segment_ids()

        next_row = min(rows)

        for row in sorted(rows, reverse=True):
            if 0 <= row < len(self.state.segments):
                del self.state.segments[row]

        after_selection: list[str] = []

        if self.state.segments:
            next_row = min(next_row, len(self.state.segments) - 1)
            after_selection = [self.state.segments[next_row].id]

        self._commit_segments_edit(
            before,
            before_selection=before_selection,
            after_selection=after_selection,
        )

    # ------------------------------------------------------------------
    # Label list inside "Edit segment"
    # ------------------------------------------------------------------

    def rebuild_label_list(self: Any) -> None:
        """
        Rebuild the small label list for selected segment(s).

        Behavior:
        - One selected segment: show that segment's labels.
        - Multiple selected segments: show the union of all labels across
          selected segments.
        """
        self.list_labels.clear()

        if not self.state:
            return

        rows = self._selected_segment_rows()
        if not rows:
            return

        labels: list[str] = []
        seen: set[str] = set()

        for row in rows:
            if not (0 <= row < len(self.state.segments)):
                continue

            segment = self.state.segments[row]

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
        selected segments.
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

        rows = self._selected_segment_rows()
        if not rows:
            return

        before = self._segments_snapshot()
        before_selection = self._selected_segment_ids()

        changed = False

        for row in rows:
            if not (0 <= row < len(self.state.segments)):
                continue

            segment = self.state.segments[row]

            while label_to_remove in segment.labels:
                segment.labels.remove(label_to_remove)
                changed = True

        if not changed:
            return

        after_selection = [
            self.state.segments[row].id
            for row in rows
            if 0 <= row < len(self.state.segments)
        ]

        self._commit_segments_edit(
            before,
            before_selection=before_selection,
            after_selection=after_selection,
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
        Return the currently selected Segment object, or None.
        """
        if not self.state:
            return None

        row = self.list.currentRow()

        if 0 <= row < len(self.state.segments):
            return self.state.segments[row]

        return None

    def _create_segment(
        self: Any,
        t_start: float,
        t_end: float,
    ) -> Segment:
        """
        Create and append a new segment for the current file.
        """
        segment = Segment(
            id=str(uuid.uuid4()),
            t_start=t_start,
            t_end=t_end,
            labels=[],
        )

        if self.state:
            self.state.segments.append(segment)

        return segment

    def _reflect_labelbar(self: Any) -> None:
        """
        Update the LabelBar toggle state.

        For multiple selected segments, only labels present on all selected
        segments are shown as active.
        """
        if not self.state:
            self.labelbar.reflect_segment([])
            return

        rows = self._selected_segment_rows()

        if not rows:
            self.labelbar.reflect_segment([])
            return

        selected = [
            self.state.segments[row]
            for row in rows
            if 0 <= row < len(self.state.segments)
        ]

        if not selected:
            self.labelbar.reflect_segment([])
            return

        common = set(selected[0].labels)

        for segment in selected[1:]:
            common &= set(segment.labels)

        self.labelbar.reflect_segment(list(common))

    def _on_labelbar_toggled(
        self: Any,
        label: str,
        checked: bool,
    ) -> None:
        """
        Add or remove a label.

        Behavior:
        - If one or more segments are selected, apply to all selected segments.
        - If no segment is selected, create a segment from the current region.
        - The whole operation is one undoable edit.
        """
        if not self.state:
            return

        before = self._segments_snapshot()
        before_selection = self._selected_segment_ids()

        rows = self._selected_segment_rows()
        affected_ids: list[str] = []

        if rows:
            for row in rows:
                if not (0 <= row < len(self.state.segments)):
                    continue

                segment = self.state.segments[row]
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
            # No selected segment: create one from the current region.
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
            getattr(self, "_auto_seg_cfg", {}).get("length_s", 3.0)
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
        Create fixed-length segments across the file, with optional overlap.
        Undoable.
        """
        if self.state is None or self.t is None or len(self.t) == 0:
            return

        dur = float(self.t[-1])
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

            a = round(start_tick * snap, 2)
            b = round(end_tick * snap, 2)

            labels = [auto_label] if auto_label else []

            new_segments.append(
                Segment(
                    id=str(uuid.uuid4()),
                    t_start=a,
                    t_end=b,
                    labels=labels,
                )
            )

            start_tick += stride_ticks

        before = self._segments_snapshot()
        before_selection = self._selected_segment_ids()

        if replace:
            self.state.segments = new_segments
        else:
            self.state.segments.extend(new_segments)

        after_selection = [new_segments[0].id] if new_segments else []

        self._commit_segments_edit(
            before,
            before_selection=before_selection,
            after_selection=after_selection,
        )
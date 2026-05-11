from __future__ import annotations

from typing import Any
import datetime
import json
import os
from pathlib import Path

from PySide6 import QtWidgets

from ..app_settings import labels_list_to_dict
from ..label_colors import LABEL_COLORS
from ..file_paths import labels_dataset_path


class LabelsMixin:
    """
    Handles label configuration and label UI state.

    Labels are loaded from labels_dataset.json.
    There are no built-in domain-specific label sets anymore.
    """

    def _show_label_info(self: Any) -> None:
        """
        Explain how labels and label buttons work.
        """
        txt = (
            "<b>How labels work</b><br><br>"
            "Labels are loaded from <code>labels_dataset.json</code>.<br><br>"
            "Click a label button to add or remove that label on the current "
            "segment. If multiple segments are selected, the label action is "
            "applied to all selected segments.<br><br>"
            "The button stays highlighted when the selected segment contains "
            "that label. For multiple selected segments, a label is highlighted "
            "only when all selected segments contain it."
        )
        QtWidgets.QMessageBox.information(self, "Label info", txt)

    def load_labels_json(self: Any) -> None:
        """
        Load labels_dataset.json and apply display/processing defaults.

        Sets:
        - self._custom_labels
        - label color map
        - session metadata defaults
        - filter defaults
        - STFT defaults
        - auto-segment defaults
        - label UI
        """
        path = labels_dataset_path()

        default_cfg = {
            "version": 1,
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
            "labels": [],
            "meta_defaults": {"environment": ""},
            "filter_defaults": {
                "lowcut": 50,
                "highcut": 3000,
                "order": 4,
                "zero_phase": True,
            },
            "stft_params": {
                "nperseg": 1024,
                "hop": 256,
                "window": "hann",
            },
            "auto_segment_defaults": {
                "length_s": 3.00,
                "overlap_s": 0.00,
                "label": "",
            },
        }

        if Path(path).is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = default_cfg
        else:
            cfg = default_cfg
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default_cfg, f, ensure_ascii=False, indent=2)

        labels = cfg.get("labels", []) or []
        self._custom_labels = list(labels) if labels else []

        LABEL_COLORS.build(self._custom_labels)

        defaults = cfg.get("meta_defaults", {})
        for key, value in defaults.items():
            self.session_meta.setdefault(key, value)

        # Backward compatibility with older configs.
        if "location" in self.session_meta and "environment" not in self.session_meta:
            self.session_meta["environment"] = self.session_meta.get("location", "")
        self.session_meta.pop("location", None)

        fdef = cfg.get("filter_defaults", {})

        if "lowcut" in fdef:
            self.sp_low.setValue(float(fdef["lowcut"]))

        if "highcut" in fdef:
            self.sp_high.setValue(float(fdef["highcut"]))

        if "order" in fdef:
            self.sp_order.setValue(int(fdef["order"]))

        if "zero_phase" in fdef:
            self.chk_zero.setChecked(bool(fdef["zero_phase"]))

        self._stft_cfg = cfg.get(
            "stft_params",
            {
                "nperseg": 1024,
                "hop": 256,
                "window": "hann",
            },
        )

        fallback_label = self._custom_labels[0] if self._custom_labels else ""
        self._auto_seg_cfg = cfg.get(
            "auto_segment_defaults",
            {
                "length_s": 3.0,
                "overlap_s": 0.0,
                "label": fallback_label,
            },
        )

        self._refresh_labelset_combo()
        self._refresh_location_choices()

    def _refresh_labelset_combo(self: Any) -> None:
        """
        Rebuild the label set combo.

        There is now one label source:
        - labels_dataset.json

        The combo is kept for compatibility with the existing UI, but it only
        shows one item: Custom.
        """
        self.labelset_combo.blockSignals(True)
        self.labelset_combo.clear()

        if self._custom_labels:
            self.labelset_combo.addItem("Custom")
            self.labelset_combo.setCurrentText("Custom")
            self.labelbar.set_labels(labels_list_to_dict(self._custom_labels))
        else:
            self.labelset_combo.addItem("Custom")
            self.labelset_combo.setCurrentText("Custom")
            self.labelbar.set_labels(
                {"No labels loaded": "Edit labels_dataset.json and reload labels"}
            )

        self.labelset_combo.blockSignals(False)

    def _apply_labelset(self: Any, name: str) -> None:
        """
        Apply labels from labels_dataset.json to the LabelBar.

        The name argument is kept because the existing combo emits it.
        """
        if not self._custom_labels:
            self.load_labels_json()

        if self._custom_labels:
            self.labelbar.set_labels(labels_list_to_dict(self._custom_labels))
        else:
            self.labelbar.set_labels(
                {"No labels loaded": "Edit labels_dataset.json and reload labels"}
            )

        self._refresh_location_choices()
        self._reflect_labelbar()

    def add_label_to_dataset(self: Any, label: str) -> None:
        """
        Append a single label to labels_dataset.json if it is not already present.
        """
        label = str(label or "").strip()
        if not label:
            return

        path = labels_dataset_path()

        data = {
            "version": 1,
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
            "labels": [],
            "meta_defaults": dict(self.session_meta or {}),
        }

        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

        labels = list(data.get("labels", []) or [])

        if label not in labels:
            labels.append(label)
            data["labels"] = labels
            data["updated"] = datetime.datetime.now().isoformat(timespec="seconds")

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        self.reload_labels_json()

    def reload_labels_json(self: Any) -> None:
        """
        Reload labels and defaults from labels_dataset.json.
        """
        self.load_labels_json()

    def _current_label_options(self: Any) -> list[str]:
        """
        Get label options for dialogs such as auto-segmentation.
        """
        if self._custom_labels:
            return list(self._custom_labels)

        return []
from __future__ import annotations

import hashlib

from PySide6 import QtGui


# High-contrast categorical palette.
# The first colors are intentionally very different.
_BASE_PALETTE = [
    "#9467bd",  # purple
    "#d62728",  # red
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#1f77b4",  # blue
    "#8c564b",  # brown
    "#e377c2",  # pink
]


def _to_qcolor_tuple(color: str, alpha: int = 120) -> tuple[int, int, int, int]:
    """
    Convert a hex color to an RGBA tuple in 0..255.
    """
    qcolor = QtGui.QColor(color)
    return (
        qcolor.red(),
        qcolor.green(),
        qcolor.blue(),
        alpha,
    )


def _qualitative_palette(n: int) -> list[tuple[int, int, int, int]]:
    """
    Return n visually distinct colors.

    Colors repeat only when more labels exist than palette entries.
    """
    if n <= 0:
        return []

    return [
        _to_qcolor_tuple(_BASE_PALETTE[i % len(_BASE_PALETTE)])
        for i in range(n)
    ]


class LabelColorMap:
    """
    Keep a stable mapping from label names to colors.

    Known labels are mapped from labels_dataset.json.
    Unknown labels still receive deterministic non-gray colors, so old sidecars
    or manually edited JSON files do not all appear as the same gray region.
    """

    def __init__(self):
        self.labels: list[str] = []
        self.colors: list[tuple[int, int, int, int]] = []
        self.map: dict[str, tuple[int, int, int, int]] = {}
        self.unknown_map: dict[str, tuple[int, int, int, int]] = {}

    def build(self, labels: list[str]) -> None:
        """
        Build the mapping with a high-contrast qualitative palette.
        """
        self.labels = list(labels)
        self.colors = _qualitative_palette(len(self.labels))
        self.map = {
            label: color
            for label, color in zip(self.labels, self.colors)
        }

        # Rebuild unknown colors after label reload.
        self.unknown_map.clear()

    def _color_for_unknown_label(self, label: str) -> tuple[int, int, int, int]:
        """
        Deterministically assign a color to a label not present in self.map.
        """
        if label in self.unknown_map:
            return self.unknown_map[label]

        palette = _qualitative_palette(len(_BASE_PALETTE))
        if not palette:
            return (120, 120, 120, 80)

        digest = hashlib.sha1(label.encode("utf-8")).hexdigest()
        start_index = int(digest[:8], 16) % len(palette)

        used = set(self.map.values()) | set(self.unknown_map.values())

        chosen = palette[start_index]
        for offset in range(len(palette)):
            candidate = palette[(start_index + offset) % len(palette)]
            if candidate not in used:
                chosen = candidate
                break

        self.unknown_map[label] = chosen
        return chosen

    def color_for(self, labels_in_segment: list[str]) -> tuple[int, int, int, int]:
        """
        Return the color for the first label in a segment.

        Priority:
        1. Known label from labels_dataset.json
        2. Deterministic generated color for unknown label
        3. Neutral gray for unlabeled segments
        """
        for label in labels_in_segment:
            if label in self.map:
                return self.map[label]

        for label in labels_in_segment:
            if str(label).strip():
                return self._color_for_unknown_label(str(label))

        return (120, 120, 120, 70)


LABEL_COLORS = LabelColorMap()
from __future__ import annotations

import os
from PySide6 import QtGui

from .config import LABELS_JSON_PATH, TIME_SNAP


# time helpers

def snap_t(x: float) -> float:
    """
    Snap a time value to the nearest TIME_SNAP step.

    Args:
        x: Time in seconds.

    Returns:
        Time rounded to the nearest step (for example 0.01 s).
    """
    return round(float(x) / TIME_SNAP) * TIME_SNAP


# path helpers

def human_relpath(root: str, path: str) -> str:
    """
    Make a readable path relative to a root folder.

    Args:
        root: Base directory.
        path: Target file or folder.

    Returns:
        Relative path using forward slashes.
    """
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        # Different drive or invalid relation, fall back to just the name
        rel = os.path.basename(path)
    return rel.replace("\\", "/")


def json_sidecar_path(wav_path: str) -> str:
    """
    Build the sidecar JSON path for a WAV file.

    Args:
        wav_path: Path to a .wav file.

    Returns:
        Path with the same base name and a .json extension.
    """
    base, _ = os.path.splitext(wav_path)
    return base + ".json"


def csv_path_for_root(root: str) -> str:
    """
    Default CSV export path inside a chosen root.

    Args:
        root: Base folder for the export.

    Returns:
        Full path to labels_export.csv in that folder.
    """
    return os.path.join(root, "labels_export.csv")


def labels_dataset_path() -> str:
    """
    Central labels JSON that ships with the app.

    Returns:
        Absolute path to labels_dataset.json.
    """
    return str(LABELS_JSON_PATH)


def ensure_dir(path: str) -> None:
    """
    Make sure the parent directory of a path exists.

    Args:
        path: File path whose directory should exist.
    """
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


# color helpers

def _to_qcolor_tuple(c, alpha: int = 60) -> tuple[int, int, int, int]:
    """
    Convert various color formats to an RGBA tuple in 0..255.

    Accepts:
      - Hex strings like "#RRGGBB" or "#RRGGBBAA".
      - Tuples/lists of 3 or 4 items. Values can be 0..1 (matplotlib) or 0..255.

    Args:
        c: Color input.
        alpha: Fallback alpha channel when none is provided.

    Returns:
        (r, g, b, a) with each value 0..255.

    Notes:
        If the input is a tuple with values <= 1, it is assumed to be normalized
        and scaled up to 0..255.
    """
    # hex input
    if isinstance(c, str) and c.startswith("#") and len(c) in (7, 9):
        q = QtGui.QColor(c)
        return (q.red(), q.green(), q.blue(), alpha)

    # tuple/list input
    if isinstance(c, (list, tuple)) and len(c) in (3, 4):
        r, g, b = c[:3]
        # handle normalized RGB from matplotlib (0..1)
        if any(v <= 1.0 for v in (r, g, b)):
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return (r, g, b, alpha)

    # unknown format, return a neutral gray
    return (120, 120, 120, alpha)


# try to load matplotlib for a stable qualitative palette
try:
    from matplotlib import cm
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


def _qualitative_palette(n: int) -> list[tuple[int, int, int, int]]:
    """
    Return n distinct colors suitable for categorical labels.

    Strategy:
      - If matplotlib is available, use 'tab20' for up to 20 items.
      - Otherwise, use a fixed fallback list and repeat when needed.

    Args:
        n: Number of colors.

    Returns:
        List of RGBA tuples in 0..255.
    """
    if _HAVE_MPL:
        # request exactly n samples from the tab20 colormap
        cmap = cm.get_cmap("tab20", max(2, n))
        return [_to_qcolor_tuple(cmap(i)) for i in range(n)]

    # fallback palette without matplotlib
    base = [
        "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
        "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
        "#ccebc5", "#ffed6f", "#1b9e77", "#d95f02", "#7570b3",
        "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666",
    ]
    if n <= len(base):
        return [_to_qcolor_tuple(h) for h in base[:n]]

    # if more colors are requested, repeat deterministically
    return [_to_qcolor_tuple(base[i % len(base)]) for i in range(n)]


class LabelColorMap:
    """
    Keep a stable mapping from label names to colors.

    The mapping is built once from the current label list
    and reused to color segments consistently across the app.
    """
    def __init__(self):
        self.labels: list[str] = []
        self.colors: list[tuple[int, int, int, int]] = []
        self.map: dict[str, tuple[int, int, int, int]] = {}

    def build(self, labels: list[str]) -> None:
        """
        Build the mapping with a qualitative palette.

        Args:
            labels: Ordered list of label names.
        """
        self.labels = list(labels)
        self.colors = _qualitative_palette(len(self.labels))
        self.map = {lab: col for lab, col in zip(self.labels, self.colors)}

    def color_for(self, labels_in_segment: list[str]) -> tuple[int, int, int, int]:
        """
        Return the color for the first known label in a segment.

        Args:
            labels_in_segment: Labels attached to a segment.

        Returns:
            RGBA tuple. Gray when none of the labels are known.
        """
        for L in labels_in_segment:
            if L in self.map:
                return self.map[L]
        return (120, 120, 120, 40)


LABEL_COLORS = LabelColorMap()

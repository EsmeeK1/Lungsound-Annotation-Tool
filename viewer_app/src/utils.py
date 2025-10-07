from __future__ import annotations

import os, json, datetime
from typing import List, Tuple, Dict
from PySide6 import QtGui
from pathlib import Path
import numpy as np

from .config import LABELS_JSON_PATH, TIME_SNAP

def snap_t(x: float) -> float:
    """Snap time to nearest TIME_SNAP (0.01s)"""
    return round(float(x) / TIME_SNAP) * TIME_SNAP

def human_relpath(root: str, path: str) -> str:
    """Get the relative path from the root directory to the specified path.

    Args:
        root (str): The root directory.
        path (str): The target file or directory path.

    Returns:
        str: The relative path from the root to the target path.
    """
    try: rel = os.path.relpath(path, root)
    except ValueError: rel = os.path.basename(path)
    return rel.replace("\\", "/")


def json_sidecar_path(wav_path: str) -> str:
    """Path to the JSON sidecar file for a given WAV file.

    Args:
        wav_path (str): Path to the input WAV file.

    Returns:
        str: Path to the corresponding JSON sidecar file.
    """
    base, _ = os.path.splitext(wav_path)
    return base + ".json"

def csv_path_for_root(root: str) -> str:
    """Default export path for CSV, inside the chosen root folder."""
    return os.path.join(root, "labels_export.csv")

def labels_dataset_path() -> str:
    """Always the central JSON file next to viewer.py."""
    return str(LABELS_JSON_PATH)

def ensure_dir(path: str):
    """Ensure the directory for the given path exists."""
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _to_qcolor_tuple(c, alpha=60):
    # accepteer matplotlib RGBA of hex; retourneer (r,g,b,a) 0-255
    if isinstance(c, str) and c.startswith("#") and len(c) in (7, 9):
        q = QtGui.QColor(c)
        return (q.red(), q.green(), q.blue(), alpha)
    if isinstance(c, (list, tuple)) and len(c) in (3, 4):
        r, g, b = c[:3]
        if any(v <= 1.0 for v in (r, g, b)):  # mpl 0..1
            r, g, b = int(r*255), int(g*255), int(b*255)
        return (r, g, b, alpha)
    return (120, 120, 120, alpha)

try:
    from matplotlib import cm
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

def _qualitative_palette(n: int):
    """
    Returns colors from a fixed palette.
    - First, try 'tab20' (stable, 20 distinct categories).
    - If not available, fall back to a handcrafted sequence (Set3-like).
    """
    if _HAVE_MPL:
        cmap = cm.get_cmap('tab20', max(2, n))
        return [_to_qcolor_tuple(cmap(i)) for i in range(n)]
    # Fallback zonder matplotlib (hexes van Set3-achtig palet)
    base = [
        "#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3",
        "#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd",
        "#ccebc5","#ffed6f","#1b9e77","#d95f02","#7570b3",
        "#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
    ]
    if n <= len(base):
        return [_to_qcolor_tuple(h) for h in base[:n]]
    # repeat (deterministic)
    return [_to_qcolor_tuple(base[i % len(base)]) for i in range(n)]

class LabelColorMap:
    """Maintains a stable mapping from label to color, based on labels_dataset.json"""
    def __init__(self):
        self.labels: list[str] = []
        self.colors: list[tuple[int,int,int,int]] = []
        self.map: dict[str, tuple[int,int,int,int]] = {}

    def build(self, labels: list[str]):
        self.labels = list(labels)
        self.colors = _qualitative_palette(len(self.labels))
        self.map = {lab: col for lab, col in zip(self.labels, self.colors)}

    def color_for(self, labels_in_segment: list[str]):
        # Kies de eerste bekende labelkleur; anders grijs
        for L in labels_in_segment:
            if L in self.map:
                return self.map[L]
        return (120,120,120,40)

LABEL_COLORS = LabelColorMap()
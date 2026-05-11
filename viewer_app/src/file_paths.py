from __future__ import annotations

import os
from PySide6 import QtGui

from .app_settings import LABELS_JSON_PATH, TIME_SNAP

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
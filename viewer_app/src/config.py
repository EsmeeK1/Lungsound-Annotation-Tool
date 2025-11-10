from pathlib import Path
from dataclasses import dataclass, asdict, field
import json, os
from typing import List

# Base directory (root of the project)
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths for datasets and config files
LABELS_JSON_PATH = BASE_DIR / "labels_dataset.json"
HEART_LOC_JSON = BASE_DIR / "heart_locations.json"
LUNG_LOC_JSON = BASE_DIR / "lung_locations.json"
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".lungsound_tool_config.json")

# Global settings
DEFAULT_SR = 16000           # Default sample rate for recordings
TIME_SNAP = 0.01             # Rounding step for timestamps in seconds

# Debug options
DEBUG_STFT = False           # Print debug info during STFT if True
DYNAMIC_SPECTRO_LEVELS = False  # Auto-adjust spectrogram levels per frame
GRAYSCALE_DEBUG = True       # Use grayscale spectrograms for debugging

# Metadata fields for recordings
METADATA_FIELDS = [
    "subject_id",
    "microphone_type",
    "sample_rate",
    "location",
]

# Standard label definitions for lung and heart sounds
LABEL_SETS = {
    "Lung": {
        "Crackle": "Short, non-musical, explosive sound (inspiratory > expiratory).",
        "Wheeze": "Narrow airway, musical tone (mostly expiratory).",
        "Rhonchi": "Low-pitched, gurgling sound often related to mucus.",
        "Stridor": "High-pitched tone, upper airway obstruction; often urgent.",
        "Normal": "Vesicular breathing without added sounds.",
    },
    "Heart": {
        "S1": "Closure of mitral and tricuspid valves (start of systole).",
        "S2": "Closure of aortic and pulmonary valves (end of systole).",
        "Murmur": "Continuous or holosystolic/diastolic blowing sound.",
        "Click": "Extra tone (for example, valve click).",
        "Gallop": "Additional S3 or S4 heart sounds.",
        "Normal": "Regular rhythm without extra tones or murmurs.",
    },
}

# Default label set to use
DEFAULT_LABEL_SET = "Lung"


def load_default_locations(label_set: str) -> list[str]:
    """
    Load the standard recording locations for a given label set.

    Args:
        label_set (str): The name of the label set ('Lung' or 'Heart').

    Returns:
        list[str]: A list of default location names, or an empty list if not found.
    """
    # Select correct file based on label set
    p = HEART_LOC_JSON if label_set.lower().startswith("heart") else LUNG_LOC_JSON
    try:
        # Read and load JSON file (expected: {"Apex": "...", "RUSB": "...", ...})
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        # Only return the location names (keys)
        return list(d.keys())
    except Exception:
        # Return empty list if loading fails
        return []


@dataclass
class SessionDefaults:
    """
    Store default values for the current recording session.
    """
    mic_type: str = ""
    location: str = ""
    lock_mic: bool = False
    lock_loc: bool = False


@dataclass
class UserPrefs:
    """
    Keep track of user preferences and recent selections.

    Attributes:
        recents_mic_types (List[str]): Recently used microphone types.
        recents_locations (List[str]): Recently used recording locations.
        session (SessionDefaults): Default session settings.
    """
    recents_mic_types: List[str] = field(default_factory=list)
    recents_locations: List[str] = field(default_factory=list)
    session: SessionDefaults = field(default_factory=SessionDefaults)

    def __post_init__(self):
        """Ensure lists are always initialized even if None."""
        if self.recents_mic_types is None:
            self.recents_mic_types = []
        if self.recents_locations is None:
            self.recents_locations = []


def load_prefs() -> UserPrefs:
    """
    Load user preferences from the config file.

    Returns:
        UserPrefs: A UserPrefs object with saved or default settings.
    """
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load session data safely
        sd = data.get("session", {})
        return UserPrefs(
            recents_mic_types=data.get("recents_mic_types", []),
            recents_locations=data.get("recents_locations", []),
            session=SessionDefaults(
                mic_type=sd.get("mic_type", ""),
                location=sd.get("location", ""),
                lock_mic=sd.get("lock_mic", False),
                lock_loc=sd.get("lock_loc", False),
            )
        )

    # If no config file exists, return defaults
    return UserPrefs()


def save_prefs(p: UserPrefs) -> None:
    """
    Save user preferences to the config file.

    Args:
        p (UserPrefs): The preferences object to save.
    """
    data = {
        "recents_mic_types": p.recents_mic_types[-8:],  # Keep last 8 entries
        "recents_locations": p.recents_locations[-8:],
        "session": asdict(p.session),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def labels_list_to_dict(labels: list[str], default_tooltip: str = "") -> dict[str, str]:
    """
    Convert a list of label names into a dictionary with tooltips.

    Args:
        labels (list[str]): The list of labels.
        default_tooltip (str): Tooltip text to assign to each label. Default is empty.

    Returns:
        dict[str, str]: Dictionary where each label points to the tooltip.
    """
    return {L: default_tooltip for L in labels}

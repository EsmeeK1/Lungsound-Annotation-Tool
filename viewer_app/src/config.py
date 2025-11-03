from pathlib import Path
from dataclasses import dataclass, asdict, field
import json, os, pathlib
from typing import List

# Project directories (labels json lives next to app.py at project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Fixed location for labels_dataset.json
LABELS_JSON_PATH = BASE_DIR / "labels_dataset.json"
HEART_LOC_JSON = BASE_DIR / "heart_locations.json"
LUNG_LOC_JSON = BASE_DIR / "lung_locations.json"
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".lungsound_tool_config.json")

# Config & constants
DEFAULT_SR = 16000
TIME_SNAP = 0.01

# Debug flags
DEBUG_STFT = False               # print debug info during STFT
DYNAMIC_SPECTRO_LEVELS = False   # auto levels per spectrogram frame
GRAYSCALE_DEBUG = True           # grayscale spectrogram (debug)

METADATA_FIELDS = [
    "subject_id",
    "microphone_type",
    "sample_rate",
    "location",
]

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
        "Click": "Extra tone (e.g., valve click).",
        "Gallop": "Additional S3 or S4 heart sounds.",
        "Normal": "Regular rhythm without extra tones or murmurs.",
    },
}
DEFAULT_LABEL_SET = "Lung"

def load_default_locations(label_set: str) -> list[str]:
    """Return standard recording locations for the current label set."""
    # label_set is de zichtbare tekst in de combobox: 'Lung', 'Heart' of 'custom (labels.json)'
    p = HEART_LOC_JSON if label_set.lower().startswith("heart") else LUNG_LOC_JSON
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)            # verwacht {"Apex":"...", "RUSB":"...", ...}
        return list(d.keys())           # we tonen alleen de locatie-namen
    except Exception:
        return []


@dataclass
class SessionDefaults:
    mic_type: str = ""
    location: str = ""
    lock_mic: bool = False
    lock_loc: bool = False

@dataclass
class UserPrefs:
    recents_mic_types: List[str] = field(default_factory=list)
    recents_locations: List[str] = field(default_factory=list)
    session: SessionDefaults = field(default_factory=SessionDefaults)

    def __post_init__(self):
        if self.recents_mic_types is None: self.recents_mic_types = []
        if self.recents_locations is None: self.recents_locations = []

def load_prefs() -> UserPrefs:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # simpele, tolerante load
        sd = data.get("session", {})
        return UserPrefs(
            recents_mic_types=data.get("recents_mic_types", []),
            recents_locations=data.get("recents_locations", []),
            session=SessionDefaults(
                mic_type=sd.get("mic_type",""),
                location=sd.get("location",""),
                lock_mic=sd.get("lock_mic", False),
                lock_loc=sd.get("lock_loc", False),
            )
        )
    return UserPrefs()

def save_prefs(p: UserPrefs) -> None:
    data = {
        "recents_mic_types": p.recents_mic_types[-8:],
        "recents_locations": p.recents_locations[-8:],
        "session": asdict(p.session),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def labels_list_to_dict(labels: list[str], default_tooltip: str = "") -> dict[str, str]:
    return {L: default_tooltip for L in labels}

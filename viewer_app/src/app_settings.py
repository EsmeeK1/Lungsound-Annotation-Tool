from pathlib import Path
from dataclasses import dataclass, asdict, field
import json
import os
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent

LABELS_JSON_PATH = BASE_DIR / "labels_dataset.json"
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".audio_annotation_tool_config.json")

DEFAULT_SR = 16000
TIME_SNAP = 0.01

DEBUG_STFT = False
DYNAMIC_SPECTRO_LEVELS = False

METADATA_FIELDS = [
    "environment",
    "notes",
]

DEFAULT_ENVIRONMENTS = [
    "",
    "indoor",
    "outdoor",
    "quiet",
    "noisy",
    "traffic",
    "home",
    "lab",
    "clinical",
    "other",
]


@dataclass
class SessionDefaults:
    environment: str = ""


@dataclass
class UserPrefs:
    recent_environments: List[str] = field(default_factory=list)
    session: SessionDefaults = field(default_factory=SessionDefaults)

    def __post_init__(self):
        if self.recent_environments is None:
            self.recent_environments = []


def load_prefs() -> UserPrefs:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        session_data = data.get("session", {})

        return UserPrefs(
            recent_environments=data.get("recent_environments", []),
            session=SessionDefaults(
                environment=session_data.get("environment", ""),
            ),
        )

    return UserPrefs()


def save_prefs(prefs: UserPrefs) -> None:
    data = {
        "recent_environments": prefs.recent_environments[-8:],
        "session": asdict(prefs.session),
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def labels_list_to_dict(labels: list[str], default_tooltip: str = "") -> dict[str, str]:
    return {label: default_tooltip for label in labels}
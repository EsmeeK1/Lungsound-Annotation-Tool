from pathlib import Path

# Project directories (labels json lives next to app.py at project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Fixed location for labels_dataset.json
LABELS_JSON_PATH = BASE_DIR / "labels_dataset.json"

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
    "gender",
    "age",
]
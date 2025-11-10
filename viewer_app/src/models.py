from dataclasses import dataclass, asdict, field
import re
from typing import Optional, Dict, Any, List


# Patterns for valid subject IDs: "001" or "P001"
SUBJECT_PATTERNS = [
    re.compile(r"^\d{3}$"),
    re.compile(r"^P\d{3}$"),
]


def normalize_subject_id(text: str) -> str:
    """
    Check if the subject ID is valid and return it in uppercase.

    Args:
        text (str): Raw subject identifier.

    Returns:
        str: Normalized subject ID.

    Raises:
        ValueError: If the ID format is not allowed.
    """
    t = text.strip().upper()
    if SUBJECT_PATTERNS[0].match(t):
        return t
    if SUBJECT_PATTERNS[1].match(t):
        return t

    # The input must match one of the two allowed formats
    raise ValueError("Subject ID must be '001' or 'P001' (3 digits, optional 'P').")


@dataclass
class Segment:
    """
    Represents a single labeled part of an audio recording.

    Attributes:
        id (str): Unique ID for the segment.
        t_start (float): Start time in seconds.
        t_end (float): End time in seconds.
        labels (List[str]): One or more labels assigned to this segment.
    """
    id: str
    t_start: float
    t_end: float
    labels: List[str] = field(default_factory=list)


@dataclass
class FileState:
    """
    Holds all information about a single audio file, including metadata and segments.

    Attributes:
        file (str): File path or name.
        sr (int): Sample rate.
        meta (dict): Extra metadata such as subject or microphone info.
        segments (List[Segment]): List of segment objects.
    """
    file: str
    sr: int
    meta: Dict[str, object] = field(default_factory=dict)
    segments: List[Segment] = field(default_factory=list)

    def to_json(self) -> dict:
        """
        Convert the file state to a JSON-safe dictionary.
        """
        return {
            "file": self.file,
            "sr": self.sr,
            "meta": self.meta,
            "segments": [asdict(s) for s in self.segments],
        }

    @staticmethod
    def from_json(d: dict) -> "FileState":
        """
        Create a FileState object from a JSON dictionary.

        Args:
            d (dict): Input dictionary with file info.

        Returns:
            FileState: Restored object with its segments.
        """
        fs = FileState(
            file=d.get("file", ""),
            sr=int(d.get("sr", 16000)),
            meta=d.get("meta", {}),
            segments=[],
        )

        # Rebuild all segment objects from dictionaries
        for s in d.get("segments", []):
            fs.segments.append(Segment(**s))

        return fs


@dataclass
class MetadataV2:
    """
    Represents the version 2 schema for metadata.

    Attributes:
        schema_version (int): The schema version number (always 2).
        subject_id (str | None): Optional subject identifier.
        mic_type (str | None): Microphone type or name.
        location (str | None): Recording location.
    """
    schema_version: int = 2
    subject_id: Optional[str] = None
    mic_type: Optional[str] = None
    location: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetadataV2":
        """
        Create a MetadataV2 instance from a dictionary.

        Handles migration from version 1 to version 2 automatically.
        """
        ver = d.get("schema_version", 1)

        if ver == 1:
            # Old schema may contain unused fields like gender or age
            return cls(
                subject_id=d.get("subject_id"),
                mic_type=d.get("mic_type"),
                location=d.get("location"),
            )

        # Directly build using v2 fields
        return cls(
            subject_id=d.get("subject_id"),
            mic_type=d.get("mic_type"),
            location=d.get("location"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this object to a simple dictionary.
        """
        return asdict(self)

    def validate(self) -> None:
        """
        Validate and fix fields if needed.

        Ensures the subject ID follows the expected pattern.
        """
        if self.subject_id:
            # Normalize to uppercase and check pattern
            self.subject_id = normalize_subject_id(self.subject_id)

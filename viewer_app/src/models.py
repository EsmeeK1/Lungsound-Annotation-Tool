from dataclasses import dataclass, asdict, field
import re
from typing import Optional, Dict, Any, List

SUBJECT_PATTERNS = [
    re.compile(r"^\d{3}$"),        # 001
    re.compile(r"^P\d{3}$"),       # P001
]

def normalize_subject_id(text: str) -> str:
    t = text.strip().upper()
    if SUBJECT_PATTERNS[0].match(t):   # 001
        return t
    if SUBJECT_PATTERNS[1].match(t):   # P001
        return t
    raise ValueError("Subject ID moet '001' of 'P001' zijn (3 cijfers, optionele 'P' prefix).")

@dataclass
class Segment:
    id: str
    t_start: float
    t_end: float
    labels: List[str] = field(default_factory=list)

@dataclass
class FileState:
    file: str
    sr: int
    meta: Dict[str, object] = field(default_factory=dict)
    segments: List[Segment] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "file": self.file,
            "sr": self.sr,
            "meta": self.meta,
            "segments": [asdict(s) for s in self.segments],
        }

    @staticmethod
    def from_json(d: dict) -> "FileState":
        fs = FileState(file=d.get("file", ""), sr=int(d.get("sr", 16000)),
                       meta=d.get("meta", {}), segments=[])
        for s in d.get("segments", []):
            fs.segments.append(Segment(**s))
        return fs

@dataclass
class MetadataV2:
    schema_version: int = 2
    subject_id: Optional[str] = None
    mic_type: Optional[str] = None
    location: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetadataV2":
        ver = d.get("schema_version", 1)
        if ver == 1:
            # migrate v1 → v2 (drop gender/age)
            return cls(
                subject_id=d.get("subject_id"),
                mic_type=d.get("mic_type"),
                location=d.get("location"),
            )
        return cls(
            subject_id=d.get("subject_id"),
            mic_type=d.get("mic_type"),
            location=d.get("location"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.subject_id:
            self.subject_id = normalize_subject_id(self.subject_id)
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional

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

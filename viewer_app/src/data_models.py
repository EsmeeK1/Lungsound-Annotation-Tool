from dataclasses import dataclass, asdict, field
from typing import Dict, List


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
            "segments": [asdict(segment) for segment in self.segments],
        }

    @staticmethod
    def from_json(data: dict) -> "FileState":
        state = FileState(
            file=data.get("file", ""),
            sr=int(data.get("sr", 16000)),
            meta=data.get("meta", {}),
            segments=[],
        )

        for segment_data in data.get("segments", []):
            state.segments.append(Segment(**segment_data))

        return state
from dataclasses import dataclass, asdict, field
from typing import Dict, List

@dataclass(frozen=True)
class AudioItem:
    """
    One item in the app's navigation queue.

    For normal short WAV files:
      - chunk_start = 0
      - chunk_end = full duration
      - chunk_count = 1

    For long WAV files:
      - multiple AudioItem objects point to the same source_path
      - each item represents a virtual 1-minute chunk
    """
    source_path: str
    chunk_start: float
    chunk_end: float
    chunk_index: int = 1
    chunk_count: int = 1

    @property
    def duration(self) -> float:
        return max(0.0, float(self.chunk_end) - float(self.chunk_start))

    @property
    def is_chunked(self) -> bool:
        return self.chunk_count > 1

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
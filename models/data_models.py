from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VideoData:
    video_id: str
    video_url: str
    title: str
    description: str
    video_path: Optional[Path]
    transcript_path: Optional[Path]


@dataclass
class VideoSegmentData:
    parent_video_id: str
    parent_video_path: Path
    parent_transcript_path: Path
    video_segment_id: int
    video_segment_path: Path
    video_segment_transcript_path: Path
    extracted_frames_path: Path
    duration_ms: float
    start_time_ms: float
    mid_time_ms: float
    end_time_ms: float

from typing import Optional

from lancedb.pydantic import LanceModel, Vector
from pydantic import Field


# Define VideoSegmentData as a Pydantic model
class VideoSegmentModel(LanceModel):
    parent_video_id: str
    parent_video_path: str  # Changed from Path to str for compatibility with PyArrow
    parent_audio_path: str  # Changed from Path to str for compatibility
    parent_vtt_path: str
    video_segment_id: int
    video_segment_path: str
    video_segment_transcript_path: str
    extracted_frame_path: str
    transcript: str
    enriched_transcript: str
    duration_ms: float
    start_ms: float
    mid_ms: float
    end_ms: float
    embeddings: Vector(1536)  # type: ignore


# Define VideoData as a LanceModel for use with LanceDB
class VideoModel(LanceModel):
    video_id: str
    video_url: str
    title: str
    description: str
    language: str = Field(default="en")
    video_path: str
    audio_path: str
    transcript_path_vtt: str
    transcript_path_text: str
    description_path: Optional[str]

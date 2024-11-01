import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from utils.logger import logger


@dataclass
class VideoSegmentData:
    parent_video_id: str
    parent_video_path: Path
    parent_audio_path: Optional[Path]
    parent_vtt_path: Optional[Path]
    video_segment_id: int
    video_segment_path: Path
    video_segment_transcript_path: Path
    extracted_frame_path: Path
    transcript: Optional[str] = field(init=False, default=None)
    enriched_transcript: Optional[str] = field(init=False, default=None)
    frame: Optional[np.ndarray] = field(init=False, default=None)
    duration_ms: float
    start_ms: float
    mid_ms: float
    end_ms: float
    embeddings: List[float] = field(init=False)

    def __post_init__(self):
        warnings.warn(
            "The `VideoSegmentData` dataclass is planned for deprecation in future versions. "
            "Please consider updating your code to use the new data model once available.",
            category=FutureWarning,
            stacklevel=2,
        )

    def __repr__(self):
        return f"VideoSegmentData(video_segment_id={self.video_segment_id}, start_ms={self.start_ms}, end_ms={self.end_ms}, duration_ms={self.duration_ms}, transcript={self.transcript}, parent_video_id={self.parent_video_id})"


@dataclass
class VideoData:
    video_id: str
    video_url: str
    title: str
    description: str
    language: str = field(init=False, default="en")
    video_path: Path
    audio_path: Optional[Path] = field(init=False, default=None)
    transcript_path_vtt: Optional[Path] = field(init=False, default=None)
    transcript_path_text: Optional[Path] = field(init=False, default=None)
    description_path: Optional[Path] = field(init=False, default=None)
    segments: List[VideoSegmentData] = field(init=False, default_factory=list)

    def __post_init__(self):
        warnings.warn(
            "The `VideoData` dataclass is planned for deprecation in future versions. "
            "Please consider updating your code to use the new data model once available.",
            category=FutureWarning,
            stacklevel=2,
        )

    def add_segement(self, segment: VideoSegmentData) -> None:
        """
        Add a segment to the video data.

        Args:
            segment (VideoSegmentData): Video segment data to be added.

        Raises:
            ValueError: If the segment parent_video_id does not match the video_id.
        """
        if segment.parent_video_id != self.video_id:
            raise ValueError(f"Segment parent_video_id {segment.parent_video_id} does not match video_id {self.video_id}")

        # Add the segment to the segments dictionary
        self.segments.append(segment)

    def get_segment(self, segment_id: int) -> Optional[VideoSegmentData]:
        """
        Get a segment from the video data based on the segment_id.

        Args:
            segment_id (int): ID of the segment to be retrieved.

        Returns:
            Optional[VideoSegmentData]: Video segment data if found, None otherwise.
        """

        # Get the segment with the specified segment_id
        return self.segments[segment_id]

    def get_segments_chronologically(self) -> List[VideoSegmentData]:
        """
        Get the video segments in chronological order.

        Returns:
            List[VideoSegmentData]: List of video segments in chronological order.
        """
        return sorted(self.segments, key=lambda segment: segment.start_ms)

    def get_nearest_neighbours(self, segment_id: int, n: int) -> List[VideoSegmentData]:
        """
        Get the nearest neighbours of a specific segment.

        Args:
            segment_id (int): ID of the segment to find nearest neighbours of.
            n (int): Number of nearest neighbours to return.

        Returns:
            List[VideoSegmentData]: List of nearest neighbours.
        """

        # Get n segments before and after the specified segment (Do not go out of bounds)
        return self.segments[max(0, segment_id - n) : min(len(self.segments), segment_id + n + 1)]

    def remove_segment(self, segment_id: int) -> None:
        """
        Remove a segment from the video data.
        Args:
            segment_id (int): ID of the segment to be removed.
        """

        # Remove the segment with the specified segment_id
        self.segments.pop(segment_id)
        logger.debug(f"Removed segment {segment_id} from video {self.video_id}")

    def clear_segments(self) -> None:
        """
        Clear all segments from the video data.
        """
        self.segments.clear()
        logger.debug(f"Cleared segments from video {self.video_id}")

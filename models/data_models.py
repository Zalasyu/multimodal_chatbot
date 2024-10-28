from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
    duration_ms: float
    start_ms: float
    mid_ms: float
    end_ms: float


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
    segments: Dict[int, VideoSegmentData] = field(init=False, default_factory=dict)

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
        self.segments[segment.video_segment_id] = segment
        logger.debug(f"Added segment {segment.video_segment_id} to video {self.video_id}")

    def get_segment(self, segment_id: int) -> Optional[VideoSegmentData]:
        """
        Get a segment from the video data based on the segment_id.

        Args:
            segment_id (int): ID of the segment to be retrieved.

        Returns:
            Optional[VideoSegmentData]: Video segment data if found, None otherwise.
        """
        return self.segments.get(segment_id)

    def get_segments_chronologically(self) -> List[VideoSegmentData]:
        """
        Get the video segments in chronological order.

        Returns:
            List[VideoSegmentData]: List of video segments in chronological order.
        """
        return sorted(self.segments.values(), key=lambda x: x.start_ms)

    def get_segments_by_timerange(self, start_ms: float, end_ms: float) -> List[VideoSegmentData]:
        """
        Get the video segments within a specific time range.

        Args:
            start_time_ms (float): Start time in milliseconds.
            end_time_ms (float): End time in milliseconds.

        Returns:
            List[VideoSegmentData]: List of video segments within the time range.
        """
        return [segment for segment in self.segments.values() if segment.start_ms <= end_ms and segment.end_ms >= start_ms]

    def remove_segment(self, segment_id: int) -> None:
        """
        Remove a segment from the video data.

        Args:
            segment_id (int): ID of the segment to be removed.
        """
        del self.segments[segment_id]

    def clear_segments(self) -> None:
        """
        Clear all segments from the video data.
        """
        self.segments.clear()
        logger.debug(f"Cleared segments from video {self.video_id}")

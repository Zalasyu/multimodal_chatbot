from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import webvtt
from colorama import Fore, Style
from tqdm import tqdm

from models.data_models import VideoData, VideoSegmentData
from utils.helpers import load_video_data
from utils.logger import logger


class Preprocessor:
    def __init__(
        self,
        base_output_path: Path,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        frame_quality: int = 95,
    ):
        """
        Initialize the Preprocessor.

        Args:
            base_output_path (Path): Base output path
            target_width (Optional[int], optional): Target width for extracted frames. Defaults to None.
            target_height (Optional[int], optional): Target height for extracted frames. Defaults to None.
            frame_quality (int, optional): JPEG quality for extracted frames[0-100]. Defaults to 95.
        """
        self.base_output_path = base_output_path
        self.target_width = target_width
        self.target_height = target_height
        self.frame_quality = frame_quality

        # Create the base output path if it doesn't exist
        self.base_output_path.mkdir(parents=True, exist_ok=True)

    def _str_to_timestamp_milliseconds(self, time_str: str) -> float:
        """
        Convert a string in the format of "HH:MM:SS.sss" to a timestamp in milliseconds

        Args:
            time_str (str): The string to be converted (format: HH:MM:SS.sss)

        Returns:
            float: The timestamp in milliseconds
        """
        hours, minutes, seconds = time_str.split(":")
        seconds, milliseconds = seconds.split(".")

        total_milliseconds = (
            int(hours) * 3600000  # hours to ms
            + int(minutes) * 60000  # minutes to ms
            + int(seconds) * 1000  # seconds to ms
            + int(milliseconds)  # already ms
        )

        return float(total_milliseconds)

    def _parse_vtt_segments(self, video_data: VideoData) -> List[Tuple[float, float, float, str]]:
        """
        Parse the WebVTT segments from the transcript file and  extract segment timings.


        Args:
            video_data (VideoData): The video data with the transcript path.

        Returns:
            List[Tuple[float, float, float]]: List of segment timings in the format (start_ms, mid_ms, end_ms)
        """

        if not video_data.transcript_path_vtt or not video_data.transcript_path_vtt.exists():
            raise FileNotFoundError(f"Transcript file not found: {video_data.transcript_path_vtt}")

        segments = []
        vtt_content: webvtt.WebVTT = webvtt.read(file=video_data.transcript_path_vtt)

        for idx, transcript_segment in enumerate(vtt_content):

            # SKip segment with only one line
            if len(transcript_segment.text.splitlines()) == 1:
                continue

            # Check if the next segment's first line matches the current segment's last line
            # If so then remove the second line
            # Check if there is a next segment
            if idx < len(vtt_content) - 1:
                next_transcript_segment = vtt_content[idx + 1]
                if transcript_segment.text.splitlines()[-1] == next_transcript_segment.text.splitlines()[0]:
                    transcript_segment.text = transcript_segment.text.splitlines()[0]

            # Get the start and end times in milliseconds
            start_ms = self._str_to_timestamp_milliseconds(transcript_segment.start)
            end_ms = self._str_to_timestamp_milliseconds(transcript_segment.end)
            mid_ms = (start_ms + end_ms) / 2

            # Add the segment to the list
            segments.append((start_ms, mid_ms, end_ms, transcript_segment.text))
            logger.debug(f"Added segment: {transcript_segment.text}")

        # Remove the last segment
        if len(segments) > 0:
            segments.pop()

        return segments

    def process_video(self, video_data: VideoData) -> VideoData:
        """
        Extract frames and corresponding metadata from a video

        Args:
            video_data (VideoData): The video data to extract the frames and metadata from

        Returns:
            VideoData: The updated video data that has the path to the extracted frames and metadata for itself.

        Raises:
        """

        if not video_data.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_data.video_path}")

        try:

            # Parse the WebVTT segments
            segment_timings: List[Tuple[float, float, float, str]] = self._parse_vtt_segments(video_data=video_data)

            video = cv2.VideoCapture(video_data.video_path)
            if not video.isOpened():
                raise RuntimeError(f"Could not open video file: {video_data.video_path}")

            # Process each segment
            for segment_id, (start_ms, mid_ms, end_ms, transcript_segment) in tqdm(
                enumerate(segment_timings),
                total=len(segment_timings),
                desc=f"{Fore.CYAN}Processing video: {video_data.title} {Style.RESET_ALL}",
            ):

                # Create segment output directory
                segment_dir: Path = self.base_output_path / video_data.video_id / f"segment_{segment_id}"
                segment_dir.mkdir(parents=True, exist_ok=True)

                # Create SegementData
                segment = VideoSegmentData(
                    parent_video_id=video_data.video_id,
                    parent_video_path=video_data.video_path,
                    parent_audio_path=video_data.audio_path,
                    parent_vtt_path=video_data.transcript_path_vtt,
                    video_segment_id=segment_id,
                    video_segment_path=Path(segment_dir / f"segment_{segment_id}.mp4"),
                    video_segment_transcript_path=Path(segment_dir / f"segment_{segment_id}.vtt"),
                    extracted_frame_path=Path(segment_dir / f"frame_{int(mid_ms)}.jpg"),
                    duration_ms=end_ms - start_ms,
                    start_ms=start_ms,
                    mid_ms=mid_ms,
                    end_ms=end_ms,
                )

                # Extract frame
                resized_frame = self._extract_frame(video=video, frame_path=segment.extracted_frame_path, timestamp_ms=mid_ms)

                # Save transcript
                if resized_frame is not None:
                    # Write transcript to file
                    with open(segment.video_segment_transcript_path, "w", encoding="utf-8") as f:
                        f.write(transcript_segment)

                    # Assign transcript
                    segment.transcript = transcript_segment

                    # Assign frame
                    segment.frame = resized_frame

                else:
                    logger.warning(
                        f"Failed to extract frame for segment {segment.video_segment_id} of video {video_data.video_id}"
                    )

                # Add segment to video data
                if resized_frame is not None:
                    video_data.add_segement(segment=segment)
                else:
                    logger.warning(
                        f"Failed to extract frame for segment {segment.video_segment_id} of video {video_data.video_id}"
                    )

            return video_data

        except Exception as e:
            logger.error(f"Error processing video {video_data.video_id}: {str(e)}")

        finally:
            video.release()

        return video_data

    def _maintain_aspect_ratio_resize(
        self, width: int, height: int, target_width: Optional[int] = None, target_height: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Maintain aspect ratio resize of an image.

        Args:
            width (int): The original width of the image
            height (int): The original height of the image
            target_width (Optional[int], optional): The target width of the image. Defaults to None.
            target_height (Optional[int], optional): The target height of the image. Defaults to None.

        Returns:
            Tuple[int, int]: The resized width and height
        """

        if target_width is None and target_height is None:
            raise ValueError("Either target_width or target_height must be provided.")

        aspect_ratio = width / height

        if target_width is not None:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)  # type: ignore
        else:
            new_width = int(target_height * aspect_ratio)  # type: ignore
            new_height = target_height  # type: ignore

        return new_width, new_height

    def _extract_frame(self, video: cv2.VideoCapture, frame_path: Path, timestamp_ms: float) -> Optional[np.ndarray]:
        """
        Extract a frame from a video.

        Args:
            video (cv2.VideoCapture): OpenCV video capture object
            frame_path (Path): Where to save the frame
            timestamp_ms (float): The timestamp of the frame in milliseconds

        Returns:
            resized_frame (Optional[np.ndarray]): The resized frame; None if the frame could not be extracted
        """
        try:

            # Get original dimensions
            original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Maintain aspect ratio resize
            new_width, new_height = self._maintain_aspect_ratio_resize(
                width=original_width, height=original_height, target_width=self.target_width, target_height=self.target_height
            )

            # Set frame position
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_position = int(timestamp_ms / 1000 * fps)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

            # Extract frame
            ret, frame = video.read()

            if not ret:
                logger.warning(f"Could not extract frame at timestamp: {timestamp_ms}")
                return None

            # Resize frame
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # Ensure parent directories exist
            frame_path.parent.mkdir(parents=True, exist_ok=True)

            # Save frame
            cv2.imwrite(str(frame_path), resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.frame_quality])

            return resized_frame

        except Exception as e:
            logger.error(f"Error extracting frame at timestamp {timestamp_ms}: {str(e)}")
            return None


if __name__ == "__main__":
    absolute_video_data_path = Path(
        "/home/zalasyu/Documents/projects/multimodal_chatbot/data/interim/video_data/KLLgGg4tmYs.json"
    )
    # Read the video data from the JSON file
    video_data = load_video_data(absolute_video_data_path)

    # Create the base output path if it doesn't exist
    base_output_path = Path(f"/home/zalasyu/Documents/projects/multimodal_chatbot/data/processed/video_frames/")

    # Create an instance of the Preprocessor
    preprocess = Preprocessor(base_output_path=base_output_path, target_height=350)

    # Extract frames and corresponding metadata from the video
    preprocess.process_video(video_data=video_data)

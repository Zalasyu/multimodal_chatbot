import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, TypeVar

from models.data_models import VideoData, VideoSegmentData
from utils.logger import logger


def format_timestamp(ms: float) -> str:
    """
    Convert millisecondss to human readable format.

    Args:
        ms (float): Timestamp in milliseconds

    Returns:
        str: Timestamp in human readable format
    """
    seconds = int(ms / 1000)
    minutes = seconds // 60
    seconds = seconds % 60

    return f"{minutes:02d}:{seconds:02d}"


T = TypeVar("T")


class VideoDataJSONEncoder:
    @staticmethod
    def encode(video_data: VideoData) -> Dict[str, Any]:
        """
        Convert a VideoData object to a JSON-serializable dictionary.

        Args:
            video_data (VideoData): The video data object to encode

        Returns:
            Dict[str, Any]: JSON-serializable dictionary
        """
        # Convert to dict using dataclasses.asdict
        data_dict = asdict(video_data)

        # Convert Path objects to strings
        for key, value in data_dict.items():
            if isinstance(value, Path):
                data_dict[key] = str(value)

            # Handle the segments list specially
            if key == "segments":
                data_dict[key] = [
                    {
                        **segment,
                        "parent_video_path": str(segment["parent_video_path"]),
                        "parent_audio_path": str(segment["parent_audio_path"]) if segment["parent_audio_path"] else None,
                        "parent_vtt_path": str(segment["parent_vtt_path"]) if segment["parent_vtt_path"] else None,
                        "video_segment_path": str(segment["video_segment_path"]),
                        "video_segment_transcript_path": str(segment["video_segment_transcript_path"]),
                        "extracted_frame_path": str(segment["extracted_frame_path"]),
                    }
                    for segment in value
                ]

        return data_dict

    @staticmethod
    def decode(data_dict: Dict[str, Any]) -> VideoData:
        """
        Convert a dictionary to a VideoData object.

        Args:
            data_dict (Dict[str, Any]): Dictionary containing video data

        Returns:
            VideoData: Reconstructed VideoData object
        """
        # First create the base VideoData object with required fields
        video_data = VideoData(
            video_id=data_dict["video_id"],
            video_url=data_dict["video_url"],
            title=data_dict["title"],
            description=data_dict["description"],
            video_path=Path(data_dict["video_path"]),
        )

        # Set optional fields if they exist
        if "audio_path" in data_dict and data_dict["audio_path"]:
            video_data.audio_path = Path(data_dict["audio_path"])

        if "transcript_path_vtt" in data_dict and data_dict["transcript_path_vtt"]:
            video_data.transcript_path_vtt = Path(data_dict["transcript_path_vtt"])

        if "transcript_path_text" in data_dict and data_dict["transcript_path_text"]:
            video_data.transcript_path_text = Path(data_dict["transcript_path_text"])

        if "description_path" in data_dict and data_dict["description_path"]:
            video_data.description_path = Path(data_dict["description_path"])

        # Set other fields
        video_data.summary_abstractive = data_dict.get("summary_abstractive", "")
        video_data.summary_extractive = data_dict.get("summary_extractive", "")
        video_data.language = data_dict.get("language", "en")
        video_data.transcribed = data_dict.get("transcribed", False)

        # Reconstruct segments if they exist
        if "segments" in data_dict:
            for segment_dict in data_dict["segments"]:
                segment = VideoSegmentData(
                    parent_video_id=segment_dict["parent_video_id"],
                    parent_video_path=Path(segment_dict["parent_video_path"]),
                    parent_audio_path=Path(segment_dict["parent_audio_path"]) if segment_dict["parent_audio_path"] else None,
                    parent_vtt_path=Path(segment_dict["parent_vtt_path"]) if segment_dict["parent_vtt_path"] else None,
                    video_segment_id=segment_dict["video_segment_id"],
                    video_segment_path=Path(segment_dict["video_segment_path"]),
                    video_segment_transcript_path=Path(segment_dict["video_segment_transcript_path"]),
                    extracted_frame_path=Path(segment_dict["extracted_frame_path"]),
                    duration_ms=segment_dict["duration_ms"],
                    start_ms=segment_dict["start_ms"],
                    mid_ms=segment_dict["mid_ms"],
                    end_ms=segment_dict["end_ms"],
                )

                # Set optional segment fields
                if "transcript" in segment_dict:
                    segment.transcript = segment_dict["transcript"]
                if "enriched_transcript" in segment_dict:
                    segment.enriched_transcript = segment_dict["enriched_transcript"]
                if "embeddings" in segment_dict:
                    segment.embeddings = segment_dict["embeddings"]

                video_data.add_segement(segment)

        return video_data


def save_video_data(video_data: VideoData, save_path: Path) -> None:
    """
    Save a VideoData object to a JSON file

    Args:
        video_data (VideoData): The video data
        save_path (Union[str]): The path to save the JSON file`:w

    """
    try:
        # Ensure the parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file=save_path, mode="w", encoding="utf-8") as f:
            json_data = VideoDataJSONEncoder.encode(video_data)
            json.dump(json_data, f, indent=4)
            logger.debug(f"Saved video data to {save_path}")
    except Exception as e:
        raise OSError(f"Failed to save video data: {e}")


def load_video_data(load_path: Path) -> VideoData:
    """
    Load a VideoData object from a JSON file

    Args:
        load_path (Union[str]): The path to load the JSON file

    Returns:
        VideoData: The VideoData object.
    """
    try:
        with open(file=load_path, encoding="utf-8") as f:
            json_data = json.load(f)
            video_data = VideoDataJSONEncoder.decode(json_data)
            logger.debgu(f"Loaded video data from {load_path}")
    except Exception as e:
        raise OSError(f"Failed to load video data: {e}")

    return video_data


def enrich_segment_transcripts(video_data: VideoData, segment: VideoSegmentData) -> VideoSegmentData:
    """Augment a transcript with transcripts of n-neighbouring segments.
    Observation: Transcripts of frames extracted from a video are usually fragemented and even with an incomplete sentence.
    - Such transripts are not meaningful and are not useful for retrieval.

    Naive Solution:
    - Extract n-neighbouring segments
    - Concatenate the transcript of the n-neighbouring segments

    Advise:
    - Should pick an individual n for each video such that updated transcripts
    say one or two meaningful facts.
    Args:
        video_data (VideoData): Video Data of the video
        segment (VideoSegmentData): Segment Data of the segment

    Returns:
        VideoSegmentData: Augmented Segment Data
    """

    # Get n segments before and after the specified segment (Use VideoData)
    neighbouring_segments: List[VideoSegmentData] = video_data.get_nearest_neighbours(segment_id=segment.video_segment_id, n=12)

    # Extract transcripts of these segments
    neighbouring_transcripts: list[str] = []
    for neighbour_segment in neighbouring_segments:
        neighbouring_transcripts.append(str(neighbour_segment.transcript))

    # Concatenate transcripts
    segment.enriched_transcript = " ".join(neighbouring_transcripts)

    return segment

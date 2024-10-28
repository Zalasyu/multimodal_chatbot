import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, TypeVar

from models.data_models import VideoData
from utils.logger import logger

T = TypeVar("T")


class VideoDataJSONEncoder:
    @staticmethod
    def encode(video_data: VideoData) -> Dict[str, Any]:
        """
        Convert a VideoData object to a dictionary

        Args:
            video_data (VideoData): The video data

        Returns:
            Dict[str, Any]: The dictionary representation of the video data
        """

        data_dict = asdict(video_data)

        # Convert VideoData to a JSON-serializable dictionary
        for key, value in data_dict.items():
            if isinstance(value, Path):
                data_dict[key] = str(value)

        return data_dict

    # TODO: Technical DEBT
    @staticmethod
    def decode(data_dict: Dict[str, Any]) -> VideoData:
        """
        Convert a dictionary to a VideoData object

        Args:
            data_dict (Dict[str, Any]): The dictionary to be converted

        Returns:
            VideoData: The VideoData object
        """
        logger.debug(f"Data dict: {data_dict}")

        # Convert string paths to Path objects
        path_fields = [
            "video_path",
            "audio_path",
            "transcript_path_vtt",
            "transcript_path_text",
            "description_path",
            "processed_video_path",
        ]

        for field in path_fields:
            if data_dict.get(field):
                data_dict[field] = Path(data_dict[field])

        logger.debug(f"Data dict: {data_dict}")

        # Only pass the fields that are allowed during VideoData initialization
        video_data = VideoData(
            video_id=data_dict["video_id"],
            video_url=data_dict["video_url"],
            title=data_dict["title"],
            description=data_dict["description"],
            video_path=data_dict["video_path"],
        )

        # Then set the other fields after initialization
        video_data.audio_path = data_dict["audio_path"]
        video_data.transcript_path_vtt = data_dict["transcript_path_vtt"]
        video_data.transcript_path_text = data_dict["transcript_path_text"]
        video_data.description_path = data_dict["description_path"]

        return video_data


def save_video_data(video_data: VideoData, save_path: Path) -> None:
    """
    Save a VideoData object to a JSON file

    Args:
        video_data (VideoData): The video data
        save_path (Union[str]): The path to save the JSON file`:w

    """
    try:
        with open(file=save_path, mode="w", encoding="utf-8") as f:
            json_data = VideoDataJSONEncoder.encode(video_data)
            json.dump(json_data, f, indent=4)
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
    except Exception as e:
        raise OSError(f"Failed to load video data: {e}")

    return video_data

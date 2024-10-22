from pathlib import Path
from typing import Optional

from models.data_models import VideoData
from utils.logger import logger


class VideoDescriptionService:
    def __init__(self, description_download_path: Path):
        self.description_download_path: Path = description_download_path
        logger.debug(f"VideoDescriptionService initialized with description_download_path: {description_download_path}")

    def describe_video(self, video_data: VideoData) -> VideoData:
        """
        Describes the video

        Args:
            video_data (VideoData): The video metadata

        Returns:
            bool: True if the video was described, False otherwise
        """
        logger.info(f"Starting description for: {video_data.title}")

        try:
            description_text: str = self._create_video_description(video_path=video_data.video_path)

            self._save_video_description(description_text=description_text, description_path=self.description_download_path)

            logger.info(f"Description for {video_data.title} was saved")

            return video_data

        except Exception as e:
            logger.error(f"Failed to describe {video_data.title}: {e}")
            return video_data

    def _create_video_description(self, video_path: Optional[Path]) -> str:
        """
        Describes the video's content.

        Args:
            video_path (Path): The path to the video file

        Returns:
            str: The description of the video
        """
        logger.debug(f"Creating description for {video_path}")
        return "Placeholder description"

    def _save_video_description(self, description_text: str, description_path: Path) -> None:
        """
        Saves the description of the video

        Args:
            description_text (str): The description of the video
            description_path (Path): The path to the description file
        """
        if not description_text:
            logger.error(f"No description for {description_path}")
            return

        logger.debug(f"Saving description for {description_path}")
        with description_path.open("w") as f:
            f.write(description_text)

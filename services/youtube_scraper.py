from pathlib import Path
from typing import Optional

from models.data_models import VideoData
from services.transcript_service import TranscriptService
from services.video_downloader import VideoDownloader
from utils.logger import logger


class YouTubeScraper:
    def __init__(self, base_download_path: Path):
        self.base_download_path: Path = base_download_path
        self.video_download_path: Optional[Path] = Path(base_download_path, "videos/YouTube")
        self.transcript_download_path: Optional[Path] = Path(base_download_path, "transcripts")
        self.transcript_service = TranscriptService(self.transcript_download_path)
        self.video_downloader = VideoDownloader(self.video_download_path)

    def process_video(self, video_url: str) -> VideoData:
        """
        Downloads a YouTube video

        Args:
            video_url (str): The URL of the YouTube video

        Returns:
            Video: The downloaded video
        """
        logger.info(f"Processing {video_url}")

        # Step 1: Download video
        video_data = self.video_downloader.download_video(video_url)
        logger.debug(f"Downloaded {video_data.title} (ID: {video_data.video_id})")

        # Step 2: Download transcript
        self.transcript_service.fetch_transcript(video_data.video_id)

        return video_data

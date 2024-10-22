from pathlib import Path

from handlers.video_processing_handler import VideoProcessingHandler
from models.data_models import VideoData
from services.transcript_service import TranscriptService
from utils.logger import logger


class TranscriptAvailableHandler(VideoProcessingHandler):

    def __init__(self, transcript_download_path: Path):
        super().__init__()
        self.transcript_service = TranscriptService(transcript_download_path=transcript_download_path)

    def handle(self, video_data: VideoData) -> VideoData:
        """
        Checks if a transcript is available for the given video data and updates the VideoData model accordingly.

        Args:
            video_data (VideoData): The video data to be processed

        Returns:
            VideoData: The processed video data
        """
        logger.info(f"Checking if transcript is available for: {video_data.title}")

        # Fetch the transcript if it does exist
        self.transcript_service.fetch_transcript(video_data=video_data)

        if video_data.transcript_path:
            logger.info(f"Transcript is available for: {video_data.title}")
            return video_data

        logger.info(f"Transcript is not available for: {video_data.title}")

        # Pass the video data to the next handler in the chain
        self.send_to_next(video_data=video_data)

        return video_data

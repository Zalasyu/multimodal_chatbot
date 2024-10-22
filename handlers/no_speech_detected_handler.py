from pathlib import Path

from handlers.video_processing_handler import VideoProcessingHandler
from models.data_models import VideoData
from services.video_description_service import VideoDescriptionService
from utils.logger import logger


class NoSpeechDetectedHandler(VideoProcessingHandler):

    def __init__(self, video_description_path: Path):
        super().__init__()
        self.video_description_service = VideoDescriptionService(description_download_path=video_description_path)

    def handle(self, video_data: VideoData) -> VideoData:
        logger.info(f"No speech detected for: {video_data.title} with ID: {video_data.video_id}")

        # Generate a description for the video
        description_generated = self.video_description_service.describe_video(video_data=video_data)

        if description_generated:
            logger.info(f"Description for {video_data.title} was generated")
        else:
            logger.info(f"Description for {video_data.title} was not generated")

        return video_data

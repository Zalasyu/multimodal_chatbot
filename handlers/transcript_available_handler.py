from handlers.video_processing_handler import VideoProcessingHandler
from models.data_models import VideoData
from services.transcript_service import TranscriptService
from utils.logger import logger


class TranscriptAvailableHandler(VideoProcessingHandler):

    def __init__(self):
        super().__init__()
        self.transcript_service = TranscriptService()

    def handle(self, video_data: VideoData) -> VideoData:
        logger.info(f"Checking if transcript is available for: {video_data.title}")
        transcript = self.transcript_service.fetch_transcript(video_data.video_id)

        if transcript is not None:
            video_data.transcript_path = transcript
            logger.info(f"Transcript for {video_data.title} was fetched")
            return video_data
        else:
            logger.info(f"Transcript for {video_data.title} is not available for {video_data.title}")
            if self.next_handler:
                return self.next_handler.handle(video_data)
            else:
                logger.info(f"No next handler found for {video_data.title}")
                return video_data

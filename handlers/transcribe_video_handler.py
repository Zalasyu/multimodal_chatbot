from pathlib import Path

from handlers.video_processing_handler import VideoProcessingHandler
from models.data_models import VideoData
from services.speech_to_text_service import SpeechToTextService
from utils.logger import logger


class TranscribeVideoHandler(VideoProcessingHandler):

    def __init__(self, transcript_download_path: Path):
        super().__init__()
        self.speech_to_text_service = SpeechToTextService(model_name="small", transcript_download_path=transcript_download_path)

    def handle(self, video_data: VideoData) -> VideoData:
        logger.debug(f"Transcribing video: {video_data.video_path}")
        transcription = self.speech_to_text_service.transcribe_video(video_data=video_data)

        if transcription:
            video_data.transcript_path = transcription.transcript_path
            return video_data

        logger.error(f"Failed to transcribe video: {video_data.video_path}")

        # Pass the video data to the next handler in the chain
        return self.send_to_next(video_data=video_data)

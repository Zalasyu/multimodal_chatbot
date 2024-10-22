from pathlib import Path

import whisper

from models.data_models import VideoData
from utils.logger import logger


class SpeechToTextService:
    def __init__(self, transcript_download_path: Path, model_name: str = "base") -> None:
        logger.debug(f"SpeechToTextService initializing with model_name: {model_name}")
        self.model = whisper.load_model(model_name)
        self.transcript_download_path: Path = transcript_download_path
        logger.info(
            f"SpeechToTextService initialized with model_name: {model_name} and subtitles_download_path: {transcript_download_path}"
        )

    def transcribe_video(self, video_data: VideoData) -> VideoData:
        """
        Transcribe a video using the Whisper model.

        Args:
            video_data (VideoData): The video metadata

        Returns:
            VideoData: The video metadata with the transcript path updated
        """
        logger.info(f"Starting transcription for: {video_data.title}")

        try:

            # Use the audio file in m4a if available; otherwise, use the video
            file_to_transcribe = video_data.audio_path if video_data.audio_path.is_file() else video_data.video_path
            result = self.model.transcribe(file_to_transcribe, fp16=False, task="transcribe", language="en")

            self._save_transcription_as_vtt(segments=result["segments"], vtt_file_path=self.transcript_download_path)

            logger.info(f"Transcription for {video_data.title} was saved")

            return video_data
        except Exception as e:
            logger.error(f"Error transcribing {video_data.title}: {e}")
            return video_data

    def _save_transcription_as_vtt(self, segments: list, vtt_file_path: Path):
        """
        Save the transcription segments as a WebVTT file.

        Args:
            segments (list): A list of transcription segments to be saved
            vtt_file_path (Path): The path to save the WebVTT file

        Returns:
            None
        """
        with open(vtt_file_path, encoding="utf-8", mode="w") as f:
            f.write("WEBVTT\n\n")
            for i, segment in enumerate(segments):
                start = self._format_timestamp(segment["start"])
                end = self._format_timestamp(segment["end"])
                text = segment["text"].strip()
                f.write(f"{start} --> {end}\n{text}\n\n")

    def _format_timestamp(self, seconds: float) -> str:
        """
        Format a timestamp given in seconds as a string in the format:
        HH:MM:SS.SSS

        Args:
            seconds (float): The timestamp to be formatted

        Returns:
            str: The formatted timestamp
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

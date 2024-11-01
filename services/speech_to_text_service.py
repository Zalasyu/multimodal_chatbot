from pathlib import Path

import whisper

from models.data_models import VideoData
from utils.logger import logger


class SpeechToTextService:
    def __init__(self, transcript_download_path: Path, model_name: str = "base") -> None:
        logger.debug(f"SpeechToTextService initializing with model_name: {model_name}")
        self.model = whisper.load_model(model_name)
        self.options = dict(
            task="transcribe",
            language="en",
            fp16=False,
            best_of=1,
            word_timestamps=True,
        )
        self.transcript_download_path: Path = transcript_download_path
        logger.debug(
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

            # Use the audio file
            file_to_transcribe = str(video_data.audio_path)

            # Transcribe the video
            results = self.model.transcribe(file_to_transcribe, **self.options)
            logger.info(f"Transcription for {video_data.title} was completed")

            # Create the VTT file path
            video_data.transcript_path_vtt = self._create_transcript_path(
                transcript_download_path=self.transcript_download_path, video_path=video_data.video_path, ext="vtt"
            )

            # Create the text file path
            video_data.transcript_path_text = self._create_transcript_path(
                transcript_download_path=self.transcript_download_path, video_path=video_data.video_path, ext="txt"
            )

            # Save the transcription as a WebVTT file
            self._save_transcription_as_vtt(segments=results["segments"], vtt_file_path=video_data.transcript_path_vtt)
            logger.info(f"Transcription for {video_data.title} was saved")

            # Save the transcription as a text file
            self._save_transcription_as_text(segments=results["segments"], text_file_path=video_data.transcript_path_text)
            logger.info(f"Transcription for {video_data.title} was saved")

            return video_data
        except Exception as e:
            logger.error(f"Error transcribing {video_data.title}: {e}")
            raise

    def _save_transcription_as_vtt(self, segments: list, vtt_file_path: Path) -> Path:
        """
        Save the transcription segments as a WebVTT file.

        Args:
            segments (list): A list of transcription segments to be saved
            vtt_file_path (Path): The path to save the WebVTT file

        Returns:
            None
        """
        logger.debug(f"Saving transcription as WebVTT file: {vtt_file_path}")
        with open(vtt_file_path, encoding="utf-8", mode="w") as f:
            f.write("WEBVTT\n\n")
            for i, segment in enumerate(segments):
                start = self._format_timestamp(segment["start"])
                end = self._format_timestamp(segment["end"])
                text = segment["text"].strip()
                f.write(f"{start} --> {end}\n{text}\n\n")

        return vtt_file_path

    def _save_transcription_as_text(self, segments: list, text_file_path: Path) -> Path:
        """
        Save the transcription segments as a text file.

        Args:
            segments (list): A list of transcription segments to be saved

        Returns:
            Path: The path to the saved text file
        """
        logger.debug("Saving transcription as text file")
        with open(text_file_path, encoding="utf-8", mode="w") as f:
            for segment in segments:
                text = segment["text"].strip()
                f.write(f"{text}\n")

        return text_file_path

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

    def _create_transcript_path(self, transcript_download_path: Path, video_path: Path, ext: str) -> Path:
        """
        Uses the video path's file name suffix and the transcript download path to
        create a new file path for the transcript.

        Args:
            transcript_download_path (Path): _description_
            video_path (Path): _description_

        Returns:
            Path: The new transcript path
        """
        new_transcript_path = Path(transcript_download_path, video_path.stem + "." + ext)
        logger.debug(f"Created transcript path: {new_transcript_path}")
        return new_transcript_path

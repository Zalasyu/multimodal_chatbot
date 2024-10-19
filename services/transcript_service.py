from pathlib import Path
from typing import Optional

from yt_dlp import YoutubeDL

from utils.logger import logger


class TranscriptService:

    def __init__(self, transcript_download_path: Path):
        self.transcript_download_path = transcript_download_path

    def fetch_transcript(self, video_id: str) -> Optional[str]:
        """
        Fetches the transcript of a YouTube video

        Args:
            video_id (str): The ID of the YouTube video

        Returns:
            Optional[str]: The transcript of the video, or None if the video does not have a transcript
        """

        ydl_opts = {
            "writesubtitles": True,
            "subtitleslangs": ["en"],
            "skip_download": True,
            "subtitlesformat": "vtt",
            "outtmpl": "subtitles/%(id)s.%(ext)s",
            "quiet": True,
        }

        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            logger.debug(f"Info dictionary: {info_dict}")

        return info_dict.get("requested_subtitles", {}).get("en")

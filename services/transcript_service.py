from pathlib import Path

from yt_dlp import YoutubeDL

from models.data_models import VideoData
from utils.logger import logger


class TranscriptService:

    def __init__(self, transcript_download_path: Path):
        self.transcript_download_path = transcript_download_path
        logger.debug(f"TranscriptService initialized with transcript_download_path: {transcript_download_path}")

    def fetch_transcript(self, video_data: VideoData) -> VideoData:
        """
        Fetches the transcript of a YouTube video

        Args:
            video_id (str): The ID of the YouTube video

        Returns:
            Optional[str]: The transcript of the video, or None if the video does not have a transcript
        """

        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "subtitleslangs": ["en", "en.*"],
            "subtitlesformat": "vtt",
            "outtmpl": f"{video_data.video_id}.%(ext)s",
            "paths": {"subtitles": self.transcript_download_path},
            "quiet": True,
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(video_data.video_id, download=False)

                if "subtitles" in info_dict and info_dict["requested_subtitles"]:
                    subtitle_info = info_dict["requested_subtitles"]
                    subtitle_file = Path(self.transcript_download_path, f"{video_data.video_id}.vtt")

                    with subtitle_file.open("w") as f:
                        f.write(subtitle_info["en"]["ext"])

                    video_data.transcript_path = subtitle_file

                    return video_data

        except Exception as e:
            logger.error(f"Failed to fetch transcript for video {video_data.video_id}: {e}")

        return video_data

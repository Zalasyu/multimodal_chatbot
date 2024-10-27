from pathlib import Path

from yt_dlp import DownloadError, YoutubeDL
from yt_dlp.utils import sanitize_filename

from models.data_models import VideoData
from utils.logger import logger


class AudioDownloader:
    def __init__(self, audio_download_path: Path):
        self.audio_download_path: Path = audio_download_path

    def download_audio(self, video_url: str, video_data: VideoData) -> VideoData:
        """
        Downloads a YouTube video

        Args:
            video_url (str): The URL of the YouTube video
            video_data (VideoData): The data of the YouTube video

        Returns:
            VideoData: The downloaded video
        """
        logger.info(f"Downloading {video_url}")

        ydl_opts = {
            "format": "bestaudio[ext=mp3]/best",
            "outtmpl": f"{self.audio_download_path}/%(title)s_%(id)s.%(ext)s",
            "writesubtitles": False,
            "writeautomaticsub": False,
            "restrictfilenames": True,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
            "quiet": True,
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:

                # Step 1: Download Video and Audio data
                info_dict = ydl.extract_info(video_url, download=True)

                logger.info(f"Downloaded audio for {info_dict['title']}")
                logger.debug(f"All the audio info: {info_dict}")

                # Sanitize the filename
                sanitized_title = sanitize_filename(info_dict["title"], restricted=True)

                # Create the video and audio paths
                audio_path = Path(self.audio_download_path, f"{sanitized_title}_{info_dict['id']}.mp3")

                # Update the video data
                video_data.audio_path = audio_path

                return video_data

        except DownloadError as e:
            logger.error(e)
            raise

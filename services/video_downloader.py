from pathlib import Path

from yt_dlp import YoutubeDL

from models.data_models import VideoData
from utils.logger import logger


class VideoDownloader:
    def __init__(self, video_download_path: Path):
        self.video_download_path: Path = video_download_path

    def download_video(self, video_url: str) -> VideoData:
        """
        Downloads a YouTube video

        Args:
            video_url (str): The URL of the YouTube video

        Returns:
            Video: The downloaded video
        """

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": f"{self.video_download_path}/%(id)s.%(ext)s",
            "quiet": True,
            "writesubtitles": False,
            "merge_output_format": "mp4",
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(video_url, download=True)
                logger.info(f"Downloaded {info_dict['title']}")

                return VideoData(
                    video_id=info_dict["id"],
                    video_url=info_dict["webpage_url"],
                    title=info_dict["title"],
                    description=info_dict["description"],
                    video_path=Path(self.video_download_path, info_dict["id"]),
                    transcript_path=None,
                )
        except Exception as e:
            logger.error(e)
            raise

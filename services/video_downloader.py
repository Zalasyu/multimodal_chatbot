from pathlib import Path

from yt_dlp import DownloadError, YoutubeDL
from yt_dlp.utils import sanitize_filename

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
        logger.info(f"Downloading {video_url}")

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": f"{self.video_download_path}/%(title)s_%(id)s.%(ext)s",
            "writesubtitles": False,
            "writeautomaticsub": False,
            "restrictfilenames": True,
            "merge_output_format": "mp4",
            "quiet": True,
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:

                # Step 1: Download Video and Audio data
                info_dict = ydl.extract_info(video_url, download=True)

                logger.info(f"Downloaded {info_dict['title']}")
                logger.debug(f"All the video info: {info_dict}")

                # Sanitize the filename
                sanitized_title = sanitize_filename(info_dict["title"], restricted=True)

                # Create the video and audio paths
                video_path = Path(self.video_download_path, f"{sanitized_title}_{info_dict['id']}.mp4")

                return VideoData(
                    video_id=info_dict["id"],
                    video_url=info_dict["webpage_url"],
                    title=info_dict["title"],
                    description=info_dict["description"],
                    video_path=video_path,
                )
        except DownloadError as e:
            logger.error(e)
            raise

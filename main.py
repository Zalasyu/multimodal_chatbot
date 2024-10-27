from pathlib import Path

from models.data_models import VideoData
from services.youtube_scraper import YouTubeScraper
from utils.logger import logger


def main():
    logger.info("Application started")

    base_download_path = Path("./data/raw")
    base_download_path.mkdir(parents=True, exist_ok=True)

    # Step 0: Initialize the scraper
    scraper = YouTubeScraper(base_download_path=base_download_path)

    # Step 1: Download and Process the video
    video_data: VideoData = scraper.process_video("https://www.youtube.com/watch?v=KLLgGg4tmYs")

    # Step 2: Embed  the Video

    logger.info("Application ended")


if __name__ == "__main__":
    main()

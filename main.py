from pathlib import Path

from services.youtube_scraper import YouTubeScraper
from utils.logger import logger


def main():
    logger.info("Application started")

    base_download_path = Path("./data/raw")
    base_download_path.mkdir(parents=True, exist_ok=True)

    scraper = YouTubeScraper(base_download_path=base_download_path)
    scraper.process_video("https://www.youtube.com/watch?v=VPiJIQJfuF8")

    logger.info("Application ended")


if __name__ == "__main__":
    main()

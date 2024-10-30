from pathlib import Path

from mm_emeddings.bridgetower_embeddings import BridgeTowerEmbedder
from models.data_models import VideoData
from preprocess.preprocessor import Preprocessor
from services.youtube_scraper import YouTubeScraper
from utils.logger import logger


def main():
    logger.info("Application started")

    # Base Path for raw data
    base_download_path = Path("./data/raw")
    base_download_path.mkdir(parents=True, exist_ok=True)

    # Base Path for processed data
    base_processed_path = Path(f"/home/zalasyu/Documents/projects/multimodal_chatbot/data/processed/video_frames/")
    base_processed_path.mkdir(parents=True, exist_ok=True)

    # Step 0: Initialize the scraper
    scraper = YouTubeScraper(base_download_path=base_download_path)

    # Step 1: Download and Process the video
    video_data: VideoData = scraper.scrape_video("https://www.youtube.com/watch?v=KLLgGg4tmYs")

    # Step 2: Preprocess the Video
    preprocessor = Preprocessor(base_output_path=base_processed_path, target_height=350, frame_quality=95)

    video_data = preprocessor.process_video(video_data=video_data)

    # Peek into Video Data's Dictionary of Segments
    logger.debug(f"Video Data: {video_data.segments}")

    # Step 3: Embed  the Video
    embedder = BridgeTowerEmbedder()
    video_data: VideoData = embedder.embed_video(video_data=video_data)

    # Peek into segments' embeddings
    segs = video_data.get_segments_chronologically()
    # logger.debug(f"Segment embeddings: {segs[0].embeddings}")

    logger.info("Application ended")


if __name__ == "__main__":
    main()

from pathlib import Path

from mm_emeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mm_vector_stores.multimodal_lancedb import MultiModalLanceDB
from models.data_models import VideoData
from preprocess.preprocessor import Preprocessor
from services.summarization_service import (
    AbstractiveSummarizationService,
    ExtractiveSummarizationService,
)
from services.youtube_scraper import YouTubeScraper
from utils.logger import logger


def main():
    logger.info("Application started")

    # Base Path for raw data
    base_download_path = Path("./data/raw")
    base_download_path.mkdir(parents=True, exist_ok=True)

    # Base Path for processed data
    base_processed_path = Path(f"/home/zalasyu/Documents/projects/multimodal_chatbot/data/processed/")
    base_processed_path.mkdir(parents=True, exist_ok=True)

    # Step 0: Initialize the scraper
    scraper = YouTubeScraper(base_download_path=base_download_path)

    # Step 1: Download and Process the video
    video_data: VideoData = scraper.scrape_video("https://www.youtube.com/watch?v=LAzKGkTIKpg")
    # video_data: VideoData = scraper.scrape_video(
    #     "https://www.youtube.com/watch?v=KLLgGg4tmYs"
    # )

    # Step 2: Summarize the video
    extractive_summarization_service = ExtractiveSummarizationService(summary_download_path=base_processed_path)
    abstractive_summarization_service = AbstractiveSummarizationService(
        summary_download_path=base_processed_path, method="refine"
    )
    logger.debug(f"Video Data: {video_data}")
    video_data: VideoData = extractive_summarization_service.summarize_video(video_data=video_data)
    # video_data: VideoData = abstractive_summarization_service.summarize_video(
    #     video_data=video_data
    # )
    abstractive_summarization_service.print_full_text_n_summaries(video_data=video_data)

    # Step 3: Preprocess the Video
    preprocessor = Preprocessor(
        base_output_path=base_processed_path,
        target_height=576,
        target_width=576,
        frame_quality=95,
    )

    video_data = preprocessor.process_video(video_data=video_data)

    # Step 4: Setup the MultiModal LanceDB
    # TODO: Major Refactor Required Embedding operations according to community are done in vectorstore classes and not intedependently.
    embedder = BridgeTowerEmbeddings()
    db = MultiModalLanceDB(uri="./data/multimodal_lancedb", embedding=embedder)

    # Step 5: Create Video Segments table and add video segments
    video_segments_table = db.create_video_segments_table(video_data=video_data)

    logger.info("Application ended")


if __name__ == "__main__":
    main()

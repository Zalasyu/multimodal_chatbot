from pathlib import Path

from mm_emeddings.bridgetower_embeddings import BridgeTowerEmbedder
from mm_vector_stores.multimodal_lancedb import MultiModalLanceDB
from models.data_models import VideoData
from models.lancedb_pydantic_models import VideoModel, VideoSegmentModel
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
    logger.debug(f"Video Data: {video_data}")
    # video_data: VideoData = abstractive_summarization_service.summarize_video(
    #     video_data=video_data
    # )
    # abstractive_summarization_service.print_full_text_n_summaries(video_data=video_data)

    # Step 2: Preprocess the Video
    preprocessor = Preprocessor(base_output_path=base_processed_path, target_height=350, frame_quality=95)

    video_data = preprocessor.process_video(video_data=video_data)

    # Peek into Video Data's Dictionary of Segments
    logger.debug(f"Video Data: {video_data.segments}")

    # Step 3: Embed  the Video
    embedder = BridgeTowerEmbedder()
    video_data: VideoData = embedder.embed_video(video_data=video_data)

    # Step 4: Send the embeddings to LanceDB
    db = MultiModalLanceDB(db_path="./data/multimodal_lancedb")

    # Create the necessary tables in the database
    videos_table, video_segments_table = db.create_tables()

    # Mistake: Should have used pydantic to define my data models and NOT dataclasses due to incompatibility with LanceDB.
    # Convert VideoData and VideoSegmentData to VideoModel and VideoSegmentModel
    video_model: VideoModel = db.video_data_to_model(video_data=video_data)

    # Convert VideoSegmentData to VideoSegmentModel
    video_segment_models: list[VideoSegmentModel] = [
        db.video_segment_data_to_model(segment=segment) for segment in video_data.segments
    ]
    logger.debug(f"The number of video segments: {len(video_segment_models)}")
    logger.debug(f"The first video segment: {video_segment_models[0]}")

    # Length of VideoSegments Embeddings
    logger.debug(f"Length of VideoSegments Embeddings: {len(video_segment_models[0].embeddings)}")

    # Insert VideoData into the database
    videos_table.add([video_model], mode="append")

    # Insert VideoSegmentData into the database
    video_segments_table.add(video_segment_models, mode="append")

    # Create indices in LanceDB tables
    video_segments_table.create_scalar_index("video_segment_id")
    video_segments_table.create_fts_index("enriched_transcript", use_tantivy=False)

    # Query Test
    query = "Who was Voltaire?"
    query_embedding = embedder.embed_query(query)
    actual = video_segments_table.search(query_embedding, vector_column_name="embeddings").limit(5).to_list()

    # Extract the Enriched Transcripts
    actual = [segment["enriched_transcript"] for segment in actual]

    # Print the results but number them
    for i, segment in enumerate(actual):
        print(f"{i+1}. {segment}\n")

    logger.info("Application ended")


if __name__ == "__main__":
    main()

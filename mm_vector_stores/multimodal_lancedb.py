import lancedb
from lancedb.pydantic import pydantic_to_schema
from lancedb.table import Table

from models.data_models import VideoData, VideoSegmentData
from models.lancedb_pydantic_models import VideoModel, VideoSegmentModel
from utils.logger import logger


class MultiModalLanceDB:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.db: lancedb.DBConnection = lancedb.connect(uri=self.db_path)

    def create_tables(self) -> Table:
        """
        Create the necessary tables in the database.

        Returns:
            None
        """
        video_schema = pydantic_to_schema(VideoModel)
        video_segment_schema = pydantic_to_schema(VideoSegmentModel)

        # Create table for Videos
        videos_table: Table = self.db.create_table(name="Videos", schema=video_schema, exist_ok=True, mode="overwrite")

        logger.debug(f"Display Table head: {videos_table.head()}")

        # Create table for Video Segments
        video_segments_table: Table = self.db.create_table(
            name="VideoSegments",
            schema=video_segment_schema,
            exist_ok=True,
            mode="overwrite",
        )

        logger.debug(f"Display Table head: {video_segments_table.head()}")

        logger.debug(f"Displaying all tables: {self.db.table_names()}")

        return videos_table, video_segments_table

    def video_segment_data_to_model(self, segment: VideoSegmentData) -> VideoSegmentModel:
        """
        Convert VideoSegmentData to VideoSegmentModel.
        For compatibility with LanceDB.
        FutureDeprecationWarning: The `VideoSegmentData` class is deprecated and will be removed in a future version.

        Args:
            segment (VideoSegmentData): _description_

        Returns:
            VideoSegmentModel: _description_
        """
        logger.debug(f"Converting VideoSegmentData to VideoSegmentModel: {segment}")
        return VideoSegmentModel(
            parent_video_id=segment.parent_video_id,
            parent_video_path=str(segment.parent_video_path),
            parent_audio_path=(str(segment.parent_audio_path) if segment.parent_audio_path else None),
            parent_vtt_path=(str(segment.parent_vtt_path) if segment.parent_vtt_path else None),
            video_segment_id=segment.video_segment_id,
            video_segment_path=str(segment.video_segment_path),
            video_segment_transcript_path=str(segment.video_segment_transcript_path),
            extracted_frame_path=str(segment.extracted_frame_path),
            transcript=segment.transcript,
            enriched_transcript=segment.enriched_transcript,
            duration_ms=segment.duration_ms,
            start_ms=segment.start_ms,
            mid_ms=segment.mid_ms,
            end_ms=segment.end_ms,
            embeddings=segment.embeddings,
        )

    def video_data_to_model(self, video_data: VideoData) -> VideoModel:
        """
        Convert VideoData to VideoModel.
        For compatibility with LanceDB.
        FutureDeprecationWarning: The `VideoData` class is deprecated and will be removed in a future version.

        Args:
            video_data (VideoData): VideoData object.

        Returns:
            VideoModel: Pydantic Version of VideoData.
        """
        return VideoModel(
            video_id=video_data.video_id,
            video_url=video_data.video_url,
            title=video_data.title,
            description=video_data.description,
            summary_extractove=video_data.summary_extractive,
            summary_abstractive=video_data.summary_abstractive,
            language=video_data.language,
            transcribed=video_data.transcribed,
            video_path=str(video_data.video_path),
            audio_path=str(video_data.audio_path) if video_data.audio_path else None,
            transcript_path_vtt=(str(video_data.transcript_path_vtt) if video_data.transcript_path_vtt else None),
            transcript_path_text=(str(video_data.transcript_path_text) if video_data.transcript_path_text else None),
            description_path=(str(video_data.description_path) if video_data.description_path else None),
        )

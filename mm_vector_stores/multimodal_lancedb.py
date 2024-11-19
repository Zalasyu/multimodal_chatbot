from typing import Any, List, Optional, Union

from lancedb.pydantic import pydantic_to_schema
from lancedb.table import Table
from langchain_community.vectorstores.lancedb import LanceDB
from langchain_core.embeddings import Embeddings

from models.data_models import VideoData, VideoSegmentData
from models.lancedb_pydantic_models import VideoModel, VideoSegmentModel
from utils.logger import logger


class MultiModalLanceDB(LanceDB):
    """`LanceDB` vector store to process multimodal data

    To use, you should have ``lancedb`` python package installed.
    You can install it with ``pip install lancedb``.

    Args:
        connection: LanceDB connection to use. If not provided, a new connection
                    will be created.
        embedding: Embedding to use for the vectorstore.
        vector_key: Key to use for the vector in the database. Defaults to ``vector``.
        id_key: Key to use for the id in the database. Defaults to ``id``.
        text_key: Key to use for the text in the database. Defaults to ``text``.
        image_path_key: Key to use for the path to image in the database. Defaults to ``image_path``.
        table_name: Name of the table to use. Defaults to ``vectorstore``.
        api_key: API key to use for LanceDB cloud database.
        region: Region to use for LanceDB cloud database.
        mode: Mode to use for adding data to the table. Defaults to ``overwrite``.



    Example:
        .. code-block:: python
            vectorstore = MultimodalLanceDB(uri='/lancedb', embedding_function)
            vectorstore.add_texts(['text1', 'text2'])
            result = vectorstore.similarity_search('text1')
    """

    def __init__(
        self,
        connection: Optional[Any] = None,
        embedding: Optional[Embeddings] = None,
        uri: Optional[str] = "/data/multimodal_lancedb",
        vector_key: Optional[str] = "embeddings",
        id_key: Optional[str] = "video_segment_id",
        text_key: Optional[str] = "enriched_transcript",
        image_path_key: Optional[str] = "frame_path",
        table_name: Optional[str] = "VideoSegments",
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        mode: Optional[str] = "append",
    ) -> None:
        super().__init__(
            connection=connection,
            embedding=embedding,
            uri=uri,
            vector_key=vector_key,
            id_key=id_key,
            text_key=text_key,
            table_name=table_name,
            api_key=api_key,
            region=region,
            mode=mode,
        )
        self._image_path_key = image_path_key

    def _initialize_tables(self) -> Union[Table, Table]:
        """
        Initialize the tables for the vectorstore.

        Returns:
            Tuple[Table, Table]: Two tables: `video_segments` and `videos`.
        """
        try:
            video_segments_table = self._connection.open_table("VideoSegments")
            videos_table = self._connection.open_table("Videos")
            logger.info("Tables opened successfully")
            return video_segments_table, videos_table
        except Exception as e:
            logger.info(f"Creating new LanceDB tables...")

            # Create Schemas
            videos_schema = pydantic_to_schema(VideoModel)
            video_segments_schema = pydantic_to_schema(VideoSegmentModel)
            # Create empty initial data matching the schemas
            empty_segment = {
                "id": 0,
                "parent_video_id": "",
                "parent_video_path": "",
                "parent_audio_path": "",
                "parent_vtt_path": "",
                "video_segment_path": "",
                "video_segment_transcript_path": "",
                "frame_path": "",
                "transcript": "",
                "enriched_transcript": "",
                "duration_ms": 0.0,
                "start_ms": 0.0,
                "mid_ms": 0.0,
                "end_ms": 0.0,
                "embeddings": [0.0] * 1536,  # Match your embedding dimension
            }

            empty_video = {
                "id": "",
                "video_url": "",
                "title": "",
                "description": "",
                "summary_abstractive": "",
                "summary_extractive": "",
                "language": "en",
                "video_path": "",
                "audio_path": "",
                "transcript_path_vtt": "",
                "transcript_path_text": "",
                "transcribed": False,
                "description_path": "",
            }

            # Create tables
            video_segments_table = self._connection.create_table(
                "VideoSegments", data=[empty_segment], schema=video_segments_schema, mode="overwrite"
            )
            videos_table = self._connection.create_table("Videos", data=[empty_video], schema=videos_schema, mode="overwrite")
            logger.info("Tables created successfully")

            # Create indices for better query performance
            video_segments_table.create_scalar_index("id")
            video_segments_table.create_fts_index("enriched_transcript", use_tantivy=False)

            return video_segments_table, videos_table

    def convert_video_segments_to_model(self, segments: List[VideoSegmentData]) -> List[VideoSegmentModel]:
        """
        Convert VideoSegmentData to VideoSegmentModel.
        For compatibility with LanceDB.
        FutureDeprecationWarning: The `VideoSegmentData` class is deprecated and will be removed in a future version.

        Args:
            segments (List[VideoSegmentData]): _description_

        Returns:
            List[VideoSegmentModel]: _description_
        """
        return [self.video_segment_data_to_model(segment) for segment in segments]

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
        return VideoSegmentModel(
            id=segment.video_segment_id,
            parent_video_id=segment.parent_video_id,
            parent_video_path=str(segment.parent_video_path),
            parent_audio_path=(str(segment.parent_audio_path) if segment.parent_audio_path else None),
            parent_vtt_path=(str(segment.parent_vtt_path) if segment.parent_vtt_path else None),
            video_segment_path=str(segment.video_segment_path),
            video_segment_transcript_path=str(segment.video_segment_transcript_path),
            frame_path=str(segment.extracted_frame_path),
            transcript=segment.transcript,
            enriched_transcript=segment.enriched_transcript,
            duration_ms=segment.duration_ms,
            start_ms=segment.start_ms,
            mid_ms=segment.mid_ms,
            end_ms=segment.end_ms,
            embeddings=segment.embeddings,
        )

    def convert_videos_to_model(self, videos: List[VideoData]) -> List[VideoModel]:
        """
        Convert VideoData to VideoModel.
        For compatibility with LanceDB.
        FutureDeprecationWarning: The `VideoData` class is deprecated and will be removed in a future version.

        Args:
            videos (List[VideoData]): _description_

        Returns:
            List[VideoModel]: _description_
        """
        return [self.video_data_to_model(video) for video in videos]

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
            id=video_data.video_id,
            video_url=video_data.video_url,
            title=video_data.title,
            description=video_data.description,
            summary_extractive=video_data.summary_extractive,
            summary_abstractive=video_data.summary_abstractive,
            video_path=str(video_data.video_path),
            audio_path=str(video_data.audio_path) if video_data.audio_path else None,
            transcript_path_vtt=(str(video_data.transcript_path_vtt) if video_data.transcript_path_vtt else None),
            transcript_path_text=(str(video_data.transcript_path_text) if video_data.transcript_path_text else None),
            description_path=(str(video_data.description_path) if video_data.description_path else None),
            language=video_data.language,
            transcribed=video_data.transcribed,
        )

    def create_video_segments_table(self, video_data: VideoData) -> Table:
        """
        Add video segments to the vector store.

        Args:
            video_data (VideoData): Video data containing segments

        Returns:
            Table: Video segments table
        """

        # Create Schema
        table_schema = pydantic_to_schema(VideoSegmentModel)

        # Create Embeddings for Video Segments
        video_data = self._embedding.embed_video(video_data=video_data)

        # Prepare data
        video_segments = self.convert_video_segments_to_model(video_data.segments)

        # Create table for Video Segments
        video_segments_table: Table = self._connection.create_table(
            name="VideoSegments",
            data=[
                {
                    "id": segment.id,
                    "embeddings": segment.embeddings,
                    "transcript": segment.transcript,
                    "enriched_transcript": segment.enriched_transcript,
                    "parent_video_id": segment.parent_video_id,
                    "parent_video_path": segment.parent_video_path,
                    "parent_audio_path": segment.parent_audio_path,
                    "parent_vtt_path": segment.parent_vtt_path,
                    "video_segment_path": segment.video_segment_path,
                    "video_segment_transcript_path": segment.video_segment_transcript_path,
                    "frame_path": segment.frame_path,
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "mid_ms": segment.mid_ms,
                    "duration_ms": segment.duration_ms,
                }
                for segment in video_segments
            ],
            schema=table_schema,
            mode="overwrite",
        )

        return video_segments_table

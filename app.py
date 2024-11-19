import re
import uuid
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import torch
from langchain_core.messages import HumanMessage
from langchain_core.outputs.chat_result import ChatResult
from loguru import logger

from LVLMS.lvlm import VideoQAChatModel
from mm_emeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mm_vector_stores.multimodal_lancedb import MultiModalLanceDB
from models.data_models import VideoData
from models.lancedb_pydantic_models import VideoModel, VideoSegmentModel
from preprocess.preprocessor import Preprocessor
from services.summarization_service import ExtractiveSummarizationService
from services.youtube_scraper import YouTubeScraper
from utils.logger import logger


class VideoQAInterface:
    def __init__(self):
        # Initialize paths
        self.base_download_path = Path("./data/raw")
        self.base_processed_path = Path("./data/processed/video_frames")
        self.video_data_path = Path("./data/raw/videos/YouTube")
        self.local_upload_path = Path("./data/raw/videos/local_upload")

        # Create necessary directories
        for path in [self.base_download_path, self.base_processed_path, self.video_data_path, self.local_upload_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Initialize services
        self.services = None
        self._initialize_services()

    def _initialize_services(self):
        """
        Initialize or re-initialize all services
        """

        # Clean up existing services first
        self._cleanup_services()

        # Initialize services
        self.services = {
            "scraper": YouTubeScraper(base_download_path=self.base_download_path),
            "preprocessor": Preprocessor(base_output_path=self.base_processed_path, target_height=350),
            "embedder": BridgeTowerEmbeddings(),
        }
        self.db = MultiModalLanceDB(
            uri="./data/multimodal_lancedb",
            embedding=self.services["embedder"],
            table_name="VideoSegments",
        )

        # Initialize video table
        self.video_segments_table, self.videos_table = self.db._initialize_tables()

        # Reset the current video data
        self.current_video_data = None

    def _cleanup_services(self):
        """Clean up GPU resources"""
        if self.services:
            logger.info("Cleaning up GPU resources...")
            if hasattr(self.services.get("chat_model", {}), "_qa_model"):
                del self.services["chat_model"]._qa_model
            if hasattr(self.services.get("embedder", {}), "model"):
                del self.services["embedder"].model
            if hasattr(self.services.get("summarizer", {}), "model"):
                del self.services["summarizer"].model

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.services = None
        logger.info("GPU resources cleaned up.")

    def _is_youtube_url(self, url: str) -> bool:
        """
        Checks if the given URL is a valid YouTube video URL

        Args:
            url (str): The URL to check

        Returns:
            bool: True if the URL is a valid YouTube video URL, False otherwise
        """
        youtube_regex = (
            r"(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/" r"(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
        )
        match = re.match(youtube_regex, url)
        return bool(match)

    def _process_uploaded_video(self, video_path: Path, progress: gr.Progress) -> VideoData:
        """
        Process a directlly uploaded video file

        Args:
            video_path (Path): Path to the uploaded video file.
            progress (gr.Progress): Gradio progress bar

        Returns:
            VideoData: The processed video data
        """

        video_data = VideoData(
            video_id=str(uuid.uuid4()),
            video_path=video_path,
            title=video_path.name,
            description="Privately-Owned Video",
            video_url="local_upload",
        )

        video_data = self.services["preprocessor"].process_video(video_data)
        progress(0.6, desc="Generating embeddings...")
        video_data = self.services["embedder"].embed_video(video_data)

        return video_data

    def process_video(self, youtube_url: str, progress: gr.Progress = gr.Progress()) -> Tuple[str, str]:
        """Process uploaded video and store in database"""
        try:
            progress(0, desc="Starting video processing...")

            # Free up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Re-initialize services fo fresh run
            self._initialize_services()

            if not self._is_youtube_url(url=youtube_url):
                return "Invalid YouTube URL", "Please provide a valid YouTube video URL."

            # Process YouTube video
            progress(0.2, desc="Downloading YoutTube video...")
            video_data: VideoData = self.services["scraper"].scrape_video(video_url=youtube_url)

            # Preprocess video
            progress(0.4, desc="Preprocessing video...")
            video_data = self.services["preprocessor"].process_video(video_data=video_data)

            # Generate embeddings
            progress(0.6, desc="Generating embeddings...")
            video_data = self.services["embedder"].embed_video(video_data=video_data)

            # Save video data to database
            logger.info("Storing video data in database...")
            progress(0.8, desc="Storing in database...")
            video_model: VideoModel = self.db.video_data_to_model(video_data=video_data)
            video_segment_models: List[VideoSegmentModel] = self.db.convert_video_segments_to_model(
                segments=video_data.segments
            )

            # Add to Tables
            self.videos_table.add([video_model], mode="overwrite")
            self.video_segments_table.add(video_segment_models, mode="overwrite")
            logger.info("Video data stored in database")

            # Set current video data
            self.current_video_data = video_data

            progress(1.0, desc="Complete!")

            return (f"Video processed: {video_data.title}", "Ask questions about the video or generate a summary.")

        except Exception as e:
            logger.exception("Error processing video")
            return f"Error processing video: {str(e)}", ""

    def answer_question(self, question: str) -> str:
        """Answer questions about the processed video"""
        try:
            # Initialize Services
            self._initialize_services()
            # Initialize QA model
            self.services["chat_model"] = VideoQAChatModel(
                lancedb_table=self.video_segments_table, embedder=self.services["embedder"], top_k_frames=3
            )
            response: ChatResult = self.services["chat_model"]._generate(messages=[HumanMessage(content=question)])
            return response.generations[0].message.content
        except Exception as e:
            logger.exception("Error answering question")
            return f"Error answering question: {str(e)}"

    def summarize_video(self) -> str:
        """Generate a summary of the entire video"""
        try:
            # Initialize Services
            self._initialize_services()
            # Initialize summarization service
            self.services["summarizer"] = ExtractiveSummarizationService(
                summary_download_path=self.base_download_path / "summaries"
            )

            # Get the most recent video from the database
            video_data = self.videos_table.to_pandas().iloc[-1]
            logger.debug(f"VideoData: {video_data}")
            logger.debug(type(video_data))

            # Generate summary
            video_data = self.services["summarizer"].summarize_video(video_data=video_data)

            return f"{video_data.summary_extractive}"
        except Exception as e:
            logger.exception("Error generating summary")
            return f"Error generating summary: {str(e)}"

    def create_interface(self) -> gr.Interface:
        """
        Create the Gradio interface

        returns:
            gr.Interface: The Gradio interface
        """
        with gr.Blocks(title="Video Question Answering System") as interface:
            gr.Markdown(
                """
                # 🎥 MultiModal Knowledge Distiller
                Upload a video or provide a YouTube URL to start analyzing and asking questions about the content.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    youtube_input = gr.Textbox(label="YouTube URL", placeholder="Paste YouTube URL here...")
                    process_button = gr.Button("Process Video", variant="primary")
                    status_output = gr.Textbox(label="Status", interactive=False)
                    instructions_output = gr.Textbox(label="Instructions", interactive=False)

                with gr.Column(scale=1):
                    summarize_button = gr.Button("📝 Generate Video Summary")
                    summary_output = gr.Textbox(label="Video Summary", interactive=False, lines=5)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ❓ Ask Questions About the Video")
                    question_input = gr.Textbox(label="Your Question", placeholder="Ask something about the video...", lines=2)
                    ask_button = gr.Button("Ask Question", variant="primary")
                    answer_output = gr.Textbox(label="Answer", interactive=False, lines=4)

            with gr.Row():
                clear_button = gr.Button("Clear")

            # Set up event handlers
            process_button.click(fn=self.process_video, inputs=[youtube_input], outputs=[status_output, instructions_output])

            ask_button.click(fn=self.answer_question, inputs=[question_input], outputs=[answer_output])

            question_input.submit(fn=self.answer_question, inputs=[question_input], outputs=[answer_output])

            summarize_button.click(fn=self.summarize_video, inputs=[], outputs=[summary_output])
            clear_button.click(
                fn=lambda: ("", "", ""),
                outputs=[status_output, instructions_output, answer_output],
            )

        return interface


def main():
    # Create and launch the interface
    qa_interface = VideoQAInterface()
    interface = qa_interface.create_interface()
    interface.launch(share=True)


if __name__ == "__main__":
    main()

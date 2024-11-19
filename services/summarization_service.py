from abc import ABC, abstractmethod
from pathlib import Path

import language_tool_python
import torch
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    pipeline,
)
from transformers.pipelines.base import Pipeline

from models.data_models import VideoData
from utils.logger import logger


class SummarizationService(ABC):
    @abstractmethod
    def summarize_video(self, video_data: VideoData) -> VideoData:
        pass

    def _load_transcript(self, video_data: VideoData) -> str:
        """
        Load the transcript text from the video data.

        Args:
            video_data (VideoData): The video data to load the transcript from.

        Returns:
            str: The transcript text.
        """
        logger.debug("Loading transcript text")

        with open(str(video_data.transcript_path_text)) as f:
            transcript_text = f.read().strip()

        return transcript_text

    def _save_summary_as_text(self, summary: str, summary_file_path: Path) -> Path:
        """
        Save the summary as a text file.

        Args:
            summary (str): The summary text to be saved
            summary_file_path (Path): The path to the summary file

        Returns:
            Path: The path to the saved summary file
        """
        logger.debug("Saving summary as text file")
        with open(summary_file_path, encoding="utf-8", mode="w") as f:
            f.write(summary)

        return summary_file_path

    def print_full_text_n_summaries(self, video_data: VideoData):
        transcript_text = self._load_transcript(video_data=video_data)
        logger.info(f"Transcript Text: {transcript_text}")
        logger.info(f"Extractive Summary: {video_data.summary_extractive}")
        logger.info(f"Abstractive Summary: {video_data.summary_abstractive}")

    def cleanup(self) -> None:
        """Release GPU memory"""
        if self.model:  # type: ignore
            del self.model  # type: ignore
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _correct_grammer(self, text: str) -> str:
        """
        Correct grammar in the text using the LanguageTool API.

        Args:
            text (str): The text to be corrected.

        Returns:
            str: The corrected text.
        """
        tool = language_tool_python.LanguageTool("en-US")
        corrected_text = tool.correct(text)
        return corrected_text


class ExtractiveSummarizationService(SummarizationService):
    def __init__(
        self,
        summary_download_path: Path,
        model_name: str = "google-t5/t5-base",
        chunk_size: int = 1024,
        overlap: int = 0,
    ) -> None:

        logger.debug(f"ExtractiveSummarizationService initializing with model_name: {model_name}")

        # Summary Download Path
        self.summary_download_path: Path = Path(summary_download_path, "summaries", "extractive")
        self.summary_download_path.mkdir(parents=True, exist_ok=True)

        # Assign the device to use for inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings()

        # Setup SemanticChunker
        self.semantic_chunker = SemanticChunker(embeddings=self.embeddings, breakpoint_threshold_type="gradient")

        # Load in T5 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load in T5 model
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)  # type: ignore
        self.model.eval()  # type: ignore

        logger.debug(
            f"ExtractiveSummarizationService initialized with model_name: {model_name} and summary_download_path: {summary_download_path}"
        )

    def summarize_video(self, video_data: VideoData) -> VideoData:
        """
        Summarize a video and add the summary to the video data.

        Args:
            video_data (VideoData): The video data to be summarized.

        Returns:
            VideoData: The video data with the summary added.
        """
        logger.info(f"Summarizing video: {video_data.title}")
        try:
            # Load in Raw Transcript Text File
            transcript_text = self._load_transcript(video_data=video_data)
            logger.debug(f"Transcript Text: {transcript_text}")

            # Split Text into Chunks
            # RecursiveCharacterTextSplitter
            # transcript_chunks = self.text_splitter.split_text(transcript_text)
            # SemanticChunker
            transcript_chunks = self.semantic_chunker.split_text(transcript_text)

            # Create Summarization Pipeline
            summarization_pipeline: Pipeline = pipeline(
                task="summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                model_kwargs={"num_beams": 8},
            )

            # Summarize Chunks
            final_summary = ""
            for chunk in transcript_chunks:
                summary = summarization_pipeline(chunk)
                final_summary += summary[0]["summary_text"]
                final_summary += " "

            # Correct Grammar
            final_summary = self._correct_grammer(text=final_summary)

            # Add Summary to Video Data
            video_data.summary_extractive = final_summary

            # Save Summary as Text File
            summary_file_path = Path(self.summary_download_path, f"{video_data.title}.txt")
            self._save_summary_as_text(summary=final_summary, summary_file_path=summary_file_path)

            return video_data
        except:
            logger.exception(f"Failed to summarize video: {video_data.title}")
            raise
        finally:
            self.cleanup()


class AbstractiveSummarizationService(SummarizationService):

    def __init__(
        self,
        summary_download_path: Path,
        model_name: str = "facebook/bart-large",
        method: str = "map_reduce",
    ) -> None:

        logger.debug(f"AbstractiveSummarizationService initializing with model_name: {model_name}")

        # Summary Download Path
        self.summary_download_path: Path = Path(summary_download_path, "summaries", "abstractive")
        self.summary_download_path.mkdir(parents=True, exist_ok=True)

        # Assign the device to use for inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load in Pegasus model
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

        # Initialize the LLM
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="summarization",
            model_kwargs={
                "num_beams": 8,
                "min_length": 64,
                "max_length": 256,
                "do_sample": True,
                "early_stopping": True,
            },
            device=0,
        )

        # Initialize Chain
        self.chain = load_summarize_chain(self.llm, chain_type=method, verbose=True)

        # Setup Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50, length_function=len)

        # Setup SemanticChunker
        # self.semantic_chunker = SemanticChunker(HuggingFaceEmbeddings, breakpoint_threshold_type='gradient')

        self.model.eval()  # type: ignore

        logger.debug(
            f"AbstractiveSummarizationService initialized with model_name: {model_name} and summary_download_path: {summary_download_path}"
        )

    def summarize_video(self, video_data: VideoData) -> VideoData:
        """
        Summarize a video and add the summary to the video data.

        Args:
            video_data (VideoData): The video data to be summarized.

        Returns:
            VideoData: The video data with the summary added.
        """
        logger.info(f"Summarizing video: {video_data.title}")

        try:
            # Load in Raw Transcript Text File
            transcript_text = self._load_transcript(video_data=video_data)
            logger.debug(f"Transcript Text: {transcript_text}")

            # Split Text into Document Chunks
            # RecursiveCharacterTextSplitter
            docs = self.text_splitter.create_documents([transcript_text])
            # SemanticChunker
            # docs = self.semantic_chunker.create_documents([transcript_text])

            # Generate Summary
            summary = self.chain.run(docs)

            # Add Summary to Video Data
            video_data.summary_abstractive = summary

            # Save Summary as Text File
            summary_file_path = Path(self.summary_download_path, f"{video_data.title}.txt")
            self._save_summary_as_text(summary=summary, summary_file_path=summary_file_path)

            return video_data

        except:
            logger.error(f"Error generating summary for video: {video_data.title}")
            raise

        finally:
            self.cleanup()

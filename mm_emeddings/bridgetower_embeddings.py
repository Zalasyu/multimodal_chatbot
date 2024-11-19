from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np
import torch
from colorama import Fore, Style
from langchain_core.embeddings import Embeddings
from PIL import Image
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import BridgeTowerModel, BridgeTowerProcessor

from models.data_models import VideoData, VideoSegmentData
from utils.logger import logger


class BridgeTowerEmbeddings(Embeddings):
    def __init__(self):
        self.processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        self.model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Max Sequence Length
        self.max_seq_length = 512

        # Create an empty image for text-only queries
        self.empty_image = Image.new("RGB", (576, 576), (128, 128, 128))

    def _extract_image_n_caption(self, segment: VideoSegmentData) -> tuple:
        """
        Extract image and caption from a segment. Parallelized.

        Args:
            segment (VideoSegmentData): _description_
        """
        return (segment.frame, segment.enriched_transcript)

    def _embed_by_batch(self, images: Tuple, captions: Tuple, batch_size: int = 2) -> list:
        """
        Embed a segment using the BridgeTower model.
        Args:
            images (Tuple): List of images
            captions (Tuple): List of captions
            batch_size (int, optional): Batch size. Defaults to 2.
        Returns:
            np.ndarray: Embeddings
        """

        # Logging
        logger.info(f"Embedding {len(images)} images")
        logger.info(f"Embedding {len(captions)} captions")

        embeddings_list = []
        for i in tqdm(range(0, len(images), batch_size)):
            batch_images = images[i : i + batch_size]
            batch_captions = captions[i : i + batch_size]

            # Preprocess inputs
            encoding = self.processor(
                text=batch_captions,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            )
            # Move tensors in 'encoding' to the device
            encoding = {key: value.to(self.device) for key, value in encoding.items()}

            with torch.no_grad():
                model_output = self.model(**encoding)

            # Extract embeddings (pooler_output has shape [batch_size, 1526])
            embeddings = model_output.pooler_output
            embeddings_list.append(embeddings)

        return embeddings_list

    def _concat_n_norm_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Concatenate and normalize embeddings.

        Args:
            embeddings (np.ndarray): Embeddings
        Returns:
            np.ndarray: Processed embeddings
        """
        # Concatenate and normalize embeddings
        embeddings_tensor = torch.cat(embeddings, dim=0)
        embeddings_array = embeddings_tensor.cpu().numpy()
        embeddings_normalized = normalize(embeddings_array, norm="l2")
        return embeddings_normalized

    def embed_video(self, video_data: VideoData, batch_size=2) -> VideoData:
        # Process the video
        video_segments: List[VideoSegmentData] = video_data.get_segments_chronologically()

        # Enrich each segment's transcript with transcripts of n-neighbouring segments
        for segment in tqdm(
            video_segments,
            total=len(video_segments),
            desc=f"{Fore.CYAN}Enriching transcripts {Style.RESET_ALL}",
        ):
            self._enrich_segment_transcripts(video_data=video_data, segment=segment)

        # Create Tuple of image and caption (maintain order)
        with ThreadPoolExecutor() as executor:
            results = executor.map(self._extract_image_n_caption, video_segments)

        images, captions = zip(*results)

        # Embed by Batch of segments
        embeddings_list = self._embed_by_batch(images=images, captions=captions, batch_size=batch_size)

        # Concatenate and normalize embeddings
        embeddings_normalized = self._concat_n_norm_embeddings(embeddings=embeddings_list)

        # Update embeddings in VideoSegmentData
        for i, segment in enumerate(video_segments):
            segment.embeddings = embeddings_normalized[i].tolist()

        return video_data

    def _enrich_segment_transcripts(self, video_data: VideoData, segment: VideoSegmentData) -> VideoSegmentData:
        """Augment a transcript with transcripts of n-neighbouring segments.
        Observation: Transcripts of frames extracted from a video are usually fragemented and even with an incomplete sentence.
        - Such transripts are not meaningful and are not useful for retrieval.

        Naive Solution:
        - Extract n-neighbouring segments
        - Concatenate the transcript of the n-neighbouring segments

        Advise:
        - Should pick an individual n for each video such that updated transcripts
        say one or two meaningful facts.
        Args:
            video_data (VideoData): Video Data of the video
            segment (VideoSegmentData): Segment Data of the segment

        Returns:
            VideoSegmentData: Augmented Segment Data
        """

        # Get n segments before and after the specified segment (Use VideoData)
        neighbouring_segments: List[VideoSegmentData] = video_data.get_nearest_neighbours(
            segment_id=segment.video_segment_id, n=12
        )

        # Extract transcripts of these segments
        neighbouring_transcripts: list[str] = []
        for neighbour_segment in neighbouring_segments:
            neighbouring_transcripts.append(str(neighbour_segment.transcript))

        # Concatenate transcripts
        segment.enriched_transcript = " ".join(neighbouring_transcripts)

        return segment

    def embed_query(self, text: str) -> list:
        """
        Embed a query.  Here we are using the BridgeTower model.

        Args:
            text (str): The query to embed

        Returns:
            list: Embeddings for the query
        """
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.  Here we are using the BridgeTower model.

        Args:
            texts (List[str]): The list of texts to embed.

        Returns:
            List[List[float]]: List of embeddings, one for each text
        """

        for text in texts:

            # Create inputs for BridgeTowerModel with query's text and placeholder image
            encoding = self.processor(
                text=text,
                images=self.empty_image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                do_rescale=False,
            )

            # Move to GPU
            encoding = {key: value.to(self.device) for key, value in encoding.items()}

            with torch.no_grad():
                model_output = self.model(**encoding)

            # Extract embeddings (pooler_output has shape [batch_size, 1536])
            embedding = model_output.pooler_output.cpu().numpy().tolist()

            # Release GPU memory
            del encoding
            torch.cuda.empty_cache()

        return embedding

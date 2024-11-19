import concurrent.futures
import io
import os
import urllib.parse as urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import numpy as np
import PIL
import pyarrow as pa
import requests  # type: ignore
import torch
from colorama import Fore, Style
from lancedb.embeddings import EmbeddingFunction, EmbeddingFunctionRegistry, registry
from lancedb.util import attempt_import_or_raise
from PIL import Image
from pydantic import PrivateAttr
from tqdm import tqdm

from models.data_models import VideoData, VideoSegmentData

IMAGES = Union[str, bytes, Image.Image, List[str], List[bytes], List[Image.Image]]

registry = EmbeddingFunctionRegistry.get_instance()


def url_retrieve(url: str) -> bytes:
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Raise an error if the request failed
    return response.content


@registry.register("open-clip")
class OpenClipEmbeddings(EmbeddingFunction):
    name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    normalize: bool = True
    _model = PrivateAttr()
    _preprocess = PrivateAttr()
    _tokenizer = PrivateAttr()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        open_clip = attempt_import_or_raise(
            "open_clip", "open-clip"
        )  # EmbeddingFunction util to import external libs and raise if not found
        model, _, preprocess = open_clip.create_model_and_transforms(self.name, pretrained=self.pretrained)
        model.to(self.device)
        self._model, self._preprocess = model, preprocess
        self._tokenizer = open_clip.get_tokenizer(self.name)
        self._ndims = None

    def ndims(self):
        if self._ndims is None:
            self._ndims = self.generate_text_embeddings("foo").shape[0]
        return self._ndims

    def compute_query_embeddings(self, query: Union[str, "PIL.Image.Image"], *args, **kwargs) -> List[np.ndarray]:
        """
        Compute the embeddings for a given user query

        Parameters
        ----------
        query : Union[str, PIL.Image.Image]
            The query to embed. A query can be either text or an image.
        """
        if isinstance(query, str):
            return [self.generate_text_embeddings(query)]
        else:
            PIL = attempt_import_or_raise("PIL", "pillow")
            if isinstance(query, PIL.Image.Image):
                return [self.generate_image_embedding(query)]
            else:
                raise TypeError("OpenClip supports str or PIL Image as query")

    def generate_text_embeddings(self, text: str) -> np.ndarray:
        torch = attempt_import_or_raise("torch")
        text = self.sanitize_input(text)  # type: ignore
        text = self._tokenizer(text)
        text.to(self.device)  # type: ignore
        with torch.no_grad():
            text_features = self._model.encode_text(text.to(self.device))  # type: ignore
            if self.normalize:
                text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().squeeze()

    def sanitize_input(self, images: IMAGES) -> Union[List[bytes], np.ndarray]:
        """
        Sanitize the input to the embedding function.
        """
        if isinstance(images, (str, bytes)):
            images = [images]
        elif isinstance(images, pa.Array):
            images = images.to_pylist()
        elif isinstance(images, pa.ChunkedArray):
            images = images.combine_chunks().to_pylist()
        return images

    def compute_source_embeddings(self, images: IMAGES, *args, **kwargs) -> List[np.array]:
        """
        Get the embeddings for the given images
        """
        images = self.sanitize_input(images)
        embeddings = []
        for i in range(0, len(images), self.batch_size):
            j = min(i + self.batch_size, len(images))
            batch = images[i:j]
            embeddings.extend(self._parallel_get(batch))
        return embeddings

    def _parallel_get(self, images: Union[List[str], List[bytes]]) -> List[np.ndarray]:
        """
        Issue concurrent requests to retrieve the image data
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.generate_image_embedding, image) for image in images]
            return [future.result() for future in futures]

    def generate_image_embedding(self, image: Union[str, bytes, "PIL.Image.Image"]) -> np.ndarray:
        """
        Generate the embedding for a single image

        Parameters
        ----------
        image : Union[str, bytes, PIL.Image.Image]
            The image to embed. If the image is a str, it is treated as a uri.
            If the image is bytes, it is treated as the raw image bytes.
        """
        torch = attempt_import_or_raise("torch")
        # TODO handle retry and errors for https
        image = self._to_pil(image)
        image = self._preprocess(image).unsqueeze(0)
        with torch.no_grad():
            return self._encode_and_normalize_image(image)

    def _to_pil(self, image: Union[str, bytes]):
        PIL = attempt_import_or_raise("PIL", "pillow")
        if isinstance(image, bytes):
            return PIL.Image.open(io.BytesIO(image))
        if isinstance(image, PIL.Image.Image):
            return image
        elif isinstance(image, str):
            parsed = urlparse.urlparse(image)
            # TODO handle drive letter on windows.
            if parsed.scheme == "file":
                return PIL.Image.open(parsed.path)
            elif parsed.scheme == "":
                return PIL.Image.open(image if os.name == "nt" else parsed.path)
            elif parsed.scheme.startswith("http"):
                return PIL.Image.open(io.BytesIO(url_retrieve(image)))
            else:
                raise NotImplementedError("Only local and http(s) urls are supported")

    def _encode_and_normalize_image(self, image_tensor: "torch.Tensor"):
        """
        encode a single image tensor and optionally normalize the output
        """
        image_features = self._model.encode_image(image_tensor)
        if self.normalize:
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().squeeze()

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

    def _extract_image_n_caption(self, segment: VideoSegmentData) -> tuple:
        """
        Extract image and caption from a segment. Parallelized.

        Args:
            segment (VideoSegmentData): _description_
        """
        return (segment.frame, segment.enriched_transcript)

    def embed_video(self, video_data: VideoData) -> VideoData:
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

        # Generate Text Embeddings
        for i, segment in enumerate(video_segments):
            segment.embeddings_from_text = self.generate_text_embeddings(captions[i])

        return video_data

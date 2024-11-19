from typing import Any, Dict, List, Mapping, Optional
from urllib import response

import language_tool_python
import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mm_emeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mm_vector_stores.multimodal_lancedb import MultiModalLanceDB
from utils.helpers import format_timestamp
from utils.logger import logger


class VideoQAChatModel(BaseChatModel):
    """
    Chat Model for video question answering using BridgeTower and direct LanceDB integration.
    """

    # Declare public attributes
    model_name: str = Field(default="meta-llama/Llama-3.2-1B-Instruct")
    max_new_tokens: int = Field(default=256)
    temperature: float = Field(default=0.7)
    top_k_frames: int = Field(default=5)

    # Declare private attributes
    _lancedb_table: Any = PrivateAttr()
    _embedder: Any = PrivateAttr()
    _qa_model = PrivateAttr()
    _tokenizer = PrivateAttr()
    _semantic_model = PrivateAttr()
    _model: Any = PrivateAttr()
    _processor: Any = PrivateAttr()
    _device: Any = PrivateAttr()

    def __init__(
        self,
        lancedb_table: Any,
        embedder: Any,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_k_frames: int = 5,
        **kwargs,
    ):
        """
        Initialize the VideoQAChatModel.

        Args:
            lancedb_table (Any): LanceDB table contianing video segments
            embedder (Any): Embedder for video segments
            model_name (str, optional): Name of the model. Defaults to "BridgeTower/bridgetower-base-itm-mlm".
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 256.
            temperature (float, optional): Temperature for the model. Defaults to 0.7.
            top_k_frames (int, optional): Top k frames to consider. Defaults to 3.
        """
        super().__init__(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k_frames=top_k_frames,
            **kwargs,
        )  # Send the kwargs to the parent class

        # Initialize Private Attributes
        self._lancedb_table = lancedb_table
        self._embedder = embedder

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Quantization config for LLaVA
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )

        # Initialize QA Model
        self._qa_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
        )

        # Move model to device
        self._qa_model.to(self._device)
        self._qa_model.eval()

        # Initialize reranker Model
        self._reranker = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def _highlight_keywords(self, text: str, query: str) -> str:
        """
        Highlight important keywords in the passed text using spaCy.

        Args:
            text (str): The text to be highlighted.

        Returns:
            str: The highlighted text.
        """
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        query_doc = nlp(query)

        # Extract keywords from the query
        query_keywords = [token.text.lower() for token in query_doc if token.is_alpha]
        logger.debug(f"Query Keywords: {query_keywords}")

        # Identify entities and key tokens fromm the text that match the query context
        keywords = []
        for ent in doc.ents:
            if ent.text.lower() in query_keywords:
                keywords.append(ent.text)

        logger.debug(f"Keywords: {keywords}")

        # Highlight relevant keywords (avoid highlighting every occurene to reduce noise)
        highlighted_text = text
        highlight_once = set()
        for keyword in keywords:
            if keyword not in highlight_once:
                highlighted_text = highlighted_text.replace(keyword, f"*{keyword}*")
                highlight_once.add(keyword)

        logger.debug(f"Highlighted Text: {highlighted_text}")

        return highlighted_text

    def _retrieve_relevant_segments(self, query: str) -> List[Dict]:
        """
        Retrieve relevant video segments based on the query using native LanceDB vector search.

        Args:
            query (str): The query to search for.

        Returns:
            List[Dict]: List of relevant video segments.
        """

        # Get the query embedding
        query_embedding = self._embedder.embed_query(query)

        # Search for relevant video segments
        results = (
            self._lancedb_table.search(query_embedding, vector_column_name="embeddings").metric("cosine").limit(3).to_list()
        )

        segments = []
        for result in results:
            segment = {
                "parent_video_id": result["parent_video_id"],
                "parent_video_path": result["parent_video_path"],
                "start_ms": result["start_ms"],
                "end_ms": result["end_ms"],
                "enriched_transcript": result["enriched_transcript"],
                "frame_path": result["frame_path"],
                "score": result.get("_distance", 0.0),  # Similarity score
            }
            segments.append(segment)

        return segments

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """
        Override the _generate method to implement the chat model logic.
        Calls an API, a local model or another implementation that generates
        response to the input prompt.

        Args:
            messages (List[BaseMessage]): The prompt composed of a list of messages.
            stop (Optional[List[str]], optional): A list of string on which the model should stop generating. Defaults to None.
            run_manager (Optional[CallbackManagerForLLMRun], optional): _description_. Defaults to None.

        Returns:
            ChatResult: The generated response.
        """
        # Extract the user's question from the messages
        query = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                query = message.content
                break

        if not query:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="I'm sorry, I didn't understand your question."))]
            )

        # Retrieve relevant video segments
        logger.debug(f"Retrieving relevant video segments for query: {query}")
        if run_manager:
            run_manager.on_text(text=f"Retrieving relevant video segments for query: {query}", verbose=True)
        segments = self._retrieve_relevant_segments(query)

        # Re-rank the segments based on their similarity to the query
        segments = self._rerank_segments(segments, query)

        # Analyze the relevance of the segments
        segments = self._analyze_segment_relevance(segments, query)

        # Pick Most Relevant Segment
        most_relevant_segment = segments[0]
        logger.info(f"Most Relevant Segment: {most_relevant_segment}")

        # Generate a response
        response = self._generate_response(most_relevant_segment, query)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=response),
                    generation_info={
                        "segments_analyzed": len(segments),
                        "most_relevant_segment": most_relevant_segment,
                    },
                )
            ]
        )

    def _generate_response(self, most_relevant_segment: Dict, query: str) -> str:
        """
        Generate a response based on the most relevant video segment.

        Args:
            most_relevant_segment (Dict): _description_
            query (str): _description_

        Returns:
            str: _description_
        """

        prompt = f"""
        Question: {query}\n\n
        Transcript: {most_relevant_segment['enriched_transcript']}\n\n
        Answer:
        """
        inputs = self._tokenizer(prompt, return_tensors="pt")

        # Move inputs to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Generate response
        with torch.inference_mode():
            outputs = self._qa_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=self.temperature,
                do_sample=True,
                num_beams=3,
                length_penalty=-0.5,
                repetition_penalty=1.2,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Answer:")[-1].strip()

        return answer

    def _analyze_segment_relevance(self, segments: List[Dict], query: str) -> List[Dict]:
        """
        Analyze the relevance of a video segment based on its similarity to the query.

        Args:
            segment (Dict): _description_
            query (str): _description_

        Returns:
            List[Dict]: _description_
        """
        logger.info(f"\nQuery: {query}")
        logger.info("\nSegment Analysis:")

        for i, segment in enumerate(segments):
            logger.info(
                f"\nSegment {i}:"
                f"\nTimestamp: {format_timestamp(segment['start_ms'])} to {format_timestamp(segment['end_ms'])}"
                f"\nTranscript Preview: {segment['enriched_transcript'][:300]}"
                f"\nInitial Score: {segment['score']}"
                f"\nSemantic Score: {segment['semntic_score']}"
                f"\nCombined Score: {segment['combined_score']}"
            )

        return segments

    def _rerank_segments(self, segments: List[Dict], query: str) -> List[Dict]:
        """
        Re-rank the video segments based on their similarity to the query.

        Args:
            segments (List[Dict]): _description_
            query (str): _description_

        Returns:
            List[Dict]: _description_
        """
        # Encode the query and segments
        query_embedding = self._reranker.encode(query, convert_to_tensor=True)
        segment_embeddings = self._reranker.encode([seg["enriched_transcript"] for seg in segments], convert_to_tensor=True)

        # Calculate semantic similarity between query and segments
        similarities = util.cos_sim(query_embedding, segment_embeddings)[0]

        # Add the similarity scores to the segments
        for i, seg in enumerate(segments):
            semantic_score = float(similarities[i].item())
            initial_score = seg.get("score", 0.0)

            # Combine the semantic and initial scores
            combined_score = 0.7 * semantic_score + 0.3 * initial_score
            seg["semntic_score"] = semantic_score
            seg["combined_score"] = combined_score

        # Sort the segments based on the combined score
        segments.sort(key=lambda seg: seg["combined_score"])

        return segments

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

    @property
    def _llm_type(self) -> str:
        """Return type of llm"""
        return "llava-video-qa"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Represent model parameterization for tracing purposes. [Optional]"""
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k_frames": self.top_k_frames,
        }


if __name__ == "__main__":
    lancedb_instance = MultiModalLanceDB(
        uri="/home/zalasyu/Documents/projects/multimodal_chatbot/data/multimodal_lancedb",
        embedding=BridgeTowerEmbeddings(),
    )
    video_segments_table = lancedb_instance._connection.open_table("VideoSegments")
    embedder = BridgeTowerEmbeddings()

    questions = [
        "What was the name of the book that Voltaire is best known for?",
        "Who did Voltaire try to save from execution?",
    ]

    # Create the Chat Model
    chat_model = VideoQAChatModel(lancedb_table=video_segments_table, embedder=embedder, top_k_frames=3)
    messages = [HumanMessage(content=questions[1])]
    response = chat_model._generate(messages=messages)

    logger.info(response.generations[0].message.content)  # type: ignore

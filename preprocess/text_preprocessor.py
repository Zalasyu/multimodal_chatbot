import re
from typing import List

import language_tool_python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


class TextPreprocessor:
    """Handles text preprocessing for video transcripts"""

    def __init__(self):
        # Download required NLTK data
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)

        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.language_tool = language_tool_python.LanguageTool("en-US")

        # Common transcription artifacts and filler words
        self.filler_patterns = [
            r"\b(um|uh|er|ah|like|you know|i mean|basically|actually|literally|sort of|kind of|well|so|right|okay|obviously)\b",
            r"\[.*?\]",  # Remove bracketed content
            r"\(.*?\)",  # Remove parenthetical content
            r"\b\d+:\d+\b",  # Remove timestamps
            r"[\(\{\[\]\}\)]",  # Remove brackets
            r"\s+",  # Normalize whitespace
        ]

    def clean_text(self, text: str) -> str:
        """Initial cleaning of the text"""
        text = text.lower()

        # Apply all cleaning patterns
        for pattern in self.filler_patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = " ".join(text.split())
        return text

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while handling common transcript issues"""
        # First clean the text
        text = self.clean_text(text)

        # Use NLTK for initial sentence splitting
        sentences = sent_tokenize(text)

        # Post-process sentences
        processed_sentences = []
        for sentence in sentences:
            # Skip very short sentences (likely noise)
            if len(sentence.split()) < 3:
                continue

            # Skip sentences that don't have a verb (likely fragments)
            tokens = nltk.pos_tag(word_tokenize(sentence))
            if not any(tag.startswith("VB") for _, tag in tokens):
                continue

            processed_sentences.append(sentence)

        return processed_sentences

    def process_sentence(self, sentence: str) -> str:
        """Process a single sentence"""
        # Tokenize
        tokens = word_tokenize(sentence)

        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip stop words and short tokens
            if token in self.stop_words or len(token) < 2:
                continue

            # Lemmatize based on POS tag
            pos = nltk.pos_tag([token])[0][1]
            if pos.startswith("V"):
                lemma = self.lemmatizer.lemmatize(token, "v")
            elif pos.startswith("N"):
                lemma = self.lemmatizer.lemmatize(token, "n")
            elif pos.startswith("J"):
                lemma = self.lemmatizer.lemmatize(token, "a")
            else:
                lemma = self.lemmatizer.lemmatize(token)

            filtered_tokens.append(lemma)

        return " ".join(filtered_tokens)

    def preprocess_transcript(self, transcript: str) -> str:
        """Main preprocessing pipeline"""
        # Split into sentences
        sentences = self.split_into_sentences(transcript)

        # Process each sentence
        processed_sentences = [self.process_sentence(sent) for sent in sentences]

        # Join sentences
        processed_text = " ".join(processed_sentences)

        # Final grammar correction
        corrected_text = self.language_tool.correct(processed_text)

        return corrected_text

import re
from pathlib import Path

import requests  # type: ignore
from yt_dlp import YoutubeDL

from models.data_models import VideoData
from utils.logger import logger


class TranscriptService:

    def __init__(self, transcript_download_path: Path):
        self.transcript_download_path = transcript_download_path
        logger.debug(f"TranscriptService initialized with transcript_download_path: {transcript_download_path}")

    def fetch_transcript(self, video_data: VideoData) -> VideoData:
        """
        Fetches the transcript of a YouTube video

        Args:
            video_id (str): The ID of the YouTube video

        Returns:
            Optional[str]: The transcript of the video, or None if the video does not have a transcript
        """

        ydl_opts = {
            "outtmpl": f"{self.transcript_download_path}/%(title)s_%(id)s.%(ext)s",
            "quiet": True,
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(video_data.video_id, download=False)

                if "automatic_captions" in info_dict:
                    automatic_captions = info_dict["automatic_captions"]

                    # Search for the english subtitle file in automatic_captions dicitonary of lists of dictionaries
                    for subtitle in automatic_captions["en"]:
                        logger.debug(f"Subtitle: {subtitle}")

                        # Extract the dictionary item where the name 'English'
                        if subtitle["name"] == "English" and subtitle["ext"] == "vtt":
                            logger.debug(f"Subtitle found: {subtitle}")
                            subtitle_file = subtitle["url"]

                            # Download the subtitle file to the transcript_download_path using requests
                            response: requests.Response = requests.get(url=subtitle_file, timeout=30)
                            logger.debug(f"Displaying transcript: {response.text}")
                            captions: str = response.text

                            # Extract the text from the VTT file
                            raw_text: str = self._extract_text_from_vtt(vtt_content=captions)

                            # Write the subtitle to a file as VTT
                            self._save_transcription(video_data=video_data, content=captions, ext="vtt")

                            # Write the subtitle to a file as TXT
                            self._save_transcription(video_data=video_data, content=raw_text, ext="txt")

                    video_data.transcript_path_vtt = Path(video_data.video_path).with_suffix(".vtt")

                    return video_data

        except Exception as e:
            logger.error(f"Failed to fetch transcript for video {video_data.video_id}: {e}")

        return video_data

    def _extract_text_from_vtt(self, vtt_content: str) -> str:
        """
        From a VTT file, extract the text. Remove the timestamps

        Args:
            vtt_content (str): The contents of the VTT file

        Returns:
            str: The text from the VTT file
        """

        lines = vtt_content.split("\n")

        # Initialize an empty list to collect lines without timestamps, tags or metadatas
        cleaned_lines = []

        for line in lines:
            # Skip lines with timestamps, tags or metadatas
            # TODO: Create a better filtering logic
            if (
                re.search(r"^\d{2}:\d{2}:\d{2}\.\d{3}", line)
                or re.search(r"[^>]+>", line)
                or line.startswith(("WEBVTT", "Kind:", "Language:"))
            ):
                continue

            # If the line passes the above filters, add it to the cleaned_lines list
            cleaned_lines.append(line.strip())

        # Join the cleaned_lines list into a single string
        cleaned_text = " ".join(cleaned_lines).strip()

        return cleaned_text

    def _save_transcription(self, video_data: VideoData, content: str, ext: str) -> VideoData:
        """
        Saves the transcript to a file

        Args:
            video_data (VideoData): The video metadata
            content (str): The transcript
            ext (str): The extension of the transcript

        Returns:
            VideoData: The video metadata
        """

        with open(Path(self.transcript_download_path, f"{video_data.title}_{video_data.video_id}.{ext}"), "w") as f:
            f.write(content)

        return video_data

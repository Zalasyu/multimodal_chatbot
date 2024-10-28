from pathlib import Path

import cv2
import webvtt

from models.data_models import VideoData
from utils.helpers import load_video_data, str_to_timestamp_milliseconds
from utils.logger import logger


class Preprocessor:
    def __init__(self):
        pass

    def extract_frames_and_metadatas(self, video_data: VideoData) -> VideoData:
        """
        Extract frames and corresponding metadata from a video

        Args:
            video_data (VideoData): The video data to extract the frames and metadata from

        Returns:
            VideoData: The updated video data that has the path to the extracted frames and metadata for itself.
        """

        video = cv2.VideoCapture(video_data.video_path)
        logger.debug(f"Displaying type: {type(video)}")

        vtt_content: webvtt.WebVTT = webvtt.read(file=video_data.transcript_path_vtt)
        logger.debug(f"Transcript type: {type(vtt_content)}")

        for idx, transcript_segment in enumerate(vtt_content):
            start_time_ms = str_to_timestamp_milliseconds(transcript_segment.start)
            logger.info(f"Start time: {start_time_ms}")
            end_time_ms = str_to_timestamp_milliseconds(transcript_segment.end)
            logger.info(f"End time: {end_time_ms}")

        return video_data


if __name__ == "__main__":
    absolute_video_data_path = Path(
        "/home/zalasyu/Documents/projects/multimodal_chatbot/data/interim/video_data/KLLgGg4tmYs.json"
    )
    preprocess = Preprocessor()

    # Read the video data from the JSON file
    video_data = load_video_data(absolute_video_data_path)

    # Extract frames and corresponding metadata from the video
    preprocess.extract_frames_and_metadatas(video_data=video_data)

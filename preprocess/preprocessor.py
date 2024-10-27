import cv2
import webvtt

from models.data_models import VideoData
from utils.helpers import str_to_timestamp_milliseconds


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

        transcript: webvtt.WebVTT = webvtt.read(file=video_data.transcript_path_vtt)

        for idx, transcript_segment in enumerate(transcript):
            start_time_ms = str_to_timestamp_milliseconds(transcript_segment.start)
            end_time_ms = str_to_timestamp_milliseconds(transcript_segment.end)

        return video_data

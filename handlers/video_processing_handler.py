from abc import ABC, abstractmethod

from models.data_models import VideoData


class VideoProcessingHandler(ABC):
    def __init__(self):
        self.next_handler = None

    def set_next(self, handler: "VideoProcessingHandler") -> "VideoProcessingHandler":
        self.next_handler = handler
        return handler

    @abstractmethod
    def handle(self, video_data: VideoData) -> VideoData:
        pass

from abc import ABC, abstractmethod

from models.data_models import VideoData


class BaseHandler(ABC):
    def __init__(self):
        self.next_handler = None

    def set_next(self, handler: "BaseHandler") -> "BaseHandler":
        """
        Sets the next handler in the chain.

        Args:
            handler: The next handler to be called in the chain

        Returns:
            The next handler
        """
        self.next_handler = handler
        return handler

    def send_to_next(self, video_data: VideoData) -> VideoData:
        """
        Sends the video data to the next handler in the chain.

        Args:
            video_data: The video data to be processed

        Returns:
            VideoData: The processed video data
        """
        if self.next_handler:
            return self.next_handler.handle(video_data=video_data)
        return video_data

    @abstractmethod
    def handle(self, video_data: VideoData) -> VideoData:
        """
        Handles a video and returns the processed video data

        Args:
            video_data: The video data to be processed

        Returns:
            VideoData: The processed video data
        """
        pass

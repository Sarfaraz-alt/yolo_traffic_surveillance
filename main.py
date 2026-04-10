"""
Traffic Object Counter Module
Counts objects crossing a defined region in a videos using YOLO and ByteTrack.
"""

import cv2
from ultralytics import solutions


class TrafficCounter:
    """
    Handles videos capture and object counting using Ultralytics ObjectCounter.

    Attributes:
        video_path (str): Path to the input videos file.
        model_path (str): Path to the YOLO model weights.
        region (list[tuple]): Counting region defined by two points.
        tracker (str): Tracker configuration file name.
        frame_size (tuple[int, int]): Width and height to resize frames to.
    """

    DEFAULT_FRAME_SIZE = (640, 640)
    DEFAULT_REGION = [(4, 370), (319, 350)]
    DEFAULT_TRACKER = "bytetrack.yaml"

    def __init__(
        self,
        video_path: str,
        model_path: str,
        region: list = DEFAULT_REGION,
        tracker: str = DEFAULT_TRACKER,
        frame_size: tuple = DEFAULT_FRAME_SIZE,
        **Kwargs
    ):
        self.video_path = video_path
        self.model_path = model_path
        self.region = region
        self.tracker = tracker
        self.frame_size = frame_size
        self.Kwargs = Kwargs
        self._cap = None
        self._model = None

    def _initialize_capture(self) -> bool:
        """Opens the videos capture source."""
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            print(f"[ERROR] Cannot open videos: {self.video_path}")
            return False
        return True

    def _initialize_model(self) -> None:
        """Initializes the Ultralytics ObjectCounter model."""
        self._model = solutions.ObjectCounter(
            show=True,
            model=self.model_path,
            region=self.region,
            tracker=self.tracker,
            **self.Kwargs
        )

    def _process_frame(self, frame) -> None:
        """Resizes and runs the object counter on a single frame."""
        resized_frame = cv2.resize(frame, self.frame_size)
        self._model.process(resized_frame)

    @staticmethod
    def _quit_requested() -> bool:
        """Returns True if the user pressed 'q' to quit."""
        return cv2.waitKey(1) & 0xFF == ord("q")

    def run(self) -> None:
        """Main loop: initializes resources, processes frames, and releases on exit."""
        if not self._initialize_capture():
            return

        self._initialize_model()

        try:
            while self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    print("[INFO] End of videos stream.")
                    break

                self._process_frame(frame)

                if self._quit_requested():
                    print("[INFO] Quit signal received.")
                    break
        finally:
            self._release_resources()

    def _release_resources(self) -> None:
        """Releases videos capture and destroys all OpenCV windows."""
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released.")


def main():
    """Entry point for the traffic counter application."""
    counter = TrafficCounter(
        video_path=(
            'videos/stock-footage-delhi-india-jul-smooth-traffic-flow-at-intersection-with-green-signal.webm'
        ),
        model_path="best.pt",
        region=[(4, 370), (319, 350)]
    )
    counter.run()


if __name__ == "__main__":
    main()
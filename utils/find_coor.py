"""
Image Coordinate Finder Module
Displays an image and prints clicked coordinates via mouse events.
"""

import cv2


class FindCoor:
    """
    Finds and prints (x, y) pixel coordinates of mouse clicks on an image.

    Attributes:
        img (str): The image path.
        img_dim (tuple[int, int]): Target (width, height) for resizing.
    """

    def __init__(self, img_path: str, img_dim: tuple[int, int] = (640, 640)) -> None:
        self.img = cv2.imread(img_path)
        self.img = cv2.resize(self.img, img_dim)
        cv2.imshow("Image", self.img)
        cv2.setMouseCallback("Image", self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def click_event(
            self,
            event: int,
            x: int,
            y: int
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Coordinates: x={x}, y={y}")
            cv2.circle(self.img, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("Image", self.img)


if __name__ == '__main__':
    FindCoor()

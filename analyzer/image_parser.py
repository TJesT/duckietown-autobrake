from abc import ABC
import numpy as np
import cv2 as cv
from typing import List, Any, Tuple

class ImageParser(ABC):
    @staticmethod
    def get_images(img: np.ndarray) -> List[np.ndarray]:
        ...


class PIDImageParser(ImageParser):
    @staticmethod
    def get_images(img):
        # RGB_MIN_THRESH = (161, 154, 157)
        RGB_MIN_THRESH = (140, 140, 140)
        RGB_MAX_THRESH = (255, 255, 255)
        KERNEL = np.full((4, 4), 255, np.uint8)

        img = cv.cvtColor(img.copy(), cv.COLOR_BGR2RGB)

        crop = img[245:, 360:]

        # Parse flow
        mask = cv.inRange(crop, RGB_MIN_THRESH, RGB_MAX_THRESH)
        dilate = cv.morphologyEx(mask, cv.MORPH_DILATE, KERNEL)
        canny = cv.Canny(dilate.copy(), 0, 0)

        mask_img = img.copy()
        mask_img[245:480, 360:640] = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

        canny_img = img.copy()
        canny_img[245:480, 360:640] = cv.cvtColor(canny, cv.COLOR_GRAY2RGB)

        return mask_img, canny_img


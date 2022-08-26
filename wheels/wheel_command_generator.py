from abc import ABC
import numpy as np
import cv2 as cv
from typing import List, Any, Tuple
import math

from submission_ws.src.auto_brake.src.gui.gui import GUI, Joystick


class WheelCmdGenerator(ABC):
    def get_commands(self, anything: Any=None, debug=False) -> Tuple[float, float]:
        ...

    @staticmethod
    def _real2imitator(vel_l, vel_r):
        return np.array([vel_r + vel_l, vel_r - vel_l]) / 2


class JoystickWheelCmdGenerator(WheelCmdGenerator):
    def __init__(self, gui: GUI):
        self.__gui = gui
        self.__joy = gui.joystick

    def get_commands(self, _=None, debug=False) -> Tuple[float, float]:
        self.__gui.render()
        len = self.__joy.length / 34.0
        angle = self.__joy.angle

        v_0 = len / 8

        vel_l = (math.sin(angle) - math.cos(angle)) * -v_0
        vel_r = (math.sin(angle) + math.cos(angle)) * -v_0

        return WheelCmdGenerator._real2imitator(vel_l, vel_r) if debug else (vel_l, vel_r)


class PIDWheelCmdGenerator(WheelCmdGenerator):
    def get_commands(self, canny: np.ndarray, debug=False) -> Tuple[float, float]:
        ANGLE_TARGET = 35.0
        k_p = 5.5 * np.pi / 180
        v_0 = 0.2

        cnt, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        cur_angle = 90

        if len(cnt) != 0:
            maximum = 0
            sel_contour = []
            for contour in cnt:
                if contour.shape[0] > maximum:
                    sel_contour = contour
                    maximum = contour.shape[0]

            x = sorted(sel_contour, key=PIDWheelCmdGenerator.__point2uint)

            max_x, max_y = x[-1][0]
            min_x, min_y = x[0][0]

            vect1 = (int(max_x) - 0, int(max_y) - int(max_y))
            vect2 = (int(min_x) - int(max_x), int(min_y) - int(max_y))

            cur_angle = 180 - PIDWheelCmdGenerator.__angle(vect1, vect2)

        vel_l = v_0 - k_p * (ANGLE_TARGET - cur_angle)
        vel_r = v_0 + k_p * (ANGLE_TARGET - cur_angle)

        return PIDWheelCmdGenerator._real2imitator(vel_l, vel_r) if debug else (vel_l, vel_r)

    @staticmethod
    def __angle(v1, v2):
        x1, y1 = v1
        x2, y2 = v2

        inner_product = x1 * x2 + y1 * y2

        len1 = math.hypot(x1, y1)
        len2 = math.hypot(x2, y2)
        len1 = max(0.01, len1)
        len2 = max(0.01, len2)

        return math.degrees(math.acos(inner_product / (len1 * len2)))

    @staticmethod
    def __point2uint(elem):
        w = 640

        x, y = elem[0]
        x = int(x)
        y = int(y)

        return w * y + x

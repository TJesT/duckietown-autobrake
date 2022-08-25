class WheelCmdGenerator(ABC):
    @staticmethod
    def get_commands(anything: Any, debug=False) -> Tuple[float, float]:
        ...


class PIDWheelCmdGenerator(WheelCmdGenerator):
    @staticmethod
    def get_commands(canny: np.ndarray, debug=False) -> Tuple[float, float]:
        ANGLE_TARGET = 35.0
        k_p = 5.5 * np.pi/180
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
            min_x, min_y = x[ 0][0]

            vect1 = (int(max_x) - 0,          int(max_y) - int(max_y))
            vect2 = (int(min_x) - int(max_x), int(min_y) - int(max_y))

            cur_angle = 180 - PIDWheelCmdGenerator.__angle(vect1, vect2)

        vel_l = v_0 - k_p * (ANGLE_TARGET - cur_angle)
        vel_r = v_0 + k_p * (ANGLE_TARGET - cur_angle)

        return PIDWheelCmdGenerator.__real2imitator(vel_l, vel_r) if debug else (vel_l, vel_r)

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
    def __real2imitator(vel_l, vel_r):
        return np.array([vel_r + vel_l, vel_r - vel_l]) / 2

    @staticmethod
    def __point2uint(elem):
        w = 640

        x, y = elem[0]
        x = int(x)
        y = int(y)

        return w * y + x

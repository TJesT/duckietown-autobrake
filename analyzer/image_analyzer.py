from typing import List

import numpy as np

from submission_ws.src.auto_brake.src.analyzer.decision_maker import DecisionMaker, NNDecisionMaker
from submission_ws.src.auto_brake.src.analyzer.image_parser import ImageParser, PIDImageParser


class ImageAnalyzer:
    def __init__(self):
        self.__image_parser: ImageParser = PIDImageParser()
        self.__decision_maker: DecisionMaker = NNDecisionMaker()

    def parse_image(self, img: np.ndarray) -> List[np.ndarray]:
        return self.__image_parser.get_images(img)

    def make_decision(self, img: np.ndarray) -> bool:
        return self.__decision_maker.make_decision(img)

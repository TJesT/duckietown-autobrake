from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np
from PIL import Image
import tensorflow as tf


class DecisionMaker(ABC):
    @abstractmethod
    def make_decision(self, img) -> bool:
        ...


class NNDecisionMaker(DecisionMaker):
    def __init__(self):
        self.out = tf.keras.models.load_model('../models/Viktor_vgg16')
        # self.__weights = tf.keras.models.load_model('my_model_weights.h5')

    def make_decision(self, img) -> bool:
        # res_img = img.swapaxes(1, 0)
        img = cv.resize(img, (200, 200))
        conv_img = Image.fromarray(img, 'RGB')
        img_array = np.array(conv_img)
        img_array = np.expand_dims(img_array, axis=0)
        print(img.shape, img_array.shape)
        predicted_val = self.out.make_decision(img_array)[0][0]
        return not predicted_val


if __name__ == '__main__':
    AI = NNDecisionMaker()
    img = cv.imread('F:/Serge/Downloads/Telegram Desktop/dataset/dataset/train/nearby/img.94.png')
    pr = AI.make_decision(img)
    print(pr)

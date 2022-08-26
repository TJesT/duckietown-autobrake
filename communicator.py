#!/usr/bin/env python3

import os
from typing import Any, Dict, Callable

import rospy
import numpy as np
import cv2 as cv
# from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage


class Subscriber:
    def __init__(self, topic: str, type: Any, decoder: Callable[[Any], Any]):
        self.msg: Any = None
        self.__sub: rospy.Subscriber = rospy.Subscriber(topic, type, self.callback)
        self.__decoder: Callable[[Any], Any] = decoder

        print(f'[{self.__class__.__name__}] initialized with name={topic}')

    def get(self):
        decoded_msg = self.__decoder(self.msg)
        self.msg = None

        return decoded_msg

    def callback(self, msg):
        self.msg = msg.data


class Publisher:
    def __init__(self, topic: str, type: Any, encoder: Callable[[Any], Any]):
        self.__pub: rospy.Publisher = rospy.Publisher(topic, type, queue_size=10)
        self.__encoder: Callable[[Any], Any] = encoder

        print(f'[{self.__class__.__name__}] initialized with name={topic}')

    def publish(self, msg: Any):
        encoded_msg = self.__encoder(msg)
        self.__pub.publish(encoded_msg)


class Communicator:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        self.__subs: Dict[str, Subscriber] = dict()
        self.__pubs: Dict[str, Publisher] = dict()

        print(f'[{self.__class__.__name__}] initialized with name={node_name}')

    def make_subscriber(self, topic: str, type: Any, decoder: Callable[[Any], Any]):
        self.__subs[topic] = Subscriber(topic, type, decoder)

    def make_publisher(self, topic: str, type: Any, encoder: Callable[[Any], Any]):
        self.__pubs[topic] = Publisher(topic, type, encoder)

    def transfer(self, topic, msg):
        self.__pubs[topic].publish(msg)

    def get(self, topic):
        return self.__subs[topic].get()


def image_decoder(msg):
    if msg is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    buf = np.ndarray(shape=(1, len(msg)), dtype=np.uint8, buffer=msg)

    img = cv.imdecode(buf, cv.IMREAD_UNCHANGED)

    return img


def image_encoder(msg):
    cmprs_img_msg = CompressedImage()
    cmprs_img_msg.format = 'jpg'
    cmprs_img_msg.data = np.array(cv.imencode('.jpg', msg)[1]).tostring()

    return cmprs_img_msg


if __name__ == '__main__':
    comm = Communicator(node_name='my_republisher_node')

    comm.make_subscriber(
        '/duck46/camera_node/image/compressed',
        CompressedImage,
        image_decoder
    )

    comm.make_publisher(
        '/a',
        CompressedImage,
        image_encoder
    )

    rate = rospy.Rate(5)

    print("[Main] starting...")
    while not rospy.is_shutdown():
        img = comm.get('/duck46/camera_node/image/compressed')
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        comm.transfer('/a', hsv)
        rate.sleep()

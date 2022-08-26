#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage, String
from submission_ws.src.auto_brake.src.communicator import *
from submission_ws.src.auto_brake.src.gui.gui import GUI

from submission_ws.src.auto_brake.src.wheels.wheel_command_generator import WheelCmdGenerator, PIDWheelCmdGenerator, \
    JoystickWheelCmdGenerator


def velocity_encoder(velocity):
    vel_l, vel_r = velocity

    return f'{float(vel_l):.2f} {float(vel_r):.2f}'


class Controller:
    def __init__(self, hostname, count=1):
        self.__count = count

        self.__gui: GUI = GUI()

        self.__pid_controller: WheelCmdGenerator = PIDWheelCmdGenerator()
        self.__joystick_controller: WheelCmdGenerator = JoystickWheelCmdGenerator(self.__gui)

        self.__long_img = np.zeros((count, 1, 3), dtype=np.uint8)
        self.__imgs = []

        for returner in self.__split_image_decoders():
            self.__gui.display.add_imshow(returner)
            self.__imgs.append(returner())

        self.__hostname = hostname
        self.__comm: Communicator = Communicator('main')
        self.__comm.make_subscriber(f'/{self.__hostname}/parser/images/compressed', CompressedImage, image_decoder)
        self.__comm.make_publisher(f'/{self.__hostname}/velocity', String, velocity_encoder)

    def __split_image_decoders(self):
        w = self.__long_img.shape[0]
        image_w = w // self.__count

        image_returners = list()
        for i in range(w // image_w):
            lower = image_w * i
            upper = image_w * (i + 1)
            image_returners.append(lambda: self.__long_img[lower:upper, :])

        return image_returners

    def run(self):
        rate = rospy.Rate(5)

        while not rospy.is_shutdown():
            self.__long_img = self.__comm.get(f'/{self.__hostname}/parser/images/compressed')

            # v_l, v_r = self.__pid_controller.get_commands(self.__imgs[0])
            v_l, v_r = self.__joystick_controller.get_commands()

            self.__comm.transfer(f'/{self.__hostname}/velocity', f'{v_l:.2f} {v_r:.2f}')

            rate.sleep()

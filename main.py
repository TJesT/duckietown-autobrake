from submission_ws.src.auto_brake.src.analyzer.image_analyzer import ImageAnalyzer
from submission_ws.src.auto_brake.src.communicator import *
import rospy
from sensor_msgs.msg import CompressedImage, String
from duckietown_msgs.msg import WheelsCmdStamped

from submission_ws.src.auto_brake.src.wheels.wheel_command_validator import WheelCmdValidator


def velocity_decoder(msg):
    vel_l, vel_r = map(float, msg.data.split())

    return vel_l, vel_r


def valid_velocity_encoder(velocity):
    vel_l, vel_r = velocity

    msg = WheelsCmdStamped()
    msg.vel_left = vel_l
    msg.vel_right = vel_r

    return msg


class Main:
    def __init__(self, hostname):
        self.__validator: WheelCmdValidator = WheelCmdValidator()
        self.__image_analyzer: ImageAnalyzer = ImageAnalyzer()

        self.__hostname = hostname
        self.__comm: Communicator = Communicator('main')
        self.__comm.make_subscriber(f'/{self.__hostname}/camera_node/image/compressed', CompressedImage, image_decoder)
        self.__comm.make_subscriber(f'/{self.__hostname}/velocity', String, velocity_decoder)

        self.__comm.make_publisher(f'/{self.__hostname}/images/compressed', CompressedImage, image_encoder)
        self.__comm.make_publisher(f'/{self.__hostname}/wheel_driver_node/wheels_cmd', WheelsCmdStamped, valid_velocity_encoder)

    def run(self):
        rate = rospy.Rate(5)

        while not rospy.is_shutdown():
            img = self.__comm.get(f'/{self.__hostname}/camera_node/image/compressed')

            access = self.__image_analyzer.make_decision(img)
            imgs = self.__image_analyzer.parse_image(img)

            comm.transfer(f'/{self.__hostname}/parser/images/compressed', imgs)

            vel_l, vel_r = self.__comm.get(f'/{self.__hostname}/velocity')
            valid_l, valid_r = self.__validator.validate(vel_l, vel_r, access)
            comm.transfer(f'/{self.__hostname}/wheel_driver_node/wheels_cmd', (valid_l, valid_r))

            rate.sleep()

if __name__ == "__main__":
    main = Main('duck46')

    main.run()

#!usr/bin/env python

import sys

import cv2

# ROS imports
import rospy
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Detector:

    def __init__(self):
        self.image_sub = rospy.Subscriber("hz_image_raw", data_class=Image, callback=self.image_callback, queue_size=1,
                                          buff_size=2**24)
        self.bridge = CvBridge()
        cv2.namedWindow("Image")

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imshow("Image", cv_image)
        cv2.waitKey(1)


def main(args):
    rospy.init_node("handler_detector")
    detector = Detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutdown")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
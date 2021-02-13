#!usr/bin/env python

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# ROS imports
import rospy
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Detector:

    def __init__(self, model_dir):
        self.image_sub = rospy.Subscriber("hz_image_raw", data_class=Image, callback=self.image_callback, queue_size=1,
                                          buff_size=2**24)
        self.bridge = CvBridge()
        self.model_dir = model_dir
        cv2.namedWindow("Image")
        self.model = None
        self.load_model()

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imshow("Image", cv_image)
        self.run_inference(cv_image)
        cv2.waitKey(1)

    def run_inference(self, cv_image):
        image = np.asarray(cv_image)
        # Convert image to tensor using tf.convert_to_tensor
        input_tensor = tf.convert_to_tensor(image)
        # Add one axis, because model expects batch of images
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        output_dict = self.model(input_tensor)

        # All outputs are batches tensors. It needs to be converted to numpy array and batch dimension needs to be
        # removed
        num_detections = int(output_dict.pop("num_detections"))
        output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
        output_dict["num_detections"] = num_detections

        # Convert detection classes to ints
        output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.int64)

    def load_model(self):
        model_dir = os.path.join(Path(self.model_dir), "saved_model")
        self.model = tf.compat.v2.saved_model.load((str(model_dir)), None)
        self.model = self.model.signatures["serving_default"]


def main(args):
    rospy.init_node("handler_detector")
    if rospy.has_param("model_dir"):
        model_dir = rospy.get_param("model_dir")
        print(model_dir)

    detector = Detector(model_dir)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutdown")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
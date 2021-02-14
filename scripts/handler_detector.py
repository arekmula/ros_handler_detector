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

# Object detection imports
import object_detection
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util


class Detector:

    def __init__(self, model_dir, label_map_path):
        self.image_sub = rospy.Subscriber("hz_image_raw", data_class=Image, callback=self.image_callback, queue_size=1,
                                          buff_size=2**24)
        self.bridge = CvBridge()
        self.model_dir = model_dir
        self.category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
        cv2.namedWindow("Image")
        self.model = None
        self.load_model()

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.run_inference(cv_image)

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

        # Visualize detections on image
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)

        cv2.imshow("Image", image)
        cv2.waitKey(1)

    def load_model(self):
        model_dir = os.path.join(Path(self.model_dir), "saved_model")
        self.model = tf.compat.v2.saved_model.load((str(model_dir)), None)
        self.model = self.model.signatures["serving_default"]


def main(args):
    rospy.init_node("handler_detector")
    if rospy.has_param("model_dir"):
        model_dir = rospy.get_param("model_dir")
        print(model_dir)
    if rospy.has_param("label_map_path"):
        label_map_path = rospy.get_param("label_map_path")
        print(label_map_path)

    detector = Detector(model_dir, label_map_path)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutdown")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
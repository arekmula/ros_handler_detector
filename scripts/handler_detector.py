#!usr/bin/env python

import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# ROS imports
import rospy
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, RegionOfInterest
from cv_bridge import CvBridge, CvBridgeError

# Object detection imports
import object_detection
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

# ROS current package import
from handler_detection.msg import HandlerPrediction


class Detector:

    def __init__(self, model_dir, label_map_path, rgb_image_topic):

        # Should publish visualization image
        self.should_publish_visualization = rospy.get_param("visualize_handler_prediction", True)
        if self.should_publish_visualization:
            self.vis_pub = rospy.Publisher("handler_visualization", Image, queue_size=1)

        # Input image subscriber
        self.rgb_image_topic = rgb_image_topic
        self.image_sub = rospy.Subscriber(self.rgb_image_topic, data_class=Image, callback=self.image_callback,
                                          queue_size=1, buff_size=2 ** 24)

        # Prediction threshold
        self.prediction_threshold = rospy.get_param("handler_prediction_threshold", 0.5)

        self.cv_bridge = CvBridge()

        # Model
        self.model_dir = model_dir
        self.category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
        self.model = None
        self.load_model()

        # Last input message
        self.last_msg = None

        # Handler detection publisher
        self.handler_detection_topic = "handler_prediction"
        self.handler_detection_pub = rospy.Publisher(self.handler_detection_topic, HandlerPrediction, queue_size=1)

    def image_callback(self, data):
        print("Received image!")
        self.last_msg = data
        self.run_inference()

    def run_inference(self):
        msg = self.last_msg
        self.last_msg = None

        if msg is not None:
            try:
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                print(e)

            image = np.asarray(cv_image)
            # Convert image to tensor using tf.convert_to_tensor
            input_tensor = tf.convert_to_tensor(image)
            # Add one axis, because model expects batch of images
            input_tensor = input_tensor[tf.newaxis, ...]

            # Run inference
            print("Running inference on input image!")
            output_dict = self.model(input_tensor)

            # All outputs are batches tensors. It needs to be converted to numpy array and batch dimension needs
            # to be removed
            num_detections = int(output_dict.pop("num_detections"))
            output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
            output_dict["num_detections"] = num_detections

            # Convert detection classes to ints
            output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.int64)

            # Handle models with masks:
            if 'detection_masks' in output_dict:
                # Reframe the the bbox mask to the image size.
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                                   tf.uint8)
                output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

            # Visualize detections on image
            if self.should_publish_visualization:
                vis_image = image.copy()
                self.visualize_prediction(vis_image, output_dict)

            # Build and publish prediction message
            prediction_msg = self.build_prediction_msg(msg, prediction=output_dict, image_shape=image.shape)
            print("Publishing inference results")
            self.handler_detection_pub.publish(prediction_msg)

    def load_model(self):
        model_dir = os.path.join(Path(self.model_dir), "saved_model")
        self.model = tf.compat.v2.saved_model.load((str(model_dir)), None)
        self.model = self.model.signatures["serving_default"]

    def visualize_prediction(self, image, output_dict):
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        visualization_msg = self.cv_bridge.cv2_to_imgmsg(image, "bgr8")
        self.vis_pub.publish(visualization_msg)

    def build_prediction_msg(self, msg, prediction, image_shape):
        img_height, img_width, _ = image_shape
        prediction_msg = HandlerPrediction()
        for i, (ymin, xmin, ymax, xmax) in enumerate(prediction["detection_boxes"]):
            # Skip detections with low score
            if prediction["detection_scores"][i] < self.prediction_threshold:
                continue

            # Create bounding box field for current prediction ROI
            box = RegionOfInterest()
            box.x_offset = np.int(np.floor(np.asscalar(xmin) * img_width))
            box.y_offset = np.int(np.floor(np.asscalar(ymin) * img_height))
            box.height = np.int(np.floor(np.asscalar(ymax - ymin) * img_height))
            box.width = np.int(np.floor(np.asscalar(xmax - xmin) * img_width))
            prediction_msg.boxes.append(box)

            # Add class_id
            class_id = prediction["detection_classes"][i]
            prediction_msg.class_ids.append(class_id)

            # Add class_name
            class_name = self.category_index[class_id]["name"]
            prediction_msg.class_names.append(class_name)

            # Add prediction score
            score = prediction["detection_scores"][i]
            prediction_msg.scores.append(score)

            # Create prediction mask for current ROI
            mask = self.create_prediction_mask(img_height, img_width, box)
            mask.header = msg.header
            prediction_msg.masks.append(mask)

        return prediction_msg

    def create_prediction_mask(self, img_height, img_width, roi: RegionOfInterest):
        """
        Creates image mask for current RoI

        :param img_height: image height
        :param img_width: image width
        :param roi: Current region of interest
        :return: image mask as ROS Image type
        """
        mask_cv2 = np.zeros((img_height, img_width, 1), np.uint8)
        mask_cv2 = cv2.rectangle(mask_cv2,
                                 (roi.x_offset, roi.y_offset),
                                 (roi.x_offset + roi.width, roi.y_offset + roi.height),
                                 255,
                                 -1)
        mask = self.cv_bridge.cv2_to_imgmsg(mask_cv2, "mono8")
        mask.encoding = "mono8"
        mask.is_bigendian = False
        mask.step = mask.width

        return mask


def main(args):
    rospy.init_node("handler_detector")
    if rospy.has_param("model_dir"):
        model_dir = rospy.get_param("model_dir")
        print(model_dir)
    if rospy.has_param("label_map_path"):
        label_map_path = rospy.get_param("label_map_path")
        print(label_map_path)
    if rospy.has_param("rgb_image_topic"):
        rgb_image_topic = rospy.get_param("rgb_image_topic")
        print(rgb_image_topic)

    detector = Detector(model_dir, label_map_path, rgb_image_topic)

    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

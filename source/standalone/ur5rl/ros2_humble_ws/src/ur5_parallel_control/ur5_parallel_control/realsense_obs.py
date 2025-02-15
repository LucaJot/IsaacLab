from rclpy import time
from rclpy.node import Node
import rclpy.wait_for_message
import pyrealsense2 as rs
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from numpy import float64
import numpy as np
import cv2
import sys
import os
from tf2_ros import Buffer, TransformListener
from typing import Tuple
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

# Add the path to the cube_detector module
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)
from cube_detector import CubeDetector

# Node should get /camera/camera/aligned_depth_to_color/image_raw and /camera/camera/color/image_raw


class realsense_obs_reciever(Node):
    def __init__(self):
        super().__init__("realsense_obs_node")
        self.create_subscription(
            Image, "/camera/depth/image_aligned_to_rgb", self.depth_callback, 10
        )
        self.create_subscription(Image, "/camera/rgb/image_raw", self.rgb_callback, 10)

        self.create_subscription(
            CameraInfo, "/camera/rgb/camera_info", self.camera_info_callback, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rgb_img: np.ndarray = None
        self.depth_img: np.ndarray = None
        self.k_matrix: np.ndarray = None

        self.cubedetector = CubeDetector(real=True)
        self.bridge = CvBridge()

        self.create_timer(10.0, self.get_cube_position)
        self.counter = 0

    def depth_callback(self, msg):
        try:
            # Convert ROS image to OpenCV image in BGR format
            self.depth_img = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def rgb_callback(self, msg):
        try:
            # Convert ROS image to OpenCV image in BGR format
            self.rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # self.rgb_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def camera_info_callback(self, msg):
        self.k_matrix = np.array(msg.k).reshape(3, 3)

    def query_transform(self) -> np.array:
        try:
            # Get the transform from parent_frame to child_frame
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                "base_link",
                "camera_link",
                time.Time(seconds=0),  # Use the latest available transform
                time.Duration(seconds=1),  # Wait for 1 seconds
            )

            # Extract translation and rotation
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            return np.array([translation.x, translation.y, translation.z]), np.array(
                [rotation.x, rotation.y, rotation.z, rotation.w]
            )

        except Exception as e:
            self.get_logger().warn(
                f"Could not transform from base_link to camera_link: {e}"
            )
            return None, None

    def get_cube_position(self):
        """
        Get the 3D position of the cube in the camera frame.
        """
        rgb_camera_pose, rgb_camera_quaternion = self.query_transform()

        # List of variables with names for logging
        variables = {
            "rgb_img": self.rgb_img,
            "depth_img": self.depth_img,
            "k_matrix": self.k_matrix,
            "rgb_camera_pose": rgb_camera_pose,
            "rgb_camera_quaternion": rgb_camera_quaternion,
        }

        # Check for None values and log
        none_variables = [name for name, value in variables.items() if value is None]
        if none_variables:
            self.get_logger().error(
                f"The following variables are None: {', '.join(none_variables)}"
            )
            return

        # "Unsqueeze" the image to a batch of 1
        rgb_images = np.expand_dims(self.rgb_img, axis=0)
        depth_images = np.expand_dims(self.depth_img, axis=0)
        k_matricies = np.expand_dims(self.k_matrix, axis=0)
        baselink_poses = np.expand_dims(np.array([0, 0, 0]), axis=0)

        rgb_camera_poses = np.expand_dims(rgb_camera_pose, axis=0)
        rgb_camera_quaternions = np.expand_dims(rgb_camera_quaternion, axis=0)

        cube_positions, cube_positions_w, data_age, distance_cam_cube, pos_sensor = (
            self.cubedetector.get_cube_positions(
                rgb_images=rgb_images,
                depth_images=depth_images,
                rgb_camera_poses=rgb_camera_poses,
                rgb_camera_quats=rgb_camera_quaternions,
                camera_intrinsics_matrices_k=k_matricies,
                base_link_poses=baselink_poses,
                CAMERA_RGB_2_D_OFFSET=0,
            )
        )

        # self.get_logger().info(
        #     f"-------------------Loop {self.counter}-------------------"
        # )
        # self.get_logger().info(f"Cube positions in camera frame: {cube_positions}")
        # self.get_logger().info(f"Cube positions in world frame: {cube_positions_w}")
        # self.get_logger().info("------------------------------------------------------")
        self.counter += 1
        return cube_positions_w, data_age, distance_cam_cube, pos_sensor


def main(args=None):
    rclpy.init(args=args)
    node = realsense_obs_reciever()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

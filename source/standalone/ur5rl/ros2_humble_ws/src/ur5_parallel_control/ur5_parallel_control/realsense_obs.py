import time
import rclpy
from rclpy.node import Node
import rclpy.wait_for_message
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from numpy import float64
import numpy as np
import cv2
import sys
import os

# Add the path to the cube_detector module
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)
from cube_detector import CubeDetector

# Node should get /camera/camera/aligned_depth_to_color/image_raw and /camera/camera/color/image_raw


class realsense_obs_reciever(Node):
    def __init__(self):  # Max v = 35 cm/s
        super().__init__("realsense_obs_node")

        self.bridge = CvBridge()
        self.cubedetector = CubeDetector()

        self.k_matrix = None

        self.depth_img: np.array = None
        self.rgb_image: np.array = None

        self.depth_subscriber = self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_img_callback,
            10,
        )

        self.rgb_subscriber = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.rgb_img_callback,
            10,
        )

        self.k_matrix_subscriber = self.create_subscription(  # Camera intrinsics matrix
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.k_matrix_callback,
            10,
        )

    def k_matrix_callback(self, msg: Image):
        camInfo = msg
        # Store k as 3x3 matrix
        self.k_matrix = np.array(camInfo.k).reshape((3, 3))  # type: ignore
        # self.k_matrix = np.array(msg.K).reshape((3, 3))
        print(self.k_matrix)

    def depth_img_callback(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        clipped_img = np.clip(self.depth_img, a_min=0.0, a_max=1.0)
        clipped_img = (clipped_img * 255).astype(np.uint8)
        cv2.imwrite(
            f"/home/luca/Pictures/isaacsimcameraframes/depth_real.png",
            clipped_img,
        )

    def rgb_img_callback(self, msg: Image):
        self.rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        print(type(self.rgb_img))

        # cv2.imwrite(
        #     f"/home/luca/Pictures/isaacsimcameraframes/rgb_real.png",
        #     self.rgb_img,
        # )

    def get_depth_img(self) -> np.array:
        """Get the most recent depth image from the realsense camera

        Returns:
            np.array: Depth image as np.array
        """
        return self.depth_img

    def get_rgb_img(self) -> np.array:
        """Get the most recent rgb image from the realsense camera

        Returns:
            np.array: RGB image as np.array
        """
        return self.rgb_img


def main(args=None):
    rclpy.init(args=args)
    node = realsense_obs_reciever()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

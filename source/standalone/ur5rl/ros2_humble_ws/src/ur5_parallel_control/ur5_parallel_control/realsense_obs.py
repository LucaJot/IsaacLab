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
from geometry_msgs.msg import TransformStamped

# Add the path to the cube_detector module
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)
from cube_detector import CubeDetector

# Node should get /camera/camera/aligned_depth_to_color/image_raw and /camera/camera/color/image_raw


class realsense_obs_reciever(Node):
    def __init__(self):  # Max v = 35 cm/s
        super().__init__("realsense_obs_node")

        # Configure the RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.y8, 30)

        self.bridge = CvBridge()
        self.cubedetector = CubeDetector()

        self.k_matrix: np.ndarray = None

        self.depth_img: np.ndarray = None
        self.rgb_img: np.ndarray = None

        # Create a buffer and TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.depth_subscriber = self.create_subscription(
            Image,
            "/camera/camera/depth/image_rect_raw",
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

        self.update_timer = self.create_timer(10, self.get_cube_position)

    def query_transform(self):
        try:
            # Get the transform from parent_frame to child_frame
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                "base_link",  # Replace with your parent frame
                "camera_link",  # Replace with your child frame
                time.Time(),  # Use the latest available transform
            )

            # Extract translation and rotation
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            # Print the position and orientation
            # self.get_logger().info(
            #     f"Position: x={translation.x}, y={translation.y}, z={translation.z}"
            # )
            # self.get_logger().info(
            #     f"Orientation (quaternion): x={rotation.x}, y={rotation.y}, z={rotation.z}, w={rotation.w}"
            # )

            return np.array([translation.x, translation.y, translation.z]), np.array(
                [rotation.x, rotation.y, rotation.z, rotation.w]
            )

        except Exception as e:
            self.get_logger().warn(
                f"Could not transform from parent_frame to child_frame: {e}"
            )

    def k_matrix_callback(self, msg: Image):
        camInfo = msg
        # Store k as 3x3 matrix
        self.k_matrix = np.array(camInfo.k).reshape((3, 3))  # type: ignore
        # self.k_matrix = np.array(msg.K).reshape((3, 3))

    def depth_img_callback(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        clipped_img = np.clip(self.depth_img, a_min=0.0, a_max=1.0)
        clipped_img = (clipped_img * 255).astype(np.uint8)
        # cv2.imwrite(
        #     f"/home/luca/Pictures/isaacsimcameraframes/depth_real.png",
        #     clipped_img,
        # )

    def rgb_img_callback(self, msg: Image):
        self.rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # cv2.imwrite(
        #     f"/home/luca/Pictures/isaacsimcameraframes/rgb_real.png",
        #     self.rgb_image,
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

    def get_cube_position(self) -> np.array:
        """Get the 3D position of the cube in the camera frame

        Returns:
            np.array: 3D position of the cube in the camera frame
        """
        # Get relative position of the camera to base_link
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

        cube_positions, cube_positions_w = self.cubedetector.get_cube_positions(
            rgb_images=rgb_images,
            depth_images=depth_images,
            rgb_camera_poses=rgb_camera_poses,
            rgb_camera_quats=rgb_camera_quaternions,
            camera_intrinsics_matrices_k=k_matricies,
            base_link_poses=baselink_poses,
            CAMERA_RGB_2_D_OFFSET=0,
        )

        # Log the cubes positions
        self.get_logger().info(f"Cube positions in world frame: {cube_positions_w}")
        return cube_positions


def main(args=None):
    rclpy.init(args=args)
    node = realsense_obs_reciever()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

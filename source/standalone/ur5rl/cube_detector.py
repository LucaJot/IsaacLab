import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class CubeDetector:
    def __init__(self, real=False, num_envs=1):
        """
                Initializes the cube detector.

                Args:
        send_joint_command
        """
        self.real = real
        #! TODO CHECK THRESH REAL!!
        self.area_thresh = 100 if real else 1000
        self.clipping_range = 2000.0 if real else 2.0
        self.data_age = np.zeros(num_envs)
        # Init with NaN to indicate that no cube has been detected yet
        self.last_pos = np.full((num_envs, 3), np.nan)
        self.last_pos_w = np.full((num_envs, 3), np.nan)

        self.distance_cam_cube = 2.0

    def return_last_pos(self, idx: int):
        # increase the age of the data by 1 if no cube detected
        self.data_age[idx] += 1
        # If the cube has been detected before, return the last known position
        if not np.all(np.isnan(self.last_pos[idx])) and not np.all(
            np.isnan(self.last_pos_w[idx])
        ):
            return self.last_pos[idx], self.last_pos_w[idx]
        else:
            # Cube not seen yet -> set it to be far away
            return [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]

    def deproject_pixel_to_point(self, cx, cy, fx, fy, pixel, z):
        """
        Deprojects pixel coordinates and depth to a 3D point relative to the same camera.

        :param intrin: A dictionary representing the camera intrinsics.
                    Example:
                    {
                        'fx': float,        # Focal length in x
                        'fy': float,        # Focal length in y
                        'cx': float,       # Principal point x
                        'cy': float,       # Principal point y
                    }
        :param pixel: Tuple or list of 2 floats representing the pixel coordinates (x, y).
        :param depth: Float representing the depth at the given pixel.
        :return: List of 3 floats representing the 3D point in space.
        """

        # Calculate normalized coordinates
        x = (pixel[0] - cx) / fx
        y = (pixel[1] - cy) / fy

        # Compute 3D point
        point = [z, -z * x, -z * y]
        return point

    def transform_frame_cam2world(self, camera_pos_w, camera_q_w, point_cam_rf):
        """
        Transforms a point from the camera frame to the world frame.

        Args:
            camera_pos_w (np.ndarray): Position of the camera in the world frame.
            camera_q_w (np.ndarray): Quaternion of the camera in the world frame.
            point_cam_rf (np.ndarray): Point in the camera frame.

        Returns:
            np.ndarray: Point in the world frame.
        """

        if self.real:
            rotation = R.from_quat(camera_q_w)  # Scipy expects [x, y, z, w]
        else:
            rotation = R.from_quat(
                [camera_q_w[1], camera_q_w[2], camera_q_w[3], camera_q_w[0]]
            )

        # Apply rotation and translation
        rotated_point = rotation.apply(point_cam_rf)
        p_world = rotated_point + camera_pos_w  # was +
        return p_world

    def get_cube_positions(
        self,
        rgb_images: np.ndarray,
        depth_images: np.ndarray,
        rgb_camera_poses: np.ndarray,
        rgb_camera_quats: np.ndarray,
        camera_intrinsics_matrices_k: np.ndarray,
        base_link_poses: np.ndarray,
        CAMERA_RGB_2_D_OFFSET: int = -35,
    ):
        """
        Extract positions of red cubes in the camera frame for all environments.

        Args:
            rgb_image (np.ndarray): RGB image of shape (n, 480, 640, 3).
            depth_image (np.ndarray): Depth image of shape (n, 480, 640, 1).

        Returns:
            list: A list of arrays containing the positions of red cubes in each environment.
        """
        rgb_images_np = rgb_images
        depth_images_np = depth_images

        # Clip and normalize to a 1m range
        depth_images_np = np.clip(depth_images_np, a_min=0.0, a_max=self.clipping_range)

        # Get the camera poses relative to world frame
        rgb_poses = rgb_camera_poses
        rgb_poses_q = rgb_camera_quats
        rgb_intrinsic_matrices = camera_intrinsics_matrices_k

        robo_rootpose = base_link_poses
        cube_positions = []
        cube_positions_w = []

        distance_cam_cube = []

        # Make the camera pose relative to the robot base link
        rel_rgb_poses = rgb_poses - robo_rootpose

        # Iterate over the images of all environments
        for env_idx in range(rgb_images.shape[0]):
            rgb_image_np = rgb_images_np[env_idx]
            # Sim images are in RGB format, real images are in BGR format
            if not self.real:
                rgb_image_np = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)
            depth_image_np = depth_images_np[env_idx]
            rgb_intrinsic_matrix = rgb_intrinsic_matrices[env_idx]

            # Get the envs camera poses from base link
            rgb_pose = rel_rgb_poses[env_idx]
            rgb_pose_q = rgb_poses_q[env_idx]
            # Make pose relative to base link (z-axis offset)
            # rgb_pose[2] -= 0.35

            hsv = cv2.cvtColor(rgb_image_np, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 50, 40])
            upper_red1 = np.array([10, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            # Find contours or the largest connected component (assuming one red cube per env)
            contours, _ = cv2.findContours(
                red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # If nothing is found, append last valid coordinates to the list
            if len(contours) == 0:
                pos, pos_w = self.return_last_pos(env_idx)
                cube_positions.append(pos)
                cube_positions_w.append(pos_w)
                distance_cam_cube.append(2.0)
            else:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Shift the contour to the left  to compensate for the offset between the rgb and depth image
                largest_contour[:, 0, 0] += CAMERA_RGB_2_D_OFFSET  # type: ignore
                # Get the moments of the largest contour
                M = cv2.moments(largest_contour)

                area = cv2.contourArea(largest_contour)

                # Check for zero division and small contours
                if M["m00"] == 0 or area < self.area_thresh:
                    # If nothing or only small objects are found, append last valid coordinates to the list
                    pos, pos_w = self.return_last_pos(env_idx)
                    cube_positions.append(pos)
                    cube_positions_w.append(pos_w)
                    distance_cam_cube.append(2.0)
                    continue

                # Get the pixel centroid of the largest contour
                cx_px = int(M["m10"] / M["m00"])
                cy_px = int(M["m01"] / M["m00"])

                # print(f"Centroid [px]: {cx_px}/1200, {cy_px}/720")

                # Get depth value at the centroid
                z = depth_image_np[cy_px, cx_px]

                if self.real:
                    # Convert the depth value to meters
                    z = z / 1000.0

                # Calculate the actual 3D position of the cube relative to the camera sensor
                #     [fx  0 cx]
                # K = [ 0 fy cy]
                #     [ 0  0  1]
                cube_pos_camera_rf = self.deproject_pixel_to_point(
                    fx=rgb_intrinsic_matrix[0, 0],
                    fy=rgb_intrinsic_matrix[1, 1],
                    cx=rgb_intrinsic_matrix[0, 2],
                    cy=rgb_intrinsic_matrix[1, 2],
                    pixel=(cx_px, cy_px),
                    z=z,
                )
                # Convert the cube position from camera to world frame
                cube_pos_w = self.transform_frame_cam2world(
                    camera_pos_w=rgb_pose,
                    camera_q_w=rgb_pose_q,
                    point_cam_rf=cube_pos_camera_rf,
                )
                cube_positions_w.append(cube_pos_w)
                distance_cam_cube.append(z)

                # Normalize thee centroid
                cx = cx_px / rgb_image_np.shape[1]
                cy = cy_px / rgb_image_np.shape[0]

                cube_positions.append(cube_pos_camera_rf)

                # Reset the data age for the current env
                self.data_age[env_idx] = 0

                # Store image with contour drawn -----------------------------------

                # Convert the depth to an 8-bit range
                # depth_vis = (depth_image_np / self.clipping_range * 255).astype(
                #     np.uint8
                # )
                # Convert single channel depth to 3-channel BGR (for contour drawing)
                # depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

                # Draw the contour of the rgb to the depth image to viz the offset
                # cv2.drawContours(depth_vis_bgr, [largest_contour], -1, (0, 255, 0), 3)
                # cv2.drawContours(rgb_image_np, [largest_contour], -1, (0, 255, 0), 3)

                # cv2.imwrite(
                #     f"/home/luca/Pictures/isaacsimcameraframes/real_maskframe_depth.png",
                #     depth_vis_bgr,
                # )

                # cv2.imwrite(
                #     f"/home/luca/Pictures/isaacsimcameraframes/real_maskframe_rgb.png",
                #     rgb_image_np,
                # )

                # --------------------------------------------------------------------
        self.last_pos = cube_positions
        self.last_pos_w = cube_positions_w

        self.distance_cam_cube = distance_cam_cube
        return (
            np.array(cube_positions),
            np.array(cube_positions_w),
            np.array(self.data_age),
            np.array(self.distance_cam_cube),
        )

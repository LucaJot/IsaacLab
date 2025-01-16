import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class CubeDetector:
    def __init__(self):
        """
        Initializes the cube detector.

        Args:

        """

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
        point = [z, -z * x, z * y]
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
        # Create a Rotation object from the quaternion
        rotation = R.from_quat(
            [camera_q_w[1], camera_q_w[2], camera_q_w[3], camera_q_w[0]]
        )  # Scipy expects [x, y, z, w]

        # Apply rotation and translation
        p_world = rotation.apply(point_cam_rf) + camera_pos_w  # was +
        return p_world

    def get_cube_positions(
        self,
        rgb_images: np.ndarray,
        depth_images: np.ndarray,
        rgb_camera_poses: np.ndarray,
        rgb_camera_quats: np.ndarray,
        camera_intrinsics_matrices_k: np.ndarray,
        base_link_poses: np.ndarray,
    ):
        """
        Extract positions of red cubes in the camera frame for all environments.

        Args:
            rgb_image (torch.Tensor): RGB image of shape (n, 480, 640, 3).
            depth_image (torch.Tensor): Depth image of shape (n, 480, 640, 1).

        Returns:
            list: A list of arrays containing the positions of red cubes in each environment.
        """
        CAMERA_RGB_2_D_OFFSET = -75
        rgb_images_np = rgb_images
        depth_images_np = depth_images

        # Clip and normalize to a 1m range
        depth_images_np = (np.clip(depth_images_np, a_min=0.0, a_max=1.0)) / (1)

        # Get the camera poses relative to world frame
        rgb_poses = rgb_camera_poses
        rgb_poses_q = rgb_camera_quats
        rgb_intrinsic_matrices = camera_intrinsics_matrices_k

        robo_rootpose = base_link_poses
        cube_positions = []
        cube_positions_w = []

        # Make the camera pose relative to the robot base link
        rel_rgb_poses = rgb_poses - robo_rootpose

        # Iterate over the images of all environments
        for env_idx in range(rgb_images.shape[0]):
            rgb_image_np = rgb_images_np[env_idx]
            depth_image_np = depth_images_np[env_idx]
            rgb_intrinsic_matrix = rgb_intrinsic_matrices[env_idx]

            # Get the envs camera poses from base link
            rgb_pose = rel_rgb_poses[env_idx]
            rgb_pose_q = rgb_poses_q[env_idx]
            # Make pose relative to base link (z-axis offset)
            # rgb_pose[2] -= 0.35

            hsv = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2HSV)
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])

            red_mask = cv2.inRange(hsv, lower_red, upper_red)

            # Find contours or the largest connected component (assuming one red cube per env)
            contours, _ = cv2.findContours(
                red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # If nothing is found, append -1 coordinates to the list
            if len(contours) == 0:
                cube_positions.append([-1, -1, -1])
                cube_positions_w.append([-1, -1, -1])
            else:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Shift the contour to the left  to compensate for the offset between the rgb and depth image
                largest_contour[:, 0, 0] += CAMERA_RGB_2_D_OFFSET  # type: ignore
                # Get the moments of the largest contour
                M = cv2.moments(largest_contour)

                # Check for zero division and small contours
                if M["m00"] == 0 or cv2.contourArea(largest_contour) < 1000:
                    cube_positions.append([-1, -1, -1])
                    cube_positions_w.append([-1, -1, -1])
                    continue

                # Get the pixel centroid of the largest contour
                cx_px = int(M["m10"] / M["m00"])
                cy_px = int(M["m01"] / M["m00"])

                print(f"Centroid [px]: {cx_px}/1200, {cy_px}/720")

                # Get depth value at the centroid
                z = depth_image_np[cy_px, cx_px]

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

                # Normalize thee centroid
                cx = cx_px / rgb_image_np.shape[1]
                cy = cy_px / rgb_image_np.shape[0]

                cube_positions.append(cube_pos_camera_rf)

                # Store image with contour drawn -----------------------------------

                # # Convert the depth to an 8-bit range
                # depth_vis = (depth_image_np * 255).astype(np.uint8)
                # # Convert single channel depth to 3-channel BGR (for contour drawing)
                # depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

                # # Draw the contour of the rgb to the depth image to viz the offset
                # cv2.drawContours(depth_vis_bgr, [largest_contour], -1, (0, 255, 0), 3)

                # cv2.imwrite(
                #     f"/home/luca/Pictures/isaacsimcameraframes/maskframe.png",
                #     depth_vis_bgr,
                # )

                # cv2.drawContours(rgb_image_np, contours, -1, (0, 255, 0), 3)
                # cv2.imwrite(
                #     f"/home/luca/Pictures/isaacsimcameraframes/maskframe.png",
                #     rgb_image_np,
                # )

                # --------------------------------------------------------------------

        return np.array(cube_positions), np.array(cube_positions_w)

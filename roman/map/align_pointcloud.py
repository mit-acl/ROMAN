###########################################################
#
# align_pointcloud.py
#
# A class to align a point cloud with a camera & project onto the camera image frame
#
# Authors: Qingyuan Li
#
# January 27. 2025
#
###########################################################


from robotdatapy.data import PointCloudData, ImgData, PoseData
import numpy as np
import cv2 as cv
from roman.utils import expandvars_recursive

class AlignPointCloud():
    pointcloud_data: PointCloudData
    img_data: ImgData
    camera_pose_data: PoseData
    T_camera_rangesens_static: np.ndarray

    def __init__(self, pointcloud_data: PointCloudData, img_data: ImgData, camera_pose_data: PoseData, tf_bag_path: str):
        """
        Class for aligning a point cloud with a camera and projecting the point cloud onto the camera's image frame

        Args:
            pointcloud_data (PointCloudData): point cloud data class
            img_data (ImgData): image data class
            camera_pose_data (PoseData): pose data class
            tf_bag_path (str):  path to bag with static tf data (needed to calculate static transform from
                                camera to range sensor)
        """
        self.pointcloud_data = pointcloud_data
        self.img_data = img_data
        self.camera_pose_data = camera_pose_data

        # calculate static transform from camera to range sensor
        self.pointcloud_frame = pointcloud_data.pointcloud(pointcloud_data.times[0]).header.frame_id
        self.camera_frame = img_data.img_header(img_data.times[0]).frame_id
        self.img_shape = img_data.img(img_data.times[0]).shape[:2]
        self.T_camera_rangesens_static = PoseData.any_static_tf_from_bag(expandvars_recursive(tf_bag_path), self.camera_frame, self.pointcloud_frame)

    def aligned_pointcloud(self, t: float):
        """
        Return the closest point cloud at time t, aligned to the camera frame

        Args:
            t (float): time to get closest point cloud to

        Returns:
            np.ndarray: (n, 3) array containing 3D point cloud in the frame of the camera (Z forward, Y down)
        """
        pcl = self.pointcloud_data.pointcloud(t)

        img_header = self.img_data.img_header(t)

        # get exact times of point cloud and image messages
        pointcloud_time = pcl.header.stamp.sec + 1e-9 * pcl.header.stamp.nanosec
        img_time = img_header.stamp.sec + 1e-9 * img_header.stamp.nanosec

        # calculate dynamic transform between robot at time of image and robot at time of pointcloud
        T_W_camera_pointcloud_time = self.camera_pose_data.T_WB(pointcloud_time)
        T_W_camera_image_time = self.camera_pose_data.T_WB(img_time)
        T_W_rangesens_pointcloud_time = T_W_camera_pointcloud_time @ self.T_camera_rangesens_static
        T_W_rangesens_image_time = T_W_camera_image_time @ self.T_camera_rangesens_static

        T_img_cloud_dynamic = np.linalg.inv(T_W_rangesens_image_time) @ T_W_rangesens_pointcloud_time

        # compose static and dynamic transforms to get approximately exact transform
        # between camera at time of image and range sensor at time of point cloud
        T_camera_rangesens = self.T_camera_rangesens_static @ T_img_cloud_dynamic

        # transform points into image frame
        points = pcl.get_xyz()
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        points_h_image_frame = points_h @ T_camera_rangesens.T
        points_camera_frame = points_h_image_frame[:, :3]

        # mask for points in front of camera (positive Z in camera frame)
        in_front_mask = points_camera_frame[:, 2] >= 0
        points_camera_frame_filtered = points_camera_frame[in_front_mask]

        return points_camera_frame_filtered
    
    def projected_pointcloud(self, points_camera_frame):
        """
        Projects a 3D point cloud in the camera frame (from aligned_pointcloud) to 
        the 2D image frame

        Args:
            points_camera_frame (np.ndarray): (n, 3) array containing 3D point cloud in the frame of the camera (Z forward, Y down)

        Returns:
            np.ndarray: (n, 2) array containing 2D projected points in (u, v) coordinates, order unchanged
        """
        # project point cloud onto 2D image
        points_camera_frame_cv = points_camera_frame.reshape(-1, 1, 3)
        points_2d_cv, _ = cv.projectPoints(points_camera_frame_cv, np.zeros(3), np.zeros(3), self.img_data.K, self.img_data.D)
        points_2d = points_2d_cv.reshape((-1, 2))

        return points_2d
    
    def filter_pointcloud_and_projection(self, points_camera_frame, points_2d):
        """
        Filters array of (u, v) coordinates in image frame and associated 3D point cloud by checking
        that it is inside the associated rgb image frame bounds and casting to int

        Args:
            points_camera_frame (np.ndarray): (n, 3) array containing 3D point cloud in the frame of the camera (Z forward, Y down)
            points_2d (np.ndarray): (n, 2) array containing 2D projected points in (u, v) coordinates, in same order
        """
        points_2d = np.round(points_2d).astype(int)
        inside_frame = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < self.img_shape[1]) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < self.img_shape[0])
        points_camera_frame = points_camera_frame[inside_frame]
        points_2d = points_2d[inside_frame]
        return points_camera_frame, points_2d

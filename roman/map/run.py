###########################################################
#
# run.py
#
# A class for processing ROMAN mapping
#
# Authors: Mason Peterson, Yulun Tian, Lucas Jia, Qingyuan Li
#
# Dec. 21, 2024
#
###########################################################

import numpy as np
import cv2 as cv

from os.path import expandvars, expanduser
from typing import Tuple
from dataclasses import dataclass
import time
from copy import deepcopy

from robotdatapy.data.img_data import ImgData
from robotdatapy.data.pose_data import PoseData
from robotdatapy.transform import transform, T_RDFFLU, T_FLURDF
from robotdatapy.data.robot_data import NoDataNearTimeException
from robotdatapy.camera import CameraParams

from roman.object.segment import Segment
from roman.viz import visualize_map_on_img, visualize_observations_on_img, visualize_3d_on_img
from roman.map.mapper import Mapper
from roman.map.fastsam_wrapper import FastSAMWrapper
from roman.map.map import ROMANMap
from roman.map.align_pointcloud import AlignPointCloud
from roman.params.data_params import DataParams
from roman.params.mapper_params import MapperParams
from roman.params.fastsam_params import FastSAMParams

@dataclass
class ProcessingTimes:
    fastsam_times: list
    map_times: list
    total_times: list

class ROMANMapRunner:
    def __init__(self, data_params: DataParams, fastsam_params: FastSAMParams, 
                 mapper_params: MapperParams, verbose=False, viz_map=False, 
                 viz_observations=False, viz_3d=False, save_viz=False):
        self.data_params = data_params
        self.fastsam_params = fastsam_params
        self.mapper_params = mapper_params

        if verbose: print("Extracting time range...")
        self.time_range = self.data_params.time_range

        if verbose: 
            print("Loading image data...")
            print(f"Time range: {self.time_range}")
        self.img_data = self.data_params.load_img_data()
        if verbose:
            self.t0 = self.img_data.t0
            self.tf = self.img_data.tf

        if verbose: print("Loading pose data...")
        self.camera_pose_data = self.data_params.load_pose_data()

        if self.data_params.use_pointcloud:
            if verbose: print("Loading point cloud data for time range {}...".format(self.time_range))
            self.pointcloud_data = self.data_params.load_pointcloud_data()
            self.align_pointcloud = AlignPointCloud(pointcloud_data=self.pointcloud_data,
                                                    img_data=self.img_data,
                                                    camera_pose_data=self.camera_pose_data,
                                                    tf_bag_path=self.data_params.pose_data_params.params_dict['path']) # TODO: clean up?
        else:
            if verbose: print("Loading depth data for time range {}...".format(self.time_range))
            self.depth_data = self.data_params.load_depth_data()
        
        if verbose: print("Setting up FastSAM...")
        self.fastsam = FastSAMWrapper.from_params(params=self.fastsam_params,
                                                  depth_cam_params=self.img_data.camera_params if self.data_params.use_pointcloud else 
                                                                   self.depth_data.camera_params,
                                                  use_pointcloud=self.data_params.use_pointcloud) # TODO: clean up?

        # TODO: start here
        if verbose: print("Setting up ROMAN mapper...")
        self.mapper = Mapper(self.mapper_params, self.img_data.camera_params)
        if self.data_params.pose_data_params.T_camera_flu is not None:
            self.mapper.set_T_camera_flu(self.data_params.pose_data_params.T_camera_flu)
        
        self.verbose = verbose
        self.viz_map = viz_map
        self.viz_observations = viz_observations
        self.viz_3d = viz_3d
        self.save_viz = save_viz
        self.viz_imgs = []
        self.processing_times = ProcessingTimes([], [], [])

    def times(self):
        return np.arange(self.t0, self.tf, self.data_params.dt)

    def update(self, t: float): 
        t0 = self.img_data.t0
        tf = self.img_data.tf

        if self.verbose: print(f"t: {t - t0:.2f} = {t}")
        img_output = None
        update_t0 = time.time()

        img_time, observations, pose_odom_camera, img = self.update_fastsam(t)
        update_t1 = time.time()
        if observations is not None and pose_odom_camera is not None and img is not None:
            img_output = self.update_segment_track(img_time, observations, pose_odom_camera, img)
        
        update_t2 = time.time()
        self.processing_times.map_times.append(update_t2 - update_t1)
        self.processing_times.fastsam_times.append(update_t1 - update_t0)
        self.processing_times.total_times.append(update_t2 - update_t0)

        return img_output

    def update_fastsam(self, t):

        try:
            img_t = self.img_data.nearest_time(t)
            img = self.img_data.img(img_t)
            T_odom_camera = self.camera_pose_data.T_WB(img_t)
            if self.data_params.use_pointcloud:
                pcl = self.align_pointcloud.aligned_pointcloud(img_t)
                pcl_proj  = self.align_pointcloud.projected_pointcloud(pcl)
                depth_data = self.align_pointcloud.filter_pointcloud_and_projection(pcl, pcl_proj)
            else:
                depth_data = self.depth_data.img(img_t)
        except NoDataNearTimeException:
            return None, None, None, None
        
        observations = self.fastsam.run(img_t, T_odom_camera, img, depth_data=depth_data)
        return img_t, observations, T_odom_camera, img

    def update_segment_track(self, t, observations, pose_odom_camera, img): 

        # collect reprojected masks
        reprojected_bboxs = []
        img_ret = None
        if self.viz_observations:
            segment: Segment
            for i, segment in enumerate(self.mapper.segments + self.mapper.inactive_segments):
                bbox = segment.reprojected_bbox(pose_odom_camera)
                if bbox is not None:
                    reprojected_bboxs.append((segment.id, bbox))

        if len(observations) > 0:
            self.mapper.update(t, pose_odom_camera, observations)
        else:
            self.mapper.poses_flu_history.append(pose_odom_camera @ self.mapper._T_camera_flu)
            self.mapper.times_history.append(t)

        if self.viz_map or self.viz_observations or self.viz_3d:
            img_ret = self.draw(t, img, pose_odom_camera, observations, reprojected_bboxs)

        if self.save_viz:
            self.viz_imgs.append(img_ret)

        return img_ret
        

    def draw(self, t, img, pose_odom_camera, observations, reprojected_bboxs):
        img_orig = deepcopy(img)
        
        if len(img_orig.shape) == 2:
            img = np.concatenate([img_orig[...,None]]*3, axis=2)
            
        if self.viz_map:
            img = visualize_map_on_img(t, pose_odom_camera, img, self.mapper)

        if self.viz_observations:
            img_fastsam = visualize_observations_on_img(t, img_orig, self.mapper, observations, reprojected_bboxs)
        
        if self.viz_3d:
            img_3d = visualize_3d_on_img(t, pose_odom_camera @ self.mapper.T_camera_flu, self.mapper)
                
        # rotate images
        img = self.fastsam.apply_rotation(img)
        if self.viz_observations:
            img_fastsam = self.fastsam.apply_rotation(img_fastsam)
                
        # concatenate images
        img_ret = np.zeros((img.shape[0], 0, img.shape[2]), dtype=np.uint8)
        if self.viz_map:
            img_ret = np.hstack([img_ret, img])
        if self.viz_3d:
            img_ret = np.hstack([img_ret, img_3d])
        if self.viz_observations:
            img_ret = np.hstack([img_ret, img_fastsam])

        return img_ret
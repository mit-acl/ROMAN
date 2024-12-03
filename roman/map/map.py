import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as Rot

import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import List
import json

from robotdatapy.data.pose_data import PoseData

from roman.align.params import SubmapAlignParams
from roman.object.segment import Segment, SegmentMinimalData
from roman.utils import transform_rm_roll_pitch

@dataclass(frozen=True)
class ROMANMap:

    segments: List[Segment]
    trajectory: List[np.ndarray]
    times: np.ndarray
    poses_are_flu: bool = True

    def __post_init__(self):
        assert len(self.trajectory) == len(self.times), \
            "Trajectory and times must have the same length"
        for pose in self.trajectory:
            assert pose.shape == (4,4), \
                "Trajectory poses must be 4x4 matrices"
            
    def minimal_data(self):
        return ROMANMap(
            segments=[seg.minimal_data() for seg in self.segments],
            trajectory=self.trajectory,
            times=self.times,
            poses_are_flu=self.poses_are_flu
        )
        
    def get_segment_by_id(self, seg_id) -> Segment:
        for seg in self.segments:
            if seg.id == seg_id:
                return seg
        return None
    
    def make_picklable(self):
        for seg in self.segments:
            seg.reset_obb()
        
    @classmethod
    def from_pickle(cls, pickle_file: str):
        # extract pickled data
        with open(os.path.expanduser(pickle_file), 'rb') as f:
            pickle_data = pickle.load(f)
            
            if type(pickle_data) == cls:
                roman_map = pickle_data
            else:
                assert len(pickle_data) == 3
                mapper, poses, times = pickle_data
                roman_map = cls(
                    segments=mapper.get_segment_map(),
                    trajectory=poses,
                    times=times
                )
        return roman_map
        
    @classmethod
    def concatenate(cls, roman_maps: list):
        reference = roman_maps[0]
        if len(roman_maps) == 1:
            return reference
        elif len(roman_maps) == 2:
            other = deepcopy(roman_maps[1])
            assert reference.poses_are_flu == other.poses_are_flu
            max_seg_id = max([seg.id for seg in reference.segments])
            for segment in other.segments:
                segment.id += max_seg_id
            return cls(
                segments=reference.segments + other.segments,
                trajectory=reference.trajectory + other.trajectory,
                times=reference.times + other.times,
                poses_are_flu=reference.poses_are_flu
            )
        
        else:
            while len(roman_maps) > 1:
                concatenated = cls.concatenate(roman_maps[:2])
                roman_maps = [concatenated]  + roman_maps[2:]
            return concatenated

@dataclass
class Submap:

    id: int
    time: float
    segments: List[Segment]
    pose_flu: np.ndarray
    pose_flu_gt: np.ndarray = None
    segment_frame: str = 'submap_gravity_aligned'

    @property
    def pose_gravity_aligned(self):
        return transform_rm_roll_pitch(self.pose_flu)
    
    @property
    def pose_gravity_aligned_gt(self):
        return transform_rm_roll_pitch(self.pose_flu_gt)
    
    @property
    def position(self):
        return self.pose_flu[:3,3]
    
    @property
    def position_gt(self):
        return self.pose_flu_gt[:3,3]
    
    @property
    def has_gt(self):
        return self.pose_flu_gt is not None

    def __len__(self):
        return len(self.segments)

@dataclass
class SubmapParams:

    radius: float = 15.0
    distance: float = 10.0
    max_size: int = 40
    time_threshold: float = np.inf
    object_center_ref: str = 'mean'
    use_minimal_data: bool = True

    @classmethod
    def from_submap_align_params(cls, submap_align_params: SubmapAlignParams):
        return cls(
            radius=submap_align_params.submap_radius,
            distance=submap_align_params.submap_center_dist,
            max_size=submap_align_params.submap_max_size,
            time_threshold=submap_align_params.submap_center_time,
        )

def load_roman_map(map_file: str) -> ROMANMap:
    """
    Load a ROMANMap from a pickled file.

    Args:
        map_file (str): File path to the pickled ROMANMap

    Returns:
        ROMANMap: map
    """
    # extract pickled data
    with open(os.path.expanduser(map_file), 'rb') as f:
        pickle_data = pickle.load(f)
        
        if type(pickle_data) == ROMANMap:
            roman_map = pickle_data
        else:
            assert len(pickle_data) == 3
            mapper, poses, times = pickle_data
            roman_map = ROMANMap(
                segments=mapper.get_segment_map(),
                trajectory=poses,
                times=times
            )
    return roman_map

def submaps_from_roman_map(roman_map: ROMANMap, submap_params: SubmapParams, 
                           gt_flu_pose_data: PoseData=None) -> List[Submap]:
    """
    Breaks a ROMANMap into submaps.

    Args:
        roman_map (ROMANMap): Full map.
        submap_params (SubmapParams): Params.
        gt_flu_pose_data (PoseData, optional): Ground truth poses in FLU frame. 
            Defaults to None.

    Returns:
        List[Submap]: List of submaps.
    """
    for segment in roman_map.segments:
        segment.set_center_ref(submap_params.object_center_ref)

    if submap_params.use_minimal_data:
        roman_map = roman_map.minimal_data()
        
    submaps = []
    # create submaps
    for i, (pose, t) in enumerate(zip(roman_map.trajectory, roman_map.times)):
        if i == 0 or np.linalg.norm(pose[:-1,-1] - submaps[-1].pose_flu[:-1,-1]) > submap_params.distance \
            or (t - submaps[-1].time > submap_params.time_threshold):
            submaps.append(Submap(
                id=len(submaps),
                time=t,
                segments=[],
                pose_flu=pose,
                pose_flu_gt=gt_flu_pose_data.pose(t) if gt_flu_pose_data is not None else None
            ))

    # add segments to submaps
    for i, sm in enumerate(submaps):
        
        # set up timing constraints
        tm1 = submaps[i-1].time if i > 0 else -np.inf
        tp1 = submaps[i+1].time if i < len(submaps) - 1 else np.inf
        meets_time_constraints = lambda seg: not (
            seg.first_seen > tp1 + submap_params.time_threshold
            or seg.last_seen < tm1 - submap_params.time_threshold
        )

        for seg in roman_map.segments:
            if norm(seg.center.flatten() - sm.pose_flu[:3,3]) < submap_params.radius \
                    and meets_time_constraints(seg):
                sm.segments.append(deepcopy(seg))

        T_center_odom = np.linalg.inv(sm.pose_gravity_aligned)
        for seg in sm.segments:
            seg.transform(T_center_odom)

        if submap_params.max_size is not None:
            segments_sorted_by_dist = sorted(sm.segments, 
                                             key=lambda seg: norm(seg.center.flatten()))
            sm.segments = segments_sorted_by_dist[:submap_params.max_size]
    return submaps



def load_segment_slam_segments(json_file: str, robot_name=None, as_dict=False):
    with open(json_file, 'r') as f: 
        data = json.load(f)
    
    segments = {}
    for seg in data['segments']:
        if robot_name is not None and seg['robot_name'] != robot_name:
            continue
        centroid = np.array([seg['centroid_odom']['x'], seg['centroid_odom']['y'], seg['centroid_odom']['z']]) #[:sm_params.dim]
        new_seg = SegmentMinimalData(
            id=seg['segment_index'],
            center=centroid,
            volume=seg['shape_attributes']['volume'],
            linearity=seg['shape_attributes']['linearity'],
            planarity=seg['shape_attributes']['planarity'],
            scattering=seg['shape_attributes']['scattering'],
            extent=None,
            semantic_descriptor=None,
            first_seen=seg['first_seen']['seconds'] + seg['first_seen']['nanoseconds'] * 1e-9,
            last_seen=seg['last_seen']['seconds'] + seg['last_seen']['nanoseconds'] * 1e-9
        )
        segments[seg['segment_index']] = new_seg
        
    if as_dict:
        return segments
    return list(segments.values())
    

def load_segment_slam_submap(json_file: str, segment_frame_is_odom=True, robot_name=None):
    
    assert segment_frame_is_odom, "Only segment frame in odom is supported"
    # TODO: support other segment frames
    
    with open(json_file, 'r') as f: 
        data = json.load(f)
            
    segments = load_segment_slam_segments(json_file, robot_name, as_dict=True)
            
    submaps = []
    for submap_json in data['submaps']:
        if robot_name is not None and submap_json['robot_name'] != robot_name:
            continue
        center = np.eye(4)
        center[:3,3] = np.array([
            submap_json['T_odom_submap']['tx'], 
            submap_json['T_odom_submap']['ty'], 
            submap_json['T_odom_submap']['tz']
        ])
        center[:3,:3] = Rot.from_quat([
            submap_json['T_odom_submap']['qx'], 
            submap_json['T_odom_submap']['qy'], 
            submap_json['T_odom_submap']['qz'], 
            submap_json['T_odom_submap']['qw']
        ]).as_matrix()
        submaps.append(Submap(
            id=submap_json['submap_index'],
            time=submap_json['stamp'] * 1e-9,
            segments=[deepcopy(segments[seg_id]) for seg_id in submap_json['segment_indices']],
            pose_flu=center,
            segment_frame='odom'
        ))
    return submaps


# def load_segment_slam_submaps(json_files: List[str], 
#         sm_params: SubmapAlignParams=SubmapAlignParams(), show_maps=False):
#     submaps = []
#     submap_centers = []
#     for json_file in json_files:
#         with open(json_file, 'r') as f:
#             smcs = []
#             sms = []
#             objs = {}
            
#             data = json.load(f)
#             for seg in data['segments']:
#                 centroid = np.array([seg['centroid_odom']['x'], seg['centroid_odom']['y'], seg['centroid_odom']['z']])[:sm_params.dim]
#                 new_obj = SegmentMinimalData(
#                     id=seg['segment_index'],
#                     center=centroid,
#                     volume=seg['shape_attributes']['volume'],
#                     linearity=seg['shape_attributes']['linearity'],
#                     planarity=seg['shape_attributes']['planarity'],
#                     scattering=seg['shape_attributes']['scattering'],
#                     extent=None,
#                     semantic_descriptor=None
#                 )
#                 objs[seg['segment_index']] = new_obj
                
#             for submap in data['submaps']:
#                 center = np.eye(4)
#                 center[:3,3] = np.array([submap['T_odom_submap']['tx'], submap['T_odom_submap']['ty'], submap['T_odom_submap']['tz']])
#                 center[:3,:3] = Rot.from_quat([submap['T_odom_submap']['qx'], submap['T_odom_submap']['qy'], submap['T_odom_submap']['qz'], submap['T_odom_submap']['qw']]).as_matrix()
#                 sm = [deepcopy(objs[idx]) for idx in submap['segment_indices']]

#                 # Transform objects to be centered at the submap center
#                 T_submap_world = np.eye(4) # transformation to move submap from world frame to centered submap frame
#                 T_submap_world[:sm_params.dim, 3] = -center[:sm_params.dim, 3]
#                 for obj in sm:
#                     obj.transform(T_submap_world)

#                 smcs.append(center)
#                 sms.append(sm)
                
#             submap_centers.append(smcs)
#             submaps.append(sms)
#     if show_maps:
#         for i in range(2):
#             for submap in submaps[i]:
#                 fig, ax = plt.subplots()
#                 for obj in submap:
#                     obj.plot2d(ax, color='blue')
                
#                 bounds = object_list_bounds(submap)
#                 if len(bounds) == 3:
#                     xlim, ylim, _ = bounds
#                 else:
#                     xlim, ylim = bounds

#                 # ax.plot([position[0] for position in submap_centers[i]], [position[1] for position in submap_centers[i]], 'o', color='black')
#                 ax.set_aspect('equal')
                
#                 ax.set_xlim(xlim)
#                 ax.set_ylim(ylim)

#             plt.show()
#         exit(0)
#     return submap_centers, submaps
        
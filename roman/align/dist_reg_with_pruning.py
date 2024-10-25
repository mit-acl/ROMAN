import numpy as np
from scipy.spatial.transform import Rotation as Rot
from typing import List

import clipperpy

from roman.align.object_registration import ObjectRegistration
from roman.object.pointcloud_object import PointCloudObject
from roman.object.object import Object

class GravityConstraintError(Exception):
    pass

class DistRegWithPruning(ObjectRegistration):
    
    def __init__(self, sigma, epsilon, mindist=0.0, volume_epsilon=0.0, dim=3, use_gravity=False, roll_pitch_thresh=np.deg2rad(5)):
        super().__init__(dim)
        self.sigma = sigma
        self.epsilon = epsilon
        self.mindist = mindist
        self.volume_epsilon = volume_epsilon
        self.use_gravity = use_gravity
        self.roll_pitch_thresh = roll_pitch_thresh
        assert not self.use_gravity or self.dim == 3, "Gravity can only be used with 3D points"
    
    def register(self, map1: List[Object], map2: List[Object]):
        clipper = self._setup_clipper()
        clipper, A_init = self._score_pruned_assoc(clipper, map1, map2) # not all to all associations
        clipper.solve()
        Ain = clipper.get_selected_associations()
        
        # if gravity constrained, check that roll/pitch is small
        if self.use_gravity:
            T_align = self.T_align(map1, map2, Ain)
            R_align = T_align[:self.dim, :self.dim] # self.dim has been checked to be 3
            yaw, pitch, roll = Rot.from_matrix(R_align).as_euler('ZYX')
            # throw error if roll/pitch is too large
            if not (np.abs(roll) < self.roll_pitch_thresh and np.abs(pitch) < self.roll_pitch_thresh):
                raise GravityConstraintError(f"Roll and pitch must be less than {self.roll_pitch_thresh} rad")

        return Ain
    
    def _setup_clipper(self):
        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.sigma = self.sigma
        iparams.epsilon = self.epsilon
        iparams.mindist = self.mindist
        
        invariant = clipperpy.invariants.EuclideanDistance(iparams)
        params = clipperpy.Params()
        clipper = clipperpy.CLIPPER(invariant, params)
        return clipper
    
    def _object_to_clipper_list(self, object: Object):
        return object.center.reshape(-1)[:self.dim].tolist()
    
    def _score_pruned_assoc(self, clipper, map1, map2):
        A_all = clipperpy.utils.create_all_to_all(len(map1), len(map2))

        volumes1 = np.array([obj.volume for obj in map1])
        volumes2 = np.array([obj.volume for obj in map2])
        volumes = np.array([volumes1[A_all[:,0]], volumes2[A_all[:,1]]]).T
        to_delete = np.min(volumes, axis=1) / np.max(volumes, axis=1) < self.volume_epsilon
                
        A_put = np.delete(A_all, to_delete, axis=0)

        map1_cl = np.array([self._object_to_clipper_list(p) for p in map1])
        map2_cl = np.array([self._object_to_clipper_list(p) for p in map2])
        self._check_clipper_arrays(map1_cl, map2_cl)

        clipper.score_pairwise_consistency(map1_cl.T, map2_cl.T, A_put)
        return clipper, A_put
    


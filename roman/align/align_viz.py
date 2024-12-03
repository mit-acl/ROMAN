import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List
import open3d as o3d

from robotdatapy.data import PoseData

from roman.object.pointcloud_object import PointCloudObject
from roman.align.results import SubmapAlignResults
from roman.map.map import submaps_from_roman_map, ROMANMap, SubmapParams, Submap

def create_ptcld_geometries(submap: Submap, color, submap_offset=np.array([0,0,0]), include_label=True):
    ocd_list = []
    label_list = []
    
    for seg in submap.segments:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg.points)
        num_pts = seg.points.shape[0]
        rand_color = np.random.uniform(0, 1) * color
        rand_color = np.repeat(rand_color, num_pts, axis=0)
        pcd.colors = o3d.utility.Vector3dVector(rand_color)
        pcd.translate(submap_offset)
        ocd_list.append(pcd)
        
        if include_label:
            label = [f"id: {seg.id}", f"volume: {seg.volume:.2f}"] 
                    # f"extent: [{ptcldobj.extent[0]:.2f}, {ptcldobj.extent[1]:.2f}, {ptcldobj.extent[2]:.2f}]"]
            for i in range(2):
                label_list.append((np.median(pcd.points, axis=0) + 
                                np.array([0, 0, -0.15*i]), label[i]))
    
    return ocd_list, label_list

parser = argparse.ArgumentParser()
parser.add_argument('output_viz_file', type=str)
parser.add_argument('--idx', '-i', type=int, nargs=2, default=None)
parser.add_argument('--offset', type=float, nargs=3, default=[20.,0,0])
parser.add_argument('--no-text', action='store_true')
args = parser.parse_args()
output_viz_file = os.path.expanduser(args.output_viz_file)

# Load result data

print('Loading data...')
pkl_file = open(output_viz_file, 'rb')
results: SubmapAlignResults
results = pickle.load(pkl_file)
roman_maps = [ROMANMap.from_pickle(results.submap_io.inputs[i]) for i in range(2)]
submap_params = SubmapParams.from_submap_align_params(results.submap_align_params)
submap_params.use_minimal_data = False
if results.submap_io.input_gt_pose_yaml != [None, None]:
    gt_pose_data = [PoseData.from_yaml(yaml_file) for yaml_file in results.submap_io.input_gt_pose_yaml]
submaps = [submaps_from_roman_map(
    roman_maps[i], submap_params, gt_pose_data[i]) for i in range(2)]
pkl_file.close()
print(f'Loaded {len(submaps[0])} and {len(submaps[1])} submaps.')

# grab variables from results
associated_objs_mat = results.associated_objs_mat

if args.idx is None:
    clipper_num_associations  =  np.zeros((len(submaps[0]), len(submaps[1])))*np.nan

    max_i = 0
    max_j = 0
    max_num = 0
    for i in range(len(submaps[0])):
        for j in range(len(submaps[1])):
            clipper_num_associations[i, j] =  len(associated_objs_mat[i][j])
            if len(associated_objs_mat[i][j]) > max_num:
                max_num = len(associated_objs_mat[i][j])
                max_i = i
                max_j = j

    plt.imshow(
        clipper_num_associations, 
        vmin=0, 
    )
    plt.colorbar(fraction=0.03, pad=0.04)
    plt.show()

if args.idx is not None:
  idx_0, idx_1 = args.idx
else:
  idx_str = input("Please input two indices, separated by a space: \n")
  idx_0, idx_1 = [int(idx) for idx in idx_str.split()]


association = associated_objs_mat[idx_0][idx_1]
print(associated_objs_mat[idx_0][idx_1])
associated_objs_mat[idx_0-1][idx_1-1]
submap_0 = submaps[0][idx_0]
submap_1 = submaps[1][idx_1]
# for obj in submap_0 + submap_1:
#   obj.use_bottom_median_as_center()
print(f'Submap pair ({idx_0}, {idx_1}) contains {len(submap_0)} and {len(submap_1)} objects.')
print(f'Clipper finds {len(association)} associations.')

# Prepare submaps for visualization
edges = []
red_color = np.asarray([1,0,0]).reshape((1,3))
blue_color = np.asarray([0,0,1]).reshape((1,3))
seg0_color = red_color
seg1_color = blue_color
submap1_offset = np.asarray(args.offset)

ocd_list_0, label_list_0 = create_ptcld_geometries(submap_0, red_color, include_label=not args.no_text)
ocd_list_1, label_list_1 = create_ptcld_geometries(submap_1, blue_color, submap1_offset, include_label=not args.no_text)
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

for obj_idx_0, obj_idx_1 in association:
    print(f'Add edge between {obj_idx_0} and {obj_idx_1}.')
    # points = [submap_0[obj_idx_0].center, submap_1[obj_idx_1].center]
    points = [ocd_list_0[obj_idx_0].get_center(), ocd_list_1[obj_idx_1].get_center()]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector([[0,1]]),
    )
    line_set.colors = o3d.utility.Vector3dVector([[0,1,0]])
    edges.append(line_set)

app = o3d.visualization.gui.Application.instance
app.initialize()
vis = o3d.visualization.O3DVisualizer()
vis.show_skybox(False)

for i, geom in enumerate(ocd_list_0 + ocd_list_1 + edges):
    vis.add_geometry(f"geom-{i}", geom)
for label in label_list_0 + label_list_1:
    vis.add_3d_label(*label)
vis.add_geometry("origin", origin)

vis.reset_camera_to_default()
app.add_window(vis)
app.run()
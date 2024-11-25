import numpy as np
import cv2 as cv
import open3d as o3d

from roman.object.segment import Segment
from roman.map.map import ROMANMap

def visualize_map_on_img(t, pose, img, mapper):
    segment: Segment
    for i, segment in enumerate(mapper.get_segment_map()):
        # only draw segments seen in the last however many seconds
        if segment.last_seen < t - mapper.params.segment_graveyard_time - 10:
            continue
        outline = segment.outline_2d(pose)
        if outline is None:
            continue
        # if i < len(tracker.segments):
        #     color = (0, 255, 0)
        # elif i < len(tracker.segments) + len(tracker.inactive_segments):
        #     color = (255, 0, 0)
        # else:
        #     color = (180, 0, 180)
        color = segment.viz_color
        for i in range(len(outline) - 1):
            start_point = tuple(outline[i].astype(np.int32))
            end_point = tuple(outline[i+1].astype(np.int32))
            img = cv.line(img, start_point, end_point, color, thickness=2)

        img = cv.putText(img, str(segment.id), (np.array(outline[0]) + np.array([10., 10.])).astype(np.int32), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def visualize_observations_on_img(t, img, mapper, observations, reprojected_bboxs):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_fastsam = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        img_fastsam = img.copy()
    img_fastsam = np.concatenate([img_fastsam[...,None]]*3, axis=2)
    
    matched_masks = []
    for segment in mapper.segments:
        if segment.last_seen == t:
            colored_mask = np.zeros_like(img)
            rand_color = segment.viz_color
            matched_masks.append(segment.last_mask)
            try:
                colored_mask = segment.last_mask.astype(np.int32)[..., np.newaxis]*rand_color
            except:
                import ipdb; ipdb.set_trace()
            colored_mask = colored_mask.astype(np.uint8)
            img_fastsam = cv.addWeighted(img_fastsam, 1.0, colored_mask, 0.5, 0)
            mass_x, mass_y = np.where(segment.last_mask >= 1)
            img_fastsam = cv.putText(img_fastsam, str(segment.id), (int(np.mean(mass_y)), int(np.mean(mass_x))), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, rand_color.tolist(), 2)
    
    for obs in observations:
        alread_shown = False
        for mask in matched_masks:
            if np.all(mask == obs.mask):
                alread_shown = True
                break
        if alread_shown:
            continue
        white_mask = obs.mask.astype(np.int32)[..., np.newaxis]*np.ones(3)*255
        white_mask = white_mask.astype(np.uint8)
        img_fastsam = cv.addWeighted(img_fastsam, 1.0, white_mask, 0.5, 0)

    for seg_id, bbox in reprojected_bboxs:
        np.random.seed(seg_id)
        rand_color = np.random.randint(0, 255, 3)
        cv.rectangle(img_fastsam, np.array([bbox[0][0], bbox[0][1]]).astype(np.int32), 
                    np.array([bbox[1][0], bbox[1][1]]).astype(np.int32), color=rand_color.tolist(), thickness=2)
    

def visualize_3d(
    roman_map: ROMANMap, 
    id_range=None, 
    time_range=None,
    points_bounds = np.array([[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]]),
    show_labels=False, 
    show_origin=True,
    show_poses=True,
    min_pose_dist=0.5
):
        
    poses_list = []
    pcd_list = []
    label_list = []
    
    if time_range is not None:
        time_range = np.array(time_range) + roman_map.times[0]

    for seg in roman_map.segments:
        # if seg.extent[0] < 2.0 or seg.extent[1] > 1.0:
        #     continue
        if id_range is not None:
            if not (seg.id > id_range[0] and seg.id < id_range[1]):
                continue
        if time_range is not None:
            if seg.first_seen > time_range[1] or seg.last_seen < time_range[0]:
                continue
        seg_points = seg.points
        if seg_points is not None:
            for i in range(3):
                points_bounds[i, 0] = min(points_bounds[i, 0], np.min(seg_points[:, i]))
                points_bounds[i, 1] = max(points_bounds[i, 1], np.max(seg_points[:, i]))
            num_pts = seg_points.shape[0]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(seg_points)
            color = np.repeat(np.array(seg.viz_color).reshape((1,3))/255., num_pts, axis=0)
            pcd.colors = o3d.utility.Vector3dVector(color)
            pcd_list.append(pcd)
            if show_labels:
                # label = [f"id: {seg.id}", f"volume: {seg.volume():.2f}", 
                #         f"extent: [{seg.extent[0]:.2f}, {seg.extent[1]:.2f}, {seg.extent[2]:.2f}]"]
                label = [f"id: {seg.id}"]
                for i in range(len(label)):
                    label_list.append((np.median(pcd.points, axis=0) + np.array([0, 0, -0.15*i]), label[i]))
                    
    print(f"Displaying {len(pcd_list)} objects.")

    displayed_positions = []
    for i, Twb in enumerate(roman_map.trajectory):
        if np.any(Twb[:3,3] < points_bounds[:,0]) or np.any(Twb[:3,3] > points_bounds[:,1]):
            continue
        if displayed_positions and \
            np.linalg.norm(Twb[:3,3] - np.array(displayed_positions[-1])) < min_pose_dist:
            continue
        if time_range is not None:
            t = roman_map.times[i]
            if t < time_range[0] or t > time_range[1]:
                continue
        
        displayed_positions.append(Twb[:3,3])
        pose_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        pose_obj.transform(Twb)
        poses_list.append(pose_obj)

    if show_origin:
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        poses_list.append(origin)
    else:
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        T = np.eye(4)
        T[:3,3] = np.mean(displayed_positions, axis=0)
        origin.transform(T)
        poses_list.append(origin)
    
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer()
    vis.show_skybox(False)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    mat.point_size = 5.0

    for i, obj in enumerate(pcd_list):
        vis.add_geometry(f"pcd-{i}", obj, mat)
    for label in label_list:
        vis.add_3d_label(*label)
    if show_poses:
        for i, obj in enumerate(poses_list):
            vis.add_geometry(f"pose-{i}", obj)

    K = np.array([[200, 0, 200],
                [0, 200, 200],
                [0, 0, 1]]).astype(np.float64)
    if show_origin:
        T_inv = np.array([[1,   0,  0, 0],
                        [0,   0,  1, -5],
                        [0,   -1, 0, 0],
                        [0,   0,  0, 1]]).astype(np.float64)
    else:
        mean_position = np.mean(displayed_positions, axis=0)
        T_inv = np.array([[1,   0,  0, 0],
                        [0,   -1,  0, 0],
                        [0,   0, -1, 20],
                        [0,   0,  0, 1]]).astype(np.float64)
        T_inv[:3,3] += mean_position
    T = np.linalg.inv(T_inv)
    vis.setup_camera(K, T, 400, 400)
    app.add_window(vis)
    app.run()
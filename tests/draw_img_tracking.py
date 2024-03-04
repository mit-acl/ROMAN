import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
import pickle
import tqdm
import yaml

from robot_utils.robot_data.img_data import ImgData
from robot_utils.robot_data.pose_data import PoseData
from robot_utils.transform import transform
from robot_utils.robot_data.general_data import GeneralData

from plot_utils import remove_ticks

from img_utils import draw_cylinder

from segment_track.observation import Observation
from segment_track.segment import Segment
from segment_track.tracker import Tracker
from segment_track.fastsam_wrapper import FastSAMWrapper

def draw(img, pose, tracker, K):
    ax = plt.gca()
    ax.clear()
    remove_ticks(ax)

    for segment in tracker.segments + tracker.segment_graveyard:
        try:
            reconstruction = segment.reconstruction3D(width_height=True)
        except:
            continue
        centroid_w, width, height = reconstruction[:3], reconstruction[3], reconstruction[4]
        centroid_c = transform(np.linalg.inv(pose), centroid_w)
        if centroid_c[2] < 0: # behind camera
            continue
        img = draw_cylinder(img, K, centroid_c, width, height, color=(0, 255, 0), id=segment.id)

    ax.imshow(img[...,::-1])
    return


def update(t, img_data, pose_data, fastsam, tracker):

    try:
        img = img_data.img(t)
        img_t = img_data.nearest_time(t)
        pose = pose_data.T_WB(img_t)
    except:
        return
    observations = fastsam.run(t, pose, img)

    if np.round(t, 1) % 1 < 1e-3:
        tracker.merge()

    print(f"observations: {len(observations)}")

    if len(observations) > 0:
        tracker.update(t, observations)

    print(f"segments: {len(tracker.segments)}")

    draw(img, pose, tracker, img_data.camera_params.K)

    return

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str, help='Path to params file', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to output file', required=False, default=None)
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)

    assert params['bag']['path'] is not None, "bag must be specified in params"
    assert params['bag']['img_topic'] is not None, "img_topic must be specified in params"
    assert params['bag']['cam_info_topic'] is not None, "cam_info_topic must be specified in params"
    assert params['bag']['pose_topic'] is not None, "pose_topic must be specified in params"
    assert params['bag']['pose_time_tol'] is not None, "pose_time_tol must be specified in params"

    assert params['fastsam']['weights'] is not None, "weights must be specified in params"
    assert params['fastsam']['imgsz'] is not None, "imgsz must be specified in params"

    print("Loading image data...")
    img_data = ImgData(
        data_file=params["bag"]["path"],
        file_type='bag',
        topic=params["bag"]["img_topic"],
        time_tol=.02,
    )
    img_data.extract_params(params['bag']['cam_info_topic'])

    if 't0' in params['bag']:
        t0 = img_data.t0 + params['bag']['t0']
    else:
        t0 = img_data.t0

    if 'tf' in params['bag']:
        tf = img_data.t0 + params['bag']['tf']
    else:
        tf = img_data.tf

    print("Loading pose data...")
    pose_data = PoseData(
        data_file=params["bag"]["path"],
        file_type='bag',
        topic=params["bag"]["pose_topic"],
        time_tol=params["bag"]["pose_time_tol"],
        interp=True
    )

    print("Setting up FastSAM...")
    fastsam = FastSAMWrapper(
        weights=params['fastsam']['weights'],
        imgsz=params['fastsam']['imgsz'],
        device='cuda'
    )
    img_area = img_data.camera_params.width * img_data.camera_params.height
    fastsam.setup_filtering(
        ignore_people=True,
        yolo_det_img_size=(128, 128),
        allow_tblr_edges=[False, False, False, False],
        area_bounds=[img_area / 20**2, img_area / 5**2]
    )

    print("Setting up segment tracker...")
    tracker = Tracker(
        camera_params=img_data.camera_params,
        pixel_std_dev=10.0,
        min_iou=0.25,
        min_sightings=10,
        max_t_no_sightings=0.25
    )

    print("Running segment tracking!")
    fig, ax = plt.subplots()
    def update_wrapper(t): update(t, img_data, pose_data, fastsam, tracker); print(f"t: {t - t0:.2f}")
    ani = FuncAnimation(fig, update_wrapper, frames=tqdm.tqdm(np.arange(t0, tf, .05)), interval=10, repeat=False)

    if not args.output:
        plt.show()
    else:
        video_file = args.output + ".mp4"
        writervideo = FFMpegWriter(fps=30)
        ani.save(video_file, writer=writervideo)

        pkl_path = args.output + ".pkl"
        pkl_file = open(pkl_path, 'wb')
        pickle.dump(tracker, pkl_file, -1)
        pkl_file.close()




    # import ipdb; ipdb.set_trace()

    # update(t0, tf, img_data, pose_data, fastsam, tracker)
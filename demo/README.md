# ROMAN Demo

Using `demo.py`, the full ROMAN pipeline can be run on a multi-robot dataset, or mapping, loop closures, or pose graph optimization can be run separately.

After following the [install instructions](../README.md/#install) and the [demo setup instructions](../README.md/#demo), the demo can be run with the following command:

```
cd roman
mkdir demo_output
python3 demo/demo.py \
    -p demo/params/demo \
    -o demo_output
```

Optionally, the mapping process can be visualized with the `-m` argument to show the map projected on the camera image as it is created or `-3` command to show a 3D visualization of the map. However, these will cause the demo to run slower.

## Demo Output

The demo output includes three directories:
- `map`:
    - `<run>.mp4`: mapping visualization video (if demo run with `-m` or `-3` argument)
    - `<run>.pkl`: pickled ROMAN map (see the [ROMANMap class](../roman/map/map.py). This can be visualized using the `o3d_viz.py` script in this directory
- `align`:
    - `<run1>_<run2>`:
        - `align.png`: global localization results. If ground truth pose data has been provided, this file shows heatmaps showing the results of attempting to perform object registration between each pair of submaps from `<run1>` and `<run2>`. Ground truth submap distance is shown, along with registration translation and rotation errors, the number of associated objects, and the ground truth heading difference between the two submap center poses. This file will be much less meaningful if ground truth pose data is not provided to the demo. 
        - `align.pkl`: pickled global localization results (see the [SubmapAlignResults class](../roman/align/results.py))
- `offline_rpgo`:
    - `result.png`: resulting trajectory estimate plots
    - `result.txt`: absolute trajectory error result if ground truth is provided
    - `<run>.csv`: optimized trajectory estimate using ROMAN loop closures

The output includes map visualization, loop closure accuracy results, and pose graph optimization results including root mean squared absolute trajectory error.

## Custom Data

Up to this point, ROMAN has mostly been used to process ROS 1/2 data (agnostic of ROS version).
This documentation gives instructions for how to image data from a ROS bag and how to load pose data from a ROS bag or from a CSV file.
To run ROMAN on played back data or in real-time, use the [roman_ros](https://github.com/mit-acl/roman_ros) repository. 
If you are interested in loading in another type of data, the [ROMANMapRunner class](../roman/map/run.py) can be edited to load other data types using the `robotdatapy` [PoseData](https://github.com/mbpeterson70/robotdatapy/blob/main/robotdatapy/data/pose_data.py) and [ImgData](https://github.com/mbpeterson70/robotdatapy/blob/main/robotdatapy/data/img_data.py) interfaces.

To run ROMAN on your data, start by copying the [demo params directory](./params/demo).
In most cases, only the `data.yaml` and `fastsam.yaml` file will need to be customized. 
Within [data.yaml](./params/demo/data.yaml), the `img_data`, `depth_data`, and `pose_data` fields will need to be edited to load your data as explained in the following sections.
Within [fastsam.yaml](./params/demo/fastsam.yaml) the `depth_scale` and `max_depth` variables may need to be updated to reflect your depth camera.
Pass the updated params directory in using the `-p` argument.

### Loading ROS Image Data

Update the following subfields within the `img_data`, and `depth_data` fields in your new `data.yaml` file.

```
path: <bag file path>
topic: <topic name>
camera_info_topic: <topic name>
compressed: <true if datatype is compressed>
compressed_rvl: <true if data is compressed using rvl (if you do not know, probably false)>
```

### Loading ROS Pose Data

Update the following subfields within the `pose_data` field in your new `data.yaml` file.

If using a Pose or Odometry topic:

```
type: "bag"
path: <bag file path>
topic: <topic name of type Pose or Odometry>
time_tol: <time in seconds> # Pose data will not be interpolated if there is not a recorded pose within time_tol of a desired time
```

If the pose is recorded using `/tf` topic rather tha a Pose or Odometry message, then use the following:

```
type: bag_tf
path: <bag file path>
parent_frame: <parent frame_id>
child_frame: <child frame_id>
inv: <Whether the transformation from /tf should be inverted or not>
```

Additionally the following subfields should be provided if they are not identity: `T_camera_flu` and `T_odombase_camera`. These are used for determining (1) the pose of the a forward-left-up coordinate frame with respect to the camera frame and (2) the pose of the camera expressed in the odometry base frame. `T_camera_flu` is used to inform ROMAN roughly along which axis gravity will point, since ROMAN uses gravity direction to help guide loop closures. `T_odombase_camera` is necessary if the camera frame is not the child of the pose topic.

These two transformation subfields can be described with the following formats:

```
input_type: "string"
string: <"T_RDFFLU" or "T_FLURDF">
```

or 

```
input_type: "tf"
parent: <parent frame_id>
child: <child frame_id>
inv: <Whether the transformation from /tf_static should be inverted or not>
```

### Using environment variables in data paths/topics

To enable using the same set of parameters on different robot data, the `run_env` field in your [data.yaml](./params/demo/data.yaml) can be used to export a desired environment variable with the run names specified in the `runs` field. As an example, if you have two robots, julius and augustus, you can use `runs: [julius, augustus]` and in the path field of `img_data`, specify the path as `"${ROBOT_ENV_VAR}.bag`. Then use `run_env: ROBOT_ENV_VAR` so that ROMAN will correctly grab `julius.bag` for the julius run and `augustus.bag` for the augustus run.

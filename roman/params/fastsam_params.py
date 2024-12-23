###########################################################
#
# fastsam_params.py
#
# Parameter class for FastSAM wrapper
#
# Authors: Mason Peterson
#
# Dec. 21, 2024
#
###########################################################

from dataclasses import dataclass
import yaml

from typing import Tuple

@dataclass
class FastSAMParams:
    
    """
    FastSAM wrapper parameters.
    
    Args:
        weights_path (str): path to FastSAM weights
        imgsz (Tuple[int, int]): size of the input image
        device (str): device to run FastSAM on
        min_mask_len_div (int): minimum mask length division. The larger this parameter is the
            more smaller masks will be kept.
        max_mask_len_div (int): maximum mask length division. The smaller this parameter is the
            more larger masks will be kept.
        ignore_people (bool): whether to ignore people
        erosion_size (int): size of the erosion kernel
        voxel_size (float): voxel size of observations.
        ignore_labels (list): list of labels to ignore
        use_keep_labels (bool): whether to use keep labels
        keep_labels (list): list of labels to keep
        keep_labels_option (dict): options for keep labels
        plane_filter_params (tuple): parameters for plane filtering
        rotate_img (str): how to rotate the image ('CW', 'CCW', '180')
        clip (bool): whether to compute clip embeddings for observations
        yolo_imgsz (Tuple[int, int]): size of the YOLO image
        depth_scale (float): depth scale factor for processing depth images
        max_depth (float): maximum depth before rejecting observation points

    Returns:
        _type_: _description_
    """
    
    weights_path: str = "$FASTSAM_WEIGHTS_PATH"
    imgsz: Tuple[int, int] = (256, 256)
    device: str = 'cuda'
    mask_downsample_factor: int = 8
    min_mask_len_div: int = 30
    max_mask_len_div: int = 3
    ignore_people: bool = False
    erosion_size: int = 3
    voxel_size: float = 0.05
    ignore_labels: list = tuple(['person'])
    use_keep_labels: bool = False
    keep_labels: list = tuple([])
    keep_labels_option: dict = None
    plane_filter_params: tuple = tuple([3.0, 1.0, 0.2])
    rotate_img: str = None
    clip: bool = True
    yolo_imgsz: Tuple[int, int] = (256, 256)
    depth_scale: float = 1e3
    max_depth: float = 7.5
    
    @classmethod
    def from_yaml(cls, yaml_path: str, run: str = None):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if run is not None:
            data = data[run]
        return cls(**data)
    
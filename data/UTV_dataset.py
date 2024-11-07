import cv2
import torch
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
import data.data_utils as data_utils
import modules.midas.utils as utils
from PIL import Image

def load_input_image(input_image_fp):
    return utils.read_image(input_image_fp)


def load_sparse_depth(input_sparse_depth_fp):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth


def random_crop(inputs, shape, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        crop_type : str
            none, horizontal, vertical, anchored, top, bottom, left, right, center
    Return:
        list[numpy[float32]] : list of cropped inputs
    '''

    n_height, n_width = shape
    o_height, o_width, _ = inputs[0].shape

    # print(inputs[0].shape)

    # Get delta of crop and original height and width

    d_height = o_height - n_height
    d_width = o_width - n_width

    # print(d_width)
    # print(d_height)

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    # If left alignment, then set starting height to 0
    if 'left' in crop_type:
        x_start = 0

    # If right alignment, then set starting height to right most position
    elif 'right' in crop_type:
        x_start = d_width

    elif 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width)

    # If top alignment, then set starting height to 0
    if 'top' in crop_type:
        y_start = 0

    # If bottom alignment, then set starting height to lowest position
    elif 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type and np.random.rand() <= 0.30:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height)

    elif 'center' in crop_type:
        pass

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width

    outputs = [
        T[y_start:y_end, x_start:x_end] for T in inputs
    ]

    # reshape to original size
    outputs = [
        cv2.resize(T, (o_width, o_height)) for T in outputs
    ]

    return outputs



class UTV_dataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths,
                 mono_pred_paths,
                 radar_paths,
                 gt_paths,
                 sparse_gt_paths,
                 rcnet_paths=None,
                 random_shape=None,
                 random_flip=False,
                 rondom_radar_noise=None,
                 random_rcnet_thr=None,
                 ):

        self.n_sample = len(image_paths)

        for paths in [image_paths, mono_pred_paths, radar_paths, gt_paths, sparse_gt_paths]:
            assert len(paths) == self.n_sample
        if rcnet_paths is not None:
            assert len(rcnet_paths) == self.n_sample

        self.image_paths = image_paths
        self.mono_pred_paths = mono_pred_paths
        self.radar_paths = radar_paths
        self.gt_paths = gt_paths
        self.sparse_gt_paths = sparse_gt_paths
        self.rcnet_paths = rcnet_paths

        self.random_shape = random_shape
        self.random_flip = random_flip
        self.rondom_radar_noise = rondom_radar_noise
        self.random_rcnet_thr = random_rcnet_thr

    def __getitem__(self, index):
        image = load_input_image(self.image_paths[index])
        mono_pred = load_sparse_depth(self.mono_pred_paths[index])

        if self.radar_paths[index].endswith('.npy'):
            radar = np.load(self.radar_paths[index])
            # n*3 (u,v,depth) to H*W*1(depth), depth_map[v,u] = depth
            # radar_map shape is the same as mono_pred
            radar_map = np.zeros_like(mono_pred)
            for i in range(radar.shape[0]):
                radar_map[int(radar[i, 1]), int(radar[i, 0])] = radar[i, 2]
            radar = radar_map
        else:
            radar = load_sparse_depth(self.radar_paths[index])

        gt = load_sparse_depth(self.gt_paths[index])
        sparse_gt = load_sparse_depth(self.sparse_gt_paths[index])

        # Convert to float32
        image, mono_pred, radar, gt, sparse_gt = [
            T.astype(np.float32)
            for T in [image, mono_pred, radar, gt, sparse_gt]
        ]

        if self.rcnet_paths is not None:
            # replace 'rcnet_0.x' in rcnet_paths[index] with new threshold
            if self.random_rcnet_thr is not None:
                cur_thr = self.rcnet_paths[index].split('rcnet_')[-1][:3]
                rcnet_thr = np.random.choice(self.random_rcnet_thr)
                self.rcnet_paths[index] = self.rcnet_paths[index].replace(cur_thr, str(rcnet_thr))
            rcnet = load_sparse_depth(self.rcnet_paths[index])
            if rcnet.sum() == 0:
                print('rcnet is all zeros ', self.rcnet_paths[index])
                rcnet = radar
            rcnet = rcnet.astype(np.float32)
        else:
            rcnet = radar

        if self.random_shape is not None:
            if np.random.random() > 0.2:
                [image, mono_pred, radar, gt, sparse_gt, rcnet] = random_crop(
                    inputs=[image, mono_pred, radar, gt, sparse_gt, rcnet],
                    shape=self.random_shape,
                    crop_type=['horizontal', 'vertical'])

        if self.random_flip:
            if np.random.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                mono_pred = np.flip(mono_pred, axis=1).copy()
                radar = np.flip(radar, axis=1).copy()
                gt = np.flip(gt, axis=1).copy()
                sparse_gt = np.flip(sparse_gt, axis=1).copy()
                rcnet = np.flip(rcnet, axis=1).copy()

        # add random noise to radar
        if self.rondom_radar_noise is not None:
            if np.random.random() > 0.5:
                radar_valid = radar > 0
                radar[radar_valid] += np.random.normal(self.rondom_radar_noise[0],
                                                       self.rondom_radar_noise[1],
                                                       radar[radar_valid].shape)

        return image, mono_pred, radar, gt, sparse_gt, rcnet




    def __len__(self):
        return self.n_sample
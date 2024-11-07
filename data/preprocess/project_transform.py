import os
import numpy as np

def homogeneous_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
This function applies the homogenous transform using the dot product.
    :param points: Points to be transformed in a Nx4 numpy array.
    :param transform: 4x4 transformation matrix in a numpy array.
    :return: Transformed points of shape Nx4 in a numpy array.
    """
    if transform.shape != (4, 4):
        raise ValueError(f"{transform.shape} must be 4x4!")
    if points.shape[1] != 4:
        raise ValueError(f"{points.shape[1]} must be Nx4!")
    return transform.dot(points.T).T


def project_3d_to_2d(points: np.ndarray, projection_matrix: np.ndarray):
    """
This function projects the input 3d ndarray to a 2d ndarray, given a projection matrix.
    :param points: Homogenous points to be projected.
    :param projection_matrix: 4x4 projection matrix.
    :return: 2d ndarray of the projected points.
    """
    if points.shape[-1] != 4:
        raise ValueError(f"{points.shape[-1]} must be 4!")

    uvw = projection_matrix.dot(points.T)
    uvw /= uvw[2]
    uvs = uvw[:2].T
    uvs = np.round(uvs).astype(np.int32)

    return uvs


def canvas_crop(points, image_size, points_depth=None):
    """
This function filters points that lie outside a given frame size.
    :param points: Input points to be filtered.
    :param image_size: Size of the frame.
    :param points_depth: Filters also depths smaller than 0.
    :return: Filtered points.
    """
    idx = points[:, 0] > 0
    idx = np.logical_and(idx, points[:, 0] < image_size[1])
    idx = np.logical_and(idx, points[:, 1] > 0)
    idx = np.logical_and(idx, points[:, 1] < image_size[0])
    if points_depth is not None:
        idx = np.logical_and(idx, points_depth > 0)

    return idx


def min_max_filter(points, max_value, min_value):
    """
This function can be used to filter points based on a maximum and minimum value.
    :param points: Points to be filtered.
    :param max_value: Maximum value.
    :param min_value: Minimum value.
    :return: Filtered points.
    """
    idx = points < max_value
    idx = np.logical_and(idx, points > min_value)
    return idx


def project_pcl_to_image(point_cloud, t_camera_pcl, camera_projection_matrix, image_shape):
    """
A helper function which projects a point clouds specific to the dataset to the camera image frame.
    :param point_cloud: Point cloud to be projected.
    :param t_camera_pcl: Transformation from the pcl frame to the camera frame.
    :param camera_projection_matrix: The 4x4 camera projection matrix.
    :param image_shape: Size of the camera image.
    :return: Projected points, and the depth of each point.
    """
    point_homo = np.hstack((point_cloud[:, :3],
                            np.ones((point_cloud.shape[0], 1),
                                    dtype=np.float32)))
    radar_points_camera_frame = homogeneous_transformation(point_homo,
                                                           transform=t_camera_pcl)
    point_depth = radar_points_camera_frame[:, 2]

    uvs = project_3d_to_2d(points=radar_points_camera_frame,
                           projection_matrix=camera_projection_matrix)
    filtered_idx = canvas_crop(points=uvs,
                               image_size=image_shape,
                               points_depth=point_depth)
    uvs = uvs[filtered_idx]
    point_depth = point_depth[filtered_idx]

    # 返回的uvs和depth按照depth从大到小排列
    # Sort the uvs and depth based on depth in descending order
    sorted_indices = np.argsort(point_depth)[::-1]
    uvs = uvs[sorted_indices]
    point_depth = point_depth[sorted_indices]

    return uvs, point_depth




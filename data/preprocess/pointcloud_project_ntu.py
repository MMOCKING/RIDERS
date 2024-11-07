import numpy as np
import os
import cv2
from project_transform import project_pcl_to_image, min_max_filter
import open3d as o3d
import warnings
from data.data_utils import *
import multiprocessing as mp
import time

warnings.filterwarnings("ignore", message="Support for FigureCanvases without a required_interactive_framework attribute was deprecated")


def process_data(file_id, thermal_file, radar_scan, lidar_scan, T_camera_radar, camera_projection_matrix, T_camera_lidar, seq_path):
    # print('file_id: ', file_id)
    visualizer = Visualization2D(image=thermal_file,
                                radar_data=radar_scan,
                                lidar_data=lidar_scan,
                                save_path=seq_path,
                                save_name=file_id,
                                t_camera_radar=T_camera_radar,
                                camera_projection_matrix=camera_projection_matrix,
                                t_camera_lidar=T_camera_lidar)

    visualizer.plot_radar_pcl()
    visualizer.plot_lidar_pcl()



class Visualization2D:
    def __init__(self,
                 radar_data, t_camera_radar,
                 camera_projection_matrix, image,
                 save_path, save_name,
                 lidar_data = None, t_camera_lidar = None):
        self.radar_data = radar_data
        self.t_camera_radar = t_camera_radar
        self.camera_projection_matrix = camera_projection_matrix
        self.image = image

        self.save_r = os.path.join(save_path, 'radar_png', save_name + '.png')
        self.save_l = os.path.join(save_path, 'lidar_png', save_name + '.png')
        self.save_r_npy = os.path.join(save_path, 'radar_npy', save_name + '.npy')
        self.save_l_int = os.path.join(save_path, 'lidar_png_int', save_name + '.png')

        os.makedirs(os.path.join(save_path, 'radar_png'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'radar_npy'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'lidar_png'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'lidar_png_int'), exist_ok=True)


        if lidar_data is not None and t_camera_lidar is not None:
            self.lidar_data = lidar_data
            self.t_camera_lidar = t_camera_lidar


    def save_depth_map(self, points_depth, uvs, save_path, save_int=False, save_path_int=None):
        height, width = self.image.shape[:2]
        depth_map = np.zeros((height, width), dtype=np.float32)

        for i in range(len(points_depth)):
            depth = points_depth[i]
            u, v = uvs[i]
            depth_map[v, u] = max(depth, 1)

        save_depth(depth_map, save_path)

        if save_int:
            # if depth map is not empty, interpolate the depth map
            if np.sum(depth_map > 0) > 5:
                depth_map_interp = interpolate_depth_delft(depth_map)
                save_depth(depth_map_interp, save_path_int)
            else:
                print('depth map is empty, save the original depth map', save_path_int)
                print(np.sum(depth_map > 0))
                cv2.imwrite(save_path_int, np.zeros((height, width), dtype=np.float32))


    def plot_lidar_pcl(self, max_distance_threshold=100.0, min_distance_threshold=1):
        uvs, points_depth = project_pcl_to_image(point_cloud=self.lidar_data,
                                                 t_camera_pcl=self.t_camera_lidar,
                                                 camera_projection_matrix=self.camera_projection_matrix,
                                                 image_shape=self.image.shape)
        min_max_idx = min_max_filter(points_depth, max_value=max_distance_threshold, min_value=min_distance_threshold)
        uvs = uvs[min_max_idx]
        points_depth = points_depth[min_max_idx]
        self.save_depth_map(points_depth, uvs, save_path= self.save_l, save_int=True, save_path_int=self.save_l_int)


    def plot_radar_pcl(self, max_distance_threshold=100.0, min_distance_threshold=1):
        uvs, points_depth = project_pcl_to_image(point_cloud=self.radar_data,
                                                 t_camera_pcl=self.t_camera_radar,
                                                 camera_projection_matrix=self.camera_projection_matrix,
                                                 image_shape=self.image.shape)

        min_max_idx = min_max_filter(points_depth, max_value=max_distance_threshold, min_value=min_distance_threshold)
        uvs = uvs[min_max_idx]
        points_depth = points_depth[min_max_idx]

        self.save_depth_map(points_depth, uvs, save_path=self.save_r)

        radar_points = np.stack((uvs[:, 0], uvs[:, 1], points_depth), axis=1)
        np.save(self.save_r_npy, radar_points)





# [x, y, z, RCS, v_r, v_r_compensated, time]
if __name__ == '__main__':
    file_root = '/media/lh/lh1/dataset/NTU/garden'
    save_root = '/media/lh/lh1/dataset/NTU/garden'

    # dirs = sorted(os.listdir(file_root))
    # dirs = dirs[1:5]
    dirs = ['garden_2022-05-13_0', 'garden_2022-05-13_1']
    print(dirs)

    # pool = mp.Pool(processes=mp.cpu_count()-4)

    for dir in dirs:
        print(dir)
        seq_path = os.path.join(file_root, dir)
        save_path = os.path.join(save_root, dir)

        lidar_path = os.path.join(seq_path, 'lidar')
        radar_path = os.path.join(seq_path, 'radar_sync')
        thermal_path = os.path.join(seq_path, 'thermal_sync')

        file_names = sorted(os.listdir(lidar_path))

        # loop through all the radar files
        for file_name in file_names:

            file_id = file_name.split('.')[0]
            print(file_id)
            lidar_file = os.path.join(lidar_path, f'{file_id}.pcd')
            lidar_scan = np.asarray(o3d.io.read_point_cloud(lidar_file).points)

            radar_file = os.path.join(radar_path, f'{file_id}.pcd')
            radar_scan = np.asarray(o3d.io.read_point_cloud(radar_file).points)

            thermal_file = plt.imread(os.path.join(thermal_path, f'{file_id}.png'))

            image_width, image_height = 640, 512
            camera_projection_matrix = np.array([[4.7196351324104091e+02, 0, 3.3903066128694218e+02, 0],
                                                 [0, 4.7248642748309049e+02, 2.7774073717116710e+02, 0],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]])

            camera_projection_matrix_33 = camera_projection_matrix[:3, :3]


            # undistort
            dist_coeffs = np.array([-1.8566954779749040e-01, 1.6745260846914475e-01, -1.8122010952647307e-04,
                                    8.6534037842673963e-05, -1.0770856460153226e-01])
            thermal_file = cv2.undistort(thermal_file, camera_projection_matrix_33, dist_coeffs)
            save_thermal_path = os.path.join(seq_path, 'thermal_undistort')
            os.makedirs(save_thermal_path, exist_ok=True)
            save_thermal_file = os.path.join(save_thermal_path, f'{file_id}.png')
            plt.imsave(save_thermal_file, thermal_file)

            # P_c= T_camera_lidar * P_l
            T_camera_lidar = np.array([[-0.01577749,-0.99987429,-0.00055128,-0.17138222],
                                       [-0.00151076,0.00057628,-0.99999762,0.09600887],
                                       [0.99987328,-0.01577772,-0.00151857,-0.10307939],
                                       [0,0,0,1]])

            T_camera_radar = np.array([[-0.0241851,-0.999665,-0.00925436,-0.0248342],
                                       [0.0404891,0.00826999,-0.999146,0.09583170000000001],
                                       [0.998887,-0.0245392,0.0402755,0.0268037],
                                       [0,0,0,1]])


            # process_data(file_id,
            #              thermal_file,
            #              radar_scan,
            #              lidar_scan,
            #              T_camera_radar,
            #              camera_projection_matrix,
            #              T_camera_lidar,
            #              save_path)

            # pool.apply_async(process_data, args=(file_id,
            #                                      thermal_file,
            #                                      radar_scan,
            #                                      lidar_scan,
            #                                      T_camera_radar,
            #                                      camera_projection_matrix,
            #                                      T_camera_lidar,
            #                                      save_path))

    # pool.close()
    # pool.join()
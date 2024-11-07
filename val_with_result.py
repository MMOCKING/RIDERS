import os
import numpy as np
import data.UTV_dataset as UTV
import utils.eval_utils as eval_utils



if __name__ == '__main__':
    root = '/media/lh/lh1/dataset/ZJU-Multispectrum'
    scenes = ['2023-10-20-10-07-22']
    min_max = [0, 50]

    n_sample = 0
    for scene in scenes:
        scene_root = os.path.join(root,scene)
        n_sample += len(os.listdir(os.path.join(scene_root, 'lidar_png')))

    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)
    abs_rel = np.zeros(n_sample)
    sq_rel = np.zeros(n_sample)
    delta1 = np.zeros(n_sample)
    idx = 0

    for scene in scenes:
        scene_root = os.path.join(root, scene)
        result_root = os.path.join(root, 'output', 'DORN/depth', scene)
        file_val_names = sorted(os.listdir(os.path.join(scene_root, 'lidar_png')))

        for file_val_name in file_val_names:
            sparse_gt = UTV.load_sparse_depth(os.path.join(scene_root, 'lidar_png', file_val_name))
            output_depth = UTV.load_sparse_depth(os.path.join(result_root, file_val_name))

            validity_map = np.where(sparse_gt > 0, 1, 0)
            validity_mask = np.where(validity_map > 0, 1, 0)
            min_max_mask = np.logical_and(
                sparse_gt > min_max[0],
                sparse_gt < min_max[1])
            mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)
            sparse_gt = sparse_gt[mask]
            output_depth = output_depth[mask]

            mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * sparse_gt)
            rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * sparse_gt)
            imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * sparse_gt)
            irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * sparse_gt)
            abs_rel[idx] = eval_utils.mean_abs_rel_err(1000.0 * output_depth, 1000.0 * sparse_gt)
            sq_rel[idx] = eval_utils.mean_sq_rel_err(1000.0 * output_depth, 1000.0 * sparse_gt)
            delta1[idx] = eval_utils.thr_acc(output_depth, sparse_gt)
            idx += 1

    mae = np.mean(mae)
    rmse = np.mean(rmse)
    imae = np.mean(imae)
    irmse = np.mean(irmse)
    abs_rel = np.mean(abs_rel)
    sq_rel = np.mean(sq_rel)
    delta1 = np.mean(delta1)

    print('imae: ', imae)
    print('irmse: ', irmse)
    print('mae: ', mae)
    print('rmse: ', rmse)
    print('abs_rel: ', abs_rel)
    print('sq_rel: ', sq_rel)
    print('delta1: ', delta1)
    imae = '%.3f' % imae
    irmse = '%.3f' % irmse
    mae = '%.3f' % mae
    rmse = '%.3f' % rmse
    abs_rel = '%.3f' % abs_rel
    sq_rel = '%.3f' % sq_rel
    delta1 = '%.3f' % delta1

    print(imae,'&',irmse,'&',mae,'&',rmse,'&',abs_rel,'&',sq_rel,'&',delta1)

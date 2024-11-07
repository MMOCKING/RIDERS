import os
import glob
import time
import torch
import torch.utils.data
import numpy as np

from PIL import Image

import modules.midas.utils as utils
import data.data_utils as data_utils
import data.UTV_dataset as UTV
import modules.midas.transforms as transforms
import utils.eval_utils as eval_utils
import utils.log_utils as log_utils

from modules.midas.midas_net_custom import MidasNet_small_videpth, MidasNet_small_depth
from modules.midas.dpt_depth import DPTDepthModel
from modules.estimator import LeastSquaresEstimator, Optimizer
from modules.interpolator import Interpolator2D



def validate(
        best_results,
        ScaleMapLearner,
        step,
        ScaleMapLearner_transform,
        min_depth_inference, max_depth_inference,
        min_depth_val, max_depth_val,
        input_path, output_path,
        scenes,
        save_output=False,
        max_save_depth=None,

        random_sample=False,
        random_sample_size=1000,
        log_path=None,

        interp = 'rcnet',
        global_alignment = 's',
        mono_type = 'inv',
        mono_model = 'leres'
        ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_image_paths = []
    val_mono_pred_paths = []
    val_radar_paths = []
    val_gt_paths = []
    val_sparse_gt_paths = []
    if 'rcnet' in interp:
        val_rcnet_paths = []
    else:
        val_rcnet_paths = None

    image_file = 'thermal_undistort'
    mono_pred_file = mono_model
    radar_file = 'radar_png'
    gt_file = 'lidar_png'
    sparse_gt_file = 'lidar_png'

    for scene in scenes:
        scene_root = os.path.join(input_path, scene)

        image_paths = sorted(os.listdir(os.path.join(scene_root, image_file)))
        mono_pred_paths = sorted(os.listdir(os.path.join(scene_root, mono_pred_file)))
        radar_paths = sorted(os.listdir(os.path.join(scene_root, radar_file)))
        gt_paths = sorted(os.listdir(os.path.join(scene_root, gt_file)))
        sparse_gt_paths = sorted(os.listdir(os.path.join(scene_root, sparse_gt_file)))

        val_image_paths += [os.path.join(scene_root, image_file, image_path) for image_path in image_paths]
        val_mono_pred_paths += [os.path.join(scene_root, mono_pred_file, mono_pred_path) for mono_pred_path in
                                mono_pred_paths]
        val_radar_paths += [os.path.join(scene_root, radar_file, radar_path) for radar_path in radar_paths]
        val_gt_paths += [os.path.join(scene_root, gt_file, gt_path) for gt_path in gt_paths]
        val_sparse_gt_paths += [os.path.join(scene_root, sparse_gt_file, sparse_gt_path) for sparse_gt_path in
                                sparse_gt_paths]

        if 'rcnet' in interp:
            rcnet_root = os.path.join(result_root, interp, scene, 'depth_predicted')
            rcnet_paths = sorted(os.listdir(rcnet_root))
            val_rcnet_paths += [os.path.join(rcnet_root, rcnet_path) for rcnet_path in rcnet_paths]

    if random_sample:
        random_sample_idx = np.random.choice(len(val_image_paths), random_sample_size, replace=False)
        val_image_paths = [val_image_paths[idx] for idx in random_sample_idx]
        val_mono_pred_paths = [val_mono_pred_paths[idx] for idx in random_sample_idx]
        val_radar_paths = [val_radar_paths[idx] for idx in random_sample_idx]
        val_gt_paths = [val_gt_paths[idx] for idx in random_sample_idx]
        val_sparse_gt_paths = [val_sparse_gt_paths[idx] for idx in random_sample_idx]
        if 'rcnet' in interp:
            val_rcnet_paths = [val_rcnet_paths[idx] for idx in random_sample_idx]

    for paths in [val_mono_pred_paths, val_radar_paths, val_gt_paths, val_sparse_gt_paths]:
        assert len(val_image_paths) == len(paths)
    if 'rcnet' in interp:
        assert len(val_image_paths) == len(val_rcnet_paths)
    print('Number of validation samples: {}'.format(len(val_image_paths)))

    val_dataloader = torch.utils.data.DataLoader(
        UTV.UTV_dataset(
            val_image_paths,
            val_mono_pred_paths,
            val_radar_paths,
            val_gt_paths,
            val_sparse_gt_paths,
            val_rcnet_paths,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=1)

    n_sample = len(val_dataloader)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)
    abs_rel = np.zeros(n_sample)
    sq_rel = np.zeros(n_sample)
    delta1 = np.zeros(n_sample)

    for idx, inputs in enumerate(val_dataloader):
        inputs = [in_.to(device) for in_ in inputs]

        image, mono_pred, sparse_depth, gt, sparse_gt, rcnet = inputs
        input_height, input_width = image.shape[1:3]

        sparse_depth_valid = (sparse_depth < max_depth_inference) * (sparse_depth > min_depth_inference)
        sparse_depth_valid = sparse_depth_valid.bool()
        sparse_depth[~sparse_depth_valid] = np.inf  # set invalid depth
        sparse_depth = 1.0 / sparse_depth

        rcnet_valid = (rcnet < max_depth_inference) * (rcnet > min_depth_inference)
        rcnet_valid = rcnet_valid.bool()
        rcnet[~rcnet_valid] = np.inf  # set invalid depth
        rcnet = 1.0 / rcnet

        sparse_depth = sparse_depth.squeeze().cpu().numpy()
        sparse_depth_valid = sparse_depth_valid.squeeze().cpu().numpy()
        rcnet = rcnet.squeeze().cpu().numpy()
        rcnet_valid = rcnet_valid.squeeze().cpu().numpy()

        # global scale/shift alignment
        if global_alignment == 'st':
            GlobalAlignment = LeastSquaresEstimator(
                estimate=mono_pred.squeeze().cpu().numpy(),
                target=sparse_depth,
                valid=sparse_depth_valid
            )
            GlobalAlignment.compute_scale_and_shift()
            GlobalAlignment.apply_scale_and_shift()
            GlobalAlignment.clamp_min_max(clamp_min=min_pred, clamp_max=max_pred)
            int_depth = GlobalAlignment.output.astype(np.float32)
        elif global_alignment == 's':
            # global scale alignment
            GlobalAlignment = Optimizer(
                estimate=mono_pred.squeeze().cpu().numpy(),
                target=sparse_depth,
                valid=sparse_depth_valid,
                depth_type=mono_type
            )
            GlobalAlignment.optimize_scale()
            GlobalAlignment.apply_scale()
            GlobalAlignment.clamp_min_max(clamp_min=min_pred, clamp_max=max_pred)
            int_depth = GlobalAlignment.output.astype(np.float32)
        else:
            raise NotImplementedError

        if 'rcnet' in interp:
            int_scales = np.ones_like(int_depth)
            int_scales[rcnet_valid] = rcnet[rcnet_valid] / int_depth[rcnet_valid]
            int_scales[sparse_depth_valid] = sparse_depth[sparse_depth_valid] / int_depth[sparse_depth_valid]
            int_scales = utils.normalize_unit_range(int_scales.astype(np.float32))
        else:
            int_scales = np.ones_like(int_depth)
            int_scales[sparse_depth_valid] = sparse_depth[sparse_depth_valid] / int_depth[sparse_depth_valid]
            int_scales = utils.normalize_unit_range(int_scales.astype(np.float32))

        # transforms
        sample = {'image': image.squeeze().cpu().numpy(),
                  'int_depth': int_depth,
                  'int_scales': int_scales,
                  'int_depth_no_tf': int_depth}

        sample = ScaleMapLearner_transform(sample)

        x = torch.cat([sample['int_depth'], sample['int_scales']], 0)

        image_gray = sample['image'][0] * 0.299 + sample['image'][1] * 0.587 + sample['image'][2] * 0.114
        image_gray = image_gray.unsqueeze(0)
        x = torch.cat([x, image_gray], 0)

        x = x.to(device)
        d = sample['int_depth_no_tf'].to(device)

        with torch.no_grad():
            sml_pred = ScaleMapLearner.forward(x.unsqueeze(0), d.unsqueeze(0))
            sml_pred = (
                torch.nn.functional.interpolate(
                    1.0 / sml_pred,
                    size=(input_height, input_width),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        sparse_gt = np.squeeze(sparse_gt.cpu().numpy())
        validity_map = np.where(sparse_gt > 0, 1, 0)

        # Select valid regions to evaluate
        validity_mask = np.where(validity_map > 0, 1, 0)
        min_max_mask = np.logical_and(
            sparse_gt > min_depth_val,
            sparse_gt < max_depth_val)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)
        output_depth = sml_pred[mask]
        sparse_gt = sparse_gt[mask]

        # Compute validation metrics
        mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * sparse_gt)
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * sparse_gt)
        imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * sparse_gt)
        irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * sparse_gt)
        abs_rel[idx] = eval_utils.mean_abs_rel_err(1000.0 * output_depth, 1000.0 * sparse_gt)
        sq_rel[idx] = eval_utils.mean_sq_rel_err(1000.0 * output_depth, 1000.0 * sparse_gt)
        delta1[idx] = eval_utils.thr_acc(output_depth, sparse_gt)
        # if mae[idx] > 3000:
        #     print(val_image_paths[idx], mae[idx], rmse[idx])

        if save_output:
            basename = os.path.basename(val_image_paths[idx]).split('.')[0] + '.png'
            scene = val_image_paths[idx].split('/')[-3]
            os.makedirs(os.path.join(output_path, 'SML'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'SML', scene), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'SML', scene, 'sml_depth'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'SML', scene, 'sml_depth_color'), exist_ok=True)

            print('Saving output {}'.format(basename))
            data_utils.save_depth(sml_pred, os.path.join(output_path, 'SML', scene, 'sml_depth', basename))
            data_utils.save_color_depth(sml_pred, os.path.join(output_path, 'SML', scene, 'sml_depth_color', basename), max_depth=max_save_depth)

    # Compute mean metrics
    mae = np.mean(mae)
    rmse = np.mean(rmse)
    imae = np.mean(imae)
    irmse = np.mean(irmse)
    abs_rel = np.mean(abs_rel)
    sq_rel = np.mean(sq_rel)
    delta1 = np.mean(delta1)

    # Print validation results to console
    log_utils.log_evaluation_results(
        title='Validation results',
        mae=mae,
        rmse=rmse,
        imae=imae,
        irmse=irmse,
        abs_rel=abs_rel,
        sq_rel=sq_rel,
        delta1=delta1,
        step=step,
        log_path=log_path)

    n_improve = 0
    if np.round(mae, 4) < np.round(best_results['mae'], 4):
        n_improve = n_improve + 1
    if np.round(rmse, 4) < np.round(best_results['rmse'], 4):
        n_improve = n_improve + 1
    if np.round(imae, 4) < np.round(best_results['imae'], 4):
        n_improve = n_improve + 1
    if np.round(irmse, 4) < np.round(best_results['irmse'], 4):
        n_improve = n_improve + 1
    if np.round(abs_rel, 4) < np.round(best_results['abs_rel'], 4):
        n_improve = n_improve + 1
    if np.round(sq_rel, 4) < np.round(best_results['sq_rel'], 4):
        n_improve = n_improve + 1
    if np.round(delta1, 4) > np.round(best_results['delta1'], 4):
        n_improve = n_improve + 1

    if n_improve > 3:
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['imae'] = imae
        best_results['irmse'] = irmse
        best_results['abs_rel'] = abs_rel
        best_results['sq_rel'] = sq_rel
        best_results['delta1'] = delta1

    log_utils.log_evaluation_results(
        title='Best results',
        mae=best_results['mae'],
        rmse=best_results['rmse'],
        imae=best_results['imae'],
        irmse=best_results['irmse'],
        step=best_results['step'],
        abs_rel=best_results['abs_rel'],
        sq_rel=best_results['sq_rel'],
        delta1=best_results['delta1'],
        log_path=log_path)

    return best_results





if __name__ == "__main__":
    root = '/media/lh/lh1/dataset/ZJU-Multispectrum'
    result_root = '/media/lh/lh1/dataset/ZJU-Multispectrum/output'
    scenes = ['2023-10-20-10-07-22', '2023-10-20-10-28-46', '2023-10-20-14-35-31']

    checkpoint_dirpath = '/media/lh/lh1/dataset/ZJU-Multispectrum/log/SML/best-any'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_pred = 0.1
    max_pred = 255.0

    # transform, DPT_beit_512:512*512, midas_small:288*288
    ScaleMapLearner_transform = transforms.get_transforms(288, 288, depth_predictor='midas_small')

    # build model
    ScaleMapLearner = MidasNet_small_videpth(
        device = device,
        min_pred=min_pred,
        max_pred=max_pred,
        in_channels=3,
    )
    # ScaleMapLearner = DPTDepthModel(backbone="beitl16_512",
    #                                 device=device,
    #                                 min_depth=min_pred,
    #                                 max_depth=max_pred)


    ScaleMapLearner.eval()
    ScaleMapLearner.to(device)
    ScaleMapLearner = torch.nn.DataParallel(ScaleMapLearner)

    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imae': np.infty,
        'irmse': np.infty,
        'abs_rel': np.infty,
        'sq_rel': np.infty,
        'delta1': 0,
    }


    for checkpoint_filename in sorted(os.listdir(checkpoint_dirpath), reverse=True):
        if not checkpoint_filename.endswith('.pth'):
            continue

        step = int(checkpoint_filename.split('/')[-1].split('.')[0].split('-')[-1])

        checkpoint_filepath = os.path.join(checkpoint_dirpath, checkpoint_filename)
        log_path = os.path.join(checkpoint_dirpath, 'results.txt')
        ScaleMapLearner.module.load(checkpoint_filepath)
        print("Model weights loaded from {}".format(checkpoint_filename))


        with torch.no_grad():
            best_results = validate(
                best_results=best_results,
                ScaleMapLearner=ScaleMapLearner,
                ScaleMapLearner_transform = ScaleMapLearner_transform,
                step = step,

                min_depth_inference = 0.0,
                max_depth_inference = 100.0,

                min_depth_val = 0.0,
                max_depth_val = 50.0,

                input_path = root,
                output_path = result_root,
                scenes = scenes,
                log_path = log_path,

                interp='rcnet_0.1',
                global_alignment='s',
                mono_type='inv',
                mono_model='any',

                save_output=True,
                random_sample=False,
                random_sample_size=1000,
                max_save_depth=None
            )




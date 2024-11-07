import os, time, datetime
import cv2
import numpy as np
import torch, torchvision
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import data.data_utils as data_utils
import data.UTV_dataset as UTV
from utils.net_utils import OutlierRemoval
import utils.log_utils as log_utils
from utils.loss import compute_loss

from modules.midas.midas_net_custom import MidasNet_small_videpth, MidasNet_small_depth
from modules.midas.dpt_depth import DPTDepthModel
from modules.estimator import LeastSquaresEstimator, Optimizer
from modules import estimator
from modules.interpolator import Interpolator2D

import modules.midas.transforms as transforms
import modules.midas.utils as utils

def train(
        # data input
        train_root,
        scenes,
        image_file,
        mono_pred_file,
        radar_file,
        gt_file,
        sparse_gt_file,
        result_root,

        # training
        learning_rates,
        learning_schedule,
        batch_size,
        n_step_per_summary,
        n_step_per_checkpoint,
        random_crop_size,
        input_random_filp,
        input_random_brightness,
        input_random_contrast,
        input_random_saturation,
        input_random_radar_noise,

        # loss
        loss_func,
        w_smoothness,
        w_weight_decay,
        sobel_filter_size,
        w_lidar_loss,
        w_edge,
        w_unsupervised,
        ground_truth_outlier_removal_kernel_size,
        ground_truth_outlier_removal_threshold,
        ground_truth_dilation_kernel_size,

        # model
        restore_path,
        min_pred,
        max_pred,
        min_depth,
        max_depth,
        checkpoint_dirpath,
        n_threads=10,

        # pipeline
        model_type = 'midas-small',
        interp = 'none',
        random_rcnet_thr=None,
        global_alignment = 's',
        mono_type = 'inv'
        ):

    if not os.path.exists(checkpoint_dirpath):
        os.makedirs(checkpoint_dirpath)

    # Set up checkpoint and event paths
    depth_model_checkpoint_path = os.path.join(checkpoint_dirpath, 'model-{}.pth')
    log_path = os.path.join(checkpoint_dirpath, 'results.txt')
    event_path = os.path.join(checkpoint_dirpath, 'events')

    log_utils.log_params(log_path, locals())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_image_paths = []
    train_mono_pred_paths = []
    train_radar_paths = []
    train_gt_paths = []
    train_sparse_gt_paths = []
    if 'rcnet' in interp:
        train_rcnet_paths = []
    else:
        train_rcnet_paths = None

    for scene in scenes:
        scene_root = os.path.join(train_root, scene)

        image_paths = sorted(os.listdir(os.path.join(scene_root, image_file)))
        mono_pred_paths = sorted(os.listdir(os.path.join(scene_root, mono_pred_file)))
        radar_paths = sorted(os.listdir(os.path.join(scene_root, radar_file)))
        gt_paths = sorted(os.listdir(os.path.join(scene_root, gt_file)))
        sparse_gt_paths = sorted(os.listdir(os.path.join(scene_root, sparse_gt_file)))

        train_image_paths += [os.path.join(scene_root, image_file, image_path) for image_path in image_paths]
        train_mono_pred_paths += [os.path.join(scene_root, mono_pred_file, mono_pred_path) for mono_pred_path in
                                  mono_pred_paths]
        train_radar_paths += [os.path.join(scene_root, radar_file, radar_path) for radar_path in radar_paths]
        train_gt_paths += [os.path.join(scene_root, gt_file, gt_path) for gt_path in gt_paths]
        train_sparse_gt_paths += [os.path.join(scene_root, sparse_gt_file, sparse_gt_path) for sparse_gt_path in
                                  sparse_gt_paths]

        if 'rcnet' in interp:
            rcnet_root = os.path.join(result_root, interp, scene, 'depth_predicted')
            rcnet_paths = sorted(os.listdir(rcnet_root))
            train_rcnet_paths += [os.path.join(rcnet_root, rcnet_path) for rcnet_path in rcnet_paths]


    n_train_sample = len(train_image_paths)
    for paths in [train_mono_pred_paths, train_radar_paths, train_gt_paths,
                  train_sparse_gt_paths]:
        assert n_train_sample == len(paths)

    if 'rcnet' in interp:
        assert n_train_sample == len(train_rcnet_paths)

    print('Number of training samples: {}'.format(n_train_sample))

    # Set up training dataloader
    n_train_step = learning_schedule[-1] * np.ceil(n_train_sample / batch_size).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        UTV.UTV_dataset(
            train_image_paths,
            train_mono_pred_paths,
            train_radar_paths,
            train_gt_paths,
            train_sparse_gt_paths,
            train_rcnet_paths,
            random_shape=random_crop_size,
            random_flip=input_random_filp,
            rondom_radar_noise=input_random_radar_noise,
            random_rcnet_thr=random_rcnet_thr,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads)

    # Initialize ground truth outlier removal
    if ground_truth_outlier_removal_kernel_size > 1 and ground_truth_outlier_removal_threshold > 0:
        ground_truth_outlier_removal = OutlierRemoval(
            kernel_size=ground_truth_outlier_removal_kernel_size,
            threshold=ground_truth_outlier_removal_threshold)
    else:
        ground_truth_outlier_removal = None

    # Initialize ground truth dilation
    if ground_truth_dilation_kernel_size > 1:
        ground_truth_dilation = torch.nn.MaxPool2d(
            kernel_size=ground_truth_dilation_kernel_size,
            stride=1,
            padding=ground_truth_dilation_kernel_size // 2)
    else:
        ground_truth_dilation = None


    # transform, DPT_beit_512:512*512, midas_small:288*288
    # build model
    if model_type == 'midas-small':
        ScaleMapLearner_transform = transforms.get_transforms(288, 288, depth_predictor='midas_small',
                                                              random_brightness = input_random_brightness,
                                                              random_contrast = input_random_contrast,
                                                              random_saturation = input_random_saturation,)
        ScaleMapLearner = MidasNet_small_videpth(device = device,
                                                 min_pred=min_pred,
                                                 max_pred=max_pred,
                                                 in_channels=3,
                                                 )
    elif model_type == 'midas-small-depth':
        ScaleMapLearner_transform = transforms.get_transforms(288, 288, depth_predictor='midas_small')
        ScaleMapLearner = MidasNet_small_depth(device=device,
                                                 min_pred=min_pred,
                                                 max_pred=max_pred,
                                                 in_channels=3,
                                                 )
    elif model_type == 'dpt-large':
        ScaleMapLearner_transform = transforms.get_transforms(512, 512, depth_predictor='dpt_beit_large_512')
        ScaleMapLearner = DPTDepthModel(backbone="beitl16_512",
                                        device=device,
                                        min_depth=min_pred,
                                        max_depth=max_pred,
                                        )
    else:
        raise NotImplementedError

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    # Initialize optimizer with starting learning rate
    parameters_model = list(ScaleMapLearner.parameters())
    optimizer = torch.optim.Adam([
        {
            'params': parameters_model,
            'weight_decay': w_weight_decay
        }],
        lr=learning_rate)

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')

    # Start training
    train_step = 0

    if restore_path is not None and restore_path != '':
        ScaleMapLearner.load(restore_path)

    for g in optimizer.param_groups:
        g['lr'] = learning_rate

    time_start = time.time()

    print('Begin training...', log_path)
    for epoch in range(1, learning_schedule[-1] + 1):
        print('Epoch: ', epoch)
        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

           # Train model for an epoch
        for batch_data in train_dataloader:
            train_step = train_step + 1
            batch_data = [
                in_.to(device) for in_ in batch_data
            ]

            image, mono_pred, sparse_depth, gt, sparse_gt, rcnet = batch_data

            # sparse radar points depth
            sparse_depth_valid = (sparse_depth < max_depth) * (sparse_depth > min_depth)
            sparse_depth_valid = sparse_depth_valid.bool()
            sparse_depth[~sparse_depth_valid] = np.inf  # set invalid depth
            sparse_depth = 1.0 / sparse_depth

            # RCNet output preprocessing
            rcnet_valid = (rcnet < max_depth) * (rcnet > min_depth)
            rcnet_valid = rcnet_valid.bool()
            rcnet[~rcnet_valid] = np.inf  # set invalid depth
            rcnet = 1.0 / rcnet

            batch_size = sparse_depth.shape[0]  # 获取批量大小

            # empty batch
            batch_x = []
            batch_d = []
            batch_image = []
            batch_mono_pred = []
            batch_gt = []
            batch_sparse_gt = []

            for i in range(batch_size):
                # single sample in batch
                sparse_depth_i = sparse_depth[i].squeeze().cpu().numpy()
                sparse_depth_valid_i = sparse_depth_valid[i].squeeze().cpu().numpy()
                rcnet_i = rcnet[i].squeeze().cpu().numpy()
                rcnet_valid_i = rcnet_valid[i].squeeze().cpu().numpy()

                # global scale/shift alignment
                if global_alignment == 'st':
                    GlobalAlignment = LeastSquaresEstimator(
                        estimate=mono_pred[i].squeeze().cpu().numpy(),
                        target=sparse_depth_i,
                        valid=sparse_depth_valid_i
                    )
                    GlobalAlignment.compute_scale_and_shift()
                    GlobalAlignment.apply_scale_and_shift()
                    GlobalAlignment.clamp_min_max(clamp_min=min_pred, clamp_max=max_pred)
                    int_depth_i = GlobalAlignment.output.astype(np.float32)
                elif global_alignment == 's':
                    # global scale alignment
                    GlobalAlignment = Optimizer(
                        estimate=mono_pred[i].squeeze().cpu().numpy(),
                        target=sparse_depth_i,
                        valid=sparse_depth_valid_i,
                        depth_type=mono_type
                    )
                    GlobalAlignment.optimize_scale()
                    GlobalAlignment.apply_scale()
                    GlobalAlignment.clamp_min_max(clamp_min=min_pred, clamp_max=max_pred)
                    int_depth_i = GlobalAlignment.output.astype(np.float32)
                else:
                    raise NotImplementedError

                if 'rcnet' in interp:
                    # use rcnet output to replace int_scales_i
                    int_scales_i = np.ones_like(int_depth_i)
                    int_scales_i[rcnet_valid_i] = rcnet_i[rcnet_valid_i] / int_depth_i[rcnet_valid_i]
                    int_scales_i[sparse_depth_valid_i] = sparse_depth_i[sparse_depth_valid_i] / int_depth_i[
                        sparse_depth_valid_i]
                    if sparse_depth_valid_i.sum() + rcnet_valid_i.sum() > 1:
                        int_scales_i = utils.normalize_unit_range(int_scales_i.astype(np.float32))
                else:
                    int_scales_i = np.ones_like(int_depth_i)
                    int_scales_i[sparse_depth_valid_i] = sparse_depth_i[sparse_depth_valid_i] / int_depth_i[
                        sparse_depth_valid_i]
                    if sparse_depth_valid_i.sum() > 1:
                        int_scales_i = utils.normalize_unit_range(int_scales_i.astype(np.float32))


                # transforms
                sample = {'image': image[i].squeeze().cpu().numpy(),
                          'mono_pred': mono_pred[i].squeeze().cpu().numpy(),
                          'gt': gt[i].squeeze().cpu().numpy(),
                          'sparse_gt': sparse_gt[i].squeeze().cpu().numpy(),
                          'int_depth': int_depth_i,
                          'int_scales': int_scales_i,
                          'int_depth_no_tf': int_depth_i}

                sample = ScaleMapLearner_transform(sample)

                x = torch.cat([sample['int_depth'], sample['int_scales']], 0)

                image_gray = sample['image'][0] * 0.299 + sample['image'][1] * 0.587 + sample['image'][2] * 0.114
                image_gray = image_gray.unsqueeze(0)
                x = torch.cat([x, image_gray], 0)

                x = x.to(device)
                d = sample['int_depth_no_tf'].to(device)
                batch_x.append(x)
                batch_d.append(d)
                batch_image.append(sample['image'].to(device))
                batch_mono_pred.append(sample['mono_pred'].to(device))
                batch_gt.append(sample['gt'].to(device))
                batch_sparse_gt.append(sample['sparse_gt'].to(device))

            x = torch.stack(batch_x, dim=0)
            d = torch.stack(batch_d, dim=0)
            batch_image = torch.stack(batch_image, dim=0)
            batch_mono_pred = torch.stack(batch_mono_pred, dim=0)
            batch_gt = torch.stack(batch_gt, dim=0)
            batch_sparse_gt = torch.stack(batch_sparse_gt, dim=0)
            # Forward pass
            # print('x.shape', x.shape) 288 384
            sml_pred = ScaleMapLearner.forward(x, d)
            # inverse depth to depth
            d = 1.0 / d
            sml_pred = 1.0 / sml_pred

            validity_map_loss_smoothness = torch.ones_like(d)

            # Compute loss function
            invalid_map_gt = batch_gt <= 0
            # validity_map_loss_smoothness = invalid_map_gt
            if ground_truth_dilation is not None:
                batch_gt = ground_truth_dilation(batch_gt)

            if ground_truth_outlier_removal is not None:
                batch_gt = ground_truth_outlier_removal.remove_outliers(batch_gt)

            # validity_map_loss_smoothness = torch.where(
            #     batch_gt > 0,
            #     torch.zeros_like(batch_gt),
            #     torch.ones_like(batch_gt))

            loss, loss_info = compute_loss(
                image=d,
                output_depth=sml_pred,
                gt_interp=batch_gt,
                gt_sparse=batch_sparse_gt,
                loss_func=loss_func,
                w_smoothness=w_smoothness,
                sobel_filter_size=sobel_filter_size,
                validity_map_loss_smoothness=validity_map_loss_smoothness,
                w_lidar_loss=w_lidar_loss,
                w_edge=w_edge,
                invalid_map_gt=invalid_map_gt,
                w_unsupervised=w_unsupervised)
            print('{}/{} epoch:{}: {}'.format(train_step % n_train_step, n_train_step, epoch, loss.item()))

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_step_per_summary) == 0:
                with torch.no_grad():
                    # Log tensorboard summary
                    log_utils.log_summary(
                        summary_writer=train_summary_writer,
                        tag='train',
                        step=train_step,
                        max_predict_depth=max_pred,
                        image=batch_image,
                        input_mono=batch_mono_pred,
                        input_depth=d,
                        output_depth=sml_pred,
                        ground_truth=batch_gt,
                        scalars=loss_info,
                        n_display=min(batch_size, 4))

            # Log results and save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                print('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)
                # Save checkpoints
                ScaleMapLearner.save(depth_model_checkpoint_path.format(train_step))

    # Save checkpoints
    ScaleMapLearner.save(depth_model_checkpoint_path.format(train_step))



if __name__ == '__main__':

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train(
        # data input
        train_root = '/media/lh/lh1/dataset/ZJU-Multispectrum',
        scenes = ['2023-10-19-19-25-47',
                  '2023-10-20-10-05-18', '2023-10-20-10-21-14',
                  '2023-10-20-10-35-20', '2023-10-20-13-56-28',
                  '2023-10-20-14-23-10', '2023-10-20-14-28-18',
                  '2023-10-20-14-38-17', '2023-10-20-14-53-28'],

        image_file = 'thermal_undistort',
        mono_pred_file = 'any',
        radar_file = 'radar_png',
        gt_file = 'lidar_png_int',
        sparse_gt_file = 'lidar_png',
        result_root = '/media/lh/lh1/dataset/ZJU-Multispectrum/output', # rcnet output

        # training
        learning_rates = [1e-4,5e-5], #[1e-4,5e-5],
        learning_schedule = [20,200], #[40,80],
        batch_size = 12,
        n_step_per_summary = 10,
        n_step_per_checkpoint = 1000,
        input_random_filp=True,
        random_crop_size=None,  # (448,560)
        input_random_brightness=None,
        input_random_contrast=None,
        input_random_saturation=None,
        input_random_radar_noise=[-0.01, 0.01], # random depth noise

        # Loss settings
        loss_func = 'l1',
        # weights
        w_lidar_loss = 1.5, # the weight of sparse lidar GT depth are heavier than interpolated lidar GT depth
        w_edge = 0.0, # edge matching loss
        w_smoothness = 0.2, # depth map are desired to be smooth except edges
        w_unsupervised= 0.0, # for areas without lidar GT depth
        # kernel sizes
        sobel_filter_size = 7, # kernel for edge detection
        w_weight_decay = 0.0, # weight decay
        ground_truth_outlier_removal_kernel_size = 3, # remove interpolated gt outliers 7/3
        ground_truth_outlier_removal_threshold = 1.5, # removal threshold in meters
        ground_truth_dilation_kernel_size = -1, # dilate ground truth (unused)

        # model
        restore_path = None,
        min_pred = 0.1, # initialize scale map learner
        max_pred = 255.0,
        min_depth = 0.0, # filter input depth
        max_depth = 100.0,
        checkpoint_dirpath = os.path.join('/media/lh/lh1/dataset/ZJU-Multispectrum/log/SML', current_time),
        n_threads=4,

        # pipeline
        model_type='midas-small',  # scale map learner network type：midas-small, dpt-large (need large GPU memory)
        interp='rcnet_0.1',  # rcnet: use quasi-dense depth from rcnet，none: use sparse radar points
        random_rcnet_thr=None,  # None or List[float]，e.g. [0.6,0.8,0.8]，rcnet random thresholds
        global_alignment='s',  # s: scale, st: scale-translation
        mono_type='inv'  # input monocular depth type: inv or pos
    )
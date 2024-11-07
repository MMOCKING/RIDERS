import os, time
import numpy as np

# Dependencies for network, loss, etc.
import torch, torchvision
import torch.utils.data
from rcnet_model import RCNetModel

# Dependencies for data loading
from data import datasets
from utils import eval_utils
from rcnet_transforms import Transforms

# Dependencies for logging
from utils.log_utils import log
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def train(root,
          scenes,
          image_file,
          radar_file,
          gt_file,
          # Input settings
          batch_size,
          patch_size,
          total_points_sampled,
          sample_probability_of_lidar,
          normalized_image_range,
          # Network settings
          encoder_type,
          n_filters_encoder_image,
          n_neurons_encoder_depth,
          decoder_type,
          n_filters_decoder,
          # Weight settings
          weight_initializer,
          activation_func,
          # Training settings
          learning_rates,
          learning_schedule,
          augmentation_probabilities,
          augmentation_schedule,
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_saturation,
          augmentation_random_noise_type,
          augmentation_random_noise_spread,
          augmentation_random_flip_type,
          # Loss settings
          w_weight_decay,
          w_positive_class,
          max_distance_correspondence,
          set_invalid_to_negative_class,
          # Checkpoint and summary settings
          checkpoint_dirpath,
          n_step_per_summary,
          n_step_per_checkpoint,
          restore_path,

          # Hardware settings
          n_thread=10):
    # Set up device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up checkpoint directory
    if not os.path.exists(checkpoint_dirpath):
        os.makedirs(checkpoint_dirpath)

    checkpoint_path = os.path.join(checkpoint_dirpath, 'model-{}.pth')
    log_path = os.path.join(checkpoint_dirpath, 'results.txt')
    event_path = os.path.join(checkpoint_dirpath, 'events')


    '''
    Set up paths for training
    '''
    train_image_paths = []
    train_radar_paths = []
    train_ground_truth_paths = []
    for scene in scenes:
        scene_root = os.path.join(root, scene)
        image_paths = sorted(os.listdir(os.path.join(scene_root, image_file)))
        radar_paths = sorted(os.listdir(os.path.join(scene_root, radar_file)))
        gt_paths = sorted(os.listdir(os.path.join(scene_root, gt_file)))
        train_image_paths += [os.path.join(scene_root, image_file, image_path) for image_path in image_paths]
        train_radar_paths += [os.path.join(scene_root, radar_file, radar_path) for radar_path in radar_paths]
        train_ground_truth_paths += [os.path.join(scene_root, gt_file, gt_path) for gt_path in gt_paths]

    print('length of train image:', len(train_image_paths))
    print('length of train radar:', len(train_radar_paths))
    print('length of train gt:', len(train_ground_truth_paths))

    n_train_sample = len(train_image_paths)

    assert n_train_sample == len(train_radar_paths)
    assert n_train_sample == len(train_ground_truth_paths)

    # Set up training dataloader
    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / batch_size).astype(np.int32)

    # Set up data loader and data transforms
    train_dataloader = torch.utils.data.DataLoader(
        datasets.RCNetTrainingDataset(
            image_paths=train_image_paths,
            radar_paths=train_radar_paths,
            ground_truth_paths=train_ground_truth_paths,
            patch_size=patch_size,
            total_points_sampled=total_points_sampled,
            sample_probability_of_lidar=sample_probability_of_lidar),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_thread)

    train_transforms = Transforms(
        normalized_image_range=normalized_image_range,
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_saturation=augmentation_random_saturation,
        random_noise_type=augmentation_random_noise_type,
        random_noise_spread=augmentation_random_noise_spread,
        random_flip_type=augmentation_random_flip_type)


    '''
    Set up the model
    '''
    # Build network
    rcnet_model = RCNetModel(
        input_channels_image=3,
        input_channels_depth=3,
        input_patch_size_image=patch_size,
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_neurons_encoder_depth=n_neurons_encoder_depth,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        device=device)

    rcnet_model.to(device)
    rcnet_model.data_parallel()

    parameters_rcnet_model = rcnet_model.parameters()

    '''
    Log settings
    '''
    log('Training input paths:', log_path)
    train_input_paths = [
        image_file,
        radar_file,
        gt_file
    ]

    for path in train_input_paths:
        log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        input_channels_image=3,
        input_channels_depth=3,
        input_patch_size_image=patch_size,
        normalized_image_range=normalized_image_range)

    log_network_settings(
        log_path,
        # Network settings
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_neurons_encoder_depth=n_neurons_encoder_depth,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_rcnet_model)

    log_training_settings(
        log_path,
        # Training settings
        batch_size=batch_size,
        n_train_sample=n_train_sample,
        n_train_step=n_train_step,
        learning_rates=learning_rates,
        learning_schedule=learning_schedule,
        # Augmentation settings
        augmentation_probabilities=augmentation_probabilities,
        augmentation_schedule=augmentation_schedule,
        augmentation_random_brightness=augmentation_random_brightness,
        augmentation_random_contrast=augmentation_random_contrast,
        augmentation_random_saturation=augmentation_random_saturation,
        augmentation_random_noise_type=augmentation_random_noise_type,
        augmentation_random_noise_spread=augmentation_random_noise_spread,
        augmentation_random_flip_type=augmentation_random_flip_type)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        w_weight_decay=w_weight_decay,
        w_positive_class=w_positive_class,
        max_distance_correspondence=max_distance_correspondence,
        set_invalid_to_negative_class=set_invalid_to_negative_class)


    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_dirpath=checkpoint_dirpath,
        n_step_per_checkpoint=n_step_per_checkpoint,
        summary_event_path=event_path,
        n_step_per_summary=n_step_per_summary,
        restore_path=restore_path,
        # Hardware settings
        device=device,
        n_thread=n_thread)

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    augmentation_schedule_pos = 0
    augmentation_probability = augmentation_probabilities[0]

    # Initialize optimizer with starting learning rate
    optimizer = torch.optim.Adam([
        {
            'params': parameters_rcnet_model,
            'weight_decay': w_weight_decay
        }],
        lr=learning_rate)

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')

    # Start training
    train_step = 0

    if restore_path is not None and restore_path != '':
        train_step, optimizer = rcnet_model.restore_model(
            restore_path,
            optimizer=optimizer)

        for g in optimizer.param_groups:
            g['lr'] = learning_rate

    time_start = time.time()

    log('Begin training...', log_path)
    for epoch in range(1, learning_schedule[-1] + 1):
        print('Epoch: ', epoch)
        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
        # Set augmentation schedule
        if -1 not in augmentation_schedule and epoch > augmentation_schedule[augmentation_schedule_pos]:
            augmentation_schedule_pos = augmentation_schedule_pos + 1
            augmentation_probability = augmentation_probabilities[augmentation_schedule_pos]

        for batch_data in train_dataloader:

            train_step = train_step + 1

            # Fetch data
            batch_data = [
                in_.to(device) for in_ in batch_data
            ]

            # Load image (N x 3 x H x W), radar point (N x 3), and ground truth (N x 1 x H x W)
            image, radar_point, bounding_boxes_list, ground_truth_depth = batch_data

            # Apply augmentations and data transforms
            [image], [ground_truth_depth], [radar_point], [bounding_boxes_list] = train_transforms.transform(
                images_arr=[image],
                labels_arr=[ground_truth_depth],
                points_arr=[radar_point],
                bounding_boxes_arr=[bounding_boxes_list],
                random_transform_probability=augmentation_probability)

            # print(ground_truth_depth.shape)
            radar_points_for_summary = radar_point.clone()
            radar_point = radar_point.view(radar_point.shape[0] * radar_point.shape[1], radar_point.shape[2])

            # for radar_depth_idx in range(0,radar_point.shape[1]):
            #     radar_depth = radar_point[:, radar_depth_idx, 2].view(radar_point.shape[0], 1, 1, 1, 1)

            radar_depth = radar_point[..., 2].view(radar_point.shape[0], 1, 1, 1)
            ground_truth_depth = ground_truth_depth.view(ground_truth_depth.shape[0] * ground_truth_depth.shape[1],
                                                         ground_truth_depth.shape[2], ground_truth_depth.shape[3],
                                                         ground_truth_depth.shape[4])

            '''
            Create ground truth labels and validity map
            '''

            distance_radar_ground_truth_depth = \
                torch.abs(ground_truth_depth - radar_depth * torch.ones_like(ground_truth_depth))

            # Correspondences are any point less than distance threshold
            ground_truth_label = torch.where(
                distance_radar_ground_truth_depth < max_distance_correspondence,
                torch.ones_like(ground_truth_depth),
                torch.zeros_like(ground_truth_depth))

            # Any missing empty points will be marked as invalid
            ground_truth_label = torch.where(
                ground_truth_depth > 0,
                ground_truth_label,
                torch.zeros_like(ground_truth_label))

            # Create valid locations to compute loss
            if set_invalid_to_negative_class:
                # Every pixel will be valid
                validity_map = torch.ones_like(ground_truth_depth)
            else:
                # Mask out invalid pixels in loss
                validity_map = torch.where(
                    ground_truth_depth <= 0,
                    torch.zeros_like(ground_truth_depth),
                    torch.ones_like(ground_truth_depth))

            '''
            Forward through network and compute loss
            '''

            bounding_boxes_list_new = []
            for bounding_box_batch_idx in range(0, bounding_boxes_list.shape[0]):
                bounding_boxes_list_new.append(bounding_boxes_list[bounding_box_batch_idx])

            logits = rcnet_model.forward(image, radar_point, bounding_boxes_list_new, return_logits=True)

            # Compute loss
            ground_truth_label = ground_truth_label.float()

            loss, loss_info = rcnet_model.compute_loss(
                logits=logits,
                ground_truth=ground_truth_label,
                validity_map=validity_map,
                w_positive_class=w_positive_class)

            # step/step per epoch/epoch/loss
            print('{}/{} epoch:{}: {}'.format(train_step%n_train_step, n_train_step, epoch, loss.item()))

            # Backwards pass and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # input()

            # Log summary
            if (train_step % n_step_per_summary) == 0:

                image_crops = []

                # Crop image and ground truth
                for batch_idx in range(0, bounding_boxes_list.shape[0]):
                    for bounding_box_idx in range(0, bounding_boxes_list.shape[1]):
                        start_x = int(bounding_boxes_list[batch_idx, bounding_box_idx, 0].item())
                        end_x = int(bounding_boxes_list[batch_idx, bounding_box_idx, 2].item())
                        start_y = int(bounding_boxes_list[batch_idx, bounding_box_idx, 1].item())
                        end_y = int(bounding_boxes_list[batch_idx, bounding_box_idx, 3].item())
                        # input()
                        image_cropped = image[batch_idx, :, start_y:end_y, start_x:end_x]
                        image_crops.append(image_cropped)

                image_cropped_for_summary = torch.stack(image_crops, dim=0)

                with torch.no_grad():
                    # Convert logits to response and label
                    response = torch.sigmoid(logits)
                    label = torch.where(
                        response > 0.50,
                        torch.ones_like(response),
                        torch.zeros_like(response))

                    n_ground_truth_label = \
                        torch.mean(torch.sum(ground_truth_label.float(), dim=[1, 2, 3]))
                    n_label = \
                        torch.mean(torch.sum(label.float(), dim=[1, 2, 3]))

                    loss_info['average_ground_truth_label_per_point'] = n_ground_truth_label
                    loss_info['average_predicted_label_per_point'] = n_label

                    # Log tensorboard summary
                    rcnet_model.log_summary(
                        summary_writer=train_summary_writer,
                        tag='train',
                        step=train_step,
                        image=image_cropped_for_summary,
                        output_response=response,
                        output_label=label,
                        validity_map=validity_map,
                        ground_truth_label=ground_truth_label,
                        ground_truth_depth=ground_truth_depth,
                        scalars=loss_info,
                        n_display=4)

            # Log results and save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{} Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, time_elapse, time_remain),
                    log_path)

                log('Loss={:.5f}'.format(loss.item()), log_path)

                # Save model to checkpoint
                rcnet_model.save_model(
                    checkpoint_path.format(train_step),
                    step=train_step,
                    optimizer=optimizer)

    # Save model to checkpoint
    rcnet_model.save_model(
        checkpoint_path.format(train_step),
        step=train_step,
        optimizer=optimizer)


def forward_output(model, image, radar_points, bounding_boxes_list, response_thr=0.5, device=torch.device('cuda')):
    # Determine crop size for possible radar correspondence
    patch_size = model.input_patch_size_image
    pad_size_x = patch_size[1] // 2
    pad_size_y = patch_size[0] // 2

    image = torchvision.transforms.functional.pad(
        image,
        (pad_size_x, pad_size_y, pad_size_x, pad_size_y),
        padding_mode='edge')

    output_tiles = []
    if radar_points.dim() == 3:
        # Convert to 1 x N x 3 to N x 3
        radar_points = torch.squeeze(radar_points, dim=0)

    x_shifts = radar_points[:, 0].clone()
    y_shifts = radar_points[:, 1].clone()

    output_crops = model.forward(
        image=image,
        point=radar_points,
        bounding_boxes=bounding_boxes_list,
        return_logits=False)

    for output_crop, x, y in zip(output_crops, x_shifts, y_shifts):
        output = torch.zeros([1, image.shape[-2], image.shape[-1]], device=device)
        output_crop = torch.where(output_crop < response_thr, torch.zeros_like(output_crop), output_crop)
        output[:, int(y)-pad_size_y:int(y)+pad_size_y, int(x)-pad_size_x:int(x)+pad_size_x] = output_crop
        output_tiles.append(output)
    output_tiles = torch.cat(output_tiles, dim=0)
    output_tiles = output_tiles[:, pad_size_y:-pad_size_y, pad_size_x:-pad_size_x]

    output_response, output = torch.max(output_tiles, dim=0, keepdim=True)

    # Fill in the map based on z value of the points chosen
    # for point_idx in range(radar_points.shape[0]):
    #     output = torch.where(
    #         output == point_idx,
    #         torch.full_like(output, fill_value=radar_points[point_idx, 2]),
    #         output)

    # use output_tiles confidence score as weights，get weighted average depth
    output = torch.sum(output_tiles * radar_points[:, 2].unsqueeze(1).unsqueeze(1), dim=0, keepdim=True) / \
                torch.sum(output_tiles, dim=0, keepdim=True)

    # Leave as 0s if we did not predict
    output_depth = torch.where(
        torch.max(output_tiles, dim=0, keepdim=True)[0] == 0,
        torch.zeros_like(output),
        output)

    return output_depth, output_response


def validate(model,
             patch_size,
             dataloader,
             transforms,
             step,
             best_results,
             min_evaluate_depth,
             max_evaluate_depth,
             device,
             summary_writer,
             log_path,
             response_thr=0.5):
    n_sample = len(dataloader)

    # Define evaluation metrics
    mae_intersection = np.zeros(n_sample)
    rmse_intersection = np.zeros(n_sample)
    imae_intersection = np.zeros(n_sample)
    irmse_intersection = np.zeros(n_sample)

    n_valid_points_output = np.zeros(n_sample)
    n_valid_points_ground_truth = np.zeros(n_sample)
    n_valid_points_intersection = np.zeros(n_sample)

    image_summaries = []
    output_depth_summaries = []
    ground_truth_summaries = []

    for sample_idx, batch_data in enumerate(dataloader):

        batch_data = [
            data.to(device) for data in batch_data
        ]

        # 1 x 3 x H x W image, 1 x N x 3 points
        image, radar_points, ground_truth = batch_data
        bounding_boxes_list = []

        pad_size_x = patch_size[1] // 2
        pad_size_y = patch_size[0] // 2
        radar_points = radar_points.squeeze(dim=0)

        if radar_points.ndim == 1:
            # Expand to 1 x 3
            radar_points = np.expand_dims(radar_points, axis=0)

        # get the shifted radar points after padding
        for radar_point_idx in range(0, radar_points.shape[0]):
            # Set radar point to the center of the patch
            radar_points[radar_point_idx, 0] = radar_points[radar_point_idx, 0] + pad_size_x
            radar_points[radar_point_idx, 1] = radar_points[radar_point_idx, 1] + pad_size_y
            bounding_box = torch.zeros(4)
            bounding_box[0] = radar_points[radar_point_idx, 0] - pad_size_x
            # bounding_box[1] = 0
            bounding_box[1] = radar_points[radar_point_idx, 1] - pad_size_y
            bounding_box[2] = radar_points[radar_point_idx, 0] + pad_size_x
            # bounding_box[3] = image.shape[-2]
            bounding_box[3] = radar_points[radar_point_idx, 1] + pad_size_y
            bounding_boxes_list.append(bounding_box)

        bounding_boxes_list = [torch.stack(bounding_boxes_list, dim=0)]

        [image], [radar_points], [bounding_boxes_list] = transforms.transform(
            images_arr=[image],
            points_arr=[radar_points],
            bounding_boxes_arr=[bounding_boxes_list],
            random_transform_probability=0.0)

        output_depth, output_response = forward_output(
            model=model,
            image=image,
            radar_points=radar_points,
            bounding_boxes_list=bounding_boxes_list,
            device=device,
            response_thr=response_thr)

        # Display summary
        if sample_idx % 500 == 0:
            image_summary = image
            ground_truth_summary = ground_truth
            output_depth_summary = torch.unsqueeze(output_depth, dim=0)

            image_summaries.append(image_summary)
            output_depth_summaries.append(output_depth_summary)
            ground_truth_summaries.append(ground_truth_summary)

        # Do evaluation against ground truth here
        ground_truth = np.squeeze(ground_truth.cpu().numpy())
        output_depth = np.squeeze(output_depth.cpu().numpy())

        # Validity map of output -> locations where output is valid
        validity_map_output = np.where(output_depth > 0, 1, 0)
        validity_map_ground_truth = np.where(ground_truth > 0, 1, 0)
        validity_map_intersection = validity_map_output * validity_map_ground_truth

        n_valid_points_intersection[sample_idx] = np.sum(validity_map_intersection)
        n_valid_points_output[sample_idx] = np.sum(validity_map_output)
        n_valid_points_ground_truth[sample_idx] = np.sum(validity_map_ground_truth)

        # Select valid regions to evaluate
        min_max_mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)
        mask_intersection = np.where(np.logical_and(validity_map_intersection, min_max_mask) > 0)

        output_depth_intersection = output_depth[mask_intersection]
        ground_truth_intersection = ground_truth[mask_intersection]

        # Compute validation metrics for intersection
        mae_intersection[sample_idx] = eval_utils.mean_abs_err(1000.0 * output_depth_intersection,
                                                               1000.0 * ground_truth_intersection)
        rmse_intersection[sample_idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth_intersection,
                                                                    1000.0 * ground_truth_intersection)
        imae_intersection[sample_idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth_intersection,
                                                                    0.001 * ground_truth_intersection)
        irmse_intersection[sample_idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth_intersection,
                                                                         0.001 * ground_truth_intersection)

    n_valid_points_output = np.mean(n_valid_points_output)
    n_valid_points_intersection = np.mean(n_valid_points_intersection)
    n_valid_points_ground_truth = np.mean(n_valid_points_ground_truth)

    # Compute mean metrics for intersection
    mae_intersection = mae_intersection[~np.isnan(mae_intersection)]
    rmse_intersection = rmse_intersection[~np.isnan(rmse_intersection)]
    imae_intersection = imae_intersection[~np.isnan(imae_intersection)]
    irmse_intersection = irmse_intersection[~np.isnan(irmse_intersection)]

    mae_intersection = np.mean(mae_intersection)
    rmse_intersection = np.mean(rmse_intersection)
    imae_intersection = np.mean(imae_intersection)
    irmse_intersection = np.mean(irmse_intersection)

    # Log to tensorboard
    if summary_writer is not None:
        scalars = {
            'mae_intersection': mae_intersection,
            'rmse_intersection': rmse_intersection,
            'imae_intersection': imae_intersection,
            'irmse_intersection': irmse_intersection,
            'n_valid_points_output': n_valid_points_output,
            'n_valid_points_intersection': n_valid_points_intersection
        }

        model.log_summary(
            summary_writer=summary_writer,
            tag='eval',
            step=step,
            image=torch.cat(image_summaries, dim=0),
            output_depth=torch.cat(output_depth_summaries, dim=0),
            ground_truth_depth=torch.cat(ground_truth_summaries, dim=0),
            scalars=scalars,
            n_display=4)

    # Print validation results to console
    log_evaluation_results(
        title='Validation results',
        mae_intersection=mae_intersection,
        rmse_intersection=rmse_intersection,
        imae_intersection=imae_intersection,
        irmse_intersection=irmse_intersection,
        n_valid_points_output=n_valid_points_output,
        n_valid_points_intersection=n_valid_points_intersection,
        n_valid_points_ground_truth=n_valid_points_ground_truth,
        step=step,
        log_path=log_path)

    n_improve = 0
    if np.round(mae_intersection, 2) <= np.round(best_results['mae_intersection'], 2):
        n_improve = n_improve + 1
    if np.round(rmse_intersection, 2) <= np.round(best_results['rmse_intersection'], 2):
        n_improve = n_improve + 1
    if np.round(imae_intersection, 2) <= np.round(best_results['imae_intersection'], 2):
        n_improve = n_improve + 1
    if np.round(irmse_intersection, 2) <= np.round(best_results['irmse_intersection'], 2):
        n_improve = n_improve + 1
    if np.round(n_valid_points_intersection, 2) >= np.round(best_results['n_valid_points_intersection'], 2):
        n_improve = n_improve + 1

    if n_improve > 3:
        best_results['step'] = step
        best_results['mae_intersection'] = mae_intersection
        best_results['rmse_intersection'] = rmse_intersection
        best_results['imae_intersection'] = imae_intersection
        best_results['irmse_intersection'] = irmse_intersection
        best_results['n_valid_points_output'] = n_valid_points_output
        best_results['n_valid_points_ground_truth'] = n_valid_points_ground_truth
        best_results['n_valid_points_intersection'] = n_valid_points_intersection

    log_evaluation_results(
        title='Best results',
        mae_intersection=best_results['mae_intersection'],
        rmse_intersection=best_results['rmse_intersection'],
        imae_intersection=best_results['imae_intersection'],
        irmse_intersection=best_results['irmse_intersection'],
        n_valid_points_output=best_results['n_valid_points_output'],
        n_valid_points_intersection=best_results['n_valid_points_intersection'],
        n_valid_points_ground_truth=best_results['n_valid_points_ground_truth'],
        step=best_results['step'],
        log_path=log_path)

    return best_results




'''
Helper functions for logging
'''


def log_input_settings(log_path,
                       input_channels_image,
                       input_channels_depth,
                       input_patch_size_image,
                       normalized_image_range):
    log('Input settings:', log_path)
    log('input_channels_image={}  input_channels_depth={}'.format(
        input_channels_image, input_channels_depth),
        log_path)
    log('input_patch_size_image={}'.format(
        input_patch_size_image),
        log_path)
    log('normalized_image_range={}'.format(normalized_image_range),
        log_path)
    log('', log_path)


def log_network_settings(log_path,
                         # Network settings
                         encoder_type,
                         n_filters_encoder_image,
                         n_neurons_encoder_depth,
                         decoder_type,
                         n_filters_decoder,
                         # Weight settings
                         weight_initializer,
                         activation_func,
                         parameters_model=[]):
    # Computer number of parameters
    n_parameter = sum(p.numel() for p in parameters_model)

    n_parameter_text = 'n_parameter={}'.format(n_parameter)
    n_parameter_vars = []

    log('Network settings:', log_path)
    log('encoder_type={}'.format(encoder_type),
        log_path)
    log('n_filters_encoder_image={}'.format(n_filters_encoder_image),
        log_path)
    log('n_neurons_encoder_depth={}'.format(n_neurons_encoder_depth),
        log_path)
    log('decoder_type={}'.format(decoder_type),
        log_path)
    log('n_filters_decoder={}'.format(
        n_filters_decoder),
        log_path)
    log('', log_path)

    log('Weight settings:', log_path)
    log(n_parameter_text.format(*n_parameter_vars),
        log_path)
    log('weight_initializer={}  activation_func={}'.format(
        weight_initializer, activation_func),
        log_path)
    log('', log_path)


def log_training_settings(log_path,
                          # Training settings
                          batch_size,
                          n_train_sample,
                          n_train_step,
                          learning_rates,
                          learning_schedule,
                          # Augmentation settings
                          augmentation_probabilities,
                          augmentation_schedule,
                          augmentation_random_brightness,
                          augmentation_random_contrast,
                          augmentation_random_saturation,
                          augmentation_random_noise_type,
                          augmentation_random_noise_spread,
                          augmentation_random_flip_type):
    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}  batch_size={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step, batch_size),
        log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // batch_size), le * (n_train_sample // batch_size), v)
                  for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('', log_path)

    log('Augmentation settings:', log_path)
    log('augmentation_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // batch_size), le * (n_train_sample // batch_size), v)
                  for ls, le, v in
                  zip([0] + augmentation_schedule[:-1], augmentation_schedule, augmentation_probabilities)),
        log_path)
    log('augmentation_random_brightness={}'.format(augmentation_random_brightness),
        log_path)
    log('augmentation_random_contrast={}'.format(augmentation_random_contrast),
        log_path)
    log('augmentation_random_saturation={}'.format(augmentation_random_saturation),
        log_path)
    log('augmentation_random_noise_type={}  augmentation_random_noise_spread={}'.format(
        augmentation_random_noise_type, augmentation_random_noise_spread),
        log_path)
    log('augmentation_random_flip_type={}'.format(augmentation_random_flip_type),
        log_path)

    log('', log_path)


def log_loss_func_settings(log_path,
                           # Loss function settings
                           w_weight_decay,
                           w_positive_class,
                           max_distance_correspondence,
                           set_invalid_to_negative_class):
    log('Loss function settings:', log_path)
    log('w_positve_class={:.1e}  w_weight_decay={:.1e}'.format(
        w_positive_class, w_weight_decay),
        log_path)
    log('max_distance_correspondence={}  set_invalid_to_negative_class={}'.format(
        max_distance_correspondence, set_invalid_to_negative_class),
        log_path)
    log('', log_path)


def log_evaluation_settings(log_path,
                            min_evaluate_depth,
                            max_evaluate_depth):
    log('Evaluation settings:', log_path)
    log('min_evaluate_depth={:.2f}  max_evaluate_depth={:.2f}'.format(
        min_evaluate_depth, max_evaluate_depth),
        log_path)
    log('', log_path)


def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_dirpath=None,
                        n_step_per_checkpoint=None,
                        summary_event_path=None,
                        n_step_per_summary=None,
                        start_step_validation=None,
                        restore_path=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):
    log('Checkpoint settings:', log_path)

    if checkpoint_dirpath is not None:
        log('checkpoint_path={}'.format(checkpoint_dirpath), log_path)

        if n_step_per_checkpoint is not None:
            log('n_step_per_checkpoint={}'.format(n_step_per_checkpoint), log_path)

        if start_step_validation is not None:
            log('start_step_validation={}'.format(start_step_validation), log_path)

        log('', log_path)

        summary_settings_text = ''
        summary_settings_vars = []

    if summary_event_path is not None:
        log('Tensorboard settings:', log_path)
        log('event_path={}'.format(summary_event_path), log_path)

    if n_step_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_step_per_summary={}'
        summary_settings_vars.append(n_step_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if len(summary_settings_text) > 0:
        log(summary_settings_text.format(*summary_settings_vars), log_path)

    if restore_path is not None and restore_path != '':
        log('restore_path={}'.format(restore_path),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)


def log_evaluation_results(title,
                           mae_intersection,
                           rmse_intersection,
                           imae_intersection,
                           irmse_intersection,
                           n_valid_points_output,
                           n_valid_points_intersection,
                           n_valid_points_ground_truth,
                           step=-1,
                           log_path=None):
    # Print evalulation results to console
    log(title + ':', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>14}  {:>14}  {:>14}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE', '# Output', '# Intersection', '# Ground truth'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:14.3f}  {:14.3f}  {:14.3f}'.format(
        step,
        mae_intersection,
        rmse_intersection,
        imae_intersection,
        irmse_intersection,
        n_valid_points_output,
        n_valid_points_intersection,
        n_valid_points_ground_truth),
        log_path)

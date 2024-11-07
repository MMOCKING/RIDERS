import os, sys
import torch.utils.data
import numpy as np

sys.path.insert(0, 'src')
from data import data_utils, datasets
from utils import eval_utils
from utils.log_utils import log
from rcnet_main import forward_output, log_network_settings, log_evaluation_results
from rcnet_model import RCNetModel
from rcnet_transforms import Transforms
import time


'''
Set up input arguments
'''

root = '/media/lh/lh1/dataset/NTU'
scenes = ['garden_2022-05-13_0', 'garden_2022-05-13_1']
# scenes = sorted(os.listdir(root))
# scenes = [scene for scene in scenes if scene.startswith('loop')]
# exclude zip
# scenes = [scene for scene in scenes if not scene.endswith('zip')]
# scenes = ['loop2_2022-06-03_1', 'loop3_2022-06-03_0']
# scenes = ['loop1_2022-06-03_0', 'loop1_2022-06-03_1', 'loop1_2022-06-03_2',
#                   'loop1_2022-06-03_3', 'loop1_2022-06-03_4',
#                   'loop2_2022-06-03_0', 'loop2_2022-06-03_2', 'loop2_2022-06-03_3',
#                   'loop3_2022-06-03_1', 'loop3_2022-06-03_2',
#                   ]
image_file = 'thermal_undistort'
radar_file = 'radar_png'
gt_file = 'lidar_png_int'
response_thr = 0.4
output_path = '/media/lh/lh1/dataset/NTU/output/rcnet_' + str(response_thr)

restore_path = '/media/lh/lh1/dataset/NTU/log/rcnet/best/model-176000.pth'
patch_size = [150, 50]
input_channels_image = 3
input_channels_depth = 3
normalized_image_range = [0, 1]

encoder_type = ['rcnet', 'batch_norm']
n_filters_encoder_image = [32, 64, 128, 128, 128]
n_neurons_encoder_depth = [32, 64, 128, 128, 128]
decoder_type = ['multiscale', 'batch_norm']
n_filters_decoder = [256, 128, 64, 32, 16]

weight_initializer = 'kaiming_uniform'
activation_func = 'leaky_relu'

min_evaluate_depth = 0.0
max_evaluate_depth = 100.0


'''
Main function
'''
if __name__ == '__main__':

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    Read input paths
    '''
    # Load training paths
    run_image_paths = []
    run_radar_paths = []
    run_ground_truth_paths = []
    print('scenes:', scenes)
    for scene in scenes:
        scene_root = os.path.join(root, scene)
        image_paths = sorted(os.listdir(os.path.join(scene_root, image_file)))
        radar_paths = sorted(os.listdir(os.path.join(scene_root, radar_file)))
        gt_paths = sorted(os.listdir(os.path.join(scene_root, gt_file)))
        run_image_paths += [os.path.join(scene_root, image_file, image_path) for image_path in image_paths]
        run_radar_paths += [os.path.join(scene_root, radar_file, radar_path) for radar_path in radar_paths]
        run_ground_truth_paths += [os.path.join(scene_root, gt_file, gt_path) for gt_path in gt_paths]

    print('length of train image:', len(run_image_paths))
    print('length of train radar:', len(run_radar_paths))
    print('length of train gt:', len(run_ground_truth_paths))

    n_run_sample = len(run_image_paths)

    assert n_run_sample == len(run_radar_paths)
    assert n_run_sample == len(run_ground_truth_paths)

    '''
    Set up inputs and outputs
    '''
    run_depth_predicted_paths = []
    run_response_predicted_paths = []
    run_depth_predicted_color_paths = []


    inputs_outputs = [
         [
            'training',
            run_image_paths,
            run_radar_paths,
            run_ground_truth_paths,
            run_depth_predicted_paths,
            run_depth_predicted_color_paths,
            run_response_predicted_paths,
        ]
    ]

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

    rcnet_model.eval()
    rcnet_model.to(device)
    rcnet_model.data_parallel()

    parameters_rcnet_model = rcnet_model.parameters()

    step, _ = rcnet_model.restore_model(restore_path)

    log('Restoring checkpoint from: \n{}\n'.format(restore_path))

    log_network_settings(
        log_path=None,
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

    '''
    Process each set of input and outputs
    '''
    for paths in inputs_outputs:
        # Unpack inputs and outputs
        tag, \
        image_paths, \
        radar_paths, \
        ground_truth_paths, \
        depth_predicted_paths, \
        depth_predicted_color_paths, \
        response_predicted_paths,  = paths

        # Create output paths for depth and response
        for radar_path in radar_paths:

            # Create path and store
            file_id = os.path.basename(radar_path).split('.')[0]
            save_scene = radar_path.split('/')[-3]
            depth_predicted_path = os.path.join(output_path, save_scene, 'depth_predicted', file_id+'.png')
            depth_predicted_paths.append(depth_predicted_path)

            depth_predicted_color_path = os.path.join(output_path, save_scene, 'depth_predicted_colors', file_id+'.png')
            depth_predicted_color_paths.append(depth_predicted_color_path)

            response_predicted_path = os.path.join(output_path, save_scene, 'response_predicted', file_id+'.png')
            response_predicted_paths.append(response_predicted_path)

        # Create directories
        depth_predicted_dirpaths = np.unique([os.path.dirname(path) for path in depth_predicted_paths])
        depth_predicted_color_dirpaths = np.unique([os.path.dirname(path) for path in depth_predicted_color_paths])
        response_predicted_dirpaths = np.unique([os.path.dirname(path) for path in response_predicted_paths])
        for dirpaths in [depth_predicted_dirpaths, depth_predicted_color_dirpaths, response_predicted_dirpaths]:
            for dirpath in dirpaths:
                os.makedirs(dirpath, exist_ok=True)

        # Set up dataloader
        dataloader = torch.utils.data.DataLoader(
            datasets.RCNetInferenceDataset(
                image_paths=image_paths,
                radar_paths=radar_paths,
                ground_truth_paths=ground_truth_paths),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

        transforms = Transforms(
            normalized_image_range=normalized_image_range)

        n_sample = len(dataloader)

        time_start = time.time()
        print('Processing {} samples...'.format(n_sample))

        # Iterate through data loader
        for sample_idx, data in enumerate(dataloader):
            with torch.no_grad():
                data = [
                    datum.to(device) for datum in data
                ]

                image, radar_points, ground_truth = data
                bounding_boxes_list = []

                pad_size_x = patch_size[1] // 2
                pad_size_y = patch_size[0] // 2
                radar_points = radar_points.squeeze(dim=0)

                if radar_points.ndim == 1:
                    # Expand to 1 x 3
                    radar_points = np.expand_dims(radar_points, axis=0)

                # get the shifted radar points after padding, padding func is in the forward()
                for radar_point_idx in range(0, radar_points.shape[0]):
                    # Set radar point to the center of the patch
                    radar_points[radar_point_idx, 0] = radar_points[radar_point_idx, 0] + pad_size_x
                    radar_points[radar_point_idx, 1] = radar_points[radar_point_idx, 1] + pad_size_y
                    bounding_box = torch.zeros(4)
                    bounding_box[0] = radar_points[radar_point_idx, 0] - pad_size_x
                    bounding_box[1] = radar_points[radar_point_idx, 1] - pad_size_y
                    bounding_box[2] = radar_points[radar_point_idx, 0] + pad_size_x
                    bounding_box[3] = radar_points[radar_point_idx, 1] + pad_size_y
                    bounding_boxes_list.append(bounding_box)

                bounding_boxes_list = [torch.stack(bounding_boxes_list, dim=0)]

                [image], [radar_points], [bounding_boxes_list] = transforms.transform(
                    images_arr=[image],
                    points_arr=[radar_points],
                    bounding_boxes_arr=[bounding_boxes_list],
                    random_transform_probability=0.0)

                output_depth, output_response = forward_output(
                    model=rcnet_model,
                    image=image,
                    radar_points=radar_points,
                    bounding_boxes_list=bounding_boxes_list,
                    device=device,
                    response_thr=response_thr)

                output_depth = np.squeeze(output_depth.cpu().numpy())
                # output_response = np.squeeze(output_response.cpu().numpy())
                thr = response_thr

                while np.sum(output_depth) == 0:
                    print('Output depth is all zeros with thr: ', thr)
                    thr = thr - 0.05
                    output_depth, output_response = forward_output(
                        model=rcnet_model,
                        image=image,
                        radar_points=radar_points,
                        bounding_boxes_list=bounding_boxes_list,
                        response_thr=thr,
                        device=device)
                    output_depth = np.squeeze(output_depth.cpu().numpy())
                    # output_response = np.squeeze(output_response.cpu().numpy())

            '''
            Save outputs
            '''
            data_utils.save_depth(output_depth, depth_predicted_paths[sample_idx])
            data_utils.save_color_depth(output_depth, depth_predicted_color_paths[sample_idx])
            # data_utils.save_response(output_response, response_predicted_paths[sample_idx])

            print('save to {}'.format(depth_predicted_paths[sample_idx]))


        time_end = time.time()
        print('Time taken: {:.4f} seconds'.format(time_end - time_start))
        print('average time per sample: {:.4f} seconds'.format((time_end - time_start)/n_run_sample))





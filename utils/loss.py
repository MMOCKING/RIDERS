import numpy as np
import torch, torchvision
import torch.nn.functional as F

def compute_loss(image,
                 output_depth,
                 gt_interp,
                 gt_sparse,
                 loss_func,
                 w_smoothness,
                 sobel_filter_size,
                 validity_map_loss_smoothness,
                 w_lidar_loss,
                 w_edge,
                 invalid_map_gt,
                 w_unsupervised,
                 ):

    loss_supervised = 0.0
    loss_lidar = 0.0
    loss_smoothness = 0.0
    loss_edge = 0.0
    loss_unsupervised = 0.0


    if w_lidar_loss > 0.0:
        # Mask out ground truth where lidar is available to avoid double counting
        mask_sparse = torch.where(
            gt_sparse > 0.0,
            torch.zeros_like(gt_sparse),
            torch.ones_like(gt_sparse))

        gt_interp = gt_interp * mask_sparse

    validity_map_ground_truth = gt_interp > 0
    validity_map_lidar = gt_sparse > 0

    if not isinstance(output_depth, list):
        output_depth = [output_depth]

    for scale, output in enumerate(output_depth):

        output_height, output_width = output.shape[-2:]
        target_height, target_width = gt_interp.shape[-2:]

        if output_height > target_height and output_width > target_width:
            output = torch.nn.functional.interpolate(
                output,
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=True)

        w_scale = 1.0 / (2 ** (len(output_depth) - scale - 1))

        if loss_func == 'l1':
            loss_supervised = loss_supervised + w_scale * l1_loss(
                output[validity_map_ground_truth],
                gt_interp[validity_map_ground_truth])

            if w_lidar_loss > 0.0:
                loss_lidar = loss_lidar + w_scale * l1_loss(
                    output[validity_map_lidar],
                    gt_sparse[validity_map_lidar])

            if w_unsupervised > 0.0:
                output_median = torch.median(output[invalid_map_gt])
                image_median = torch.median(image[invalid_map_gt])
                loss_unsupervised = loss_unsupervised + w_scale * l1_loss(
                    output[invalid_map_gt]/output_median,
                    image[invalid_map_gt]/image_median)


        elif loss_func == 'l2':
            loss_supervised = loss_supervised + w_scale * l2_loss(
                output[validity_map_ground_truth],
                gt_interp[validity_map_ground_truth])

            if w_lidar_loss > 0.0:
                loss_lidar = loss_lidar + w_scale * l2_loss(
                    output[validity_map_lidar],
                    gt_sparse[validity_map_lidar])

            if w_unsupervised > 0.0:
                output_median = torch.median(output[invalid_map_gt])
                image_median = torch.median(image[invalid_map_gt])
                loss_unsupervised = loss_unsupervised + w_scale * l2_loss(
                    output[invalid_map_gt] / output_median,
                    image[invalid_map_gt] / image_median)


        elif loss_func == 'smoothl1':
            loss_supervised = loss_supervised + w_scale * smooth_l1_loss(
                output[validity_map_ground_truth],
                gt_interp[validity_map_ground_truth])

            if w_lidar_loss > 0.0:
                loss_lidar = loss_lidar + w_scale * smooth_l1_loss(
                    output[validity_map_lidar],
                    gt_sparse[validity_map_lidar])

            if w_unsupervised > 0.0:
                output_median = torch.median(output[invalid_map_gt])
                image_median = torch.median(image[invalid_map_gt])
                loss_unsupervised = loss_unsupervised + w_scale * smooth_l1_loss(
                    output[invalid_map_gt] / output_median,
                    image[invalid_map_gt] / image_median)

        else:
            raise ValueError('No such loss: {}'.format(loss_func))

        if w_smoothness > 0.0 or w_edge > 0.0:

            sobel_filter_size = [1, 1, sobel_filter_size, sobel_filter_size]

            temp_smooth, temp_edge = sobel_smoothness_loss_func(predict=output,
                                                                image=image,
                                                                weights=validity_map_loss_smoothness,
                                                                filter_size=sobel_filter_size)
            loss_smoothness = loss_smoothness + w_scale * temp_smooth
            loss_edge = loss_edge + w_scale * temp_edge

    loss = loss_supervised + w_lidar_loss * loss_lidar + \
           w_smoothness * loss_smoothness + w_edge * loss_edge + \
           w_unsupervised * loss_unsupervised

    loss_info = {
        'loss': loss,
        'loss_supervised': loss_supervised,
        'loss_lidar': loss_lidar,
        'loss_smoothness': loss_smoothness,
        'loss_edge': loss_edge,
        'loss_unsupervised': loss_unsupervised,
    }

    return loss, loss_info



def smooth_l1_loss(src, tgt):
    '''
    Computes smooth_l1 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
    Returns:
        float : mean smooth l1 loss across batch
    '''

    return torch.nn.functional.smooth_l1_loss(src, tgt, reduction='mean')


def l1_loss(src, tgt):
    '''
    Computes l1 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
    Returns:
        float : mean l1 loss across batch
    '''

    return torch.nn.functional.l1_loss(src, tgt, reduction='mean')


def l2_loss(src, tgt):
    '''
    Computes l2 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
    Returns:
        float : mean l2 loss across batch
    '''

    return torch.nn.functional.mse_loss(src, tgt, reduction='mean')


def sobel_smoothness_loss_func(predict, image, weights,
                               filter_size=[1, 1, 7, 7]):
    '''
    Computes the local smoothness loss using sobel filter

    Arg(s):
        predict : tensor
            N x 1 x H x W predictions
        image : tensor
            N x 3 x H x W RGB image
        w : tensor
            N x 1 x H x W weights
    Returns:
        tensor : smoothness loss
    '''

    device = predict.device

    if image.shape[1] == 3:
        image = image[:, 0, :, :] * 0.299 + image[:, 1, :, :] * 0.587 + image[:, 2, :, :] * 0.114
        image = torch.unsqueeze(image, 1)

    image_pad = torch.nn.functional.pad(image,
                                    (filter_size[-1]//2, filter_size[-1]//2, filter_size[-2]//2, filter_size[-2]//2),
                                    mode='replicate')
    image_smooth = torch.nn.functional.pad(image, (1, 1, 1, 1), mode='replicate')

    predict_pad = torch.nn.functional.pad(predict,
                                      (filter_size[-1]//2, filter_size[-1]//2, filter_size[-2]//2, filter_size[-2]//2),
                                      mode='replicate')

    gx, gy = sobel_filter(filter_size)
    gx_smooth, gy_smooth = sobel_filter([1, 1, 3, 3])
    gx = gx.to(device)
    gy = gy.to(device)
    gx_smooth = gx_smooth.to(device)
    gy_smooth = gy_smooth.to(device)

    image_dy = torch.nn.functional.conv2d(image_pad, gy)
    image_dx = torch.nn.functional.conv2d(image_pad, gx)

    image_smooth_dy = torch.nn.functional.conv2d(image_smooth, gy_smooth)
    image_smooth_dx = torch.nn.functional.conv2d(image_smooth, gx_smooth)

    predict_dy = torch.nn.functional.conv2d(predict_pad, gy)
    predict_dx = torch.nn.functional.conv2d(predict_pad, gx)

    # Create edge awareness weights
    weights_x = torch.exp(-torch.mean(torch.abs(image_smooth_dy), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_smooth_dx), dim=1, keepdim=True))

    smoothness_x = torch.mean(weights * weights_x * torch.abs(predict_dx))
    smoothness_y = torch.mean(weights * weights_y * torch.abs(predict_dy))

    smoothness_loss = (smoothness_x + smoothness_y) / float(filter_size[-1] * filter_size[-2])

    # Calculate edge-matching loss
    loss_dx = torch.mean(weights * torch.abs(abs(predict_dx) - abs(image_dx)))
    loss_dy = torch.mean(weights * torch.abs(abs(predict_dy) - abs(image_dy)))
    # loss_dx = torch.mean(weights * torch.abs(predict_dx - image_dx))
    # loss_dy = torch.mean(weights * torch.abs(predict_dy - image_dy))

    # Total edge-matching loss
    edge_matching_loss = (loss_dx + loss_dy) / float(filter_size[-1] * filter_size[-2])

    return  smoothness_loss, edge_matching_loss




'''
Helper functions for constructing loss functions
'''
def sobel_filter(filter_size=[1, 1, 3, 3]):
    Gx = torch.ones(filter_size)
    Gy = torch.ones(filter_size)

    Gx[:, :, :, filter_size[-1] // 2] = 0
    Gx[:, :, (filter_size[-2] // 2), filter_size[-1] // 2 - 1] = 2
    Gx[:, :, (filter_size[-2] // 2), filter_size[-1] // 2 + 1] = 2
    Gx[:, :, :, filter_size[-1] // 2:] = -1*Gx[:, :, :, filter_size[-1] // 2:]

    Gy[:, :, filter_size[-2] // 2, :] = 0
    Gy[:, :, filter_size[-2] // 2 - 1, filter_size[-1] // 2] = 2
    Gy[:, :, filter_size[-2] // 2 + 1, filter_size[-1] // 2] = 2
    Gy[:, :, filter_size[-2] // 2+1:, :] = -1*Gy[:, :, filter_size[-2] // 2+1:, :]

    return Gx, Gy

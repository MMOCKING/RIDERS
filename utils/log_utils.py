import os
import torch, torchvision
import numpy as np
from matplotlib import pyplot as plt


def log(s, filepath=None, to_console=True):
    '''
    Logs a string to either file or console
    Arg(s):
        s : str
            string to log
        filepath
            output filepath for logging
        to_console : bool
            log to console
    '''

    if to_console:
        print(s)

    if filepath is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, 'w+') as o:
                o.write(s + '\n')
        else:
            with open(filepath, 'a+') as o:
                o.write(s + '\n')


def colorize(T, colormap='magma', return_numpy=False):
    '''
    Colorizes a 1-channel tensor with matplotlib colormaps
    Arg(s):
        T : torch.Tensor[float32]
            1-channel tensor
        colormap : str
            matplotlib colormap
    '''

    cm = plt.cm.get_cmap(colormap)
    shape = T.shape

    # Convert to numpy array and transpose
    if shape[0] > 1:
        T = np.squeeze(np.transpose(T.cpu().numpy(), (0, 2, 3, 1)))
    else:
        T = np.squeeze(np.transpose(T.cpu().numpy(), (0, 2, 3, 1)), axis=-1)

    # Colorize using colormap
    color = np.concatenate([
        np.expand_dims(cm(T[n, ...])[..., 0:3], 0) for n in range(T.shape[0])],
        axis=0)

    if return_numpy:
        return color
    else:
        # Transpose back to torch format
        color = np.transpose(color, (0, 3, 1, 2))

        # Convert back to tensor
        return torch.from_numpy(color.astype(np.float32))



def log_params(log_path, params_dict):
    with open(log_path, 'w') as log_file:
        for param_name, param_value in params_dict.items():
            log_file.write(f"{param_name}: {param_value}\n")
            
            
            
def log_evaluation_results(title,
                           mae,
                           rmse,
                           imae,
                           irmse,
                           abs_rel=None,
                           sq_rel=None,
                           delta1=None,
                           step=-1,
                           log_path=None):
    # Print evalulation results to console
    log(title + ':', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE', 'Abs_Rel', 'Sq_Rel', 'Delta1'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step,
        mae,
        rmse,
        imae,
        irmse,
        abs_rel,
        sq_rel,
        delta1),
        log_path)


    
def log_summary(summary_writer,
                tag,
                step,
                max_predict_depth,
                image=None,
                input_depth=None,
                input_mono=None,
                output_depth=None,
                ground_truth=None,
                scalars={},
                n_display=4):
    '''
    Logs summary to Tensorboard

    Arg(s):
        summary_writer : SummaryWriter
            Tensorboard summary writer
        tag : str
            tag that prefixes names to log
        step : int
            current step in training
        image : torch.Tensor[float32]
            N x 3 x H x W image
        input_depth : torch.Tensor[float32]
            N x 1 x H x W input depth
        output_depth : torch.Tensor[float32]
            N x 1 x H x W output depth
        ground_truth : torch.Tensor[float32]
            N x 1 x H x W ground truth depth
        scalars : dict[str, float]
            dictionary of scalars to log
        n_display : int
            number of images to display
    '''

    with torch.no_grad():

        display_summary_image = []
        display_summary_depth = []

        display_summary_image_text = tag
        display_summary_depth_text = tag

        if image is not None:
            image_summary = image[0:n_display, ...]

            display_summary_image_text += '_image'
            display_summary_depth_text += '_image'

            # Add to list of images to log
            display_summary_image.append(
                torch.cat([
                    image_summary.cpu(),
                    torch.zeros_like(image_summary, device=torch.device('cpu'))],
                    dim=-1))

            display_summary_depth.append(display_summary_image[-1])

        if output_depth is not None:
            output_depth_summary = output_depth[0:n_display, ...]

            display_summary_depth_text += '_output_depth'

            # Add to list of images to log
            n_batch, _, n_height, n_width = output_depth_summary.shape

            display_summary_depth.append(
                torch.cat([
                    colorize(
                        (output_depth_summary / max_predict_depth).cpu(),
                        colormap='viridis'),
                    torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                    dim=3))

            # Log distribution of output depth
            summary_writer.add_histogram(tag + '_output_depth_distro', output_depth, global_step=step)

        if output_depth is not None and input_depth is not None:
            input_depth_summary = input_depth[0:n_display, ...]

            display_summary_depth_text += '_input_depth-error'

            # Compute output error w.r.t. input depth
            input_depth_error_summary = \
                torch.abs(output_depth_summary - input_depth_summary)

            input_depth_error_summary = torch.where(
                input_depth_summary > 0.0,
                input_depth_error_summary / (input_depth_summary + 1e-8),
                input_depth_summary)

            # Add to list of images to log
            input_depth_summary = colorize(
                (input_depth_summary / max_predict_depth).cpu(),
                colormap='viridis')
            input_depth_error_summary = colorize(
                (input_depth_error_summary / 0.05).cpu(),
                colormap='inferno')

            display_summary_depth.append(
                torch.cat([
                    input_depth_summary,
                    input_depth_error_summary],
                    dim=3))

            # Log distribution of input depth
            summary_writer.add_histogram(tag + '_input_depth_distro', input_depth, global_step=step)


        if output_depth is not None and input_mono is not None:
            mono_summary = input_mono[0:n_display, ...]

            display_summary_depth_text += '_mono'

            # Add to list of images to log
            mono_summary = colorize(
                mono_summary.cpu(),
                colormap='viridis')

            display_summary_depth.append(
                torch.cat([
                    mono_summary,
                    torch.zeros_like(mono_summary)],
                    dim=3))

            # Log distribution of input depth
            # summary_writer.add_histogram(tag + '_mono_distro', input_depth, global_step=step)

        if output_depth is not None and ground_truth is not None:
            ground_truth = ground_truth[0:n_display, ...]
            ground_truth = torch.unsqueeze(ground_truth[:, 0, :, :], dim=1)

            ground_truth_summary = ground_truth[0:n_display]
            validity_map_summary = torch.where(
                ground_truth > 0,
                torch.ones_like(ground_truth),
                torch.zeros_like(ground_truth))

            display_summary_depth_text += '_ground_truth-error'

            # Compute output error w.r.t. ground truth
            ground_truth_error_summary = \
                torch.abs(output_depth_summary - ground_truth_summary)

            ground_truth_error_summary = torch.where(
                validity_map_summary == 1.0,
                (ground_truth_error_summary + 1e-8) / (ground_truth_summary + 1e-8),
                validity_map_summary)

            # Add to list of images to log
            ground_truth_summary = colorize(
                (ground_truth_summary / max_predict_depth).cpu(),
                colormap='viridis')
            ground_truth_error_summary = colorize(
                (ground_truth_error_summary / 0.05).cpu(),
                colormap='inferno')

            display_summary_depth.append(
                torch.cat([
                    ground_truth_summary,
                    ground_truth_error_summary],
                    dim=3))

            # Log distribution of ground truth
            summary_writer.add_histogram(tag + '_ground_truth_distro', ground_truth, global_step=step)

        # Log scalars to tensorboard
        for (name, value) in scalars.items():
            summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

        # Log image summaries to tensorboard
        if len(display_summary_image) > 1:
            display_summary_image = torch.cat(display_summary_image, dim=2)

            summary_writer.add_image(
                display_summary_image_text,
                torchvision.utils.make_grid(display_summary_image, nrow=n_display),
                global_step=step)

        if len(display_summary_depth) > 1:
            display_summary_depth = torch.cat(display_summary_depth, dim=2)

            summary_writer.add_image(
                display_summary_depth_text,
                torchvision.utils.make_grid(display_summary_depth, nrow=n_display),
                global_step=step)
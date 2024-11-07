import numpy as np
import cv2
import math
import torch
import torchvision.transforms as transforms

import torchvision.transforms.functional as functional
from modules.midas.utils import normalize_unit_range
import modules.midas.normalization as normalization
import random
from PIL import Image

class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        for item in sample.keys():
            interpolation_method = self.__image_interpolation_method
            sample[item] = cv2.resize(
                sample[item],
                (width, height),
                interpolation=interpolation_method,
            )

        if self.__resize_target:
            if "radar" in sample:
                sample["radar"] = cv2.resize(
                    sample["radar"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

            if "gt" in sample:
                sample["gt"] = cv2.resize(
                    sample["gt"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

            if "sparse_gt" in sample:
                sample["sparse_gt"] = cv2.resize(
                    sample["sparse_gt"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )
            if "mono_pred" in sample:
                sample["mono_pred"] = cv2.resize(
                    sample["mono_pred"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

        return sample



class NormalizeIntermediate(object):
    """Normalize intermediate data by given mean and std.
    """

    def __init__(self, mean, std):

        self.__int_depth_mean = mean["int_depth"]
        self.__int_depth_std = std["int_depth"]

        self.__int_scales_mean = mean["int_scales"]
        self.__int_scales_std = std["int_scales"]

    def __call__(self, sample):

        if "int_depth" in sample and sample["int_depth"] is not None:
            sample["int_depth"] = (sample["int_depth"] - self.__int_depth_mean) / self.__int_depth_std

        if "int_scales" in sample and sample["int_scales"] is not None:
            sample["int_scales"] = (sample["int_scales"] - self.__int_scales_mean) / self.__int_scales_std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):

        for item in sample.keys():

            if sample[item] is None:
                pass
            elif item == "image":
                image = np.transpose(sample["image"], (2, 0, 1))
                sample["image"] = np.ascontiguousarray(image).astype(np.float32)
            else:
                array = sample[item].astype(np.float32)
                array = np.expand_dims(array, axis=0) # add channel dim
                sample[item] = np.ascontiguousarray(array)

        return sample


class Tensorize(object):
    """Convert sample to tensor.
    """

    def __init__(self):
        pass

    def __call__(self, sample):

        for item in sample.keys():

            if sample[item] is None:
                pass
            else:
                # before tensorizing, verify that data is clean
                assert not np.any(np.isnan(sample[item])) 
                sample[item] = torch.Tensor(sample[item])

        return sample


class Random_adjust_brightness(object):

    def __init__(self, brightness_range):
        self.__brightness_range = brightness_range

    def __call__(self, sample):
        if self.__brightness_range is not None and random.random() < 0.5:
            # print('brightness')
            image = sample["image"]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            random_brightness = random.uniform(
                self.__brightness_range[0], self.__brightness_range[1]
            )
            hsv[:, :, 2] = hsv[:, :, 2] * random_brightness

            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            sample["image"] = image

        return sample


class Random_adjust_contrast(object):

    def __init__(self, contrast_range):
        self.__contrast_range = contrast_range

    def __call__(self, sample):
        if self.__contrast_range is not None and random.random() < 0.5:
            # print('contrast')
            image = sample["image"]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            random_contrast = random.uniform(
                self.__contrast_range[0], self.__contrast_range[1]
            )
            hsv[:, :, 1] = hsv[:, :, 1] * random_contrast

            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            sample["image"] = image

        return sample


class Random_adjust_saturation(object):

        def __init__(self, saturation_range):
            self.__saturation_range = saturation_range

        def __call__(self, sample):
            if self.__saturation_range is not None and random.random() < 0.5:
                # print('saturation')
                image = sample["image"]
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                random_saturation = random.uniform(
                    self.__saturation_range[0], self.__saturation_range[1]
                )
                hsv[:, :, 1] = hsv[:, :, 1] * random_saturation

                image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                sample["image"] = image

            return sample







def get_transforms(net_w, net_h,
                   depth_predictor = 'midas_small',
                   sparsifier = 'void',
                   nsamples = '500',
                   random_brightness = None,
                   random_contrast = None,
                   random_saturation = None,):

    sml_model_transform_steps = [
        Resize(
            width=net_w,
            height=net_h,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_NEAREST,
        ),
        Random_adjust_brightness(random_brightness),
        Random_adjust_contrast(random_contrast),
        Random_adjust_saturation(random_saturation),
        NormalizeIntermediate(
            # mean=normalization.VOID_INTERMEDIATE[depth_predictor][f"{sparsifier}_{nsamples}"]["mean"],
            # std=normalization.VOID_INTERMEDIATE[depth_predictor][f"{sparsifier}_{nsamples}"]["std"],
            mean={"int_depth": 0.729, "int_scales": 0.404},
            std={"int_depth": 0.210, "int_scales": 0.117},
        ),
        PrepareForNet(),
        Tensorize(),
    ]

    return transforms.Compose(sml_model_transform_steps)


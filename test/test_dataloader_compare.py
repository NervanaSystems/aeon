# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import itertools
import math
import os

import cv2
import numpy as np
from aeon import DataLoader

MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]


def random_crop_cv(img, scale, aspect_ratio, origin):
    """Crops image with specified 'aspect_ratio' at specified 'scale'.
    If 'scale' is too big to fit the 'aspect_ratio', the 'scale' will be reduced to fit.
    Crop starts at 'origin' if it is provided, otherwise its random.
    Returns the cropped image.
    """
    ar_sqr = math.sqrt(np.random.uniform(*aspect_ratio))
    width = 1. * ar_sqr
    height = 1. / ar_sqr

    bound = min((float(img.shape[0]) / img.shape[1]) / (height**2),
                (float(img.shape[1]) / img.shape[0]) / (width**2))

    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.shape[0] * img.shape[1] * np.random.uniform(
        scale_min, scale_max)
    target_size = math.sqrt(target_area)
    width = int(round(target_size * width))
    height = int(round(target_size * height))
    i = origin[1] if origin != None else np.random.randint(
        0, img.shape[0] - height + 1)
    j = origin[0] if origin != None else np.random.randint(
        0, img.shape[1] - width + 1)

    img = img[i:i + height, j:j + width, :]
    return img


def rotate_image_cv(img, angle_range):
    """Rotates image by angle specified in 'angle_range'.
    Returns the rotated image.
    """
    (height, width) = img.shape[:2]
    center = (width / 2, height / 2)
    angle = angle_range[0] if angle_range[0] == angle_range[
        1] else np.random.randint(angle_range[0], angle_range[1])
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot, (width, height))
    return rotated


def transform_image_cv(img_path, output_size, angle, crop, interpolation_type,
                       flip, bgr_to_rgb, standardize, channel_major,
                       data_type):
    """Reads image and applies transformations using OpenCV.
    Returns the transformed image.
    """
    assert os.path.isfile(img_path)
    img = cv2.imread(img_path)
    if not angle[0] == angle[1] == 0:
        img = rotate_image_cv(img, angle)
    if crop["enable"]:
        img = random_crop_cv(img, crop["scale"], crop["ratio"], crop["origin"])
    img = cv2.resize(img, (output_size, output_size),
                     interpolation=interpolation_type)
    if flip:
        img = cv2.flip(img, 1)
    if bgr_to_rgb:
        img = img[:, :, ::-1]

    img = np.array(img).astype(data_type)

    if channel_major:
        img = img.transpose((2, 0, 1))

    if standardize:
        if channel_major:
            img /= 255
            img -= np.array(MEAN).reshape((3, 1, 1))
            img /= np.array(STDDEV).reshape((3, 1, 1))
        else:
            img /= 255
            img -= np.array(MEAN).reshape((1, 1, 3))
            img /= np.array(STDDEV).reshape((1, 1, 3))
    return img


def get_reference_images_cv(manifest_name, count, output_size, angle, crop,
                            interpolation_type, flip, bgr_to_rgb, standardize,
                            channel_major, data_type):
    """Reads manifest file and transforms all images listed inside using OpenCV reference code.
    Returns the transformed images.
    """
    with open(manifest_name) as manifest_file:
        lines = [line.strip() for line in manifest_file][1::]
    flist = []
    while len(flist) < count:
        for line in lines:
            img_path, _ = line.split()
            flist.append(
                transform_image_cv(img_path, output_size, angle, crop,
                                   interpolation_type, flip, bgr_to_rgb,
                                   standardize, channel_major, data_type))
            if len(flist) >= count:
                break
    return np.array(flist)


def get_aeon_images(manifest_name, count, output_size, angle, crop,
                    interpolation_type, flip, bgr_to_rgb, standardize,
                    channel_major, data_type):
    """Transforms all images in manifest file using aeon.
    Returns the transformed images.
    """
    config = {
        "manifest_filename":
        manifest_name,
        "shuffle_enable":
        False,
        "shuffle_manifest":
        False,
        "batch_size":
        count,
        "iteration_mode":
        "ONCE",
        "cache_directory":
        "",
        "cpu_list":
        "0",
        "augmentation": [{
            "type": "image",
            "angle": angle,
            "flip_enable": flip,
            "center": False,
            "crop_enable": crop["enable"],
            "crop_origin": crop["origin"],
            "horizontal_distortion": crop["ratio"],
            "do_area_scale": True,
            "scale": crop["scale"],
            "mean": MEAN if standardize else [],
            "stddev": STDDEV if standardize else [],
            "resize_short_size": 0,
            "interpolation_method": interpolation_type,
            "debug_output_directory": "out"
        }],
        "etl": [{
            "type": "image",
            "height": output_size,
            "width": output_size,
            "channels": 3,
            "output_type": data_type,
            "channel_major": channel_major,
            "bgr_to_rgb": bgr_to_rgb
        }, {
            "type": "label",
            "binary": False
        }]
    }
    data_loader = DataLoader(config)
    raw_aeon_data = data_loader.next()[0][1]
    return np.copy(raw_aeon_data)


ANGLE_OPTIONS = [[0, 0], [-10, -10], [11, 11]]
FLIP_OPTIONS = [False]
BGR_TO_RGB_OPTIONS = [False, True]
STANDARDIZE_OPTIONS = [False, True]
CHANNEL_MAJOR_OPTIONS = [True, False]
CROP_OPTIONS = [{
    "enable": True,
    "origin": [0, 0],
    "scale": [1, 1],
    "ratio": [1, 1]
}, {
    "enable": True,
    "origin": [0, 0],
    "scale": [0.1, 0.1],
    "ratio": [0.1, 0.1]
}, {
    "enable": True,
    "origin": [0, 0],
    "scale": [0.5, 0.5],
    "ratio": [0.8, 0.8]
}]


def test_loader_compare_various():
    """Compare images provided by aeon with the reference code and \
        assert max abs difference is small.
    Use all combinations of provided arguments to the following transformations:
    rotate, crop, flip, bgr_to_rgb, standardize and channel_major.
    """
    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path + '/test_data')
    manifest_name = "manifest_for_compare.tsv"
    assert os.path.getsize(manifest_name) > 0

    count = 12
    output_size = 224
    for values in itertools.product(ANGLE_OPTIONS, CROP_OPTIONS, FLIP_OPTIONS,
                                    BGR_TO_RGB_OPTIONS, STANDARDIZE_OPTIONS,
                                    CHANNEL_MAJOR_OPTIONS):
        angle = values[0]
        crop = values[1]
        flip = values[2]
        bgr_to_rgb = values[3]
        standardize = values[4]
        channel_major = values[5]
        print("angle: {},\ncrop: {},\nflip: {},\nbgr_to_rgb: {},\n"
              "standardize: {},\nchannel_major: {}\n".format(
                  angle, crop, flip, bgr_to_rgb, standardize, channel_major))

        # get all reference transformed images
        data_type = 'float32'
        interpolation_type = cv2.INTER_LANCZOS4
        reference_images = get_reference_images_cv(manifest_name, count,
                                                   output_size, angle, crop,
                                                   interpolation_type, flip,
                                                   bgr_to_rgb, standardize,
                                                   channel_major, data_type)
        # get aeon transformed images
        interpolation_type = "LANCZOS4"
        data_type = 'float'
        aeon_images = get_aeon_images(manifest_name, count, output_size, angle,
                                      crop, interpolation_type, flip,
                                      bgr_to_rgb, standardize, channel_major,
                                      data_type)
        # compare them
        expected_shape = (count, 3, output_size,
                          output_size) if channel_major else (count,
                                                              output_size,
                                                              output_size, 3)
        error_message = "Assert failed: {} == {} == {}".format(
            reference_images.shape, aeon_images.shape, expected_shape)
        assert reference_images.shape == aeon_images.shape == expected_shape, error_message
        eps_err = 0.0001 if standardize else 1
        list_of_errors = []
        for i in range(count):
            max_abs_diff = np.amax(
                np.abs((reference_images[i] - aeon_images[i])))
            if max_abs_diff >= eps_err:
                list_of_errors.append(
                    "Assertion {} < {} failed for image {}.".format(
                        max_abs_diff, eps_err, i))
        assert not list_of_errors, "Images provided by aeon do not match reference:\n{}".format(
            "\n".join(list_of_errors))

    os.chdir(cwd)

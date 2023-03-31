"""Zindi data normalizer, reads each image in the training set,
and calculates the mean pixel value and std for each channel,
 returns the average of the mean and std to be used as image normalisation e.g.

[INFO] Mean: [0.41210787 0.50030631 0.34875169]
[INFO] Standard deviation: [0.15202952 0.15280726 0.1288698 ]
"""

import cv2
import numpy as np
import pandas as pd

import os

from pathlib import Path
from PIL import Image


from core.utils import csv_read, check_empty_images
from core.logs import ProjectLogger
logger = ProjectLogger(__name__)


def setup_args(subparsers):
    """Argument paser for data_normalizer."""
    subparsers.add_parser("data_normalizer")
    return None


def extract_image_stats(images_dir, images_files, round_decimals=None):
    """
    Iterate through images in a directory, and extract the image min, max, mean, and std for each channel and
    averaged across all channels, add these data as rows in a pandas dataframe

    Parameters:
    images_dir (str): Directory containing the images

    Returns:
    stats_df (pandas.DataFrame): Dataframe containing image stats for each image
    """

    stats = []
    for image_file in images_files:
        img = cv2.imread(os.path.join(images_dir, image_file))

        channels = {'b': img[:, :, 0], 'g': img[:, :, 1], 'r': img[:, :, 2]}

        img_stats = {
            'ImageId': image_file,
            'img_min': img.min(),
            'img_max': img.max(),
            'img_mean': img.mean(),
            'img_var': img.var(),
            'img_std': img.std()}

        channel_stats = [{f"min_{c}": v.min(),
                          f"max_{c}": v.max(),
                          f"mean_{c}": v.mean(),
                          f"var_{c}": v.var(),
                          f"std_{c}": v.std()} for c,
                         v in channels.items()]
        channel_stats = {k: v for d in channel_stats for k, v in d.items()}
        stats.append(dict(**img_stats, **channel_stats))

    df = pd.DataFrame(stats)

    if round_decimals:
        df = df.round(round_decimals)

    return df


def main(kwargs):
    df_train = csv_read(kwargs.get("train_csv"))

    image_files = df_train["ImageId"].to_list()

    empty_images = check_empty_images(kwargs['train_images'])

    logger.i(f"Images in {kwargs['train_images']}: {len(image_files)}")
    logger.i(f"Empty images in {kwargs['train_images']}: {len(empty_images)}")
    logger.i(
        f"Images to process from training data: {kwargs['train_images']}: "
        f"{len(image_files) - len(empty_images)}")

    image_files = [i for i in image_files if i not in empty_images]

    image_paths = [Path(kwargs.get("train_images"), img)
                   for img in image_files]

    num_channels = 3

    mean = np.zeros(num_channels)
    variance = np.zeros(num_channels)
    count = 0

    for i, image_file in enumerate(image_paths):
        with Image.open(image_file) as img:
            img_arr = np.array(img) / 255
            mean += np.mean(img_arr, axis=(0, 1))
            variance += np.var(img_arr, axis=(0, 1))
            if i % 20 == 0:
                logger.i(
                    f"Processing image {image_file} {i} / {len(image_paths)}")
            count += 1

    final_mean = mean / count
    final_std = np.sqrt(variance / count)
    logger.i(f"Mean: {final_mean}")
    logger.i(f"Standard deviation: {final_std}")

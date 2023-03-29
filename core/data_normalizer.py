"""Zindi data normalizer
https://zindi.africa/learn/how-to-download-data-files-from-zindi-to-colab
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import os

from core.logs import ProjectLogger
from core.utils import csv_read
from PIL import Image

logger = ProjectLogger(__name__)


def setup_args(subparsers):
    """Argument paser for data_normalizer."""
    subparser_data_downloader = subparsers.add_parser("data_normalizer")
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


def normalize(kwargs):
    df_train = csv_read(kwargs.get("train_csv"))

    image_files = df_train["ImageId"].to_list()
    image_paths = [Path(kwargs.get("train_images"), img) for img in image_files]

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
                    f"Processing image {image_file} {i} / {len(df_train)}")
            count += 1

    final_mean = mean / count
    final_std = np.sqrt(variance / count)
    logger.i(f"Mean: {final_mean}")
    logger.i(f"Standard deviation: {final_std}")



def main(kwargs):
    normalize(kwargs)


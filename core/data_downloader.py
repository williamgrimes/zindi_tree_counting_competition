"""Zindi data downloader
https://zindi.africa/learn/how-to-download-data-files-from-zindi-to-colab
"""

import os
from pathlib import Path

import requests
import shutil

from core.logs import ProjectLogger
from core.utils import check_file_exists

logger = ProjectLogger(__name__)


def setup_args(subparsers):
    """Argument paser for downloading project data."""
    subparser_data_downloader = subparsers.add_parser("data_downloader")
    subparser_data_downloader.add_argument(
        "--train_url",
        default="https://api.zindi.africa/v1/competitions/digital-africa-plantation-counting-challenge/files/Train.csv",
        help="URL of Train data csv.")
    subparser_data_downloader.add_argument(
        "--test_url",
        default="https://api.zindi.africa/v1/competitions/digital-africa-plantation-counting-challenge/files/Test.csv",
        help="URL of Test data csv.")
    subparser_data_downloader.add_argument(
        "--images_url",
        default="https://api.zindi.africa/v1/competitions/digital-africa-plantation-counting-challenge/files/TreeImages.zip",
        help="URL of TreeImages.zip.")
    subparser_data_downloader.add_argument(
        "--auth_token",
        required=True,
        help="Required token to access data.")
    return subparsers


def zindi_data_downloader(
        data_url: str,
        token: str,
        unzip: bool = False,
        **kwargs):
    """Function to download project data from zindi website."""
    file_name = data_url.split("/")[-1]
    file_path = Path(kwargs.get('data_dir'), file_name)

    if not check_file_exists(file_path):
        logger.i(f"Downloading {file_path}.")
        token = {"auth_token": f"{token}"}

        competition_data = requests.post(url=data_url, data=token, stream=True)

        handle = open(file_path, "wb")
        for chunk in competition_data.iter_content(
                chunk_size=512):  # Download the data in chunks
            if chunk:  # filter out keep-alive new chunks
                handle.write(chunk)
        handle.close()
        logger.i(f"Finished downloading {file_path}.")

        if unzip:
            extract_dir = Path(file_path.parent, file_path.stem)
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir)
                logger.i(f"Unpacking {file_path} to {extract_dir}.")
                shutil.unpack_archive(file_path, extract_dir)
            else:
                logger.i(f"Directory {extract_dir} already exists.")


def create_data_dir(kwargs):
    """Create data directory if it does not exist."""
    if not os.path.exists(kwargs.get("data_dir")):
        os.makedirs(kwargs.get("data_dir"))
        logger.i(f"Directory {kwargs.get('data_dir')} created successfully!")
    else:
        logger.i(f"Directory {kwargs.get('data_dir')} already exists.")


def main(kwargs):
    create_data_dir(kwargs)
    zindi_data_downloader(
        kwargs.get("train_url"),
        kwargs.get("auth_token"),
        **kwargs)
    zindi_data_downloader(
        kwargs.get("test_url"),
        kwargs.get("auth_token"),
        **kwargs)
    zindi_data_downloader(
        kwargs.get("images_url"),
        kwargs.get("auth_token"),
        unzip=True,
        **kwargs)


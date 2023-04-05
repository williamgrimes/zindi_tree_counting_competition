"""Utility functions for project."""
import csv
import gc
import os
import subprocess

import pandas as pd
import psutil
import torch
import yaml

from pathlib import Path
from typing import Dict, Union, List
from sklearn.model_selection import train_test_split

from core.logs import ProjectLogger

logger = ProjectLogger(__name__)


def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.i(f"Set seed to {seed}, in numpy, torch.")


def csv_read(csv_file, sep=','):
    df = pd.read_csv(csv_file, sep=sep)
    logger.i(f"Read {csv_file} with {len(df)} rows, columns {list(df.columns)}.")
    return df


def csv_write(df, csv_file, sep=',', index=True):
    df.to_csv(csv_file, sep=sep, index=index)
    logger.i(f"Writing {csv_file} with {len(df)} rows, columns {list(df.columns)}.")
    return None


def train_val_split(df, val_split, seed):
    df_train, df_val = train_test_split(df, test_size=val_split, random_state=seed)
    logger.i(f"Split {len(df)} rows, to train {len(df_train)}, rows and "
             f"validation set {len(df_val)}, columns {list(df.columns)}.")
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.i(f"CUDA memory before flush cache:")
        logger.i(f"--- memory:  {get_device_mem_used(device)} / {get_device_mem_total(device)} Mb")
        torch.cuda.empty_cache()
        gc.collect()
        logger.i(f"CUDA memory after flush cache:")
        logger.i(f"--- memory:  {get_device_mem_used(device)} / {get_device_mem_total(device)} Mb")
    else:
        device = torch.device("cpu")
    logger.i(f"{device.type.upper()} device available.")
    return device


def log_dict(d, prefix=''):
    """Explode dicitonary and log outputs"""
    for key, value in d.items():
        if isinstance(value, dict):
            log_dict(value, f"{prefix}{key} - ")
        else:
            logger.i(f"--- Parameter: {prefix}{key}: {str(value)}")


def get_params(kwargs):
    """Load parameters from yaml file."""
    with open(kwargs.get("params_file")) as file:
        params = yaml.safe_load(file)
        params = {**params, **params['net'][kwargs.get('net')]}
        params = {k: v for k, v in params.items() if k != "net"}
        params = {**{"net": kwargs.get("net")}, **params}
        logger.i(f"Loaded {kwargs.get('command')} params for {kwargs.get('net')} "
                 f"from {file.name}.")
    log_dict(params)
    return params


def check_file_exists(file_path: str):
    """Function to check if file exists."""
    if os.path.exists(file_path):
        logger.i(f"{file_path} exists.")
        return True
    else:
        logger.i(f"{file_path} does not exist.")
        return False


def get_device_mem_used(device):
    if device.type == "cuda":
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits'],
            encoding='utf-8')
        mem_usage = result.strip().split('\n')[1:]
        mem_usage = sum([int(''.join(x for x in s if x.isdigit())) for s in mem_usage])
    else:
        process = psutil.Process()
        memory_info = process.memory_info()
        mem_usage = memory_info.rss / 1024 ** 2  # Convert to MB
    return int(mem_usage)

def get_device_mem_total(device):
    if device.type == "cuda":
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits'],
            encoding='utf-8')
        mem_total = result.strip().split('\n')[1:]
        mem_total = sum([int(''.join(x for x in s if x.isdigit())) for s in mem_total])
    else:
        mem_total = psutil.virtual_memory().total / 1024 ** 2
    return int(mem_total)


def device_memory_usage(device):
    """Function to check if file exists."""
    if device.type == "cuda":
        result = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=index,memory.used,memory.total',
            '--format=csv,nounits',
            '--format=csv',
            '--unit=G',
        ], encoding='utf-8')
        gpu_memory = result.strip().split('\n')[1:]
        gpu_memory = [x.split(',') for x in gpu_memory]
        gpu_memory = {int(x[0]): {'used': x[1], 'total': x[2]} for x in gpu_memory}
        mem_string = f"GPU memory: {gpu_memory}"
    else:
        mem = psutil.virtual_memory()
        mem_string = f"CPU memory: {mem.used / 1024 ** 3:.2f} GB / " \
                     f"{mem.total / 1024 ** 3:.2f} GB (total)"
    return mem_string


def check_empty_images(images_dir: Union[str, Path]) -> List[str]:
    """
    Checks for empty images in the given directory and returns a list of their filenames.
    The function considers an image empty if its file size is 4813 bytes.

    :param images_dir: Path or str of the directory containing the images
    :return: empty_images: list of empty images
    """
    empty_images = []
    for image_file in os.listdir(images_dir):
        img_size = os.stat(Path(images_dir, image_file)).st_size
        if img_size == 4813:  # 4813 is the size of empty image
            logger.d(f"Image {image_file} is empty")
            empty_images.append(image_file)
    return empty_images


def write_dict_to_csv(data: dict, file_path: str) -> None:
    """
    Write a dictionary as a line in a CSV file.

    Args:
        data: A dictionary containing the data to write.
        file_path: The path to the CSV file.

    Returns:
        None
    """
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        logger.i(f"Writing {data} to {file_path}")
        writer.writerow(data)

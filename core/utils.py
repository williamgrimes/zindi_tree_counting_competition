import gc
import os

import pandas as pd
import psutil
import torch
import yaml
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
        logger.i(f"CUDA memory before flush cache: {device_memory_usage(device)}.")
        torch.cuda.empty_cache()
        gc.collect()
        logger.i(f"CUDA memory after flush cache: {device_memory_usage(device)}.")
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
        params = params.get(kwargs.get("command"))
        logger.i(f"Loaded {kwargs.get('command')} params: {file.name}.")
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

def device_memory_usage(device):
    """Function to check if file exists."""
    if device.type == "cuda" :
        mem_string = f"GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB / " \
        f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB (max)"
    else:
        mem = psutil.virtual_memory()
        mem_string = f"CPU memory: {mem.used / 1024 ** 3:.2f} GB / " \
        f"{mem.total  / 1024 ** 3:.2f} GB (total)"
    return mem_string


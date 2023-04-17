"""Palm tree counting training model and evaluate"""

import PIL.Image
import cv2
import math
import os
import numpy as np
import pandas as pd
import shutil
import time
import torch


from efficientnet_pytorch import EfficientNet
from pathlib import Path
from PIL import Image
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from typing import Callable, List, Tuple, Dict

from core import argparser
from core.logs import ProjectLogger
from core.utils import csv_read, train_val_split, check_device, get_params, get_device_mem_used, get_device_mem_total, \
    csv_write, write_dict_to_csv

logger = ProjectLogger(__name__)


def setup_args(subparsers: argparser) -> None:
    """Argument paser for efficientnet."""
    subparser_train = subparsers.add_parser("train")
    subparser_train.add_argument(
        "--net",
        required=True,
        help="Model architecture (Net) to train.")
    return None


def transform_rescale(img: PIL.Image.Image) -> PIL.Image.Image:
    """
    Rescale an image using perspective correction, strecth to cover maximum area.

    Args:
        img (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: Rescaled image.
    """

    img = np.array(img)

    H, W = img.shape[0:2]

    if img.max():  # image has data
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea)

        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        approx = np.reshape(approx, (approx.shape[0], -1)).astype('float32')

        if len(approx) != 4:
            x, y, w, h = cv2.boundingRect(largest_contour)
            approx = np.array([[x, y], [x, y + h - 1], [x + w - 1, y + h - 1], [x + w - 1, y]], dtype=np.float32)

        pts_dst = np.array([[0, 0], [0, H - 1], [W - 1, H - 1], [W - 1, 0]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(approx, pts_dst)

        img_warped = cv2.warpPerspective(img, M, (H, W))
    else:
        img_warped = img

    return Image.fromarray(img_warped)


def transform_images(params: Dict) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Transform images, creates data augmentation pipelines for training and
    validation set. If rescale is true then  the image is rescaled using the
    transform_rescale function.

    Args:
        params (Dict): Dictionary of parameters for preprocessing / training.

    Returns:
        train_transform, val_transform: Compose pipelines for training and val.
    """

    normalizer = transforms.Normalize(
        mean=params["normalize"]["rgb_means"],
        std=params["normalize"]["rgb_stds"],
    )
    train_transform = [
        transforms.ToTensor(),
        transforms.Resize(params["transforms"]["resize"]),
        transforms.GaussianBlur(kernel_size=params["transforms"]["blur_kernel"],
                                sigma=params["transforms"]["blur_sigma"]),
        transforms.RandomHorizontalFlip(p=params["transforms"]["h_flip_probability"]),
        transforms.RandomVerticalFlip(p=params["transforms"]["v_flip_probability"]),
        transforms.ColorJitter(brightness=params["transforms"]["jitter_brightness"],
                               contrast=params["transforms"]["jitter_contrast"],
                               saturation=params["transforms"]["jitter_saturation"],
                               hue=params["transforms"]["jitter_hue"]),
        normalizer,
        ]
    val_transform = [
            transforms.ToTensor(),
            transforms.Resize(params["transforms"]["resize"]),
            normalizer,
    ]
    if params["transforms"]["rescaler"]:
        logger.i("Rescaling images.")
        train_transform.insert(0, transform_rescale)
        val_transform.insert(0, transform_rescale)
    logger.i(f"{train_transform=}")
    logger.i(f"{val_transform=}")
    return transforms.Compose(train_transform), transforms.Compose(val_transform)


class TreeImagesDataset(Dataset):
    """
    A PyTorch Dataset for loading tree images and their corresponding labels.

    Args:
        df (pandas.DataFrame): DataFrame containing the image IDs and labels.
        images_dir (str): Root directory path of the images.
        is_inference (bool, optional): Whether the dataset is being used for inference or not. Default is False.
        transform (bool, optional): Whether to apply data augmentation transforms to the images. Default is False.

    Returns:
        torch.Tensor or tuple: If `is_inference` is False, returns a tuple containing the transformed image and its label as a tensor.
                               If `is_inference` is True, returns only the transformed image as a tensor.
    """
    def __init__(self, df: pd.DataFrame, images_dir: str, is_inference: bool = False, transform: bool = False):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform
        self.is_inference = is_inference

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.df)

    def __getitem__(self, index):
        """
        Returns a transformed image and its label as a tensor, or only a
        transformed image as a tensor.

        Args:
            index (int): Index of the item to be retrieved from the dataset.

        Returns:
            torch.Tensor or tuple:
                If `is_inference` is False, returns a tuple containing the transformed image and its label as a tensor.
                If `is_inference` is True, returns only the transformed image as a tensor.
        """

        image_name = self.df['ImageId'][index]
        images_path = Path(self.images_dir, image_name)
        image = Image.open(images_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_inference is False:
            label = self.df['Target'][index]
            return image, torch.as_tensor(label)

        return image


class EfficientNetCounter(nn.Module):
    """
    A PyTorch module for counting objects using EfficientNet.

    Args:
        params (dict): A dictionary containing the configuration parameters for the model.
            - model_name (str): The name of the EfficientNet model to use.

    Attributes:
        model (EfficientNet): The pre-trained EfficientNet model.
        fc (nn.Linear): The fully-connected layer for counting objects.
        relu (nn.ReLU): The activation function for the fully-connected layer.

    """
    def __init__(self, params=None):
        super(EfficientNetCounter, self).__init__()
        self.model = EfficientNet.from_pretrained(params.get("model_name"))
        self.fc = nn.Linear(1000, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return self.relu(x)

class ResNetCounter(nn.Module):
    """
    A PyTorch module for counting objects using ResNet.

    Args:
        params (dict): A dictionary containing the configuration parameters for the model.
            - model_name (str): The name of the ResNet model to use.

    Attributes:
        model (nn.Sequential): The pre-trained ResNet model, with its final layers removed.
        fc1 (nn.Linear): The first fully-connected layer for counting objects.
        fc2 (nn.Linear): The second fully-connected layer for counting objects.

    """
    def __init__(self, params=None):
        super(ResNetCounter, self).__init__()
        self.model = getattr(models, params.get("model_name"))(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.fc1 = nn.Linear(512*7*7, 256)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(256, 1)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, batch):
        x = self.model(batch)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        return x


def compute_rmse_loss(loader, model, device):
    """
    Computes the Root Mean Squared Error (RMSE) loss between the model's
    predictions and the true labels over a given data loader.

    Args:
        loader: A PyTorch data loader containing a dataset of images and labels.
        model: A PyTorch model used for making predictions on the images.
        device: The device (CPU or GPU) used for training the model.

    Returns:
        The average RMSE loss over all predictions and true labels in the loader.
    """
    loss = 0
    criterion = nn.MSELoss().to(device)
    model.eval()  # Set model to evaluation mode, disables gradients
    with torch.no_grad():  # no_grad context disables gradient computation and reduce memory usage.
        for X, y in loader:
            X = X.to(device).to(torch.float32)
            y = y.to(device).to(torch.float).unsqueeze(1)

            model = model.to(device)
            preds = model(X)
            loss += torch.sqrt(criterion(preds, y)).item()
    model.train()  # Set model back to train mode
    return loss / len(loader)


def load_checkpoint(checkpoint, model):
    """Loads a model's state dictionary from a checkpoint.

    Args:
        checkpoint: A dictionary containing a model's state dictionary.
        model: A PyTorch model to load the checkpoint into.

    Returns:
        None

    Raises:
        KeyError: If the state dictionary key 'state_dict' is not found in the checkpoint.
    """
    try:
        model.load_state_dict(checkpoint['state_dict'])
        logger.i(f"Checkpoint loaded successfully")
    except KeyError:
        logger.e(f"Checkpoint does not contain state_dict key")
        raise


def train_epoch(data_loader: torch.utils.data.DataLoader, model: torch.nn.module,
                opt: torch.optim.Optimizer, criterion: Callable, device: torch.device):
    """
    Train the given PyTorch model for one epoch using the provided data loader,
    optimization algorithm, and loss function.

    Parameters:
        data_loader (torch.utils.data.DataLoader): A PyTorch DataLoader object containing the training data.
        model (torch.nn.Module): The PyTorch model to train.
        opt (torch.optim.Optimizer): The PyTorch optimization algorithm to use for training.
        criterion (callable): The loss function to use for training, which should take in the model predictions and ground truth labels as inputs.
        device (torch.device): The device (CPU or GPU) to use for training.

    Returns:
        None
    """
    intervals = np.linspace(0, len(data_loader), num=11).astype(int)
    for i, (X, y) in enumerate(data_loader):
        X = X.to(device).to(torch.float32)
        y = y.to(torch.float).unsqueeze(1).to(device)

        preds = model(X).to(torch.float)

        loss = criterion(preds, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if i in intervals:
            logger.i(f" --- batch: {i + 1} / {len(data_loader)}: RMSE {math.sqrt(loss):.4f}")


def training_loop(dataloaders: Dict[str: torch.utils.data.DataLoader],
                  device: torch.device, params: Dict, model_file_path: Path)\
        -> Tuple[torch.nn.Module, float]:
    """
    Train a PyTorch model using the provided training, validation and test dataloaders.

    Args:
        dataloaders: A dictionary containing PyTorch DataLoader objects for training, validation and test sets.
        device: A torch.device object specifying the device to use for training.
        params: A dictionary containing hyperparameters for training the model.
        model_file_path: A Path object specifying the path to save the trained model.

    Returns:
        A tuple containing the trained PyTorch model and the final validation loss.

    Raises:
        None.
    """
    criterion = nn.MSELoss().to(device)

    if params.get("net") == "efficientnet":
        model = EfficientNetCounter(params).to(device)
    elif params.get("net") == "resnet":
        model = ResNetCounter(params).to(device)
    else:
        logger.e(f"Unknown net param supplied: {params.get('net')}")
    opt = optim.Adam(model.parameters(), lr=params.get("learning_rate"))
    loss = 999999999
    es = 0
    for epoch in range(params.get("max_epochs")):
        logger.i(f"")
        logger.i(f"Epoch {epoch} / {params.get('max_epochs')}: val_loss: {loss:.4f}")

        train_epoch(dataloaders["train"], model, opt, criterion, device)
        val_loss = compute_rmse_loss(dataloaders["val"], model, device)

        logger.i(f" --- val loss: {val_loss:.2f}")
        logger.i(f" --- memory:  {get_device_mem_used(device)} / "
                 f"{get_device_mem_total(device)} Mb")

        if val_loss < loss:
            loss = val_loss
            es = 0
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict()
            }
            logger.i(f'Writing checkpoint -> {model_file_path.name}')
            torch.save(checkpoint, model_file_path)  # save checkpoint

        else:
            es += 1

        if es == params.get("early_stopping_patience"):
            logger.i(f"Early stopping at epoch {epoch} with best {loss=}")
            break
    return model, loss


def inference(dataloaders: Dict[str: torch.utils.data.DataLoader],
              model: torch.nn.Module, df_test: pd.DataFrame,
              device: torch.device) -> pd.DataFrame:
    """
    Runs inference on a test set and returns predictions in a pandas DataFrame.

    Args:
        dataloaders: A dictionary containing PyTorch data loaders for train, validation, and test sets.
        model: A PyTorch model for making predictions.
        df_test: A pandas DataFrame containing the test set.
        device: A string specifying the device to run the model on ('cpu' or 'cuda').

    Returns:
        A pandas DataFrame with predictions for the test set.
    """
    loader = dataloaders["test"]
    intervals = np.linspace(0, len(loader), num=11).astype(int)
    model.eval()
    all_preds = np.array([])
    with torch.no_grad():
        for i, x in enumerate(loader):
            x = x.to(device).to(torch.float32)
            all_preds = np.append(all_preds, model(x).cpu())
            if i in intervals:
                logger.i(f" --- inference: {i} / {len(loader)}")

    df_test['Target'] = all_preds
    logger.i('Finished Inference!')
    return df_test


def main(kwargs):
    start_time = time.time()
    params = get_params(kwargs)
    device = check_device()

    run_name = f"{logger.now}_{params['model_name']}"
    run_dir = Path(kwargs["runs_dir"], f"{run_name}")
    logger.d(f"{run_dir=}")

    df = csv_read(kwargs.get("train_csv"))
    df_test = csv_read(kwargs.get("test_csv"))

    df_train, df_val = train_val_split(
        df, params.get("val_size"), kwargs.get("seed"))

    train_transform, val_transform = transform_images(params)

    datasets = {
        "train": TreeImagesDataset(
            df_train,
            kwargs.get("train_images"),
            transform=train_transform),
        "val": TreeImagesDataset(
            df_val,
            kwargs.get("train_images"),
            transform=val_transform),
        "test": TreeImagesDataset(
            df_test,
            kwargs.get("train_images"),
            is_inference=True,
            transform=val_transform),
    }
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=params.get("batch_size"),
            shuffle=True),
        "val": DataLoader(
            datasets["val"],
            batch_size=params.get("batch_size"),
            shuffle=False),
        "test": DataLoader(
            datasets["test"],
            batch_size=params.get("batch_size"),
            shuffle=False)}

    logger.i(f"Creating {run_dir} directory.")
    os.makedirs(run_dir)

    model_file_path = Path(run_dir, f"{run_name}_checkpoint.pth.tar")
    preds_file_path = Path(run_dir, f"{run_name}_preds.csv")
    params_file_path = Path(run_dir, f"{run_name}_params.yaml")
    logs_file_path = Path(run_dir, f"{run_name}.log")

    model, loss = training_loop(dataloaders, device, params, model_file_path)

    load_checkpoint(torch.load(model_file_path), model)

    df_test = inference(dataloaders, model, df_test, device)

    csv_write(df_test, preds_file_path, index=False)

    logger.i(f"Copying params {run_dir} directory.")
    shutil.copy(kwargs.get("params_file"), params_file_path)

    if logger.log_file:
        logger.i(f"Log file found, copying to {logs_file_path}.")
        shutil.copy(logger.log_file, logs_file_path)

    run_data = {"run": run_name,
                "loss": loss,
                "model_name": params["model_name"],
                "learning_rate": params["learning_rate"],
                "batch_size": params["batch_size"],
                "image_size": params["transforms"]["resize"],
                "image_rescaler": params["transforms"]["rescaler"],
                "blur_kernel": params["transforms"]["blur_kernel"],
                "blur_sigma": params["transforms"]["blur_sigma"],
                "mem_usage": get_device_mem_used(device),
                "elapsed_time": round(time.time() - start_time)
                }

    write_dict_to_csv(run_data, kwargs["runs_csv"])

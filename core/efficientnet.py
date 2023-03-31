"""EfficientNet approach"""

import os
import shutil
from pathlib import Path

import torch

import numpy as np
from PIL import Image

from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from efficientnet_pytorch import EfficientNet
from sklearn.metrics import mean_squared_error

from core.logs import ProjectLogger
from core.utils import csv_read, train_val_split, check_device, get_params, get_device_mem_used, get_device_mem_total, \
    csv_write, write_dict_to_csv

logger = ProjectLogger(__name__)


def setup_args(subparsers):
    """Argument paser for efficientnet."""
    subparser_data_downloader = subparsers.add_parser("efficientnet")
    return None


def rescaler(img):

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

def transform_images(params):
    normalizer = transforms.Normalize(
        mean=params["normalize"]["rgb_means"],
        std=params["normalize"]["rgb_stds"],
    )
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                [
                    params["transforms"]["height"],
                    params["transforms"]["width"]]),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=params["transforms"]["blur_kernel"])],
                p=params["transforms"]["blur_probability"]),
            transforms.RandomHorizontalFlip(
                p=params["transforms"]["horizontal_flip_probability"]),
            transforms.RandomVerticalFlip(
                p=params["transforms"]["vertical_flip_probability"]),
            normalizer,
        ])
    logger.i(f"{train_transform=}")
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize([params["transforms"]["height"], params["transforms"]["width"]]),
            normalizer,
        ]
    )
    logger.i(f"{val_transform=}")
    return train_transform, val_transform


class TreeImagesDataset(Dataset):
    def __init__(self, df, root_dir, is_inference=False, transform=False):
        self.df = df
        self.images_dir = root_dir
        self.transform = transform
        self.is_inference = is_inference

        for img in self.df.iterrows():
            if not os.path.exists(Path(self.images_dir, img)):
                logger.e(f"Image {img} does not exist")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        image_name = self.df['ImageId'][index]
        image = Image.open(Path(self.images_dir, image_name)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if not self.is_inference:
            label = self.df['Target'][index]
            return image, torch.as_tensor(label)

        return image


class Net(nn.Module):
    def __init__(self, params=None):
        super(Net, self).__init__()
        self.model = EfficientNet.from_pretrained(params.get("model_name"))
        self.fc = nn.Linear(1000, 1)
        self.relu = nn.ReLU()

    def forward(self, image):
        x = self.model(image)
        x = self.fc(x)
        return self.relu(x)


def check_acc(loader, model, device):
    loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device).to(torch.float32)
            y = y.to(torch.float).unsqueeze(1)

            preds = model(X)
            loss += np.sqrt(mean_squared_error(preds.cpu(), y))
    model.train()
    return loss / len(loader)


def load_checkpoint(checkpoint, model):
    logger.i(f'Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def train_fn(loader, model, opt, loss_fn, device):
    intervals = np.linspace(0, len(loader), num=11).astype(int)
    for i, (X, y) in enumerate(loader):
        X = X.to(device).to(torch.float32)
        y = y.to(torch.float).unsqueeze(1).to(device)

        preds = model(X).to(torch.float)

        loss = loss_fn(preds, y)

        model.zero_grad()
        loss.backward()
        opt.step()
        if i in intervals:
            logger.i(f" --- iter: {i} / {len(loader)}")


def train(dataloaders, device, params, model_file_path):
    loss_fn = nn.MSELoss().to(device)
    model = Net(params).to(device)
    opt = optim.Adam(model.parameters(), lr=params.get("learning_rate"))
    loss = 999999999
    es = 0
    for epoch in range(params.get("max_epochs")):
        logger.i(f"Epoch: {epoch}")

        train_fn(dataloaders["train"], model, opt, loss_fn, device)
        train_loss = check_acc(dataloaders["train"], model, device)
        val_loss = check_acc(dataloaders["val"], model, device)

        logger.i(f" --- train loss: {train_loss:.2f}")
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
            logger.i(f'Writing checkpoint to {model_file_path}')
            torch.save(checkpoint, model_file_path)  # save checkpoint

        else:
            es += 1

        if es == params.get("early_stopping_patience"):
            break
    return model


def inference(dataloaders, model, df_test, device):
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
    params = get_params(kwargs)
    device = check_device()

    run_name = Path(kwargs["runs_dir"], f"{logger.now}_{params['model_name']}")
    logger.d(f"{run_name=}")

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

    logger.i(f"Creating {run_name} directory.")
    os.makedirs(run_name)

    model_file_path = Path(run_name, f"{run_name.name}_checkpoint.pth.tar")
    preds_file_path = Path(run_name, f"{run_name.name}_preds.csv")
    params_file_path = Path(run_name, f"{run_name.name}_params.yaml")
    logs_file_path = Path(run_name, f"{run_name.name}.log")

    model = train(dataloaders, device, params, model_file_path)

    load_checkpoint(torch.load(model_file_path), model)

    df_test = inference(dataloaders, model, df_test, device)

    csv_write(df_test, preds_file_path, index=False)

    logger.i(f"Copying params {run_name} directory.")
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


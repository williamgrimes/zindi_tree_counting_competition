"""EfficientNet approach"""

import os
import random
import torch

import albumentations as A
import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

from PIL import Image
import pandas as pd

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from efficientnet_pytorch import EfficientNet
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from core.logs import ProjectLogger
from core.utils import csv_read, set_seed, train_val_split, check_device, get_params, device_memory_usage

logger = ProjectLogger(__name__)


# TODO treat empty images where std < 0.1 separately
# TODO transform empty regions shrink bounding box

def transform_images(params):
    normalizer = A.Normalize(
        mean=params.get("normalize").get("rgb_means"),
        std=params.get("normalize").get("rgb_stds"),
        max_pixel_value=params.get("normalize").get("rgb_stds"),
    )
    train_transform = A.Compose([
        A.Resize(params.get("transforms").get("height"), params.get("transforms").get("width")),
        A.Blur(p=params.get("transforms").get("blur")),
        A.HorizontalFlip(p=params.get("transforms").get("horizontal_flip")),
        A.VerticalFlip(p=params.get("transforms").get("vertical_flip")),
        normalizer,
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(params.get("transforms").get("height"), params.get("transforms").get("width")),
        normalizer,
        ToTensorV2(),
    ])
    return train_transform, val_transform

class CustomDataset(Dataset):
    def __init__(self, df, images_dir, is_inference=False, transform=False):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform
        self.is_inference = is_inference

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        root_and_dir = self.df['ImageId'][index]
        if not self.is_inference:
            label = self.df['Target'][index]

        image = np.array(
            Image.open(
                os.path.join(
                    self.images_dir,
                    root_and_dir)).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations['image']

        if not self.is_inference:
            return image, torch.as_tensor(label)

        return image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
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


def save_checkpoint(state, filename):
    logger.i(f'--> Saving checkpoint to {filename}')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    logger.i(f'--> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def inference(loader, model, device):
    model.eval()
    all_preds = np.array([])
    with torch.no_grad():
        for x in enumerate(loader):
            x = x.to(device).to(torch.float32)
            all_preds = np.append(all_preds, model(x).cpu())

    logger.i('Finished Inference!')
    return all_preds


def train_fn(loader, model, opt, loss_fn, device):
    loader_len = len(loader)
    intervals = np.linspace(0, loader_len, num=11).astype(int)
    for i, (X, y) in enumerate(loader):
        X = X.to(device).to(torch.float32)
        y = y.to(torch.float).unsqueeze(1).to(device)

        preds = model(X).to(torch.float)

        loss = loss_fn(preds, y)

        model.zero_grad()
        loss.backward()
        opt.step()
        if i in intervals:
            logger.i(f" --- iter: {i} / {loader_len}")


def main(kwargs):
    set_seed(kwargs.get("seed"))
    params = get_params(kwargs)
    device = check_device()

    root_filename = f"{logger.now}_{kwargs['command']}_"
    logger.d(f"{root_filename=}")

    df = csv_read(kwargs.get("train_csv"))
    df_test = csv_read(kwargs.get("test_csv"))

    df_train, df_val = train_val_split(df, params.get("val_size"), kwargs.get("seed"))

    train_transform, val_transform = transform_images(params)

    train_ds = CustomDataset(
        df_train,
        kwargs.get("train_images"),
        is_inference=False,
        transform=train_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=params.get("batch_size"),
        shuffle=True)

    val_ds = CustomDataset(
        df_val,
        kwargs.get("train_images"),
        is_inference=False,
        transform=val_transform)
    val_loader = DataLoader(
        val_ds,
        batch_size=params.get("batch_size"),
        shuffle=False)

    loss_fn = nn.MSELoss().to(device)
    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=params.get("learning_rate"))
    loss = 999999999
    es = 0

    for epoch in range(params.get("max_epochs")):
        logger.i(f"Epoch: {epoch}")

        train_fn(train_loader, model, opt, loss_fn, device)
        train_loss = check_acc(train_loader, model, device)
        val_loss = check_acc(val_loader, model, device)

        logger.i(f"--- train loss: {train_loss:.2f} "
                 f"--- val loss: {val_loss:.2f} "
                 f"--- memory:  {device_memory_usage(device)}")

        if val_loss < loss:
            loss = val_loss
            es = 0
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict()
            }
            save_checkpoint(checkpoint, filename=f"{root_filename}.pth.tar")

        else:
            es += 1

        if es == params.get("early_stopping_patience"):
            break

    load_checkpoint(torch.load(f"{root_filename}.pth.tar"), model)

    test_ds = CustomDataset(
        df_test,
        kwargs.get("train_images"),
        is_inference=True,
        transform=val_transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=params.get("batch_size"),
        shuffle=False)
    preds = inference(test_loader, model, device)
    df_test['Target'] = preds
    df_test.to_csv(f"{root_filename}_preds.csv", index=False)



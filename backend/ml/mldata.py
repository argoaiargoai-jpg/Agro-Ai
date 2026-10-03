"""Dataset + class bookkeeping shared by train.py, evaluate.py and calibrate.py."""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from common import *  # noqa: F401,F403


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST)


def class_names(df: pd.DataFrame) -> list[str]:
    """34 supported PlantVillage classes (alphabetical) + unknown_plant + non_plant."""
    known = sorted(df[(df.source == "plantvillage") & (df.role == "known")].label.unique())
    return known + [UNKNOWN, NON_PLANT]


def train_transform():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.35, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(0.3),
        T.RandomApply([T.RandomRotation(35)], p=0.5),
        T.ColorJitter(0.4, 0.4, 0.4, 0.04),
        T.RandomGrayscale(0.05),
        T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.25),
        T.ToTensor(), T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def eval_transform():
    return T.Compose([T.Resize(256), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])


class ImageSet(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list[str], transform):
        self.df = df.reset_index(drop=True)
        self.idx = {c: i for i, c in enumerate(classes)}
        self.t = transform
        self.y = np.array([self.idx[l] for l in self.df.label_out])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        im = Image.open(DATA / self.df.path[i]).convert("RGB")
        return self.t(im), int(self.y[i])


def split_frame(df: pd.DataFrame, split: str, *, exclude_novel_from_train=True) -> pd.DataFrame:
    s = df[df.split == split]
    if split == "train" and exclude_novel_from_train:
        s = s[s.role != "unknown_novel"]          # novel crops must stay unseen
    return s

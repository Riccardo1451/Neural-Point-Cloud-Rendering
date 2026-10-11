# Dataset per il training: per ogni frame ritorna l'immagine RGB (ground truth) e la pose.
# La point cloud non sta qui perché è una sola per tutta la scena e viene passata a parte.

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.pointcloud_utils import load_color, load_pose


class SceneFrameDataset(Dataset):
    """height e width devono essere le stesse usate per la voxelizzazione nel training."""

    def __init__(self, sequence_path: str, frame_ids: list, height: int, width: int):
        self.sequence_path = sequence_path
        self.frame_ids = frame_ids
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        fid = self.frame_ids[idx]
        frame_str = f"frame-{fid:06d}"

        color = load_color(os.path.join(self.sequence_path, f"{frame_str}.color.png"))
        pose = load_pose(os.path.join(self.sequence_path, f"{frame_str}.pose.txt"))

        color_resized = cv2.resize(color, (self.width, self.height), interpolation=cv2.INTER_AREA)

        image_tensor = torch.from_numpy(color_resized).permute(2, 0, 1).float()  # (3, H, W) già in [0, 1]
        pose_tensor = torch.from_numpy(pose).float()

        return image_tensor, pose_tensor


def split_train_val(frame_ids: list, val_every: int = 10):
    """Un frame ogni val_every va in validazione, gli altri in training."""
    val_ids = frame_ids[::val_every]
    train_ids = [f for f in frame_ids if f not in set(val_ids)]
    return train_ids, val_ids

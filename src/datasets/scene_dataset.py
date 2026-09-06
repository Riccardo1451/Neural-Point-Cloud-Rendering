"""
scene_dataset.py

Dataset PyTorch minimale: per ogni frame della sequenza, restituisce
l'immagine RGB (il ground truth su cui calcolare la loss) e la pose della
camera (per sapere da dove voxelizzare la point cloud globale della scena).

NON restituisce la point cloud - quella e' UNA SOLA, globale, condivisa
per tutta la scena (vedi precompute_scene_pointcloud.py), non cambia da
un frame all'altro. Viene passata separatamente al training loop.
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.pointcloud_utils import load_color, load_pose


class SceneFrameDataset(Dataset):
    """
    Args:
        sequence_path: cartella della sequenza
        frame_ids: quali frame includere (per gestire lo split train/val)
        height, width: risoluzione target (deve combaciare con quella
                       usata per la voxelizzazione nel training loop)
    """

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

        image_tensor = torch.from_numpy(color_resized).permute(2, 0, 1).float()  # (3, H, W), gia' in [0, 1]
        pose_tensor = torch.from_numpy(pose).float()  # (4, 4)

        return image_tensor, pose_tensor


def split_train_val(frame_ids: list, val_every: int = 10):
    """
    Split semplice: 1 frame ogni `val_every` va in validazione, il resto
    in training. Sufficiente per monitorare l'overfitting durante il
    training, non serve nulla di piu' sofisticato per questo progetto.
    """
    val_ids = frame_ids[::val_every]
    train_ids = [f for f in frame_ids if f not in set(val_ids)]
    return train_ids, val_ids
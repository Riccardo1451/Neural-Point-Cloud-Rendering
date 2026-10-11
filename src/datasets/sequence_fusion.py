# Fonde i punti di più frame della stessa sequenza in un'unica point cloud world space.
# Così i buchi di un frame vengono riempiti dai frame che vedono la superficie da un'altra angolazione.

import glob
import os

import numpy as np
from tqdm import tqdm

from src.datasets.pointcloud_utils import (
    load_color,
    load_depth,
    load_pose,
    unproject_depth_to_pointcloud,
)


def list_available_frame_ids(sequence_path: str) -> list:
    """Lista ordinata dei frame_id presenti nella cartella (guardando i .color.png)."""
    pattern = os.path.join(sequence_path, "frame-*.color.png")
    files = sorted(glob.glob(pattern))
    frame_ids = []
    for f in files:
        basename = os.path.basename(f)
        # nome tipo frame-000000.color.png
        num_str = basename.split("-")[1].split(".")[0]
        frame_ids.append(int(num_str))
    return sorted(frame_ids)


def select_frames_by_baseline(
    sequence_path: str,
    all_frame_ids: list,
    target_frame_id: int,
    min_baseline: float = 0.05,
    max_baseline: float = 0.5,
    max_frames: int = 20,
) -> list:
    """
    Sceglie i frame in base a quanto si è spostata la camera rispetto al target
    (distanza tra min_baseline e max_baseline, in metri), non in base all'indice.
    I frame vicini nel tempo vedono quasi la stessa cosa, quelli troppo lontani rischiano drift.
    Il target è sempre incluso.
    """
    target_pose = load_pose(
        os.path.join(sequence_path, f"frame-{target_frame_id:06d}.pose.txt")
    )
    target_pos = target_pose[:3, 3]

    candidates = []
    for fid in all_frame_ids:
        if fid == target_frame_id:
            continue
        pose = load_pose(os.path.join(sequence_path, f"frame-{fid:06d}.pose.txt"))
        pos = pose[:3, 3]
        dist = float(np.linalg.norm(pos - target_pos))
        if min_baseline <= dist <= max_baseline:
            candidates.append((fid, dist))

    candidates.sort(key=lambda x: x[1])
    selected = [fid for fid, _ in candidates[:max_frames]]
    selected.append(target_frame_id)

    return sorted(set(selected))


def fuse_frames_to_pointcloud(
    sequence_path: str,
    frame_ids: list,
    pixel_stride: int = 4,
):
    """Carica i frame indicati e concatena le loro point cloud in una sola (punti e colori)."""
    all_points = []
    all_colors = []

    for fid in tqdm(frame_ids, desc="Fusione frame"):
        frame_str = f"frame-{fid:06d}"
        color_path = os.path.join(sequence_path, f"{frame_str}.color.png")
        depth_path = os.path.join(sequence_path, f"{frame_str}.depth.png")
        pose_path = os.path.join(sequence_path, f"{frame_str}.pose.txt")

        color = load_color(color_path)
        depth = load_depth(depth_path)
        pose = load_pose(pose_path)

        # con tanti frame lo stride va tenuto alto, con 4 si tiene 1 pixel su 16
        points, colors = unproject_depth_to_pointcloud(
            depth, color, pose, stride=pixel_stride
        )
        all_points.append(points)
        all_colors.append(colors)

    points_world = np.concatenate(all_points, axis=0)
    colors_all = np.concatenate(all_colors, axis=0)

    return points_world, colors_all

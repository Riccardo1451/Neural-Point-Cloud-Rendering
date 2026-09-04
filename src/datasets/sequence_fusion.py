"""
sequence_fusion.py

Fonde i punti di più frame di una sequenza 7-Scenes in un'unica point cloud
world-space persistente della scena. Dato che ogni frame ha una pose nota,
possiamo proiettare i punti di frame diversi nello stesso sistema di
coordinate world e sommarli - questo riduce drasticamente i "buchi" dovuti
al rumore/occlusioni di un singolo frame, perché un punto mancante in un
frame può essere presente in un altro frame che ha visto la stessa
superficie da un angolo diverso.
"""

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
    """
    Trova tutti i frame_id disponibili in una cartella di sequenza,
    ispezionando i file *.color.png presenti.

    Returns:
        Lista ordinata di interi (frame_id).
    """
    pattern = os.path.join(sequence_path, "frame-*.color.png")
    files = sorted(glob.glob(pattern))
    frame_ids = []
    for f in files:
        basename = os.path.basename(f)
        # formato: frame-000000.color.png
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
    Seleziona frame in base a quanto la posizione della camera differisce
    da quella del frame bersaglio, invece che in base alla vicinanza
    nell'indice/tempo.

    Motivazione: frame temporalmente vicini spesso hanno una camera che si
    e' mossa pochissimo, quindi vedono la scena dalla stessa identica
    angolazione - poco utile per colmare buchi dovuti a riflessi IR che
    dipendono dall'angolo di vista. Serve abbastanza "baseline" (differenza
    di posizione) per catturare angolazioni diverse, ma non troppa, per
    evitare drift di pose o cambiamenti di scena tra frame molto distanti.

    Args:
        sequence_path: cartella della sequenza
        all_frame_ids: lista di tutti i frame_id disponibili
        target_frame_id: frame bersaglio
        min_baseline: distanza minima (metri) tra le posizioni di camera
        max_baseline: distanza massima (metri)
        max_frames: numero massimo di frame da selezionare (i piu' vicini
                    al range [min_baseline, max_baseline], ordinati per
                    distanza crescente)

    Returns:
        Lista di frame_id selezionati (include sempre il target).
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
    """
    Carica una lista di frame e fonde le loro point cloud world-space
    in un'unica nuvola.

    Args:
        sequence_path: cartella della sequenza (es. .../chess/seq-01)
        frame_ids: lista di interi, quali frame includere nella fusione
        pixel_stride: sottocampionamento pixel PER FRAME - importante per
                      tenere sotto controllo la dimensione totale quando si
                      fondono molti frame insieme (es. stride=4 -> 1/16 dei
                      pixel per frame, ma un frame in piu' nella fusione)

    Returns:
        points_world: (N, 3) point cloud fusa
        colors: (N, 3) colori corrispondenti
    """
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

        points, colors = unproject_depth_to_pointcloud(
            depth, color, pose, stride=pixel_stride
        )
        all_points.append(points)
        all_colors.append(colors)

    points_world = np.concatenate(all_points, axis=0)
    colors_all = np.concatenate(all_colors, axis=0)

    return points_world, colors_all
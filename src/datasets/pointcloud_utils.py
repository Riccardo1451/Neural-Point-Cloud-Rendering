"""
pointcloud_utils.py

Funzioni per trasformare depth map + pose camera in point cloud world-space.

Parametri di calibrazione dalla pagina ufficiale di 7-Scenes
(Microsoft Research): principal point (320, 240), focal length (585, 585).
Nota: RGB e depth in 7-Scenes NON sono calibrate tra loro.

Possiamo leggere un frame del dataset e trasformarlo in una pointcloud world space
"""

import numpy as np
import cv2


# Intrinseci di default della depth camera in 7-Scenes (Kinect v1)
DEFAULT_INTRINSICS = {
    "fx": 585.0,
    "fy": 585.0,
    "cx": 320.0,
    "cy": 240.0,
}

# In 7-Scenes il depth è salvato come PNG a 16 bit, in millimetri.
# Il valore 65535 indica un pixel invalido (nessuna lettura di profondità).
INVALID_DEPTH_VALUE = 65535


def load_depth(depth_path: str, depth_scale: float = 1000.0) -> np.ndarray:
    """
    Legge una depth map di 7-Scenes e la converte in metri.

    Args:
        depth_path: percorso al file .depth.png
        depth_scale: fattore di conversione (7-Scenes salva in mm -> /1000 per metri)

    Returns:
        depth (H, W) float32, in metri. I pixel invalidi diventano NaN,
        così sono facili da filtrare più avanti con np.isnan.
    """
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Impossibile leggere il depth: {depth_path}")

    depth = raw.astype(np.float32)
    # Due marcatori di pixel invalido nei sensori Kinect: 65535 (nessuna lettura,
    # documentato da 7-Scenes) E 0 (altro marcatore comune di "no return" che
    # altrimenti collasserebbe i punti sull'origine della camera).
    invalid_mask = (raw == INVALID_DEPTH_VALUE) | (raw == 0)
    depth[invalid_mask] = np.nan
    depth = depth / depth_scale
    return depth


def load_color(color_path: str) -> np.ndarray:
    """
    Legge l'immagine RGB corrispondente, normalizzata in [0, 1].

    Returns:
        color (H, W, 3) float32, canali in ordine RGB.
    """
    bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Impossibile leggere il colore: {color_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def load_pose(pose_path: str) -> np.ndarray:
    """
    Legge la pose della camera da un file .pose.txt di 7-Scenes.

    In 7-Scenes il file contiene direttamente la matrice camera-to-world
    4x4 (righe separate da newline, valori separati da spazi) - NON serve
    invertirla, porta già da coordinate camera a coordinate world.

    Returns:
        pose (4, 4) float32, matrice camera-to-world.
    """
    pose = np.loadtxt(pose_path, dtype=np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"Pose con shape inattesa {pose.shape} in {pose_path}")
    return pose


def unproject_depth_to_pointcloud(
    depth: np.ndarray,
    color: np.ndarray,
    pose_cam_to_world: np.ndarray,
    intrinsics: dict = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Il cuore del modulo: da una depth map + pose, genera la point cloud
    in coordinate world.

    Procedimento (esattamente la matematica della Lezione 5 - pinhole camera,
    solo percorsa "al contrario": da pixel + depth a punto 3D, invece che
    da punto 3D a pixel):

    1. Per ogni pixel (u, v) con depth d valida, calcolo le coordinate
       camera-space (x_cam, y_cam, z_cam) invertendo il modello pinhole:
           x_cam = (u - cx) * d / fx
           y_cam = (v - cy) * d / fy
           z_cam = d
    2. Porto il punto da camera-space a world-space applicando la pose
       camera-to-world (rotazione + traslazione, coordinate omogenee).

    Args:
        depth: (H, W) depth map in metri, con NaN sui pixel invalidi
        color: (H, W, 3) immagine RGB in [0, 1]
        pose_cam_to_world: (4, 4) matrice camera-to-world
        intrinsics: dict con fx, fy, cx, cy (default: DEFAULT_INTRINSICS)
        stride: sottocampiona la griglia di pixel (utile per debug veloce
                o per limitare la densità della point cloud)

    Returns:
        points_world: (N, 3) coordinate XYZ in world space
        colors: (N, 3) colori RGB corrispondenti, in [0, 1]
    """
    if intrinsics is None:
        intrinsics = DEFAULT_INTRINSICS

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    h, w = depth.shape

    # Griglia di coordinate pixel (u = colonna, v = riga)
    u_grid, v_grid = np.meshgrid(
        np.arange(0, w, stride), np.arange(0, h, stride)
    )

    d = depth[v_grid, u_grid]
    valid_mask = ~np.isnan(d)

    u_valid = u_grid[valid_mask]
    v_valid = v_grid[valid_mask]
    d_valid = d[valid_mask]

    # Inversione del modello pinhole: da pixel a camera-space
    x_cam = (u_valid - cx) * d_valid / fx
    y_cam = (v_valid - cy) * d_valid / fy
    z_cam = d_valid

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)

    # Coordinate omogenee per applicare la trasformazione 4x4
    points_cam_h = np.concatenate(
        [points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float32)], axis=1
    )  # (N, 4)

    points_world_h = (pose_cam_to_world @ points_cam_h.T).T  # (N, 4)
    points_world = points_world_h[:, :3]

    colors = color[v_valid, u_valid]  # (N, 3)

    return points_world.astype(np.float32), colors.astype(np.float32)


def add_synthetic_depth_noise(
    depth: np.ndarray, sigma: float = 0.01, seed: int = None
) -> np.ndarray:
    """
    Aggiunge rumore gaussiano sintetico alla depth map, in metri.
    Utile più avanti per l'esperimento di robustità (confronto col
    renderer z-buffer classico su point cloud rumorosa) - non serve
    ora, ma la mettiamo già pronta qui vicino al resto della logica depth.

    Args:
        depth: (H, W) depth map in metri (può contenere NaN)
        sigma: deviazione standard del rumore, in metri
        seed: seed per riproducibilità

    Returns:
        depth rumorosa, stessa shape, NaN preservati dove erano già NaN
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=depth.shape).astype(np.float32)
    noisy_depth = depth + noise
    noisy_depth[np.isnan(depth)] = np.nan
    return noisy_depth
# Funzioni per leggere i frame di 7-Scenes e passare da depth + pose a point cloud in world space.
# Intrinseci presi dalla pagina di 7-Scenes. Attenzione: RGB e depth non sono calibrate tra loro.

import numpy as np
import cv2


# intrinseci della depth camera di 7-Scenes (Kinect v1)
DEFAULT_INTRINSICS = {
    "fx": 585.0,
    "fy": 585.0,
    "cx": 320.0,
    "cy": 240.0,
}

# la depth è un PNG a 16 bit in millimetri, 65535 vuol dire pixel senza lettura
INVALID_DEPTH_VALUE = 65535


def load_depth(depth_path: str, depth_scale: float = 1000.0) -> np.ndarray:
    """Legge la depth e la converte in metri, i pixel invalidi diventano NaN."""
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Impossibile leggere il depth: {depth_path}")

    depth = raw.astype(np.float32)
    # anche lo 0 va scartato, altrimenti i punti finiscono tutti sull'origine della camera
    invalid_mask = (raw == INVALID_DEPTH_VALUE) | (raw == 0)
    depth[invalid_mask] = np.nan
    depth = depth / depth_scale
    return depth


def load_color(color_path: str) -> np.ndarray:
    """Legge l'immagine a colori in RGB, valori in [0, 1]."""
    bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Impossibile leggere il colore: {color_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def load_pose(pose_path: str) -> np.ndarray:
    """Legge la pose 4x4 dal .pose.txt. È già camera-to-world, non va invertita."""
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
    Da depth map + pose a point cloud in world space.
    Inverte il modello pinhole per avere i punti in camera space e poi applica la pose.
    stride serve a prendere un pixel ogni tanto per avere meno punti.
    """
    if intrinsics is None:
        intrinsics = DEFAULT_INTRINSICS

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    h, w = depth.shape

    # u = colonna, v = riga
    u_grid, v_grid = np.meshgrid(
        np.arange(0, w, stride), np.arange(0, h, stride)
    )

    d = depth[v_grid, u_grid]
    valid_mask = ~np.isnan(d)

    u_valid = u_grid[valid_mask]
    v_valid = v_grid[valid_mask]
    d_valid = d[valid_mask]

    # pinhole al contrario: da pixel a camera space
    x_cam = (u_valid - cx) * d_valid / fx
    y_cam = (v_valid - cy) * d_valid / fy
    z_cam = d_valid

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    # coordinate omogenee per moltiplicare per la 4x4
    points_cam_h = np.concatenate(
        [points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float32)], axis=1
    )

    points_world_h = (pose_cam_to_world @ points_cam_h.T).T
    points_world = points_world_h[:, :3]

    colors = color[v_valid, u_valid]

    return points_world.astype(np.float32), colors.astype(np.float32)


def add_synthetic_depth_noise(
    depth: np.ndarray, sigma: float = 0.01, seed: int = None
) -> np.ndarray:
    """Aggiunge rumore gaussiano (sigma in metri) alla depth, i NaN restano NaN."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=depth.shape).astype(np.float32)
    noisy_depth = depth + noise
    noisy_depth[np.isnan(depth)] = np.nan
    return noisy_depth

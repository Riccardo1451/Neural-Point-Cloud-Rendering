"""
multiplane.py

Implementa la Sezione 3.3 del paper "Neural Point Cloud Rendering via
Multi-Plane Projection": data una point cloud world-space e una vista di
camera specifica, costruisce un volume di feature P x H x W dividendo il
frustum in P piani lungo la profondità z, e aggrega i punti in ciascun
voxel con una media pesata (invece della scelta "secca" dello z-buffer
classico).

Prototipo in NumPy puro, per debug facile - la versione PyTorch
differenziabile (per il training) arriverà come step successivo, una
volta validata la logica qui.
"""

import numpy as np


def scale_intrinsics(intrinsics: dict, orig_size: tuple, new_size: tuple) -> dict:
    """
    Riscala fx, fy, cx, cy quando si vuole renderizzare a una risoluzione
    diversa da quella con cui sono stati calibrati (es. depth camera 480x640,
    ma vogliamo un volume a 128x128 per velocita').

    Args:
        intrinsics: dict originale con fx, fy, cx, cy
        orig_size: (orig_height, orig_width)
        new_size: (new_height, new_width)

    Returns:
        dict con intrinseci riscalati
    """
    orig_h, orig_w = orig_size
    new_h, new_w = new_size
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    return {
        "fx": intrinsics["fx"] * scale_x,
        "fy": intrinsics["fy"] * scale_y,
        "cx": intrinsics["cx"] * scale_x,
        "cy": intrinsics["cy"] * scale_y,
    }


def world_to_camera(points_world: np.ndarray, pose_cam_to_world: np.ndarray) -> np.ndarray:
    """
    Porta i punti da world-space a camera-space.

    E' l'operazione opposta a quella fatta in pointcloud_utils durante
    l'unprojection: li' andavamo da camera-space a world-space applicando
    la pose; qui andiamo nella direzione contraria, quindi applichiamo
    l'inversa della pose (che per una trasformazione rigida rotazione+
    traslazione si inverte facilmente).

    Args:
        points_world: (N, 3)
        pose_cam_to_world: (4, 4)

    Returns:
        points_cam: (N, 3)
    """
    pose_world_to_cam = np.linalg.inv(pose_cam_to_world)

    n = points_world.shape[0]
    points_world_h = np.concatenate([points_world, np.ones((n, 1), dtype=np.float32)], axis=1)
    points_cam_h = (pose_world_to_cam @ points_world_h.T).T

    return points_cam_h[:, :3].astype(np.float32)


def project_to_pixels(points_cam: np.ndarray, intrinsics: dict):
    """
    Proietta i punti camera-space sul piano immagine (modello pinhole
    standard, Lezione 5).

    Args:
        points_cam: (N, 3)
        intrinsics: dict con fx, fy, cx, cy

    Returns:
        u, v: (N,) coordinate pixel float (NON ancora arrotondate a intero)
        z_cam: (N,) profondita' in camera space (serve per il binning sui piani)
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    # Punti dietro la camera non sono proiettabili, li marchiamo con z invariato
    # e lasciamo che il chiamante li filtri via z <= 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * x / z + cx
        v = fy * y / z + cy

    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32)


def compute_multiplane_voxelization(
    points_world: np.ndarray,
    features: np.ndarray,
    pose_cam_to_world: np.ndarray,
    intrinsics: dict,
    height: int,
    width: int,
    num_planes: int,
    near: float = None,
    far: float = None,
    a: float = 1.0,
    b: float = 1.0,
):
    """
    Il cuore del modulo: costruisce il volume di feature P x H x W.

    Passi (rispecchiano la Sezione 3.3 del paper):
    1. Proietta i punti world nella camera di questa vista specifica.
    2. Scarta i punti fuori dal frustum (dietro la camera, o fuori
       dall'immagine).
    3. Determina near/far come minimo/massimo della profondita' dei punti
       proiettati (esattamente come descritto nel paper), poi divide questo
       range in `num_planes` fette uniformi.
    4. Assegna ogni punto al suo voxel (piano p, riga v, colonna u).
    5. Calcola il peso di ogni punto (Eq. 2: combina D1 = distanza dal
       centro pixel, D2 = distanza in profondita' dal punto piu' vicino nel
       voxel) e fa la media pesata delle feature per voxel (Eq. 1).

    Args:
        points_world: (N, 3) point cloud world-space
        features: (N, C) feature per punto (es. RGB, o feature apprese 8-dim)
        pose_cam_to_world: (4, 4) pose della vista per cui vogliamo il volume
        intrinsics: dict con fx, fy, cx, cy
        height, width: risoluzione dell'immagine target
        num_planes: P, numero di piani di profondita'
        near, far: se None, calcolati automaticamente dai dati (come nel paper)
        a, b: esponenti dei due termini di peso nell'Eq. 2

    Returns:
        volume: (num_planes, height, width, C) float32, feature aggregate
                (zero dove non cade nessun punto)
        occupancy: (num_planes, height, width) bool, True dove almeno un
                   punto e' caduto in quel voxel (utile alla rete/debug per
                   distinguere "voxel vuoto" da "voxel con feature zero")
    """
    points_cam = world_to_camera(points_world, pose_cam_to_world)
    u, v, z = project_to_pixels(points_cam, intrinsics)

    # --- Filtro 1: punti davanti alla camera e dentro l'immagine ---
    in_front = z > 1e-6
    u_int = np.floor(u).astype(np.int64)
    v_int = np.floor(v).astype(np.int64)
    in_image = (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)

    valid = in_front & in_image
    if not np.any(valid):
        C = features.shape[1]
        return (
            np.zeros((num_planes, height, width, C), dtype=np.float32),
            np.zeros((num_planes, height, width), dtype=bool),
        )

    u, v, z = u[valid], v[valid], z[valid]
    u_int, v_int = u_int[valid], v_int[valid]
    feats = features[valid]

    # --- Determina near/far dai dati, come nel paper ---
    if near is None:
        near = float(z.min())
    if far is None:
        far = float(z.max())
    far = max(far, near + 1e-6)  # evita divisione per zero se near == far

    # --- Assegna ogni punto al suo piano di profondita' ---
    plane_idx = np.floor((z - near) / (far - near) * num_planes).astype(np.int64)
    plane_idx = np.clip(plane_idx, 0, num_planes - 1)

    # --- Indice lineare univoco per ogni voxel (piano, riga, colonna) ---
    voxel_id = plane_idx * (height * width) + v_int * width + u_int
    num_voxels = num_planes * height * width

    # --- D2: differenza di profondita' dal punto piu' vicino nello stesso voxel ---
    # Per farlo servono i valori minimi di z per ciascun voxel_id.
    z_min_per_voxel = np.full(num_voxels, np.inf, dtype=np.float32)
    np.minimum.at(z_min_per_voxel, voxel_id, z)
    z_min_of_point = z_min_per_voxel[voxel_id]
    d2 = np.abs(z - z_min_of_point)

    # --- D1: distanza (normalizzata in [0, 1]) tra la proiezione del punto
    #     e il centro del pixel in cui cade ---
    du = u - (u_int + 0.5)
    dv = v - (v_int + 0.5)
    dist_pixel = np.sqrt(du ** 2 + dv ** 2)
    d1 = np.clip(dist_pixel / (np.sqrt(2) * 0.5), 0.0, 1.0)  # normalizzato su distanza max nel pixel

    # --- Peso finale (Eq. 2) ---
    weight = ((1.0 - d1) ** a) * (1.0 / (1.0 + d2)) ** b
    weight = np.maximum(weight, 1e-8)  # evita pesi esattamente zero (voxel "morti")

    # --- Media pesata delle feature per voxel (Eq. 1), via scatter-add ---
    C = feats.shape[1]
    weighted_feats_sum = np.zeros((num_voxels, C), dtype=np.float32)
    weights_sum = np.zeros(num_voxels, dtype=np.float32)

    np.add.at(weighted_feats_sum, voxel_id, feats * weight[:, None])
    np.add.at(weights_sum, voxel_id, weight)

    occupied = weights_sum > 0
    aggregated = np.zeros((num_voxels, C), dtype=np.float32)
    aggregated[occupied] = weighted_feats_sum[occupied] / weights_sum[occupied, None]

    volume = aggregated.reshape(num_planes, height, width, C)
    occupancy = occupied.reshape(num_planes, height, width)

    return volume, occupancy
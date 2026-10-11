# Voxelizzazione multi-piano (Sezione 3.3 del paper), versione NumPy.
# La uso per il debug, per il training c'è quella in multiplane_torch.py

import numpy as np


def scale_intrinsics(intrinsics: dict, orig_size: tuple, new_size: tuple) -> dict:
    """Riscala fx, fy, cx, cy quando si lavora a una risoluzione diversa da quella originale."""
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
    """Porta i punti da world space a camera space usando l'inversa della pose."""
    pose_world_to_cam = np.linalg.inv(pose_cam_to_world)

    n = points_world.shape[0]
    points_world_h = np.concatenate([points_world, np.ones((n, 1), dtype=np.float32)], axis=1)
    points_cam_h = (pose_world_to_cam @ points_world_h.T).T

    return points_cam_h[:, :3].astype(np.float32)


def project_to_pixels(points_cam: np.ndarray, intrinsics: dict):
    """Proiezione pinhole, ritorna u, v (float) e la z in camera space."""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    # i punti dietro la camera li filtra poi il chiamante con z <= 0
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
    Costruisce il volume di feature (P, H, W, C) per una vista.
    Ogni voxel è la media pesata dei punti che ci cadono dentro (Eq. 1 e 2 del paper).
    Ritorna il volume e la maschera di occupancy (P, H, W).
    """
    points_cam = world_to_camera(points_world, pose_cam_to_world)
    u, v, z = project_to_pixels(points_cam, intrinsics)

    # teniamo solo i punti davanti alla camera e dentro l'immagine
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

    # near e far presi dai dati come nel paper
    if near is None:
        near = float(z.min())
    if far is None:
        far = float(z.max())
    far = max(far, near + 1e-6)  # se near == far si dividerebbe per zero

    plane_idx = np.floor((z - near) / (far - near) * num_planes).astype(np.int64)
    plane_idx = np.clip(plane_idx, 0, num_planes - 1)

    # indice lineare del voxel (piano, riga, colonna)
    voxel_id = plane_idx * (height * width) + v_int * width + u_int
    num_voxels = num_planes * height * width

    # D2: distanza in profondità dal punto più vicino nello stesso voxel
    z_min_per_voxel = np.full(num_voxels, np.inf, dtype=np.float32)
    np.minimum.at(z_min_per_voxel, voxel_id, z)
    z_min_of_point = z_min_per_voxel[voxel_id]
    d2 = np.abs(z - z_min_of_point)

    # D1: distanza dal centro del pixel, normalizzata sulla massima possibile (mezza diagonale)
    du = u - (u_int + 0.5)
    dv = v - (v_int + 0.5)
    dist_pixel = np.sqrt(du ** 2 + dv ** 2)
    d1 = np.clip(dist_pixel / (np.sqrt(2) * 0.5), 0.0, 1.0)

    # Eq. 2
    weight = ((1.0 - d1) ** a) * (1.0 / (1.0 + d2)) ** b
    weight = np.maximum(weight, 1e-8)  # così nessun voxel occupato ha peso totale zero

    # Eq. 1: media pesata per voxel con scatter add
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

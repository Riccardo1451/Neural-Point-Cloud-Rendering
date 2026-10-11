# Renderer z-buffer classico usato come baseline: stessa proiezione della voxelizzazione,
# ma per ogni pixel vince il punto più vicino alla camera, senza medie pesate.

import numpy as np

from src.voxelization.multiplane import world_to_camera, project_to_pixels


def render_zbuffer(
    points_world: np.ndarray,
    colors: np.ndarray,
    pose_cam_to_world: np.ndarray,
    intrinsics: dict,
    height: int,
    width: int,
) -> tuple:
    """
    Renderizza la point cloud con i colori RGB veri dalla pose data.
    Ritorna l'immagine (H, W, 3), nera dove non c'è nessun punto, e la maschera dei pixel coperti.
    """
    points_cam = world_to_camera(points_world, pose_cam_to_world)
    u, v, z = project_to_pixels(points_cam, intrinsics)

    u_int = np.floor(u).astype(np.int64)
    v_int = np.floor(v).astype(np.int64)

    in_front = z > 1e-6
    in_image = (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)
    valid = in_front & in_image

    u_int, v_int, z = u_int[valid], v_int[valid], z[valid]
    point_colors = colors[valid]

    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    image = np.zeros((height, width, 3), dtype=np.float32)

    # invece del test z < depth_buffer punto per punto ordino dal più lontano al più vicino
    # e scrivo tutto insieme: sui pixel ripetuti resta l'ultimo scritto, cioè il più vicino
    order = np.argsort(-z)
    u_sorted = u_int[order]
    v_sorted = v_int[order]
    z_sorted = z[order]
    colors_sorted = point_colors[order]

    image[v_sorted, u_sorted] = colors_sorted
    depth_buffer[v_sorted, u_sorted] = z_sorted

    valid_mask = np.isfinite(depth_buffer)

    return image, valid_mask

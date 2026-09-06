"""
zbuffer_renderer.py

Renderer "classico": stessa identica proiezione camera pinhole usata nella
voxelizzazione (world_to_camera, project_to_pixels - riusate direttamente
da multiplane.py, stessa matematica delle Lezioni 2-3-5), ma con la
regola di selezione più semplice: per ogni pixel, vince il punto più
vicino alla camera (il classico z-buffer), niente media pesata.

E' il punto di riferimento contro cui confrontare il rendering neurale:
la differenza di robustezza al rumore tra questo e il modello allenato
e' il cuore dell'esperimento del progetto.
"""

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
    Args:
        points_world: (N, 3)
        colors: (N, 3) - qui usiamo il colore RGB vero (non le feature
                apprese: lo z-buffer classico non ha nulla da allenare,
                e' puramente geometrico)
        pose_cam_to_world: (4, 4) - la vista da cui renderizzare
        intrinsics: dict con fx, fy, cx, cy
        height, width: risoluzione dell'immagine di output

    Returns:
        image: (H, W, 3) float32, nero (0,0,0) dove nessun punto e' visibile
        valid_mask: (H, W) bool, True dove almeno un punto ha "vinto" quel pixel
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

    # Buffer di profondita': inizializzato a +infinito, ogni pixel tiene
    # traccia della profondita' minima vista finora e del colore associato -
    # esattamente l'algoritmo dello z-buffer classico.
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    image = np.zeros((height, width, 3), dtype=np.float32)

    # Ordino i punti per profondita' DECRESCENTE e li disegno in quell'ordine:
    # cosi' l'ultimo scritto (il piu' vicino) e' quello che rimane nell'immagine -
    # equivalente al test "z < depth_buffer[pixel]" ma vettorizzato, senza loop
    # Python esplicito su ogni punto (che sarebbe lentissimo su centinaia di
    # migliaia di punti).
    order = np.argsort(-z)  # decrescente: i piu' lontani per primi
    u_sorted = u_int[order]
    v_sorted = v_int[order]
    z_sorted = z[order]
    colors_sorted = point_colors[order]

    image[v_sorted, u_sorted] = colors_sorted
    depth_buffer[v_sorted, u_sorted] = z_sorted

    valid_mask = np.isfinite(depth_buffer)

    return image, valid_mask
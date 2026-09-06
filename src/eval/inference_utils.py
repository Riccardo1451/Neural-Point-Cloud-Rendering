"""
inference_utils.py

Funzioni condivise per caricare un modello allenato e la sua point cloud,
e renderizzare una singola vista - usate sia da infer.py sia da
render_novel_view.py, per non duplicare la stessa logica di caricamento.
"""

import torch

from src.models.point_features import PointFeatureTable, build_point_input_features
from src.models.render_head import MultiPlaneRenderNetwork
from src.voxelization.multiplane_torch import compute_multiplane_voxelization


def load_scene_pointcloud(pointcloud_path: str, device: torch.device):
    """Carica la point cloud fusa precalcolata di UNA scena specifica."""
    pc_data = torch.load(pointcloud_path)
    points_world = pc_data["points_world"].to(device)
    colors_np = pc_data["colors"].numpy()
    return points_world, colors_np


def load_trained_model(
    checkpoint_path: str,
    num_points: int,
    device: torch.device,
    feature_dim: int = 8,
    base_channels: int = 16,
    num_levels: int = 3,
):
    """Carica i pesi allenati (rete + feature dei punti) da un checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    point_feature_table = PointFeatureTable(num_points, feature_dim=feature_dim).to(device)
    point_feature_table.load_state_dict(checkpoint["point_features_state_dict"])

    in_channels = feature_dim + 3
    model = MultiPlaneRenderNetwork(
        in_channels=in_channels, base_channels=base_channels, num_levels=num_levels
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return point_feature_table, model, checkpoint["epoch"]


@torch.no_grad()
def render_neural_view(
    model,
    point_feature_table,
    points_world: torch.Tensor,
    pose: torch.Tensor,
    intrinsics: dict,
    height: int,
    width: int,
    num_planes: int,
):
    """
    Renderizza UNA vista con il modello allenato, per una pose qualsiasi
    (non deve necessariamente corrispondere a un frame esistente della
    sequenza - puo' essere una posa costruita/interpolata a mano).

    Returns:
        (H, W, 3) numpy array, l'immagine renderizzata
    """
    camera_position = pose[:3, 3]
    features = build_point_input_features(point_feature_table, points_world, camera_position)

    volume, _ = compute_multiplane_voxelization(
        points_world, features, pose, intrinsics,
        height=height, width=width, num_planes=num_planes,
    )
    volume_input = volume.permute(3, 0, 1, 2).unsqueeze(0)

    output = model(volume_input)
    return output["blended_image"][0].permute(1, 2, 0).cpu().numpy()


def add_gaussian_noise_to_points(points_world: torch.Tensor, sigma: float, seed: int = None) -> torch.Tensor:
    """
    Inietta rumore gaussiano isotropo (stessa deviazione standard su x, y, z)
    nelle posizioni 3D dei punti - la versione "sui punti gia' fusi" della
    stessa idea di add_synthetic_depth_noise (che invece perturbava il
    depth PRIMA dell'unprojection). Usata per l'esperimento di robustezza:
    stessa point cloud, stesso rumore, applicato identicamente sia al
    rendering neurale sia allo z-buffer di baseline - l'unica differenza
    tra i due e' come AGGREGANO punti rumorosi, non i dati stessi.

    Args:
        points_world: (N, 3)
        sigma: deviazione standard del rumore, in metri
        seed: per riproducibilita'

    Returns:
        Nuova tensor (N, 3), rumorosa - l'originale non viene modificato
    """
    if seed is not None:
        generator = torch.Generator(device=points_world.device).manual_seed(seed)
        noise = torch.randn(points_world.shape, device=points_world.device, generator=generator)
    else:
        noise = torch.randn_like(points_world)
    return points_world + noise * sigma
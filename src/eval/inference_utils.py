# Funzioni in comune tra infer.py e render_novel_view.py per caricare
# modello e point cloud e renderizzare una vista.

import torch

from src.models.point_features import PointFeatureTable, build_point_input_features
from src.models.render_head import MultiPlaneRenderNetwork
from src.voxelization.multiplane_torch import compute_multiplane_voxelization


def load_scene_pointcloud(pointcloud_path: str, device: torch.device):
    """Carica la point cloud fusa precalcolata di una scena."""
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
    """Carica rete e feature dei punti da un checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    point_feature_table = PointFeatureTable(num_points, feature_dim=feature_dim).to(device)
    point_feature_table.load_state_dict(checkpoint["point_features_state_dict"])

    in_channels = feature_dim + 3  # feature + direzione di vista
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
    """Renderizza una vista da una pose qualsiasi, anche inventata. Ritorna (H, W, 3) numpy."""
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
    Aggiunge rumore gaussiano (sigma in metri) alle posizioni dei punti già fusi.
    Serve per l'esperimento di robustezza, lo stesso rumore va sia al modello neurale che allo z-buffer.
    Ritorna un nuovo tensore, l'originale non viene toccato.
    """
    if seed is not None:
        generator = torch.Generator(device=points_world.device).manual_seed(seed)
        noise = torch.randn(points_world.shape, device=points_world.device, generator=generator)
    else:
        noise = torch.randn_like(points_world)
    return points_world + noise * sigma

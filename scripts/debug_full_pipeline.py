# Controlla che il gradiente arrivi fino a PointFeatureTable passando per
# voxelizzazione, rete e loss. Se non arriva il training non dà errori ma non impara le feature.
# Uso: python scripts/debug_full_pipeline.py

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.point_features import PointFeatureTable, build_point_input_features
from src.voxelization.multiplane_torch import compute_multiplane_voxelization
from src.models.render_head import MultiPlaneRenderNetwork


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # point cloud finta, punti a caso in un cubo davanti alla camera
    num_points = 5000
    torch.manual_seed(0)
    points_world = torch.rand(num_points, 3, device=device)
    points_world[:, 0] = points_world[:, 0] * 2 - 1  # x in [-1, 1]
    points_world[:, 1] = points_world[:, 1] * 2 - 1  # y in [-1, 1]
    points_world[:, 2] = points_world[:, 2] * 2 + 1  # z in [1, 3] (davanti alla camera)

    pose = torch.eye(4, device=device)  # camera nell'origine

    height, width, num_planes = 32, 32, 8
    intrinsics = {"fx": float(width), "fy": float(height), "cx": width / 2, "cy": height / 2}

    # sono queste che devono ricevere il gradiente
    feature_dim = 8
    point_features_table = PointFeatureTable(num_points, feature_dim).to(device)

    camera_position = pose[:3, 3]
    point_indices = torch.arange(num_points, device=device)

    features = build_point_input_features(
        point_features_table, points_world, camera_position, point_indices
    )
    print(f"Shape feature combinate (punto + direzione vista): {features.shape}")  # (N, 11)

    volume, occupancy = compute_multiplane_voxelization(
        points_world, features, pose, intrinsics,
        height=height, width=width, num_planes=num_planes,
    )
    print(f"Shape volume: {volume.shape}, voxel occupati: {occupancy.sum().item()} / {occupancy.numel()}")
    assert occupancy.sum() > 0, "Nessun voxel occupato - controlla intrinseci/pose/range punti"

    volume_input = volume.permute(3, 0, 1, 2).unsqueeze(0)  # la rete vuole (B, C, P, H, W)
    print(f"Shape input rete: {volume_input.shape}")

    in_channels = feature_dim + 3  # 8 feature + 3 direzione di vista
    model = MultiPlaneRenderNetwork(in_channels=in_channels, base_channels=16, num_levels=2).to(device)

    out = model(volume_input)
    print(f"Shape immagine renderizzata: {out['blended_image'].shape}")

    # loss finta, basta uno scalare qualsiasi per vedere se il gradiente passa
    loss = out["blended_image"].mean()
    print(f"Loss: {loss.item():.6f}")

    loss.backward()

    grad = point_features_table.features.grad
    grad_exists = grad is not None and grad.abs().sum().item() > 0
    print(f"Gradiente su PointFeatureTable.features: {'presente' if grad_exists else 'ASSENTE!'}")
    if grad_exists:
        print(f"  Norma del gradiente: {grad.norm().item():.6f}")
        n_points_with_grad = (grad.abs().sum(dim=1) > 0).sum().item()
        print(f"  Punti con gradiente non nullo: {n_points_with_grad} / {num_points}")
        print("  (atteso: solo i punti effettivamente proiettati in un voxel valido, non tutti i 5000)")

    assert grad_exists, "PROBLEMA: il gradiente non arriva alle feature dei punti!"

    print("\nPipeline end-to-end completamente differenziabile - test superato.")


if __name__ == "__main__":
    main()
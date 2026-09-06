"""
infer.py

Lo script che chiude il cerchio: carica un checkpoint allenato, sceglie
uno o piu' frame della sequenza come "vista bersaglio", e confronta tre
cose fianco a fianco:
1. Ground truth (l'immagine reale)
2. Rendering del modello neurale allenato
3. Rendering z-buffer classico (baseline), usando gli stessi punti ma
   con la regola "vince il piu' vicino" invece della rete

Calcola anche PSNR/SSIM per entrambi i metodi rispetto al ground truth.

Uso:
    python scripts/infer.py \
        --sequence_path data/raw/7scenes/chess/seq-01 \
        --pointcloud_path data/processed/chess_seq01_pointcloud.pt \
        --checkpoint_path checkpoints/checkpoint_latest.pt \
        --frame_ids 0 50 100
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.pointcloud_utils import DEFAULT_INTRINSICS, load_color, load_depth, load_pose
from src.voxelization.multiplane import scale_intrinsics
from src.voxelization.multiplane_torch import compute_multiplane_voxelization
from src.models.point_features import PointFeatureTable, build_point_input_features
from src.models.render_head import MultiPlaneRenderNetwork
from src.baseline.zbuffer_renderer import render_zbuffer
from src.eval.metrics import compute_psnr, compute_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, required=True)
    parser.add_argument("--pointcloud_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--frame_ids", type=int, nargs="+", required=True, help="Uno o piu' frame_id da renderizzare")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num_planes", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=8)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--num_levels", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="outputs/renders")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Point cloud e feature table ---
    pc_data = torch.load(args.pointcloud_path)
    points_world = pc_data["points_world"].to(device)
    colors_np = pc_data["colors"].numpy()  # servono come RGB per lo z-buffer di baseline
    points_world_np = points_world.cpu().numpy()

    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    point_feature_table = PointFeatureTable(points_world.shape[0], feature_dim=args.feature_dim).to(device)
    point_feature_table.load_state_dict(checkpoint["point_features_state_dict"])

    in_channels = args.feature_dim + 3
    model = MultiPlaneRenderNetwork(
        in_channels=in_channels, base_channels=args.base_channels, num_levels=args.num_levels
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint caricato (epoca {checkpoint['epoch']})")

    # --- Intrinseci riscalati ---
    sample_depth = load_depth(os.path.join(args.sequence_path, f"frame-{args.frame_ids[0]:06d}.depth.png"))
    orig_h, orig_w = sample_depth.shape
    intrinsics = scale_intrinsics(DEFAULT_INTRINSICS, (orig_h, orig_w), (args.height, args.width))

    os.makedirs(args.output_dir, exist_ok=True)

    for frame_id in args.frame_ids:
        frame_str = f"frame-{frame_id:06d}"
        target_color = load_color(os.path.join(args.sequence_path, f"{frame_str}.color.png"))
        pose_np = load_pose(os.path.join(args.sequence_path, f"{frame_str}.pose.txt"))
        pose = torch.from_numpy(pose_np).float().to(device)

        target_resized = cv2.resize(target_color, (args.width, args.height), interpolation=cv2.INTER_AREA)

        # --- Rendering neurale ---
        with torch.no_grad():
            camera_position = pose[:3, 3]
            features = build_point_input_features(point_feature_table, points_world, camera_position)
            volume, _ = compute_multiplane_voxelization(
                points_world, features, pose, intrinsics,
                height=args.height, width=args.width, num_planes=args.num_planes,
            )
            volume_input = volume.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, P, H, W)
            output = model(volume_input)
            neural_render = output["blended_image"][0].permute(1, 2, 0).cpu().numpy()

        # --- Rendering z-buffer di baseline (stessa risoluzione, stessa point cloud) ---
        zbuffer_render, zbuffer_mask = render_zbuffer(
            points_world_np, colors_np, pose_np, intrinsics, args.height, args.width
        )

        # --- Metriche ---
        psnr_neural = compute_psnr(neural_render, target_resized)
        psnr_zbuffer = compute_psnr(zbuffer_render, target_resized, mask=zbuffer_mask)
        ssim_neural = compute_ssim(neural_render, target_resized)
        ssim_zbuffer = compute_ssim(zbuffer_render, target_resized)

        print(f"\nFrame {frame_id}:")
        print(f"  Neurale  - PSNR: {psnr_neural:.2f} dB, SSIM: {ssim_neural:.4f}")
        print(f"  Z-buffer - PSNR: {psnr_zbuffer:.2f} dB, SSIM: {ssim_zbuffer:.4f} (calcolato solo sui pixel coperti)")

        # --- Visualizzazione a confronto ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(target_resized)
        axes[0].set_title("Ground truth")
        axes[0].axis("off")

        axes[1].imshow(np.clip(neural_render, 0, 1))
        axes[1].set_title(f"Rendering neurale\nPSNR={psnr_neural:.2f} SSIM={ssim_neural:.3f}")
        axes[1].axis("off")

        axes[2].imshow(np.clip(zbuffer_render, 0, 1))
        axes[2].set_title(f"Z-buffer classico\nPSNR={psnr_zbuffer:.2f} SSIM={ssim_zbuffer:.3f}")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"comparison_frame{frame_id:06d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Salvato: {out_path}")


if __name__ == "__main__":
    main()
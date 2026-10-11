# Voxelizza un frame dalla stessa camera che l'ha catturato: appiattendo il volume
# si dovrebbe riottenere qualcosa di simile all'immagine originale.
# Salva il grafico dei voxel occupati per piano e la ricostruzione grezza.
# Uso: python scripts/debug_voxelization.py --frame_path data/raw/7scenes/chess/seq-01 --frame_id 0

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.pointcloud_utils import (
    load_depth,
    load_color,
    load_pose,
    unproject_depth_to_pointcloud,
    DEFAULT_INTRINSICS,
)
from src.voxelization.multiplane import (
    compute_multiplane_voxelization,
    scale_intrinsics,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str, required=True)
    parser.add_argument("--frame_id", type=int, default=0)
    parser.add_argument("--height", type=int, default=120, help="Risoluzione target del volume (piu' bassa = piu' veloce da ispezionare)")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--num_planes", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    frame_str = f"frame-{args.frame_id:06d}"
    color_path = os.path.join(args.frame_path, f"{frame_str}.color.png")
    depth_path = os.path.join(args.frame_path, f"{frame_str}.depth.png")
    pose_path = os.path.join(args.frame_path, f"{frame_str}.pose.txt")

    print(f"Carico frame: {frame_str}")
    color = load_color(color_path)
    depth = load_depth(depth_path)
    pose = load_pose(pose_path)
    orig_h, orig_w = depth.shape

    # stride 1 per avere una densità realistica
    points_world, colors = unproject_depth_to_pointcloud(depth, color, pose, stride=1)
    print(f"Punti nella point cloud: {points_world.shape[0]}")

    scaled_intrinsics = scale_intrinsics(
        DEFAULT_INTRINSICS, orig_size=(orig_h, orig_w), new_size=(args.height, args.width)
    )
    print(f"Intrinseci riscalati per {args.height}x{args.width}: {scaled_intrinsics}")

    volume, occupancy = compute_multiplane_voxelization(
        points_world,
        colors,  # come feature uso direttamente l'RGB così si vede subito se torna
        pose,
        scaled_intrinsics,
        height=args.height,
        width=args.width,
        num_planes=args.num_planes,
    )
    print(f"Shape volume: {volume.shape}")
    print(f"Percentuale voxel totali occupati: {occupancy.mean() * 100:.2f}%")

    occ_per_plane = occupancy.reshape(args.num_planes, -1).sum(axis=1)
    for p in range(args.num_planes):
        print(f"  Piano {p:2d}: {occ_per_plane[p]:6d} voxel occupati")

    os.makedirs(args.output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(args.num_planes), occ_per_plane)
    ax.set_xlabel("Indice piano (0 = piu' vicino alla camera)")
    ax.set_ylabel("Numero di voxel occupati")
    ax.set_title("Distribuzione dei punti lungo i piani di profondita'")
    plt.tight_layout()
    occ_plot_path = os.path.join(args.output_dir, "voxelization_occupancy_per_plane.png")
    plt.savefig(occ_plot_path, dpi=150)
    plt.close(fig)
    print(f"Grafico occupazione salvato in: {occ_plot_path}")

    # media dei piani occupati per pixel, non è il rendering vero (quello lo fa la rete)
    occ_count = occupancy.sum(axis=0, keepdims=True).astype(np.float32)
    occ_count_safe = np.where(occ_count > 0, occ_count, 1.0)
    rough_reconstruction = volume.sum(axis=0) / occ_count_safe[0, :, :, None]

    import cv2
    color_resized = cv2.resize(color, (args.width, args.height), interpolation=cv2.INTER_AREA)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(color_resized)
    axes[0].set_title("Immagine originale (ridimensionata)")
    axes[0].axis("off")

    axes[1].imshow(np.clip(rough_reconstruction, 0, 1))
    axes[1].set_title("Ricostruzione grezza dal volume\n(media dei piani, NON il rendering finale)")
    axes[1].axis("off")

    plt.tight_layout()
    recon_path = os.path.join(args.output_dir, "voxelization_rough_reconstruction.png")
    plt.savefig(recon_path, dpi=150)
    plt.close(fig)
    print(f"Confronto ricostruzione salvato in: {recon_path}")


if __name__ == "__main__":
    main()
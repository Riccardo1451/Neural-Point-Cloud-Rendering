# Confronta la ricostruzione grezza fatta con il solo frame target e con più frame fusi,
# per vedere se la fusione chiude i buchi (schermi, superfici riflettenti...).
# Uso: python scripts/debug_voxelization_fused.py --sequence_path data/raw/7scenes/chess/seq-01 --target_frame_id 0 --num_fusion_frames 20

import argparse
import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.pointcloud_utils import (
    load_color,
    load_depth,
    load_pose,
    unproject_depth_to_pointcloud,
    DEFAULT_INTRINSICS,
)
from src.datasets.sequence_fusion import (
    list_available_frame_ids,
    select_frames_by_baseline,
    fuse_frames_to_pointcloud,
)
from src.voxelization.multiplane import (
    compute_multiplane_voxelization,
    scale_intrinsics,
)


def rough_reconstruction(volume, occupancy):
    """Media dei piani occupati per ogni pixel, come in debug_voxelization.py."""
    occ_count = occupancy.sum(axis=0, keepdims=True).astype(np.float32)
    occ_count_safe = np.where(occ_count > 0, occ_count, 1.0)
    recon = volume.sum(axis=0) / occ_count_safe[0, :, :, None]
    return recon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, required=True)
    parser.add_argument("--target_frame_id", type=int, default=0)
    parser.add_argument("--num_fusion_frames", type=int, default=20, help="Numero massimo di frame da fondere")
    parser.add_argument("--min_baseline", type=float, default=0.05, help="Distanza minima (metri) tra le posizioni di camera dei frame selezionati e il target")
    parser.add_argument("--max_baseline", type=float, default=0.5, help="Distanza massima (metri)")
    parser.add_argument("--pixel_stride", type=int, default=4, help="Sottocampionamento pixel per frame durante la fusione")
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--num_planes", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    frame_str = f"frame-{args.target_frame_id:06d}"
    target_color = load_color(os.path.join(args.sequence_path, f"{frame_str}.color.png"))
    target_depth = load_depth(os.path.join(args.sequence_path, f"{frame_str}.depth.png"))
    target_pose = load_pose(os.path.join(args.sequence_path, f"{frame_str}.pose.txt"))
    orig_h, orig_w = target_depth.shape

    scaled_intrinsics = scale_intrinsics(
        DEFAULT_INTRINSICS, orig_size=(orig_h, orig_w), new_size=(args.height, args.width)
    )

    # (a) solo il frame target
    points_single, colors_single = unproject_depth_to_pointcloud(
        target_depth, target_color, target_pose, stride=1
    )
    volume_single, occ_single = compute_multiplane_voxelization(
        points_single, colors_single, target_pose, scaled_intrinsics,
        height=args.height, width=args.width, num_planes=args.num_planes,
    )
    recon_single = rough_reconstruction(volume_single, occ_single)

    # (b) point cloud fusa da più frame
    all_frame_ids = list_available_frame_ids(args.sequence_path)
    print(f"Frame disponibili nella sequenza: {len(all_frame_ids)}")

    fusion_frame_ids = select_frames_by_baseline(
        args.sequence_path,
        all_frame_ids,
        args.target_frame_id,
        min_baseline=args.min_baseline,
        max_baseline=args.max_baseline,
        max_frames=args.num_fusion_frames,
    )
    print(f"Frame usati per la fusione ({len(fusion_frame_ids)}, baseline [{args.min_baseline}, {args.max_baseline}] m): {fusion_frame_ids}")

    points_fused, colors_fused = fuse_frames_to_pointcloud(
        args.sequence_path, fusion_frame_ids, pixel_stride=args.pixel_stride
    )
    print(f"Punti totali nella point cloud fusa: {points_fused.shape[0]}")

    volume_fused, occ_fused = compute_multiplane_voxelization(
        points_fused, colors_fused, target_pose, scaled_intrinsics,
        height=args.height, width=args.width, num_planes=args.num_planes,
    )
    recon_fused = rough_reconstruction(volume_fused, occ_fused)

    print(f"Occupazione voxel - singolo frame: {occ_single.mean() * 100:.2f}%")
    print(f"Occupazione voxel - fusione:       {occ_fused.mean() * 100:.2f}%")

    color_resized = cv2.resize(target_color, (args.width, args.height), interpolation=cv2.INTER_AREA)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(color_resized)
    axes[0].set_title("Immagine originale")
    axes[0].axis("off")

    axes[1].imshow(np.clip(recon_single, 0, 1))
    axes[1].set_title(f"Ricostruzione - solo frame {args.target_frame_id}")
    axes[1].axis("off")

    axes[2].imshow(np.clip(recon_fused, 0, 1))
    axes[2].set_title(f"Ricostruzione - {len(fusion_frame_ids)} frame fusi")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "voxelization_fusion_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confronto salvato in: {out_path}")


if __name__ == "__main__":
    main()
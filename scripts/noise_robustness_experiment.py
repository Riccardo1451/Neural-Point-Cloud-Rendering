"""
noise_robustness_experiment.py

L'ESPERIMENTO CENTRALE del progetto: per una serie crescente di livelli
di rumore sintetico iniettato nelle posizioni 3D dei punti, confronta
PSNR/SSIM del rendering neurale vs z-buffer classico rispetto al ground
truth. Se il metodo funziona come promesso dal paper, il rendering
neurale dovrebbe degradare molto piu' lentamente dello z-buffer
all'aumentare del rumore.

Uso:
    python scripts/noise_robustness_experiment.py \
        --sequence_path data/raw/7scenes/chess/seq-01 \
        --pointcloud_path data/processed/chess_seq01_pointcloud.pt \
        --checkpoint_path checkpoints/checkpoint_latest.pt \
        --frame_ids 0 50 100 150 200 \
        --noise_levels 0.0 0.01 0.02 0.05 0.1 0.2
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
from src.baseline.zbuffer_renderer import render_zbuffer
from src.eval.metrics import compute_psnr, compute_ssim
from src.eval.inference_utils import (
    load_scene_pointcloud,
    load_trained_model,
    render_neural_view,
    add_gaussian_noise_to_points,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, required=True)
    parser.add_argument("--pointcloud_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--frame_ids", type=int, nargs="+", required=True, help="Frame di test su cui mediare le metriche a ogni livello di rumore")
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2], help="Deviazioni standard del rumore, in metri")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num_planes", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=8)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--num_levels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/renders")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    points_world_clean, colors_np = load_scene_pointcloud(args.pointcloud_path, device)
    point_feature_table, model, epoch = load_trained_model(
        args.checkpoint_path, points_world_clean.shape[0], device,
        feature_dim=args.feature_dim, base_channels=args.base_channels, num_levels=args.num_levels,
    )
    print(f"Checkpoint caricato (epoca {epoch})")

    sample_depth = load_depth(os.path.join(args.sequence_path, f"frame-{args.frame_ids[0]:06d}.depth.png"))
    orig_h, orig_w = sample_depth.shape
    intrinsics = scale_intrinsics(DEFAULT_INTRINSICS, (orig_h, orig_w), (args.height, args.width))

    # --- Pre-carico ground truth e pose per i frame di test, una volta sola ---
    targets = {}
    poses = {}
    for fid in args.frame_ids:
        color = load_color(os.path.join(args.sequence_path, f"frame-{fid:06d}.color.png"))
        targets[fid] = cv2.resize(color, (args.width, args.height), interpolation=cv2.INTER_AREA)
        poses[fid] = load_pose(os.path.join(args.sequence_path, f"frame-{fid:06d}.pose.txt"))

    results = {"noise_level": [], "psnr_neural": [], "psnr_zbuffer": [], "ssim_neural": [], "ssim_zbuffer": []}

    for sigma in args.noise_levels:
        print(f"\n--- Livello di rumore: sigma={sigma} m ---")

        # STESSA point cloud rumorosa usata da ENTRAMBI i metodi, per un
        # confronto onesto (la variabile che cambia e' solo l'aggregazione)
        noisy_points = add_gaussian_noise_to_points(points_world_clean, sigma, seed=args.seed)
        noisy_points_np = noisy_points.cpu().numpy()

        psnr_neural_list, psnr_zbuffer_list = [], []
        ssim_neural_list, ssim_zbuffer_list = [], []

        for fid in args.frame_ids:
            pose_np = poses[fid]
            pose_tensor = torch.from_numpy(pose_np).float().to(device)
            target = targets[fid]

            neural_render = render_neural_view(
                model, point_feature_table, noisy_points, pose_tensor, intrinsics,
                args.height, args.width, args.num_planes,
            )
            zbuffer_render, zbuffer_mask = render_zbuffer(
                noisy_points_np, colors_np, pose_np, intrinsics, args.height, args.width
            )

            psnr_neural_list.append(compute_psnr(neural_render, target))
            psnr_zbuffer_list.append(compute_psnr(zbuffer_render, target, mask=zbuffer_mask))
            ssim_neural_list.append(compute_ssim(neural_render, target))
            ssim_zbuffer_list.append(compute_ssim(zbuffer_render, target))

        mean_psnr_neural = float(np.mean(psnr_neural_list))
        mean_psnr_zbuffer = float(np.mean(psnr_zbuffer_list))
        mean_ssim_neural = float(np.mean(ssim_neural_list))
        mean_ssim_zbuffer = float(np.mean(ssim_zbuffer_list))

        print(f"  Neurale  - PSNR medio: {mean_psnr_neural:.2f} dB, SSIM medio: {mean_ssim_neural:.4f}")
        print(f"  Z-buffer - PSNR medio: {mean_psnr_zbuffer:.2f} dB, SSIM medio: {mean_ssim_zbuffer:.4f}")

        results["noise_level"].append(sigma)
        results["psnr_neural"].append(mean_psnr_neural)
        results["psnr_zbuffer"].append(mean_psnr_zbuffer)
        results["ssim_neural"].append(mean_ssim_neural)
        results["ssim_zbuffer"].append(mean_ssim_zbuffer)

    # --- Grafici finali: PSNR e SSIM in funzione del rumore ---
    os.makedirs(args.output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(results["noise_level"], results["psnr_neural"], "o-", label="Rendering neurale")
    axes[0].plot(results["noise_level"], results["psnr_zbuffer"], "s-", label="Z-buffer classico")
    axes[0].set_xlabel("Deviazione standard del rumore (m)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR vs rumore")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results["noise_level"], results["ssim_neural"], "o-", label="Rendering neurale")
    axes[1].plot(results["noise_level"], results["ssim_zbuffer"], "s-", label="Z-buffer classico")
    axes[1].set_xlabel("Deviazione standard del rumore (m)")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM vs rumore")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "noise_robustness_experiment.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nGrafico salvato: {out_path}")

    # --- Salvo anche i numeri grezzi, utili per la presentazione/tabelle ---
    results_path = os.path.join(args.output_dir, "noise_robustness_results.npz")
    np.savez(results_path, **{k: np.array(v) for k, v in results.items()})
    print(f"Dati grezzi salvati: {results_path}")


if __name__ == "__main__":
    main()
"""
render_novel_view.py

Dimostra la sintesi di VERE nuove viste: interpola tra le pose di due
frame esistenti della sequenza (posizione: interpolazione lineare;
rotazione: SLERP, la stessa tecnica di interpolazione sferica vista per i
quaternioni) e chiede al modello di renderizzare i punti intermedi -
pose che NON esistono come frame reali nella sequenza.

Nota: per queste pose intermedie non esiste un ground truth con cui fare
un confronto quantitativo (nessun frame reale corrisponde esattamente a
quella pose) - questo e' normale, l'obiettivo qui e' qualitativo: vedere
se il modello produce una transizione plausibile e coerente tra le due
viste note, non calcolare metriche.

Uso:
    python scripts/render_novel_view.py \
        --sequence_path data/raw/7scenes/chess/seq-01 \
        --pointcloud_path data/processed/chess_seq01_pointcloud.pt \
        --checkpoint_path checkpoints/checkpoint_latest.pt \
        --frame_id_a 0 --frame_id_b 50 --num_steps 7
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.pointcloud_utils import DEFAULT_INTRINSICS, load_depth, load_pose
from src.voxelization.multiplane import scale_intrinsics
from src.eval.inference_utils import load_scene_pointcloud, load_trained_model, render_neural_view


def interpolate_pose(pose_a: np.ndarray, pose_b: np.ndarray, t: float) -> np.ndarray:
    """
    Interpola tra due pose camera-to-world.
    - Posizione: interpolazione lineare (lerp)
    - Rotazione: SLERP (interpolazione sferica) - stessa idea della Lezione 4
      sui quaternioni, qui applicata alla rotazione della camera invece che
      a un oggetto animato
    """
    R_a, R_b = pose_a[:3, :3], pose_b[:3, :3]
    t_a, t_b = pose_a[:3, 3], pose_b[:3, 3]

    rotations = Rotation.from_matrix([R_a, R_b])
    slerp = Slerp([0, 1], rotations)
    R_interp = slerp([t]).as_matrix()[0]
    t_interp = (1 - t) * t_a + t * t_b

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R_interp
    pose[:3, 3] = t_interp
    return pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, required=True)
    parser.add_argument("--pointcloud_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--frame_id_a", type=int, required=True)
    parser.add_argument("--frame_id_b", type=int, required=True)
    parser.add_argument("--num_steps", type=int, default=7, help="Quante viste intermedie generare, incluse le due pose di partenza/arrivo")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num_planes", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=8)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--num_levels", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="outputs/renders")
    parser.add_argument("--orig_height", type=int, default=480, help="Risoluzione originale del sensore (fissa per 7-Scenes)")
    parser.add_argument("--orig_width", type=int, default=640)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    points_world, _ = load_scene_pointcloud(args.pointcloud_path, device)
    point_feature_table, model, epoch = load_trained_model(
        args.checkpoint_path, points_world.shape[0], device,
        feature_dim=args.feature_dim, base_channels=args.base_channels, num_levels=args.num_levels,
    )
    print(f"Checkpoint caricato (epoca {epoch})")

    intrinsics = scale_intrinsics(DEFAULT_INTRINSICS, (args.orig_height, args.orig_width), (args.height, args.width))

    pose_a = load_pose(os.path.join(args.sequence_path, f"frame-{args.frame_id_a:06d}.pose.txt"))
    pose_b = load_pose(os.path.join(args.sequence_path, f"frame-{args.frame_id_b:06d}.pose.txt"))

    os.makedirs(args.output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, args.num_steps, figsize=(3 * args.num_steps, 3))
    for i, t in enumerate(np.linspace(0, 1, args.num_steps)):
        pose_interp = interpolate_pose(pose_a, pose_b, t)
        pose_tensor = torch.from_numpy(pose_interp).to(device)

        image = render_neural_view(
            model, point_feature_table, points_world, pose_tensor, intrinsics,
            args.height, args.width, args.num_planes,
        )

        axes[i].imshow(np.clip(image, 0, 1))
        axes[i].set_title(f"t={t:.2f}")
        axes[i].axis("off")
        print(f"  Vista t={t:.2f} renderizzata")

    plt.tight_layout()
    out_path = os.path.join(
        args.output_dir, f"novel_view_{args.frame_id_a:06d}_to_{args.frame_id_b:06d}.png"
    )
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato: {out_path}")


if __name__ == "__main__":
    main()
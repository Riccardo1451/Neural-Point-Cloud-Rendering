# Genera la point cloud di un singolo frame, stampa qualche statistica e la salva
# in .ply (da aprire con MeshLab o CloudCompare) più uno snapshot PNG.
# Uso: python scripts/debug_pointcloud.py --frame_path data/raw/7scenes/chess/seq-01 --frame_id 0

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# per poter importare src/ lanciando lo script dalla root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.pointcloud_utils import (
    load_depth,
    load_color,
    load_pose,
    unproject_depth_to_pointcloud,
)


def save_projection_snapshot(points: np.ndarray, colors: np.ndarray, output_path: str, max_points: int = 30000):
    """Salva tre proiezioni (alto, fronte, lato) in PNG con matplotlib, funziona anche senza display."""
    n = points.shape[0]
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    views = [
        ("Vista dall'alto (X-Z)", 0, 2),
        ("Vista frontale (X-Y)", 0, 1),
        ("Vista laterale (Z-Y)", 2, 1),
    ]

    for ax, (title, i, j) in zip(axes, views):
        ax.scatter(points[:, i], points[:, j], c=colors, s=0.5)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # come nelle immagini, y verso il basso

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Snapshot delle proiezioni salvato in: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame_path",
        type=str,
        required=True,
        help="Cartella della sequenza, es. data/raw/7scenes/chess/seq-01",
    )
    parser.add_argument(
        "--frame_id",
        type=int,
        default=0,
        help="Numero del frame da testare (es. 0 per frame-000000)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Sottocampionamento pixel, per non generare troppi punti nel test rapido",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/debug_pointcloud.ply",
        help="Dove salvare la point cloud generata",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Tenta di aprire la finestra interattiva Open3D (può non funzionare "
        "su alcuni setup Linux/Wayland - default: disattivato, usa un viewer esterno "
        "tipo MeshLab sul file .ply salvato)",
    )
    args = parser.parse_args()

    frame_str = f"frame-{args.frame_id:06d}"
    color_path = os.path.join(args.frame_path, f"{frame_str}.color.png")
    depth_path = os.path.join(args.frame_path, f"{frame_str}.depth.png")
    pose_path = os.path.join(args.frame_path, f"{frame_str}.pose.txt")

    for p in [color_path, depth_path, pose_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File non trovato: {p}")

    print(f"Carico frame: {frame_str}")
    color = load_color(color_path)
    depth = load_depth(depth_path)
    pose = load_pose(pose_path)

    valid_ratio = np.mean(~np.isnan(depth))
    print(f"Depth shape: {depth.shape}")
    print(f"Percentuale pixel di depth validi: {valid_ratio * 100:.1f}%")
    print(
        f"Range depth valido: [{np.nanmin(depth):.3f}, {np.nanmax(depth):.3f}] metri"
    )
    print(f"Pose camera-to-world:\n{pose}")

    points, colors = unproject_depth_to_pointcloud(
        depth, color, pose, stride=args.stride
    )
    print(f"Punti generati: {points.shape[0]}")
    print(f"Bounding box XYZ min: {points.min(axis=0)}")
    print(f"Bounding box XYZ max: {points.max(axis=0)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    o3d.io.write_point_cloud(args.output, pcd)
    print(f"Point cloud salvata in: {args.output}")

    snapshot_path = os.path.splitext(args.output)[0] + "_snapshot.png"
    save_projection_snapshot(points, colors, snapshot_path)

    # la finestra di Open3D su alcuni Linux (Wayland) va in segfault, per questo è solo con --show
    if args.show:
        print("Apro la finestra di visualizzazione Open3D (chiudila per terminare)...")
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
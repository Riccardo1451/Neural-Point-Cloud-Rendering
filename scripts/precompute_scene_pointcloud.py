# Fonde i frame di tutta la sequenza in un'unica point cloud e la salva su disco.
# Va lanciato una volta sola prima del training.
# Qui non si sceglie per baseline come nel debug: si vuole coprire tutta la scena,
# eventuali punti incoerenti tra viste li gestisce la rete con i pesi di blending.
# Uso: python scripts/precompute_scene_pointcloud.py --sequence_path data/raw/7scenes/chess/seq-01 --output_path data/processed/chess_seq01_pointcloud.pt --frame_stride 5 --pixel_stride 2

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.sequence_fusion import list_available_frame_ids, fuse_frames_to_pointcloud


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--frame_stride", type=int, default=5, help="Prende 1 frame ogni N (copertura ampia senza ridondanza eccessiva)")
    parser.add_argument("--pixel_stride", type=int, default=2, help="Sottocampionamento pixel per frame")
    parser.add_argument("--max_points", type=int, default=2_000_000, help="Se superato, sottocampiona casualmente (limite di memoria/velocita')")
    args = parser.parse_args()

    all_ids = list_available_frame_ids(args.sequence_path)
    selected_ids = all_ids[:: args.frame_stride]
    print(f"Frame totali nella sequenza: {len(all_ids)}, selezionati (stride {args.frame_stride}): {len(selected_ids)}")

    points_world, colors = fuse_frames_to_pointcloud(
        args.sequence_path, selected_ids, pixel_stride=args.pixel_stride
    )
    print(f"Punti totali dopo la fusione: {points_world.shape[0]:,}")

    if points_world.shape[0] > args.max_points:
        print(f"Sottocampiono a {args.max_points:,} punti (random)...")
        idx = torch.randperm(points_world.shape[0])[: args.max_points].numpy()
        points_world = points_world[idx]
        colors = colors[idx]

    print(f"Punti finali salvati: {points_world.shape[0]:,}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(
        {
            "points_world": torch.from_numpy(points_world).float(),
            "colors": torch.from_numpy(colors).float(),  # non usati nel training, servono a z-buffer e debug
            "sequence_path": args.sequence_path,
            "source_frame_ids": selected_ids,
        },
        args.output_path,
    )
    print(f"Salvato in: {args.output_path}")


if __name__ == "__main__":
    main()
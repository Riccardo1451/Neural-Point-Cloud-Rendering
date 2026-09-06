"""
train.py

Entry point del training. Mette insieme tutti i pezzi verificati finora:
point cloud precalcolata, feature dei punti, voxelizzazione PyTorch, rete,
perceptual loss - e li allena insieme.

Prerequisito: aver gia' lanciato precompute_scene_pointcloud.py per
generare il file di point cloud fusa.

Uso:
    python scripts/train.py \
        --sequence_path data/raw/7scenes/chess/seq-01 \
        --pointcloud_path data/processed/chess_seq01_pointcloud.pt \
        --num_epochs 30
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.pointcloud_utils import DEFAULT_INTRINSICS, load_depth
from src.datasets.scene_dataset import SceneFrameDataset, split_train_val
from src.datasets.sequence_fusion import list_available_frame_ids
from src.voxelization.multiplane import scale_intrinsics
from src.models.point_features import PointFeatureTable
from src.models.render_head import MultiPlaneRenderNetwork
from src.losses.perceptual import PerceptualLoss
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, required=True)
    parser.add_argument("--pointcloud_path", type=str, required=True)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num_planes", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=8)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--num_levels", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--val_every_n_frames", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=5, help="Ogni quante epoche salvare un checkpoint")
    parser.add_argument("--log_dir", type=str, default="outputs/logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Percorso di un checkpoint da cui riprendere")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Point cloud globale precalcolata ---
    print(f"Carico point cloud da: {args.pointcloud_path}")
    pc_data = torch.load(args.pointcloud_path)
    points_world = pc_data["points_world"]
    print(f"Punti nella scena: {points_world.shape[0]:,}")

    # --- Intrinseci riscalati alla risoluzione di training ---
    sample_depth_path = None
    all_ids = list_available_frame_ids(args.sequence_path)
    sample_depth = load_depth(os.path.join(args.sequence_path, f"frame-{all_ids[0]:06d}.depth.png"))
    orig_h, orig_w = sample_depth.shape
    intrinsics = scale_intrinsics(DEFAULT_INTRINSICS, (orig_h, orig_w), (args.height, args.width))

    # --- Split train/val e DataLoader ---
    train_ids, val_ids = split_train_val(all_ids, val_every=args.val_every_n_frames)
    print(f"Frame di training: {len(train_ids)}, validazione: {len(val_ids)}")

    train_dataset = SceneFrameDataset(args.sequence_path, train_ids, args.height, args.width)
    val_dataset = SceneFrameDataset(args.sequence_path, val_ids, args.height, args.width)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- Modello, feature dei punti, loss ---
    point_feature_table = PointFeatureTable(points_world.shape[0], feature_dim=args.feature_dim)
    in_channels = args.feature_dim + 3  # + direzione di vista

    model = MultiPlaneRenderNetwork(
        in_channels=in_channels, base_channels=args.base_channels, num_levels=args.num_levels
    )
    loss_fn = PerceptualLoss()

    trainer = Trainer(
        model=model,
        point_feature_table=point_feature_table,
        loss_fn=loss_fn,
        points_world=points_world,
        intrinsics=intrinsics,
        height=args.height,
        width=args.width,
        num_planes=args.num_planes,
        device=device,
        lr=args.lr,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    start_epoch = 0
    if args.resume_from is not None:
        start_epoch = trainer.load_checkpoint(args.resume_from) + 1

    # --- Loop di training ---
    for epoch in range(start_epoch, args.num_epochs):
        epoch_train_losses = []
        for batch_idx, (images, poses) in enumerate(train_loader):
            loss_value, _ = trainer.train_step(images, poses)
            epoch_train_losses.append(loss_value)
            trainer.writer.add_scalar("loss/train_step", loss_value, trainer.global_step)

        mean_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)

        # --- Validazione ---
        epoch_val_losses = []
        last_val_images = None
        last_val_preds = None
        for images, poses in val_loader:
            loss_value, preds = trainer.eval_step(images, poses)
            epoch_val_losses.append(loss_value)
            last_val_images, last_val_preds = images, preds

        mean_val_loss = sum(epoch_val_losses) / len(epoch_val_losses) if epoch_val_losses else float("nan")

        print(f"Epoca {epoch+1}/{args.num_epochs} - train loss: {mean_train_loss:.4f} - val loss: {mean_val_loss:.4f}")
        trainer.writer.add_scalar("loss/train_epoch", mean_train_loss, epoch)
        trainer.writer.add_scalar("loss/val_epoch", mean_val_loss, epoch)

        # --- Log visivo: confronto target vs predizione su un campione di validazione ---
        if last_val_images is not None:
            trainer.writer.add_images("val/target", last_val_images[:4], epoch)
            trainer.writer.add_images("val/predicted", last_val_preds[:4].clamp(0, 1), epoch)

        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.num_epochs:
            trainer.save_checkpoint(epoch, tag=f"epoch{epoch+1}")
            trainer.save_checkpoint(epoch, tag="latest")

    print("Training completato.")


if __name__ == "__main__":
    main()
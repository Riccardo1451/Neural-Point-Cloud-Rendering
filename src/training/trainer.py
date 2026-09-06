"""
trainer.py

Il training loop vero: per ogni batch di viste target,
1. per ciascuna vista, voxelizza la point cloud globale della scena usando
   le feature apprese correnti + la direzione di vista di quella camera
2. impila i volumi risultanti in un batch, li passa alla rete
3. confronta l'immagine renderizzata con il ground truth (perceptual loss)
4. backpropaga - il gradiente aggiorna SIA i pesi della rete SIA le feature
   dei punti (stesso optimizer, stessi passi, esattamente come descritto
   nella Sezione 3.6 del paper)
"""

import os

import torch
from torch.utils.tensorboard import SummaryWriter

from src.models.point_features import build_point_input_features
from src.voxelization.multiplane_torch import compute_multiplane_voxelization


class Trainer:
    def __init__(
        self,
        model,
        point_feature_table,
        loss_fn,
        points_world: torch.Tensor,
        intrinsics: dict,
        height: int,
        width: int,
        num_planes: int,
        device: torch.device,
        lr: float = 1e-4,
        log_dir: str = "outputs/logs",
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.point_feature_table = point_feature_table.to(device)
        self.loss_fn = loss_fn.to(device)
        self.points_world = points_world.to(device)
        self.intrinsics = intrinsics
        self.height = height
        self.width = width
        self.num_planes = num_planes
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Stesso optimizer per i pesi della rete E per le feature dei punti -
        # vengono aggiornati insieme, come descritto nel paper (Sezione 3.6)
        params = list(self.model.parameters()) + list(self.point_feature_table.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def _build_volume_batch(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Voxelizza la point cloud globale per ciascuna posa nel batch, e
        impila i risultati. Ogni vista ha una voxelizzazione DIVERSA
        (dipende dalla posizione/orientamento della camera), anche se la
        point cloud sottostante e' sempre la stessa.

        Returns:
            (B, C, P, H, W) - pronto per il forward della rete
        """
        volumes = []
        for i in range(poses.shape[0]):
            pose = poses[i]
            camera_position = pose[:3, 3]

            features = build_point_input_features(
                self.point_feature_table, self.points_world, camera_position
            )

            volume, _occupancy = compute_multiplane_voxelization(
                self.points_world,
                features,
                pose,
                self.intrinsics,
                height=self.height,
                width=self.width,
                num_planes=self.num_planes,
            )
            volumes.append(volume.permute(3, 0, 1, 2))  # (C, P, H, W)

        return torch.stack(volumes, dim=0)  # (B, C, P, H, W)

    def train_step(self, target_images: torch.Tensor, poses: torch.Tensor) -> tuple:
        """
        Un passo di training su un batch di viste.

        Args:
            target_images: (B, 3, H, W) ground truth
            poses: (B, 4, 4)

        Returns:
            (loss_value: float, predicted_images: (B, 3, H, W) tensor, staccato dal grafo)
        """
        self.model.train()
        target_images = target_images.to(self.device)
        poses = poses.to(self.device)

        volume_batch = self._build_volume_batch(poses)
        output = self.model(volume_batch)
        predicted = output["blended_image"]

        loss = self.loss_fn(predicted, target_images)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1
        return loss.item(), predicted.detach()

    @torch.no_grad()
    def eval_step(self, target_images: torch.Tensor, poses: torch.Tensor) -> tuple:
        """Come train_step, ma senza backward - per il set di validazione."""
        self.model.eval()
        target_images = target_images.to(self.device)
        poses = poses.to(self.device)

        volume_batch = self._build_volume_batch(poses)
        output = self.model(volume_batch)
        predicted = output["blended_image"]

        loss = self.loss_fn(predicted, target_images)
        return loss.item(), predicted

    def save_checkpoint(self, epoch: int, tag: str = "latest"):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "point_features_state_dict": self.point_feature_table.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"Checkpoint salvato: {path}")

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.point_feature_table.load_state_dict(checkpoint["point_features_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint caricato da: {path} (epoca {checkpoint['epoch']})")
        return checkpoint["epoch"]
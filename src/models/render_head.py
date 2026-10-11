# Rete completa: UNet3D + attivazioni sull'output + blending dei piani nell'immagine finale.

import torch
import torch.nn as nn

from src.models.unet3d import UNet3D


class MultiPlaneRenderNetwork(nn.Module):
    """Dal volume di feature (B, C, P, H, W) all'immagine renderizzata (B, 3, H, W)."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 16,
        num_levels: int = 3,
        bottleneck_dilation: int = 2,
    ):
        super().__init__()
        self.backbone = UNet3D(
            in_channels=in_channels,
            out_channels=4,  # 3 RGB + 1 peso di blending
            base_channels=base_channels,
            num_levels=num_levels,
            bottleneck_dilation=bottleneck_dilation,
        )

    def forward(self, volume: torch.Tensor) -> dict:
        """Ritorna l'immagine finale e, per il debug, RGB e pesi di ogni piano."""
        raw = self.backbone(volume)  # (B, 4, P, H, W)

        rgb = torch.sigmoid(raw[:, :3])
        alpha_logits = raw[:, 3:4]
        # softmax sui piani così per ogni pixel i pesi sommano a 1 (il paper non lo specifica)
        alpha = torch.softmax(alpha_logits, dim=2)

        blended_image = (rgb * alpha).sum(dim=2)  # (B, 3, H, W)

        return {
            "blended_image": blended_image,
            "per_plane_rgb": rgb,
            "per_plane_alpha": alpha,
        }

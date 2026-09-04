"""
render_head.py

Avvolge il backbone UNet3D con:
1. Le attivazioni corrette sui suoi 4 canali di output per voxel: RGB
   (sigmoid, per stare in [0, 1]) + peso di blending (softmax lungo la
   dimensione dei piani, cosi' i pesi di ciascun pixel sommano a 1 tra
   tutti i piani - una scelta implementativa ragionevole non specificata
   esplicitamente nel paper, ma coerente con l'idea di "quanto contribuisce
   ciascun piano" descritta nella Sezione 3.4).
2. Il blending finale: I = somma sui piani di (RGB_piano * peso_piano),
   esattamente la formula del paper.
"""

import torch
import torch.nn as nn

from src.models.unet3d import UNet3D


class MultiPlaneRenderNetwork(nn.Module):
    """
    Rete completa: dal volume di feature aggregate (output della
    voxelizzazione) all'immagine finale renderizzata.

    Args:
        in_channels: canali di input per voxel (es. 8 feature apprese del
                     punto + 3 di direzione di vista = 11, vedi Sezione 3.2)
        base_channels, num_levels, bottleneck_dilation: passati al backbone
                     UNet3D, vedi unet3d.py per i dettagli
    """

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
        """
        Args:
            volume: (B, in_channels, P, H, W) - il volume aggregato dalla
                    voxelizzazione multi-piano

        Returns:
            dict con:
                "blended_image": (B, 3, H, W) - l'immagine finale renderizzata
                "per_plane_rgb": (B, 3, P, H, W) - RGB predetto per ciascun piano
                                 (utile per debug/visualizzazione, es. per
                                 vedere cosa "vede" ogni singolo piano)
                "per_plane_alpha": (B, 1, P, H, W) - peso di blending per
                                   ciascun piano (softmax lungo P, sommano
                                   a 1 per ogni pixel)
        """
        raw = self.backbone(volume)  # (B, 4, P, H, W)

        rgb = torch.sigmoid(raw[:, :3])       # (B, 3, P, H, W)
        alpha_logits = raw[:, 3:4]            # (B, 1, P, H, W)
        alpha = torch.softmax(alpha_logits, dim=2)  # softmax lungo la dimensione P

        blended_image = (rgb * alpha).sum(dim=2)  # (B, 3, H, W)

        return {
            "blended_image": blended_image,
            "per_plane_rgb": rgb,
            "per_plane_alpha": alpha,
        }
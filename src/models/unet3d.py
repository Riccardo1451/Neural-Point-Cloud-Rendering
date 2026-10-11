# U-Net 3D che lavora sul volume della voxelizzazione (Sezione 3.4 del paper).
# Le conv 3D guardano anche i piani vicini, utile quando un punto finisce nel piano sbagliato.
# Shape dei tensori: (B, C, P, H, W), cioè la (N, C, D, H, W) di Conv3d.

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Due conv 3D con InstanceNorm e ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation  # con kernel 3 tiene invariata la dimensione
        # InstanceNorm e non BatchNorm perché il batch size è piccolo
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down3D(nn.Module):
    """MaxPool che dimezza P, H, W e poi ConvBlock3D."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.conv = ConvBlock3D(in_channels, out_channels, dilation=dilation)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    """Conv trasposta che raddoppia P, H, W, concatena la skip connection e fa un ConvBlock3D."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # dopo la concat con la skip i canali tornano in_channels
        self.conv = ConvBlock3D(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = self._pad_to_match(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

    @staticmethod
    def _pad_to_match(x, skip):
        """Se le dimensioni non erano divisibili per 2 aggiunge padding per far tornare le shape."""
        diff_d = skip.shape[2] - x.shape[2]
        diff_h = skip.shape[3] - x.shape[3]
        diff_w = skip.shape[4] - x.shape[4]
        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x = nn.functional.pad(
                x,
                [0, diff_w, 0, diff_h, 0, diff_d],  # l'ordine di F.pad parte dall'ultima dimensione (W)
            )
        return x


class UNet3D(nn.Module):
    """
    Encoder di num_levels livelli, bottleneck con conv dilatata come nel paper e decoder con skip.
    P, H, W dovrebbero essere divisibili per 2**num_levels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 4,
        base_channels: int = 16,
        num_levels: int = 3,
        bottleneck_dilation: int = 2,
    ):
        super().__init__()

        self.in_conv = ConvBlock3D(in_channels, base_channels)

        # encoder, i canali raddoppiano a ogni livello
        self.downs = nn.ModuleList()
        ch = base_channels
        for _ in range(num_levels):
            self.downs.append(Down3D(ch, ch * 2))
            ch *= 2

        self.bottleneck = ConvBlock3D(ch, ch, dilation=bottleneck_dilation)

        # decoder, i canali si dimezzano
        self.ups = nn.ModuleList()
        for _ in range(num_levels):
            self.ups.append(Up3D(ch, ch // 2))
            ch //= 2

        self.out_conv = nn.Conv3d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.in_conv(x)

        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.out_conv(x)

"""
unet3d.py

Backbone CNN 3D stile U-Net (Sezione 3.4 del paper). Opera direttamente
sul volume prodotto dalla voxelizzazione multi-piano: le convoluzioni 3D
lavorano contemporaneamente sulle dimensioni (piano di profondita', riga,
colonna), invece che solo su (riga, colonna) come farebbe una CNN 2D
classica - questo permette alla rete di sfruttare anche l'informazione
tra piani di profondita' vicini, utile per gestire il rumore di
proiezione (un punto leggermente sbagliato in profondita' "sconfina" nel
piano vicino, e la conv 3D puo' comunque usarlo).

Convenzione forma tensori: (Batch, Canali, Piani, Altezza, Larghezza)
- e' la convenzione standard di PyTorch per Conv3d: (N, C, D, H, W).
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """
    Due convoluzioni 3D consecutive con normalizzazione e attivazione.
    E' il "mattone" ripetuto sia nell'encoder sia nel decoder, come in un
    U-Net 2D classico ma esteso alla terza dimensione (piani di profondita').

    Uso InstanceNorm3d invece di BatchNorm3d: con batch size piccoli (comuni
    quando si allena su volumi 3D, che occupano molta memoria) BatchNorm
    diventa instabile, InstanceNorm e' la scelta standard in questi casi.
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation  # mantiene invariata la dimensione spaziale con kernel 3
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
    """Downsampling (dimezza P, H, W) seguito da un ConvBlock3D."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.conv = ConvBlock3D(in_channels, out_channels, dilation=dilation)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    """
    Upsampling (raddoppia P, H, W) via convoluzione trasposta, poi
    concatena la skip connection dal livello encoder corrispondente
    (l'ingrediente chiave di un U-Net: recupera dettagli fini persi
    durante il downsampling) e applica un ConvBlock3D.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # dopo la concatenazione con la skip connection i canali raddoppiano,
        # per questo il ConvBlock3D riceve in_channels (non in_channels // 2)
        self.conv = ConvBlock3D(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = self._pad_to_match(x, skip)
        x = torch.cat([skip, x], dim=1)  # concatena lungo i canali
        return self.conv(x)

    @staticmethod
    def _pad_to_match(x, skip):
        """
        Se P, H o W non sono divisibili esattamente per 2 ai vari livelli,
        upsampling e skip possono avere shape leggermente diverse - questo
        le allinea con un padding minimo, invece di far crashare la concat.
        """
        diff_d = skip.shape[2] - x.shape[2]
        diff_h = skip.shape[3] - x.shape[3]
        diff_w = skip.shape[4] - x.shape[4]
        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x = nn.functional.pad(
                x,
                [0, diff_w, 0, diff_h, 0, diff_d],  # (W_left,W_right, H_left,H_right, D_left,D_right)
            )
        return x


class UNet3D(nn.Module):
    """
    Backbone completo: encoder che scende di `num_levels` livelli
    (dimezzando P,H,W a ogni livello), un bottleneck con convoluzione
    dilatata (come specificato nel paper, per catturare piu' contesto senza
    ulteriore downsampling), e un decoder simmetrico con skip connection.

    Args:
        in_channels: canali di input per voxel (es. feature del punto +
                     direzione di vista, vedi Sezione 3.2 del paper)
        out_channels: canali di output per voxel (nel nostro caso 4:
                      RGB + peso di blending, vedi render_head.py)
        base_channels: canali del primo livello encoder, raddoppiati a
                       ogni downsampling
        num_levels: quante volte si scende (P, H, W devono essere
                    divisibili per 2**num_levels per risultati puliti)
        bottleneck_dilation: dilatazione della convoluzione nel bottleneck
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

        # --- Encoder: un Down3D per livello, raddoppiando i canali ---
        self.downs = nn.ModuleList()
        ch = base_channels
        for _ in range(num_levels):
            self.downs.append(Down3D(ch, ch * 2))
            ch *= 2

        # --- Bottleneck con convoluzione dilatata (Sezione 3.4 del paper) ---
        self.bottleneck = ConvBlock3D(ch, ch, dilation=bottleneck_dilation)

        # --- Decoder: un Up3D per livello, dimezzando i canali ---
        self.ups = nn.ModuleList()
        for _ in range(num_levels):
            self.ups.append(Up3D(ch, ch // 2))
            ch //= 2

        self.out_conv = nn.Conv3d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, P, H, W)
        Returns:
            (B, out_channels, P, H, W) - stessa risoluzione spaziale
            dell'input, grazie alle skip connection
        """
        x = self.in_conv(x)

        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.out_conv(x)
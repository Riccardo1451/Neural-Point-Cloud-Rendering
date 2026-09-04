"""
perceptual.py

Perceptual loss basata su VGG-19 (Sezione 3.5 del paper): invece di
confrontare pixel per pixel (L1/L2), confronta le feature map estratte da
una VGG-19 pre-allenata su ImageNet a diversi livelli di profondita' -
cattura similarita' percettiva (texture, forme, struttura) invece che
somiglianza pixel-esatta, che e' empiricamente piu' efficace per compiti
di rendering/generazione immagini.

Livelli usati (nomenclatura standard "reluX_Y"): input grezzo + relu1_2,
relu2_2, relu3_2, relu4_2, relu5_2 - corrispondono ai layer che il paper
chiama "conv1-2, conv2-2, ..., conv5-2" (la loss viene calcolata
sull'attivazione DOPO la ReLU che segue quella convoluzione).
"""

import torch
import torch.nn as nn
import torchvision.models as models


# Indici dei layer in vgg19(...).features corrispondenti a relu1_2, relu2_2,
# relu3_2, relu4_2, relu5_2 (verificati contro l'architettura standard di
# torchvision - vedi commento in fondo al file per l'elenco completo)
VGG19_LAYER_INDICES = {
    "relu1_2": 3,
    "relu2_2": 8,
    "relu3_2": 13,
    "relu4_2": 22,
    "relu5_2": 31,
}

# Normalizzazione ImageNet standard, richiesta dai pesi pre-allenati di VGG
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PerceptualLoss(nn.Module):
    """
    Args:
        layer_weights: dict {nome_layer: peso} - quali layer VGG usare e
                       con che peso relativo. Default: pesi uguali su tutti
                       i 5 livelli - il paper non specifica pesi espliciti,
                       questa e' una scelta implementativa ragionevole
                       (documentala come tale in sede di presentazione).
        input_pixel_weight: peso della componente L1 sui pixel grezzi
                            (il livello "input" citato nel paper, oltre
                            alle feature VGG)
    """

    def __init__(self, layer_weights: dict = None, input_pixel_weight: float = 1.0):
        super().__init__()

        if layer_weights is None:
            layer_weights = {name: 1.0 for name in VGG19_LAYER_INDICES}
        self.layer_weights = layer_weights
        self.input_pixel_weight = input_pixel_weight

        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False  # VGG e' solo un estrattore di feature fisso, non si allena

        self.vgg = vgg

        # Buffer, non parametri - si spostano automaticamente col .to(device)
        # del modulo, ma non compaiono tra i parametri allenabili
        self.register_buffer("mean", IMAGENET_MEAN)
        self.register_buffer("std", IMAGENET_STD)

        self.max_layer_idx = max(VGG19_LAYER_INDICES[name] for name in self.layer_weights)

    def _extract_features(self, x: torch.Tensor) -> dict:
        """
        Passa l'immagine attraverso VGG-19 UNA SOLA VOLTA, salvando le
        attivazioni ai layer che ci interessano lungo il percorso - molto
        piu' efficiente che rifare N forward pass separati per N layer.
        """
        x = (x - self.mean) / self.std

        features = {}
        target_indices = {idx: name for name, idx in VGG19_LAYER_INDICES.items() if name in self.layer_weights}

        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in target_indices:
                features[target_indices[i]] = x
            if i >= self.max_layer_idx:
                break

        return features

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted: (B, 3, H, W) immagine renderizzata, valori in [0, 1]
            target: (B, 3, H, W) immagine ground truth, valori in [0, 1]

        Returns:
            loss scalare
        """
        loss = self.input_pixel_weight * torch.nn.functional.l1_loss(predicted, target)

        pred_feats = self._extract_features(predicted)
        with torch.no_grad():
            target_feats = self._extract_features(target)

        for name, weight in self.layer_weights.items():
            loss = loss + weight * torch.nn.functional.l1_loss(pred_feats[name], target_feats[name])

        return loss


# Riferimento - architettura vgg19(...).features di torchvision (indici 0-36):
# 0 Conv 1 ReLU 2 Conv 3 ReLU(relu1_2) 4 MaxPool
# 5 Conv 6 ReLU 7 Conv 8 ReLU(relu2_2) 9 MaxPool
# 10 Conv 11 ReLU 12 Conv 13 ReLU(relu3_2) 14 Conv 15 ReLU 16 Conv 17 ReLU 18 MaxPool
# 19 Conv 20 ReLU 21 Conv 22 ReLU(relu4_2) 23 Conv 24 ReLU 25 Conv 26 ReLU 27 MaxPool
# 28 Conv 29 ReLU 30 Conv 31 ReLU(relu5_2) 32 Conv 33 ReLU 34 Conv 35 ReLU 36 MaxPool
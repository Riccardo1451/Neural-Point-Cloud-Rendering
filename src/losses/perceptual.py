# Perceptual loss con VGG-19 (Sezione 3.5 del paper).
# L1 sui pixel + L1 sulle feature di relu1_2, relu2_2, relu3_2, relu4_2, relu5_2.

import torch
import torch.nn as nn
import torchvision.models as models


# indici dei layer dentro vgg19().features, la lista completa è in fondo al file
VGG19_LAYER_INDICES = {
    "relu1_2": 3,
    "relu2_2": 8,
    "relu3_2": 13,
    "relu4_2": 22,
    "relu5_2": 31,
}

# normalizzazione di ImageNet, serve per i pesi pre-allenati
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PerceptualLoss(nn.Module):
    """
    layer_weights: peso di ogni layer VGG, di default tutti a 1 (il paper non dà pesi).
    input_pixel_weight: peso della L1 sui pixel.
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
            param.requires_grad = False  # la VGG resta congelata

        self.vgg = vgg

        # buffer così seguono il .to(device) ma non finiscono tra i parametri
        self.register_buffer("mean", IMAGENET_MEAN)
        self.register_buffer("std", IMAGENET_STD)

        self.max_layer_idx = max(VGG19_LAYER_INDICES[name] for name in self.layer_weights)

    def _extract_features(self, x: torch.Tensor) -> dict:
        """Un solo passaggio nella VGG salvando le attivazioni che servono."""
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
        """predicted e target sono (B, 3, H, W) con valori in [0, 1]."""
        loss = self.input_pixel_weight * torch.nn.functional.l1_loss(predicted, target)

        pred_feats = self._extract_features(predicted)
        with torch.no_grad():
            target_feats = self._extract_features(target)

        for name, weight in self.layer_weights.items():
            loss = loss + weight * torch.nn.functional.l1_loss(pred_feats[name], target_feats[name])

        return loss


# vgg19().features di torchvision, indici 0..36:
# 0 Conv 1 ReLU 2 Conv 3 ReLU(relu1_2) 4 MaxPool
# 5 Conv 6 ReLU 7 Conv 8 ReLU(relu2_2) 9 MaxPool
# 10 Conv 11 ReLU 12 Conv 13 ReLU(relu3_2) 14 Conv 15 ReLU 16 Conv 17 ReLU 18 MaxPool
# 19 Conv 20 ReLU 21 Conv 22 ReLU(relu4_2) 23 Conv 24 ReLU 25 Conv 26 ReLU 27 MaxPool
# 28 Conv 29 ReLU 30 Conv 31 ReLU(relu5_2) 32 Conv 33 ReLU 34 Conv 35 ReLU 36 MaxPool

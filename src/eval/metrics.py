# PSNR e SSIM per confrontare le immagini renderizzate con il ground truth.

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
    """
    PSNR su immagini (H, W, 3) in [0, 1].
    Con la mask si considerano solo i pixel coperti, altrimenti lo sfondo nero gonfia il punteggio.
    """
    if mask is not None:
        predicted = predicted[mask]
        target = target[mask]
    return float(peak_signal_noise_ratio(target, predicted, data_range=1.0))


def compute_ssim(predicted: np.ndarray, target: np.ndarray) -> float:
    """SSIM sull'immagine intera, qui la mask non si può usare perché lavora su finestre 2D."""
    return float(structural_similarity(target, predicted, channel_axis=-1, data_range=1.0))

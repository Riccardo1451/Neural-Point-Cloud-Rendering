"""
metrics.py

PSNR e SSIM - le due metriche standard per confrontare quantitativamente
un'immagine renderizzata con il ground truth, usate per il confronto
numerico tra rendering neurale e z-buffer classico.

- PSNR (Peak Signal-to-Noise Ratio): piu' alto e' meglio, misura l'errore
  pixel-per-pixel su scala logaritmica (dB). Semplice ma non cattura bene
  la qualita' percepita.
- SSIM (Structural Similarity Index): tra -1 e 1 (1 = identiche), confronta
  struttura/luminosita'/contrasto in finestre locali - piu' vicino a come
  un occhio umano giudica la somiglianza.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Args:
        predicted, target: (H, W, 3) float in [0, 1]
        mask: (H, W) bool opzionale - se fornita, calcola il PSNR SOLO sui
              pixel in maschera (utile per confrontare equamente z-buffer
              e rendering neurale sulle stesse zone effettivamente coperte
              da punti, escludendo lo sfondo nero comune a entrambi, che
              altrimenti gonfierebbe artificialmente il punteggio)
    """
    if mask is not None:
        predicted = predicted[mask]
        target = target[mask]
    return float(peak_signal_noise_ratio(target, predicted, data_range=1.0))


def compute_ssim(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    SSIM richiede finestre 2D spaziali, quindi NON si applica una maschera
    a pixel sparsi come per il PSNR - va calcolato sull'immagine intera
    (channel_axis=-1 indica che l'ultima dimensione sono i canali colore).
    """
    return float(structural_similarity(target, predicted, channel_axis=-1, data_range=1.0))
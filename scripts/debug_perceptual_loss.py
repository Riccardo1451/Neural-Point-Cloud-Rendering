# Test veloce della perceptual loss. La prima volta scarica i pesi di VGG-19 (circa 550 MB).
# Uso: python scripts/debug_perceptual_loss.py

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.losses.perceptual import PerceptualLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Carico VGG-19 (al primo avvio scarica i pesi pre-allenati, puo' richiedere un minuto)...")
    loss_fn = PerceptualLoss().to(device)
    print("VGG-19 caricata.")

    pred = torch.rand(2, 3, 64, 64, device=device, requires_grad=True)
    target = torch.rand(2, 3, 64, 64, device=device)

    loss = loss_fn(pred, target)
    print(f"Loss (immagini casuali, dovrebbe essere > 0): {loss.item():.6f}")
    assert loss.item() > 0, "Loss a zero con immagini casuali diverse - qualcosa non torna"

    # un'immagine confrontata con se stessa deve dare loss circa 0
    loss_self = loss_fn(target, target)
    print(f"Loss (stessa immagine confrontata con se stessa, dovrebbe essere ~0): {loss_self.item():.8f}")
    assert loss_self.item() < 1e-5, "Loss non nulla confrontando un'immagine con se stessa - bug!"

    loss.backward()
    grad_exists = pred.grad is not None and pred.grad.abs().sum().item() > 0
    print(f"Gradiente propagato su 'pred': {grad_exists}")
    assert grad_exists, "Il gradiente non si propaga - bug nel grafo computazionale"

    print("\nPerceptual loss verificata correttamente.")


if __name__ == "__main__":
    main()
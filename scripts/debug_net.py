# Test della rete da sola su un volume finto: shape di output, alpha che somma a 1,
# immagine in [0, 1], gradiente fino all'input e tempo del forward.
# Uso: python scripts/debug_net.py

import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.render_head import MultiPlaneRenderNetwork


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # P, H, W divisibili per 2**num_levels = 8, così non serve padding nella U-Net
    batch_size = 2
    in_channels = 11  # 8 feature del punto + 3 direzione di vista
    num_planes = 16
    height, width = 64, 64

    print(f"Shape input: (B={batch_size}, C={in_channels}, P={num_planes}, H={height}, W={width})")

    volume = torch.randn(batch_size, in_channels, num_planes, height, width, device=device)

    model = MultiPlaneRenderNetwork(
        in_channels=in_channels, base_channels=16, num_levels=3
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Numero di parametri: {n_params:,}")

    model.eval()
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = model(volume)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

    print(f"Tempo forward pass: {t1 - t0:.3f} s")

    print(f"Shape blended_image: {out['blended_image'].shape} (atteso: ({batch_size}, 3, {height}, {width}))")
    print(f"Shape per_plane_rgb: {out['per_plane_rgb'].shape} (atteso: ({batch_size}, 3, {num_planes}, {height}, {width}))")
    print(f"Shape per_plane_alpha: {out['per_plane_alpha'].shape} (atteso: ({batch_size}, 1, {num_planes}, {height}, {width}))")

    assert out["blended_image"].shape == (batch_size, 3, height, width), "Shape blended_image sbagliata!"
    assert out["per_plane_rgb"].shape == (batch_size, 3, num_planes, height, width), "Shape per_plane_rgb sbagliata!"
    assert out["per_plane_alpha"].shape == (batch_size, 1, num_planes, height, width), "Shape per_plane_alpha sbagliata!"

    # col softmax alpha deve sommare a 1 lungo i piani
    alpha_sum = out["per_plane_alpha"].sum(dim=2)
    max_deviation = (alpha_sum - 1.0).abs().max().item()
    print(f"Deviazione massima di sum(alpha) da 1.0: {max_deviation:.6f}")
    assert max_deviation < 1e-4, "I pesi di blending non sommano a 1 - bug nel softmax!"

    img_min = out["blended_image"].min().item()
    img_max = out["blended_image"].max().item()
    print(f"Range blended_image: [{img_min:.4f}, {img_max:.4f}] (atteso: dentro [0, 1])")
    assert 0.0 <= img_min and img_max <= 1.0, "L'immagine finale esce dal range [0, 1] - bug nel sigmoid/blending!"

    # il gradiente deve arrivare fino all'input
    model.train()
    volume_grad = torch.randn(1, in_channels, num_planes, height, width, device=device, requires_grad=True)
    out_grad = model(volume_grad)
    loss = out_grad["blended_image"].mean()
    loss.backward()
    grad_exists = volume_grad.grad is not None and volume_grad.grad.abs().sum().item() > 0
    print(f"Gradiente propagato correttamente fino all'input: {grad_exists}")
    assert grad_exists, "Il gradiente non si propaga fino all'input - problema nel grafo computazionale!"

    print("\nTutti i controlli superati.")


if __name__ == "__main__":
    main()
# Feature dei punti (Sezione 3.2 del paper): ogni punto ha un vettore a 8 dimensioni
# allenato insieme alla rete, a cui si attacca la direzione di vista.

import torch
import torch.nn as nn


class PointFeatureTable(nn.Module):
    """Tabella (num_points, feature_dim) di feature allenabili, una riga per punto."""

    def __init__(self, num_points: int, feature_dim: int = 8):
        super().__init__()
        # nn.Parameter così l'optimizer la vede insieme ai pesi della rete
        self.features = nn.Parameter(torch.randn(num_points, feature_dim) * 0.01)

    def forward(self, point_indices: torch.Tensor = None) -> torch.Tensor:
        """Ritorna le feature dei punti richiesti, o tutta la tabella se point_indices è None."""
        if point_indices is None:
            return self.features
        return self.features[point_indices]


def compute_view_directions(
    points_world: torch.Tensor, camera_position_world: torch.Tensor
) -> torch.Tensor:
    """Versore camera -> punto per ogni punto. La posizione della camera è pose[:3, 3]."""
    directions = points_world - camera_position_world.unsqueeze(0)
    norms = torch.norm(directions, dim=1, keepdim=True).clamp(min=1e-8)
    return directions / norms


def build_point_input_features(
    point_feature_table: PointFeatureTable,
    points_world: torch.Tensor,
    camera_position_world: torch.Tensor,
    point_indices: torch.Tensor = None,
) -> torch.Tensor:
    """
    Feature apprese + direzione di vista, cioè l'input della voxelizzazione (N, feature_dim + 3).
    Se point_indices è None si assume che points_world sia tutta la tabella, nello stesso ordine.
    """
    point_feats = point_feature_table(point_indices)
    view_dirs = compute_view_directions(points_world, camera_position_world)
    return torch.cat([point_feats, view_dirs], dim=1)

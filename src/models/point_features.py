"""
point_features.py

Implementa la Sezione 3.2 del paper: invece di un colore RGB fisso, ogni
punto della point cloud ha un vettore di feature a 8 dimensioni, allenabile
insieme ai pesi della rete (non calcolato, ma ottimizzato via
backpropagation - la rete impara "cosa deve rappresentare" ogni punto,
non solo come renderizzarlo). A queste si concatena la direzione di vista
normalizzata (calcolata geometricamente, non allenabile), perche' l'aspetto
di una superficie puo' dipendere da dove la si osserva (riflessi, luci
speculari).
"""

import torch
import torch.nn as nn


class PointFeatureTable(nn.Module):
    """
    Tabella di feature allenabili, una per punto della point cloud.

    E' semplicemente un tensore di parametri (B, feature_dim) ottimizzato
    dalla backpropagation esattamente come i pesi della rete - per questo
    e' un nn.Module con un nn.Parameter, non un tensore qualsiasi: cosi'
    l'optimizer lo vede automaticamente insieme al resto dei parametri.

    Args:
        num_points: quanti punti ha la point cloud (fissato in anticipo,
                    dipende dalla scena/sequenza che stai usando)
        feature_dim: dimensione del vettore di feature per punto (8 nel paper)
    """

    def __init__(self, num_points: int, feature_dim: int = 8):
        super().__init__()
        # Inizializzazione a piccola varianza, come e' prassi per feature
        # allenabili che partono "da zero" (non abbiamo nessuna informazione
        # a priori su cosa debba rappresentare ogni punto)
        self.features = nn.Parameter(torch.randn(num_points, feature_dim) * 0.01)

    def forward(self, point_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            point_indices: (N,) indici dei punti da recuperare, oppure None
                            per restituire l'intera tabella (utile quando si
                            usa tutta la point cloud in una volta)

        Returns:
            (N, feature_dim) feature dei punti richiesti
        """
        if point_indices is None:
            return self.features
        return self.features[point_indices]


def compute_view_directions(
    points_world: torch.Tensor, camera_position_world: torch.Tensor
) -> torch.Tensor:
    """
    Calcola la direzione di vista normalizzata per ogni punto: il versore
    che va dalla camera al punto (o equivalentemente dal punto alla camera,
    a seconda della convenzione - qui uso camera->punto).

    NON e' allenabile, e' pura geometria - la stessa idea della camera
    pinhole che gia' conosci, solo che qui ci interessa la *direzione*,
    non la proiezione.

    Args:
        points_world: (N, 3) coordinate world-space dei punti
        camera_position_world: (3,) posizione della camera in world-space
                                (la puoi leggere direttamente dalla pose:
                                e' la colonna di traslazione, pose[:3, 3])

    Returns:
        (N, 3) versori direzione, norma unitaria
    """
    directions = points_world - camera_position_world.unsqueeze(0)  # (N, 3)
    norms = torch.norm(directions, dim=1, keepdim=True).clamp(min=1e-8)
    return directions / norms


def build_point_input_features(
    point_feature_table: PointFeatureTable,
    points_world: torch.Tensor,
    camera_position_world: torch.Tensor,
    point_indices: torch.Tensor = None,
) -> torch.Tensor:
    """
    Combina le due cose sopra nell'input completo per la voxelizzazione:
    feature apprese del punto + direzione di vista concatenate.

    Args:
        point_feature_table: la tabella di feature allenabili
        points_world: (N, 3) posizioni world-space dei punti (per calcolare
                      la direzione di vista - servono SOLO quelli visibili
                      nella vista corrente, non l'intera point cloud)
        camera_position_world: (3,) posizione della camera per questa vista
        point_indices: (N,) indici nella tabella globale corrispondenti a
                       questi punti (se None, si assume che points_world
                       copra l'intera tabella, in ordine)

    Returns:
        (N, feature_dim + 3) feature pronte per la voxelizzazione
    """
    point_feats = point_feature_table(point_indices)  # (N, feature_dim)
    view_dirs = compute_view_directions(points_world, camera_position_world)  # (N, 3)
    return torch.cat([point_feats, view_dirs], dim=1)
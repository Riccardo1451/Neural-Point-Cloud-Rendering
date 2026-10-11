# Stessa voxelizzazione di multiplane.py ma in PyTorch, così il gradiente
# arriva fino alle feature dei punti durante il training.
# Le posizioni dei punti non sono mai allenabili, il gradiente passa solo dalle feature.

import torch


def world_to_camera(points_world: torch.Tensor, pose_cam_to_world: torch.Tensor) -> torch.Tensor:
    """Come in multiplane.py."""
    pose_world_to_cam = torch.inverse(pose_cam_to_world)
    n = points_world.shape[0]
    ones = torch.ones(n, 1, device=points_world.device, dtype=points_world.dtype)
    points_world_h = torch.cat([points_world, ones], dim=1)
    points_cam_h = (pose_world_to_cam @ points_world_h.T).T
    return points_cam_h[:, :3]


def project_to_pixels(points_cam: torch.Tensor, intrinsics: dict):
    """Come in multiplane.py."""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return u, v, z


def compute_multiplane_voxelization(
    points_world: torch.Tensor,
    features: torch.Tensor,
    pose_cam_to_world: torch.Tensor,
    intrinsics: dict,
    height: int,
    width: int,
    num_planes: int,
    near: float = None,
    far: float = None,
    a: float = 1.0,
    b: float = 1.0,
):
    """
    Versione differenziabile di compute_multiplane_voxelization in multiplane.py.
    np.add.at diventa index_add e np.minimum.at diventa scatter_reduce("amin").
    """
    device = points_world.device
    points_cam = world_to_camera(points_world, pose_cam_to_world)
    u, v, z = project_to_pixels(points_cam, intrinsics)

    in_front = z > 1e-6
    u_int = torch.floor(u).long()
    v_int = torch.floor(v).long()
    in_image = (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)
    valid = in_front & in_image

    C = features.shape[1]

    if valid.sum() == 0:
        volume = torch.zeros(num_planes, height, width, C, device=device, dtype=features.dtype)
        occupancy = torch.zeros(num_planes, height, width, dtype=torch.bool, device=device)
        return volume, occupancy

    u, v, z = u[valid], v[valid], z[valid]
    u_int, v_int = u_int[valid], v_int[valid]
    feats = features[valid]  # qui il gradiente resta attaccato alle feature

    near_t = z.min() if near is None else torch.as_tensor(near, device=device, dtype=z.dtype)
    far_t = z.max() if far is None else torch.as_tensor(far, device=device, dtype=z.dtype)
    far_t = torch.maximum(far_t, near_t + 1e-6)

    plane_idx = torch.floor((z - near_t) / (far_t - near_t) * num_planes).long()
    plane_idx = torch.clamp(plane_idx, 0, num_planes - 1)

    voxel_id = plane_idx * (height * width) + v_int * width + u_int
    num_voxels = num_planes * height * width

    # D2, la z viene dalla geometria quindi non serve gradiente
    z_min_per_voxel = torch.full((num_voxels,), float("inf"), device=device, dtype=z.dtype)
    z_min_per_voxel = z_min_per_voxel.scatter_reduce(
        0, voxel_id, z.detach(), reduce="amin", include_self=True
    )
    z_min_of_point = z_min_per_voxel[voxel_id]
    d2 = (z - z_min_of_point).abs()

    # D1
    du = u - (u_int.to(u.dtype) + 0.5)
    dv = v - (v_int.to(v.dtype) + 0.5)
    dist_pixel = torch.sqrt(du ** 2 + dv ** 2)
    d1 = torch.clamp(dist_pixel / (2 ** 0.5 * 0.5), max=1.0)

    # Eq. 2
    weight = ((1.0 - d1) ** a) * (1.0 / (1.0 + d2)) ** b
    weight = torch.clamp(weight, min=1e-8)

    # Eq. 1, index_add è differenziabile rispetto a feats
    weighted_feats_sum = torch.zeros(num_voxels, C, device=device, dtype=feats.dtype)
    weighted_feats_sum = weighted_feats_sum.index_add(0, voxel_id, feats * weight.unsqueeze(1))

    weights_sum = torch.zeros(num_voxels, device=device, dtype=weight.dtype)
    weights_sum = weights_sum.index_add(0, voxel_id, weight)

    occupied = weights_sum > 0
    # nei voxel vuoti divido per 1 invece che per 0 per evitare NaN
    weights_sum_safe = torch.where(occupied, weights_sum, torch.ones_like(weights_sum))
    aggregated = weighted_feats_sum / weights_sum_safe.unsqueeze(1)

    volume = aggregated.reshape(num_planes, height, width, C)
    occupancy = occupied.reshape(num_planes, height, width)

    return volume, occupancy

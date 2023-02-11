import einops
import torch
import numpy as np
import torch.nn.functional as F


def normal_to_depth(pt_map, n_map, fx, fy, cx, cy):
    """
    :param cx, cy, fx, fy: camera parameter
    :param pt_map: 3d point map, size: (B, 3, h, w)
    :param n_map: surface normal map, size: (B, 3, h, w)
    :return: depth from surface normal, size: (B, 1, h, w)

    """
    bs, _, h, w = pt_map.shape
    uv = np.indices((h, w), dtype=np.float32)
    uv = torch.FloatTensor(np.array([uv] * bs)).to(pt_map.device)
    u = (uv[:, 1, :, :] - einops.repeat(cx, 'bs -> bs h w', h=h, w=w))
    u = u / einops.repeat(fx, 'bs -> bs h w', h=h, w=w) * n_map[:, 0, :, :]
    v = (uv[:, 0, :, :] - einops.repeat(cy, 'bs -> bs h w', h=h, w=w))
    v = v / einops.repeat(fy, 'bs -> bs h w', h=h, w=w) * n_map[:, 1, :, :]

    n_map_right = F.pad(n_map, [0, 1, 0, 0])[:, :, :, 1:]  # shift to right
    n_map_bottom = F.pad(n_map, [0, 0, 0, 1])[:, :, 1:, :]  # shift to bottom

    pt_map_right = F.pad(pt_map, [0, 1, 0, 0])[:, :, :, 1:]
    pt_map_bottom = F.pad(pt_map, [0, 0, 0, 1])[:, :, 1:, :]

    z_right = torch.sum(n_map * pt_map_right, dim=1) / (u + v + n_map[:, 2, :, :] + 1e-6)
    z_bottom = torch.sum(n_map * pt_map_bottom, dim=1) / (u + v + n_map[:, 2, :, :] + 1e-6)

    weight_bottom = torch.sum(n_map * n_map_bottom, dim=1)
    weight_right = torch.sum(n_map * n_map_right, dim=1)
    z = (weight_right * z_right + weight_bottom * z_bottom) / (weight_right + weight_bottom + 1e-6)
    z = z.unsqueeze(1)
    return z

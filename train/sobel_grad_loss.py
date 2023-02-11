import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from train.consistency_loss import mean_on_mask


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


def sobel_normal_loss(pred, gt, mask, sobel_fn):
    ones = torch.ones_like(pred).float().to(pred.device)
    ones = torch.autograd.Variable(ones)
    depth_grad = sobel_fn(pred)
    output_grad = sobel_fn(gt)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(gt)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(gt)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(gt)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(gt)

    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

    loss_dx = torch.abs(output_grad_dx - depth_grad_dx)
    loss_dy = torch.abs(output_grad_dy - depth_grad_dy)
    loss_grad = mean_on_mask(loss_dx, mask) + mean_on_mask(loss_dy, mask)
    # loss_grad = (loss_dx[mask > 0]).mean() + (loss_dy[mask > 0]).mean()
    loss_normal = 1 - F.cosine_similarity(output_normal, depth_normal)
    b_mask = mask.squeeze(1)
    loss_normal = mean_on_mask(torch.abs(loss_normal), b_mask)
    return loss_grad + loss_normal

import torch
import torch.nn as nn
import torch.nn.functional as F

def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
    """標準的光流對齊函數 (Optical Flow Warping)"""
    # x 的形狀: (N, C, H, W)
    # flow 的形狀: (N, H, W, 2) (因為呼叫時用了 permute)
    
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float().to(x.device)
    grid.requires_grad = False
    
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    
    output = F.grid_sample(x, vgrid_scaled, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return output

def make_layer(block, num_blocks, **kwarg):
    """快速生成多層神經網路的工具"""
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)

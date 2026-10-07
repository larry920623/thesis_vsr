"""Causal encoded-depth consistency prior; no learned weights.

Cross-view z-buffer disagreement is a visibility RISK proxy, not ground-truth
occlusion. Camera motion/object motion can cause valid correspondences to differ.
"""
import torch
import torch.nn.functional as F


def sampling_grid(flow):
    if flow.ndim != 4 or flow.size(1) != 2:
        raise ValueError("flow must be [B,2,H,W] current-to-previous, in pixels")
    b, _, h, w = flow.shape
    if h < 2 or w < 2:
        raise ValueError("Confidence requires H,W >= 2")
    flow = flow.float()
    finite = torch.isfinite(flow).all(dim=1, keepdim=True)
    flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    y, x = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype), indexing="ij"
    )
    sx, sy = x[None] + flow[:, 0], y[None] + flow[:, 1]
    valid = finite & (sx[:, None] >= 0) & (sx[:, None] <= w - 1)
    valid = valid & (sy[:, None] >= 0) & (sy[:, None] <= h - 1)
    grid = torch.stack((2 * sx / (w - 1) - 1, 2 * sy / (h - 1) - 1), dim=-1)
    return grid, valid.float()


def sample_history(value, grid, mode="bilinear"):
    # Float32 geometry calculations even if a caller uses mixed precision.
    return F.grid_sample(value.float(), grid, mode=mode,
                         padding_mode="zeros", align_corners=True)


@torch.no_grad()
def depth_motion_confidence(current_depth, previous_depth, motion, tau=0.001):
    """Return confidence and diagnostics, all [B,1,H,W].

    tau is measured in ENCODED z-buffer units. 0.001 is a starting value,
    not a dataset-validated threshold. Nearest depth sampling avoids creating
    artificial interpolated depths across foreground/background edges.
    Depth 0/1 are not automatically invalid: sky/far-plane may use endpoints.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    expected = (motion.size(0), 1, *motion.shape[-2:])
    if tuple(current_depth.shape) != expected or tuple(previous_depth.shape) != expected:
        raise ValueError(f"Depths must have shape {expected}")
    grid, valid = sampling_grid(motion)
    current = current_depth.float()
    previous = previous_depth.float()
    cur_ok = torch.isfinite(current) & (current >= 0) & (current <= 1)
    prev_ok = torch.isfinite(previous) & (previous >= 0) & (previous <= 1)
    warped_ok = sample_history(prev_ok.float(), grid, "nearest")
    warped = sample_history(torch.nan_to_num(previous), grid, "nearest")
    delta = (torch.nan_to_num(current) - warped).abs()
    valid = valid * cur_ok.float() * (warped_ok > 0.5).float()
    confidence = valid * torch.exp(-delta / float(tau))
    return confidence, {"warped_depth": warped, "depth_delta": delta, "valid": valid}


@torch.no_grad()
def compose_second_order_confidence(first, previous_first, motion, second_flow):
    """C(t->t-2) = C(t->t-1) * warp(C(t-1->t-2), flow(t->t-1)).

    Explicitly validate composed t-2 coordinates as well. Do not compute an
    independent direct depth test which could bypass an unreliable t-1 hop.
    """
    grid, valid1 = sampling_grid(motion)
    _, valid2 = sampling_grid(second_flow)
    return (first * sample_history(previous_first, grid) * valid1 * valid2).clamp(0, 1)

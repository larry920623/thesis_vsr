"""Causal, streaming BasicVSR++ for zero-lookahead video super-resolution.

This module intentionally keeps the original bidirectional ``basicvsr_pp.py``
untouched.  It reuses the same SPyNet, residual blocks, pixel-shuffle blocks,
and second-order deformable alignment implementation, but replaces the four
grid-propagation branches with two forward-only branches.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basicvsr_net import ResidualBlocksWithInputConv, SPyNet
from .basicvsr_pp import SecondOrderDeformableAlignment
from .pixel_shuffle_pack import PixelShufflePack
from .utils import flow_warp


State = Dict[str, Any]


class CausalBasicVSRPlusPlus(nn.Module):
    """Forward-only BasicVSR++ with two-stage second-order propagation.

    At time ``t`` the model consumes only ``I_t`` and a state produced from
    frames ``I_0 ... I_{t-1}``.  It never reads future frames.  ``forward`` is
    a clip-mode wrapper around ``forward_step``, so training and streaming use
    exactly the same temporal update.

    Args:
        mid_channels: Number of channels in each latent feature.
        num_blocks: Residual blocks in each propagation branch.
        max_residue_magnitude: Maximum learned DCN offset residue.
        is_low_res_input: If True, produce a 4x output.  Otherwise output has
            the same spatial size as the input, matching the original model.
        spynet_pretrained: Optional SPyNet checkpoint path.
    """

    branch_names = ("forward_1", "forward_2")

    def __init__(
        self,
        mid_channels: int = 64,
        num_blocks: int = 7,
        max_residue_magnitude: int = 10,
        is_low_res_input: bool = True,
        spynet_pretrained: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input

        self.spynet = SPyNet(pretrained=spynet_pretrained)

        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5),
            )

        self.deform_align = nn.ModuleDict()
        for branch_name in self.branch_names:
            self.deform_align[branch_name] = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude,
            )

        # forward_1: [spatial, propagated_1] -> 2C
        # forward_2: [spatial, current_forward_1, propagated_2] -> 3C
        self.backbone = nn.ModuleDict(
            {
                "forward_1": ResidualBlocksWithInputConv(
                    2 * mid_channels, mid_channels, num_blocks
                ),
                "forward_2": ResidualBlocksWithInputConv(
                    3 * mid_channels, mid_channels, num_blocks
                ),
            }
        )

        # Reconstruction sees only current spatial, forward_1, and forward_2.
        self.reconstruction = ResidualBlocksWithInputConv(
            3 * mid_channels, mid_channels, 5
        )
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3
        )
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3
        )
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def _flow_input(self, frame: torch.Tensor) -> torch.Tensor:
        """Return the low-resolution RGB used by SPyNet."""
        if self.is_low_res_input:
            return frame
        h, w = frame.shape[-2:]
        return F.interpolate(
            frame,
            size=(h // 4, w // 4),
            mode="bicubic",
            align_corners=False,
        )

    @staticmethod
    def _validate_frame(frame: torch.Tensor) -> None:
        if frame.ndim != 4:
            raise ValueError(
                "forward_step expects [B, C, H, W], "
                f"but received shape {tuple(frame.shape)}"
            )
        if frame.size(1) != 3:
            raise ValueError(f"Expected 3 RGB channels, got {frame.size(1)}")

    def _validate_state(self, frame_for_flow: torch.Tensor, state: State) -> None:
        required = {"prev_lq", "prev_flow", "forward_1", "forward_2", "num_frames"}
        missing = required.difference(state)
        if missing:
            raise KeyError(f"Streaming state is missing keys: {sorted(missing)}")
        prev_lq = state["prev_lq"]
        if prev_lq.shape != frame_for_flow.shape:
            raise ValueError(
                "The new frame does not match the existing stream state: "
                f"current flow input {tuple(frame_for_flow.shape)} vs "
                f"previous {tuple(prev_lq.shape)}. Reset state for a new stream."
            )

    def _align_branch(
        self,
        branch_name: str,
        feat_current: torch.Tensor,
        branch_state: Optional[Dict[str, torch.Tensor]],
        flow_n1: Optional[torch.Tensor],
        flow_n2: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Align the two preceding hidden features to the current frame."""
        if branch_state is None:
            return feat_current.new_zeros(
                feat_current.size(0), self.mid_channels, *feat_current.shape[-2:]
            )

        feat_n1 = branch_state["prev"]
        feat_n2 = branch_state["prev2"]
        if flow_n1 is None or flow_n2 is None:
            raise RuntimeError("A non-empty temporal state requires optical flow")

        cond_n1 = flow_warp(feat_n1, flow_n1.permute(0, 2, 3, 1))
        cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
        extra_feat = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
        second_order_input = torch.cat([feat_n1, feat_n2], dim=1)
        return self.deform_align[branch_name](
            second_order_input, extra_feat, flow_n1, flow_n2
        )

    def _reconstruct(
        self,
        frame: torch.Tensor,
        spatial: torch.Tensor,
        feat_forward_1: torch.Tensor,
        feat_forward_2: torch.Tensor,
    ) -> torch.Tensor:
        hr = torch.cat([spatial, feat_forward_1, feat_forward_2], dim=1)
        hr = self.reconstruction(hr)
        hr = self.lrelu(self.upsample1(hr))
        hr = self.lrelu(self.upsample2(hr))
        hr = self.lrelu(self.conv_hr(hr))
        hr = self.conv_last(hr)
        if self.is_low_res_input:
            return hr + self.img_upsample(frame)
        return hr + frame

    def forward_step(
        self, frame: torch.Tensor, state: Optional[State] = None
    ) -> Tuple[torch.Tensor, State]:
        """Process one frame and return ``(sr_frame, next_state)``.

        Args:
            frame: Current RGB frame with shape ``[B, 3, H, W]``.
            state: State returned by the preceding call, or ``None`` at the
                beginning of a sequence.
        """
        self._validate_frame(frame)
        spatial = self.feat_extract(frame)
        frame_for_flow = self._flow_input(frame)

        flow_n1: Optional[torch.Tensor] = None
        flow_n2: Optional[torch.Tensor] = None
        if state is not None:
            self._validate_state(frame_for_flow, state)
            if frame_for_flow.size(-2) < 64 or frame_for_flow.size(-1) < 64:
                raise ValueError(
                    "SPyNet input height and width must be at least 64, got "
                    f"{tuple(frame_for_flow.shape[-2:])}"
                )

            # Current -> previous.  Both frames are already available at t.
            flow_n1 = self.spynet(frame_for_flow, state["prev_lq"])

            # Current -> t-2 = (current -> previous) composed with
            # (previous -> t-2).  At the second frame, the t-2 feature is zero.
            if state["prev_flow"] is None:
                flow_n2 = torch.zeros_like(flow_n1)
            else:
                flow_n2 = flow_n1 + flow_warp(
                    state["prev_flow"], flow_n1.permute(0, 2, 3, 1)
                )

        branch_1_state = None if state is None else state["forward_1"]
        aligned_1 = self._align_branch(
            "forward_1", spatial, branch_1_state, flow_n1, flow_n2
        )
        feat_forward_1 = aligned_1 + self.backbone["forward_1"](
            torch.cat([spatial, aligned_1], dim=1)
        )

        branch_2_state = None if state is None else state["forward_2"]
        aligned_2 = self._align_branch(
            "forward_2", spatial, branch_2_state, flow_n1, flow_n2
        )
        feat_forward_2 = aligned_2 + self.backbone["forward_2"](
            torch.cat([spatial, feat_forward_1, aligned_2], dim=1)
        )

        sr_frame = self._reconstruct(
            frame, spatial, feat_forward_1, feat_forward_2
        )

        zero_1 = torch.zeros_like(feat_forward_1)
        zero_2 = torch.zeros_like(feat_forward_2)
        next_state: State = {
            "prev_lq": frame_for_flow,
            "prev_flow": flow_n1,
            "forward_1": {
                "prev": feat_forward_1,
                "prev2": zero_1 if branch_1_state is None else branch_1_state["prev"],
            },
            "forward_2": {
                "prev": feat_forward_2,
                "prev2": zero_2 if branch_2_state is None else branch_2_state["prev"],
            },
            "num_frames": 1 if state is None else state["num_frames"] + 1,
        }
        return sr_frame, next_state

    @staticmethod
    def detach_state(state: Optional[State]) -> Optional[State]:
        """Detach a streaming state, useful for truncated-BPTT training."""
        if state is None:
            return None

        def detach(value: Any) -> Any:
            if torch.is_tensor(value):
                return value.detach()
            if isinstance(value, dict):
                return {key: detach(item) for key, item in value.items()}
            return value

        return detach(state)

    def forward(
        self,
        lqs: torch.Tensor,
        state: Optional[State] = None,
        return_state: bool = False,
    ):
        """Process a clip causally.

        Args:
            lqs: RGB sequence with shape ``[B, T, 3, H, W]``.
            state: Optional state for continuing an earlier clip.
            return_state: Also return the final state when True.

        Returns:
            By default, an SR sequence ``[B, T, 3, 4H, 4W]``.  When
            ``return_state=True``, returns ``(sequence, final_state)``.
        """
        if lqs.ndim != 5:
            raise ValueError(
                f"forward expects [B, T, C, H, W], got {tuple(lqs.shape)}"
            )
        if lqs.size(1) == 0:
            raise ValueError("The input sequence must contain at least one frame")

        outputs = []
        next_state = state
        for frame_idx in range(lqs.size(1)):
            sr_frame, next_state = self.forward_step(
                lqs[:, frame_idx], next_state
            )
            outputs.append(sr_frame)
        output_sequence = torch.stack(outputs, dim=1)
        if return_state:
            return output_sequence, next_state
        return output_sequence

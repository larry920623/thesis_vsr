"""C1: Causal BasicVSR++ driven by QRISP renderer motion vectors.

The architecture is identical to the RGB-only causal baseline except that
SPyNet is removed completely.  ``motion_t`` must be a current-to-previous flow
in LR pixel units with shape [B, 2, H, W].
"""

from typing import Optional, Tuple

import torch

from .causal_basicvsr_pp import CausalBasicVSRPlusPlus, State
from .utils import flow_warp


class CausalBasicVSRPlusPlusMV(CausalBasicVSRPlusPlus):
    """Causal two-stage BasicVSR++ using renderer MV instead of SPyNet."""

    def __init__(
        self,
        mid_channels: int = 64,
        num_blocks: int = 7,
        max_residue_magnitude: int = 10,
        is_low_res_input: bool = True,
    ) -> None:
        super().__init__(
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            max_residue_magnitude=max_residue_magnitude,
            is_low_res_input=is_low_res_input,
            spynet_pretrained=None,
        )
        # C1 must not retain unused SPyNet parameters or computation.
        del self.spynet

    @staticmethod
    def _validate_motion(frame: torch.Tensor, motion: torch.Tensor) -> None:
        if motion.ndim != 4:
            raise ValueError(
                "motion must have shape [B, 2, H, W], "
                f"but received {tuple(motion.shape)}"
            )
        expected = (frame.size(0), 2, frame.size(-2), frame.size(-1))
        if tuple(motion.shape) != expected:
            raise ValueError(
                f"Expected motion shape {expected}, got {tuple(motion.shape)}"
            )
        if motion.device != frame.device:
            raise ValueError("frame and motion must be on the same device")

    def _validate_renderer_state(self, spatial: torch.Tensor, state: State) -> None:
        required = {"prev_flow", "forward_1", "forward_2", "num_frames"}
        missing = required.difference(state)
        if missing:
            raise KeyError(f"Streaming state is missing keys: {sorted(missing)}")
        expected = (spatial.size(0), self.mid_channels, *spatial.shape[-2:])
        for branch_name in self.branch_names:
            branch = state[branch_name]
            for history_name in ("prev", "prev2"):
                if tuple(branch[history_name].shape) != expected:
                    raise ValueError(
                        f"State {branch_name}/{history_name} has shape "
                        f"{tuple(branch[history_name].shape)}, expected {expected}. "
                        "Reset state when starting a new stream."
                    )

    def forward_step(
        self,
        frame: torch.Tensor,
        motion: torch.Tensor,
        state: Optional[State] = None,
    ) -> Tuple[torch.Tensor, State]:
        """Process one RGB frame using its current-to-previous renderer MV.

        ``motion`` is ignored for the first frame because no history exists,
        but a correctly shaped tensor is still required for a uniform API.
        """
        self._validate_frame(frame)
        self._validate_motion(frame, motion)
        spatial = self.feat_extract(frame)

        flow_n1 = None
        flow_n2 = None
        if state is not None:
            self._validate_renderer_state(spatial, state)
            # Renderer MV has already been decoded into LR pixel units.
            flow_n1 = motion.to(dtype=spatial.dtype)
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

    def forward(
        self,
        lqs: torch.Tensor,
        motions: torch.Tensor,
        state: Optional[State] = None,
        return_state: bool = False,
    ):
        """Process a clip causally with renderer motion vectors.

        Args:
            lqs: RGB sequence [B, T, 3, H, W].
            motions: Current-to-previous flow [B, T, 2, H, W] in pixels.
            state: Optional state when continuing an earlier clip.
            return_state: Return the final state together with outputs.
        """
        if lqs.ndim != 5:
            raise ValueError(f"lqs must be [B, T, 3, H, W], got {tuple(lqs.shape)}")
        if motions.ndim != 5:
            raise ValueError(
                f"motions must be [B, T, 2, H, W], got {tuple(motions.shape)}"
            )
        expected = (lqs.size(0), lqs.size(1), 2, lqs.size(3), lqs.size(4))
        if tuple(motions.shape) != expected:
            raise ValueError(
                f"Expected motions shape {expected}, got {tuple(motions.shape)}"
            )
        if lqs.size(1) == 0:
            raise ValueError("The input sequence must contain at least one frame")

        outputs = []
        next_state = state
        for frame_index in range(lqs.size(1)):
            output, next_state = self.forward_step(
                lqs[:, frame_index], motions[:, frame_index], next_state
            )
            outputs.append(output)
        output_sequence = torch.stack(outputs, dim=1)
        if return_state:
            return output_sequence, next_state
        return output_sequence

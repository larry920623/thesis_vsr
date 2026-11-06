"""C3: Geometry-Conditioned Learned History Gate.

Preserve C1 DCN alignment and gate the aligned features before each backbone.
Each propagation stage has an independent learned spatial gate.

Depth input:
    Decoded QRISP encoded z-buffer [B, 1, H, W].
    Use the original decoder without per-frame normalization.

Geometry cues:
    Reuse depth_motion_confidence() from C2.
    Nearest depth sampling; confidence = valid * exp(-abs(delta_z) / tau).
    This is a consistency/risk cue, not ground-truth occlusion.

API:
    forward_step(frame, motion, depth, state=None) -> (sr, state)
    forward(lqs, motions, depths, state=None, return_state=False)
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn

from .causal_basicvsr_pp import State
from .causal_basicvsrpp_mv import CausalBasicVSRPlusPlusMV
from .utils import flow_warp
from .depth_confidence import depth_motion_confidence


class LearnedHistoryGate(nn.Module):
    """Predict one history-retention value per spatial location."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 16,
        initial_gate: float = 0.99,
    ) -> None:
        super().__init__()

        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be between 0 and 1")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")

        # current + history + abs(current - history) + confidence + valid
        input_channels = channels * 3 + 2

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )

        # Start near C1's full-history behavior.
        # Only initialize the new gate; preserve inherited C1 modules.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(
            self.net[-1].bias,
            math.log(initial_gate / (1.0 - initial_gate)),
        )

    def forward(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        gate_input = torch.cat(
            [
                current,
                history,
                torch.abs(current - history),
                confidence.to(dtype=current.dtype),
                valid.to(dtype=current.dtype),
            ],
            dim=1,
        )
        return torch.sigmoid(self.net(gate_input))


class CausalBasicVSRPlusPlusMVDepthGate(CausalBasicVSRPlusPlusMV):
    """C1 plus independent learned history gates for its two stages."""

    GATE_MODES = ("learned", "zero", "one")

    def __init__(
        self,
        mid_channels: int = 64,
        num_blocks: int = 7,
        max_residue_magnitude: int = 10,
        is_low_res_input: bool = True,
        gate_hidden_channels: int = 16,
        initial_gate: float = 0.99,
        depth_tau: float = 0.001,
        gate_mode: str = "learned",
    ) -> None:
        super().__init__(
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            max_residue_magnitude=max_residue_magnitude,
            is_low_res_input=is_low_res_input,
        )

        if not math.isfinite(depth_tau) or depth_tau <= 0:
            raise ValueError("depth_tau must be finite and positive")


        self.depth_tau = float(depth_tau)

        self.history_gates = nn.ModuleDict(
            {
                name: LearnedHistoryGate(
                    channels=mid_channels,
                    hidden_channels=gate_hidden_channels,
                    initial_gate=initial_gate,
                )
                for name in ("forward_1", "forward_2")
            }
        )

        self.set_gate_mode(gate_mode)

    def set_gate_mode(self, mode: str) -> None:
        """Choose learned gating or a fixed gate ablation."""
        if mode not in self.GATE_MODES:
            raise ValueError(
                f"gate_mode must be one of {self.GATE_MODES}, got {mode!r}"
            )
        self.gate_mode = mode

    def confidence_config(self):
        return {
            "type": "encoded_depth_feature_gate_v2",
            "depth_tau": self.depth_tau,
            "gate_mode": self.gate_mode,
            "gate_hidden_channels": (
                self.history_gates["forward_1"].net[0].out_channels
            ),
            "depth_space": "qrisp_packed_zbuffer",
            "depth_sampling": "nearest",
            "confidence_rule": "exp(-abs(delta_z) / tau)",
            "history_cue": "first_order",
        } 

    def load_c1_state_dict(self, state_dict):
        """Load a plain C1 state_dict, allowing only new gate keys.

        Pass the actual model state_dict, not a checkpoint wrapper.
        DataParallel/DDP 'module.' prefixes must be removed beforehand.
        """
        expected = {
            key
            for key in self.state_dict()
            if not key.startswith("history_gates.")
        }
        provided = set(state_dict)

        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)

        if missing or unexpected:
            raise RuntimeError(
                "C1 checkpoint does not match the inherited architecture.\n"
                f"Missing keys: {missing}\n"
                f"Unexpected keys: {unexpected}"
            )

        return self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _validate_depth(
        frame: torch.Tensor,
        depth: torch.Tensor,
    ) -> None:
        expected = (
            frame.size(0),
            1,
            frame.size(-2),
            frame.size(-1),
        )
        if tuple(depth.shape) != expected:
            raise ValueError(
                f"Expected decoded depth shape {expected}, "
                f"got {tuple(depth.shape)}"
            )
        if depth.device != frame.device:
            raise ValueError("frame and depth must be on the same device")
        if not depth.is_floating_point():
            raise TypeError(
                "depth must be decoded floating-point linear depth"
            )

    def _depth_cues(
        self,
        depth: torch.Tensor,
        motion: torch.Tensor,
        previous_depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use the same encoded-depth confidence helper as C2."""
        if previous_depth.device != depth.device:
            raise ValueError(
                "Stored depth is on a different device; reset state"
            )

        with torch.no_grad():
            with torch.autocast(
                device_type=depth.device.type,
                enabled=False,
            ):
                confidence, diagnostics = depth_motion_confidence(
                    current_depth=depth.float(),
                    previous_depth=previous_depth.float(),
                    motion=motion.float(),
                    tau=self.depth_tau,
                )

        return confidence, diagnostics["valid"]

    def _gate_history(
        self,
        branch_name: str,
        spatial: torch.Tensor,
        aligned: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        has_history: bool,
    ) -> torch.Tensor:
        if not has_history or self.gate_mode == "one":
            return aligned

        if self.gate_mode == "zero":
            return torch.zeros_like(aligned)

        gate = self.history_gates[branch_name](
            spatial, aligned, confidence, valid
        )
        return aligned * gate

    def forward_step(
        self,
        frame: torch.Tensor,
        motion: torch.Tensor,
        depth: torch.Tensor,
        state: Optional[State] = None,
    ) -> Tuple[torch.Tensor, State]:
        """Process the current frame using only current/past inputs."""
        self._validate_frame(frame)
        self._validate_motion(frame, motion)
        self._validate_depth(frame, depth)

        spatial = self.feat_extract(frame)

        if spatial.shape[-2:] != depth.shape[-2:]:
            raise ValueError(
                "This C3 implementation requires depth/MV at feature "
                "resolution; use the LR-input configuration."
            )

        flow_n1 = None
        flow_n2 = None
        confidence = spatial.new_zeros(
            spatial.size(0), 1, *spatial.shape[-2:]
        )
        valid = torch.zeros_like(confidence)

        if state is not None:
            self._validate_renderer_state(spatial, state)

            if "prev_depth" not in state:
                raise KeyError(
                    "C3 state requires prev_depth; reset state when "
                    "switching from C1/C2 to C3."
                )

            # Keep C1 flow composition unchanged.
            flow_n1 = motion.to(dtype=spatial.dtype)

            if state["prev_flow"] is None:
                flow_n2 = torch.zeros_like(flow_n1)
            else:
                flow_n2 = flow_n1 + flow_warp(
                    state["prev_flow"],
                    flow_n1.permute(0, 2, 3, 1),
                )

            if self.gate_mode == "learned":
                confidence, valid = self._depth_cues(
                    depth, motion, state["prev_depth"]
                )

        # Stage 1: original C1 DCN, then history gate.
        branch_1_state = None if state is None else state["forward_1"]

        aligned_1 = self._align_branch(
            "forward_1",
            spatial,
            branch_1_state,
            flow_n1,
            flow_n2,
        )
        aligned_1 = self._gate_history(
            "forward_1",
            spatial,
            aligned_1,
            confidence,
            valid,
            has_history=state is not None,
        )

        feat_forward_1 = aligned_1 + self.backbone["forward_1"](
            torch.cat([spatial, aligned_1], dim=1)
        )

        # Stage 2: original C1 DCN, then an independent history gate.
        branch_2_state = None if state is None else state["forward_2"]

        aligned_2 = self._align_branch(
            "forward_2",
            spatial,
            branch_2_state,
            flow_n1,
            flow_n2,
        )
        aligned_2 = self._gate_history(
            "forward_2",
            spatial,
            aligned_2,
            confidence,
            valid,
            has_history=state is not None,
        )

        feat_forward_2 = aligned_2 + self.backbone["forward_2"](
            torch.cat(
                [spatial, feat_forward_1, aligned_2],
                dim=1,
            )
        )

        sr_frame = self._reconstruct(
            frame, spatial, feat_forward_1, feat_forward_2
        )

        zero_1 = torch.zeros_like(feat_forward_1)
        zero_2 = torch.zeros_like(feat_forward_2)

        next_state: State = {
            "prev_flow": flow_n1,
            "prev_depth": depth.detach().float(),
            "forward_1": {
                "prev": feat_forward_1,
                "prev2": (
                    zero_1
                    if branch_1_state is None
                    else branch_1_state["prev"]
                ),
            },
            "forward_2": {
                "prev": feat_forward_2,
                "prev2": (
                    zero_2
                    if branch_2_state is None
                    else branch_2_state["prev"]
                ),
            },
            "num_frames": (
                1 if state is None else state["num_frames"] + 1
            ),
        }
        return sr_frame, next_state

    def forward(
        self,
        lqs: torch.Tensor,
        motions: torch.Tensor,
        depths: torch.Tensor,
        state: Optional[State] = None,
        return_state: bool = False,
    ):
        """Process a clip causally.

        Args:
            lqs: RGB [B, T, 3, H, W].
            motions: Current-to-previous LR pixel flow [B, T, 2, H, W].
            depths: Decoded linear depth [B, T, 1, H, W].
            state: C3 state from an earlier contiguous clip.
            return_state: Also return the final streaming state.
        """
        if lqs.ndim != 5:
            raise ValueError(
                f"lqs must be [B, T, 3, H, W], got {tuple(lqs.shape)}"
            )
        if lqs.size(1) == 0:
            raise ValueError("The input sequence must contain a frame")

        batch, length, _, height, width = lqs.shape

        expected_motion = (batch, length, 2, height, width)
        expected_depth = (batch, length, 1, height, width)

        if tuple(motions.shape) != expected_motion:
            raise ValueError(
                f"Expected motions {expected_motion}, "
                f"got {tuple(motions.shape)}"
            )
        if tuple(depths.shape) != expected_depth:
            raise ValueError(
                f"Expected depths {expected_depth}, "
                f"got {tuple(depths.shape)}"
            )

        outputs = []
        next_state = state

        for frame_index in range(length):
            output, next_state = self.forward_step(
                frame=lqs[:, frame_index],
                motion=motions[:, frame_index],
                depth=depths[:, frame_index],
                state=next_state,
            )
            outputs.append(output)

        output_sequence = torch.stack(outputs, dim=1)

        if return_state:
            return output_sequence, next_state
        return output_sequence
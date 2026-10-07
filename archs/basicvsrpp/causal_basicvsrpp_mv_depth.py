"""C2: depth confidence gates the two DCN history groups of saved C1.

API: forward_step(frame, motion, depth, state=None) -> (sr, state).
No new trainable weights. The C1 state_dict loads with strict=True.
"""
import torch
from mmcv.ops import modulated_deform_conv2d

from .basicvsr_pp import SecondOrderDeformableAlignment
from .causal_basicvsrpp_mv import CausalBasicVSRPlusPlusMV
from .depth_confidence import depth_motion_confidence, compose_second_order_confidence
from .utils import flow_warp


class DepthAwareSecondOrderAlignment(SecondOrderDeformableAlignment):
    def forward(self, x, extra_feat, flow_1, flow_2, confidence_1=None, confidence_2=None):
        if confidence_1 is None:
            return super().forward(x, extra_feat, flow_1, flow_2)
        out = self.conv_offset(torch.cat([extra_feat, flow_1, flow_2], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        mask = torch.sigmoid(mask)
        # The first half of input channels/deform groups belongs to t-1,
        # the second half to t-2, matching the inherited C1 offset layout.
        m1, m2 = torch.chunk(mask, 2, dim=1)
        mask = torch.cat([m1 * confidence_1.to(m1), m2 * confidence_2.to(m2)], dim=1)
        return modulated_deform_conv2d(
            x, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deform_groups
        )


class CausalBasicVSRPlusPlusMVDepth(CausalBasicVSRPlusPlusMV):
    def __init__(self, mid_channels=64, num_blocks=7, max_residue_magnitude=10,
                 is_low_res_input=True, depth_tau=0.001, gate_strength=1.0):
        if not is_low_res_input:
            raise ValueError("C2 supports the QRISP LR input / 4x output path only")
        if depth_tau <= 0 or not 0 <= gate_strength <= 1:
            raise ValueError("depth_tau > 0 and gate_strength in [0,1] required")
        super().__init__(mid_channels, num_blocks, max_residue_magnitude, is_low_res_input)
        self.depth_tau = float(depth_tau)
        self.gate_strength = float(gate_strength)
        # Same module/parameter names, shapes and architecture as C1.
        for name in self.branch_names:
            previous = self.deform_align[name]
            replacement = DepthAwareSecondOrderAlignment(
                2 * mid_channels, mid_channels, 3, padding=1, deform_groups=16,
                max_residue_magnitude=max_residue_magnitude
            )
            replacement.load_state_dict(previous.state_dict(), strict=True)
            self.deform_align[name] = replacement

    def confidence_config(self):
        return {"depth_tau": self.depth_tau, "gate_strength": self.gate_strength,
                "depth_space": "qrisp_packed_zbuffer", "depth_sampling": "nearest"}

    def _align_depth(self, name, spatial, history, flow1, flow2, c1, c2):
        if history is None:
            return torch.zeros_like(spatial)
        g1 = (1 - self.gate_strength) + self.gate_strength * c1
        g2 = (1 - self.gate_strength) + self.gate_strength * c2
        f1, f2 = history["prev"], history["prev2"]
        # Confidence maps live on CURRENT coordinates. Apply after warping
        # for offset conditions, and to destination-grid DCN masks, never to
        # unwarped source features using these destination coordinates.
        cond1 = flow_warp(f1, flow1.permute(0, 2, 3, 1)) * g1.to(f1)
        cond2 = flow_warp(f2, flow2.permute(0, 2, 3, 1)) * g2.to(f2)
        return self.deform_align[name](
            torch.cat([f1, f2], dim=1), torch.cat([cond1, spatial, cond2], dim=1),
            flow1, flow2, g1, g2
        )

    def forward_step(self, frame, motion, depth, state=None):
        self._validate_frame(frame)
        self._validate_motion(frame, motion)
        if tuple(depth.shape) != (frame.size(0), 1, *frame.shape[-2:]):
            raise ValueError("depth must be [B,1,H,W] matching the LR RGB")
        if depth.device != frame.device:
            raise ValueError("RGB/MV/depth must share device")
        if state is not None:
            for key in ("prev_depth", "prev_confidence_1"):
                if key not in state or state[key].shape != depth.shape:
                    raise ValueError("Reset stream state; C2 requires matching depth history")
        # An exact control path: gate_strength=0 executes the inherited C1.
        if self.gate_strength == 0:
            output, next_state = super().forward_step(frame, motion, state)
            next_state["prev_depth"] = depth.detach().float()
            next_state["prev_confidence_1"] = torch.zeros_like(depth, dtype=torch.float32)
            return output, next_state

        spatial = self.feat_extract(frame)
        flow1 = flow2 = None
        c1 = torch.zeros_like(depth, dtype=torch.float32)
        c2 = torch.zeros_like(c1)
        if state is not None:
            self._validate_renderer_state(spatial, state)
            flow1 = motion.to(dtype=spatial.dtype)
            c1, _ = depth_motion_confidence(depth, state["prev_depth"], motion, self.depth_tau)
            if state["prev_flow"] is None:
                flow2 = torch.zeros_like(flow1)
            else:
                flow2 = flow1 + flow_warp(state["prev_flow"], flow1.permute(0, 2, 3, 1))
                c2 = compose_second_order_confidence(
                    c1, state["prev_confidence_1"], motion, flow2
                )
        h1 = None if state is None else state["forward_1"]
        aligned1 = self._align_depth("forward_1", spatial, h1, flow1, flow2, c1, c2)
        feat1 = aligned1 + self.backbone["forward_1"](torch.cat([spatial, aligned1], dim=1))
        h2 = None if state is None else state["forward_2"]
        aligned2 = self._align_depth("forward_2", spatial, h2, flow1, flow2, c1, c2)
        feat2 = aligned2 + self.backbone["forward_2"](torch.cat([spatial, feat1, aligned2], dim=1))
        output = self._reconstruct(frame, spatial, feat1, feat2)
        next_state = {
            "prev_flow": flow1,
            "forward_1": {"prev": feat1, "prev2": torch.zeros_like(feat1) if h1 is None else h1["prev"]},
            "forward_2": {"prev": feat2, "prev2": torch.zeros_like(feat2) if h2 is None else h2["prev"]},
            "num_frames": 1 if state is None else state["num_frames"] + 1,
            "prev_depth": depth.detach().float(),
            "prev_confidence_1": c1.detach(),
        }
        return output, next_state

    def forward(self, lqs, motions, depths, state=None, return_state=False):
        if lqs.ndim != 5 or lqs.size(1) < 1:
            raise ValueError("lqs must be nonempty [B,T,3,H,W]")
        b, t, _, h, w = lqs.shape
        if tuple(motions.shape) != (b, t, 2, h, w) or tuple(depths.shape) != (b, t, 1, h, w):
            raise ValueError("Expected motions [B,T,2,H,W], depths [B,T,1,H,W]")
        outputs = []
        for i in range(t):
            y, state = self.forward_step(lqs[:, i], motions[:, i], depths[:, i], state)
            outputs.append(y)
        output = torch.stack(outputs, dim=1)
        return (output, state) if return_state else output

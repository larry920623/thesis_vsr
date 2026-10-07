import argparse
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import torch

from archs.basicvsrpp.causal_basicvsrpp_mv import CausalBasicVSRPlusPlusMV
from train_causal_basicvsrpp_mv import QRISPRendererMVDataset


def parse_args():
    parser = argparse.ArgumentParser(description="C1 renderer-MV smoke test")
    parser.add_argument(
        "--qrisp-root",
        default="/home/larry/ssd_data/sr_project/datasets/QRISP",
    )
    return parser.parse_args()


@torch.no_grad()
def stream(model, frames, motions):
    state = None
    outputs = []
    for frame_index in range(frames.size(1)):
        output, state = model.forward_step(
            frames[:, frame_index], motions[:, frame_index], state
        )
        outputs.append(output)
    return torch.stack(outputs, dim=1)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Smoke test requires CUDA and MMCV DCN")
    torch.manual_seed(0)
    device = torch.device("cuda")

    dataset = QRISPRendererMVDataset(
        args.qrisp_root,
        split="train",
        num_frames=3,
        lr_patch_size=64,
    )
    frames, motions, targets = dataset[0]
    frames = frames.unsqueeze(0).to(device)
    motions = motions.unsqueeze(0).to(device)
    targets = targets.unsqueeze(0).to(device)

    model = CausalBasicVSRPlusPlusMV(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
    ).to(device).eval()
    spynet_keys = [key for key in model.state_dict() if key.startswith("spynet.")]
    if spynet_keys:
        raise AssertionError(f"C1 still contains SPyNet parameters: {spynet_keys[:3]}")

    with torch.no_grad():
        clip_output = model(frames, motions)
        stream_output = stream(model, frames, motions)

        changed_frames = frames.clone()
        changed_motions = motions.clone()
        changed_frames[:, 2:] = torch.rand_like(changed_frames[:, 2:])
        changed_motions[:, 2:] = torch.randn_like(changed_motions[:, 2:])
        changed_output = model(changed_frames, changed_motions)

        prefix_output = model(frames[:, :2], motions[:, :2])

    clip_stream_error = (clip_output - stream_output).abs().max().item()
    future_error = (clip_output[:, :2] - changed_output[:, :2]).abs().max().item()
    prefix_error = (clip_output[:, 1] - prefix_output[:, -1]).abs().max().item()

    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(f"LR shape:       {tuple(frames.shape)}")
    print(f"MV shape:       {tuple(motions.shape)}")
    print(f"GT shape:       {tuple(targets.shape)}")
    print(f"Output shape:   {tuple(clip_output.shape)}")
    print(
        f"MV pixels:      min={motions.min().item():.3f}, "
        f"max={motions.max().item():.3f}, "
        f"mean_abs={motions.abs().mean().item():.3f}"
    )
    print(f"Parameters:     {params_m:.2f} M")
    print(f"SPyNet keys:    {len(spynet_keys)}")
    print(f"Future error:   {future_error:.3e}")
    print(f"Prefix error:   {prefix_error:.3e}")
    print(f"Stream error:   {clip_stream_error:.3e}")

    tolerance = 1e-5
    if max(future_error, prefix_error, clip_stream_error) > tolerance:
        raise AssertionError("C1 causality/equivalence smoke test failed")
    expected_output_shape = (
        frames.size(0),
        frames.size(1),
        3,
        frames.size(-2) * 4,
        frames.size(-1) * 4,
    )
    if tuple(clip_output.shape) != expected_output_shape:
        raise AssertionError(
            f"Expected output {expected_output_shape}, got {tuple(clip_output.shape)}"
        )
    print("✅ C1 Renderer-MV smoke test passed")


if __name__ == "__main__":
    main()

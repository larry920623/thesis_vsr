import argparse
import glob
import os
import time

import cv2
import lpips
import numpy as np
import torch
from thop import profile
from torchmetrics.functional.image import structural_similarity_index_measure

from archs.basicvsrpp.causal_basicvsr_pp import CausalBasicVSRPlusPlus


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify and evaluate streaming Causal BasicVSR++"
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "/home/larry/ssd_data/sr_project/checkpoints/"
            "causal_basicvsrpp/best_model.pth"
        ),
    )
    parser.add_argument(
        "--input-dir",
        default="/home/larry/ssd_data/sr_project/datasets/test_frames",
    )
    parser.add_argument(
        "--gt-dir",
        default="/home/larry/ssd_data/sr_project/datasets/test_gt",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/larry/ssd_data/sr_project/experiments/"
            "causal_basicvsrpp/spaceship_test"
        ),
    )
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--causality-frames", type=int, default=10)
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--skip-causality-tests", action="store_true")
    return parser.parse_args()


def read_rgb_sequence(directory):
    paths = sorted(glob.glob(os.path.join(directory, "*.png")))
    if not paths:
        raise FileNotFoundError(f"找不到 PNG 影像: {directory}")
    images = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"無法讀取影像: {path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(rgb.astype(np.float32) / 255.0)
    tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    return tensor.unsqueeze(0), paths


def clean_state_dict(checkpoint):
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    clean = {}
    for key, value in state_dict.items():
        if key in {"step_counter", "meta"}:
            continue
        if key.startswith("generator."):
            key = key[len("generator.") :]
        if key.startswith("module."):
            key = key[len("module.") :]
        clean[key] = value
    return clean


@torch.no_grad()
def streaming_inference(model, sequence):
    state = None
    outputs = []
    for frame_idx in range(sequence.size(1)):
        sr_frame, state = model.forward_step(sequence[:, frame_idx], state)
        outputs.append(sr_frame)
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def verify_causality(model, sequence, prefix_length=None, atol=1e-5):
    """Run future-leakage, prefix, and clip/stream equivalence tests."""
    num_frames = sequence.size(1)
    if num_frames < 2:
        raise ValueError("Causality tests require at least two input frames")
    if prefix_length is None:
        prefix_length = min(5, num_frames - 1)
    prefix_length = max(1, min(prefix_length, num_frames - 1))

    output_reference = model(sequence)
    changed_future = sequence.clone()
    changed_future[:, prefix_length:] = torch.rand_like(
        changed_future[:, prefix_length:]
    )
    output_changed = model(changed_future)
    future_error = (
        output_reference[:, :prefix_length]
        - output_changed[:, :prefix_length]
    ).abs().max().item()

    prefix_error = 0.0
    for frame_idx in range(num_frames):
        output_prefix = model(sequence[:, : frame_idx + 1])
        error = (
            output_reference[:, frame_idx] - output_prefix[:, -1]
        ).abs().max().item()
        prefix_error = max(prefix_error, error)

    output_stream = streaming_inference(model, sequence)
    stream_error = (output_reference - output_stream).abs().max().item()

    print("🧪 因果性驗證")
    print(
        f"   Future Leakage Test (前 {prefix_length} 幀): "
        f"max error = {future_error:.3e}"
    )
    print(f"   Prefix Test:                         max error = {prefix_error:.3e}")
    print(f"   Clip/Streaming Equivalence:          max error = {stream_error:.3e}")

    errors = {
        "future_leakage": future_error,
        "prefix": prefix_error,
        "clip_stream": stream_error,
    }
    failed = {name: value for name, value in errors.items() if value > atol}
    if failed:
        raise AssertionError(f"因果性驗證失敗 (atol={atol}): {failed}")
    print(f"✅ 全部通過 (atol={atol})")


@torch.no_grad()
def benchmark_streaming(
    model, sequence, warmup_repeats=1, benchmark_repeats=3
):
    if benchmark_repeats < 1:
        raise ValueError("benchmark_repeats must be at least 1")

    for _ in range(warmup_repeats):
        streaming_inference(model, sequence)
    torch.cuda.synchronize()

    all_times = []
    final_output = None
    for _ in range(benchmark_repeats):
        state = None
        outputs = []
        events = []
        for frame_idx in range(sequence.size(1)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            sr_frame, state = model.forward_step(sequence[:, frame_idx], state)
            end.record()
            events.append((start, end))
            outputs.append(sr_frame)
        torch.cuda.synchronize()
        all_times.append([start.elapsed_time(end) for start, end in events])
        final_output = torch.stack(outputs, dim=1)

    timings = np.asarray(all_times, dtype=np.float64)
    first_frame_ms = float(timings[:, 0].mean())
    if timings.shape[1] > 1:
        steady_timings = timings[:, 1:].reshape(-1)
    else:
        steady_timings = timings.reshape(-1)
    steady_ms = float(steady_timings.mean())
    steady_std_ms = float(steady_timings.std())
    return final_output, first_frame_ms, steady_ms, steady_std_ms


def save_outputs(outputs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images = outputs.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()
    for frame_idx, image in enumerate(images):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        path = os.path.join(output_dir, f"spaceship_out_{frame_idx:04d}.png")
        if not cv2.imwrite(path, bgr):
            raise RuntimeError(f"輸出影像寫入失敗: {path}")


@torch.no_grad()
def evaluate(outputs, gt_sequence):
    if outputs.shape != gt_sequence.shape:
        raise ValueError(
            f"Output/GT shape 不一致: {tuple(outputs.shape)} vs "
            f"{tuple(gt_sequence.shape)}"
        )

    lpips_fn = lpips.LPIPS(net="alex").to(outputs.device).eval()
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "std_diff": 0.0}
    num_frames = outputs.size(1)
    for frame_idx in range(num_frames):
        output = outputs[:, frame_idx]
        target = gt_sequence[:, frame_idx]

        mse = torch.mean((output - target) ** 2)
        psnr = (-10.0 * torch.log10(mse.clamp_min(1e-12))).item()
        ssim = structural_similarity_index_measure(
            output, target, data_range=1.0
        ).item()
        perceptual = lpips_fn(output * 2.0 - 1.0, target * 2.0 - 1.0).item()
        # This is a spatial contrast statistic, not temporal consistency.
        std_diff = abs(output.std().item() - target.std().item())

        print(
            f"Frame {frame_idx:04d} | PSNR {psnr:.2f} dB | "
            f"SSIM {ssim:.4f} | LPIPS {perceptual:.4f} | "
            f"Spatial Std Diff {std_diff:.4f}"
        )
        totals["psnr"] += psnr
        totals["ssim"] += ssim
        totals["lpips"] += perceptual
        totals["std_diff"] += std_diff

    print("-" * 72)
    print("🏆 序列平均成績")
    print(f"Mean PSNR:             {totals['psnr'] / num_frames:.2f} dB")
    print(f"Mean SSIM:             {totals['ssim'] / num_frames:.4f}")
    print(f"Mean LPIPS:            {totals['lpips'] / num_frames:.4f}")
    print(f"Mean Spatial Std Diff: {totals['std_diff'] / num_frames:.4f}")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("此推論程式需要 CUDA 與可用的 MMCV DCN operator")

    device = torch.device("cuda")
    print("🔄 初始化 RGB-only Causal BasicVSR++...")
    model = CausalBasicVSRPlusPlus(
        mid_channels=64, num_blocks=7, is_low_res_input=True
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint), strict=True)
    model = model.to(device).eval()
    print(f"📦 已載入 causal checkpoint: {args.checkpoint}")

    inputs, input_paths = read_rgb_sequence(args.input_dir)
    inputs = inputs.to(device)
    print(f"📂 輸入 {len(input_paths)} 幀，tensor shape = {tuple(inputs.shape)}")

    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"🧠 Parameters: {total_params / 1e6:.2f} M")

    if not args.skip_causality_tests:
        test_length = min(args.causality_frames, inputs.size(1))
        verify_causality(model, inputs[:, :test_length])

    if not args.skip_flops:
        print("🧮 估算 FLOPs（THOP 可能低估自訂 MMCV DCN operator）...")
        start_time = time.time()
        try:
            macs, _ = profile(model, inputs=(inputs,), verbose=False)
            flops_per_frame = macs * 2.0 / 1e9 / inputs.size(1)
            print(f"⚡ Approx. FLOPs/Frame: {flops_per_frame:.2f} G")
        except Exception as error:
            print(f"⚠️ THOP 無法分析此模型，略過 FLOPs: {error}")
        print(f"   FLOPs profiling wall time: {time.time() - start_time:.1f} s")

    outputs, first_ms, steady_ms, steady_std_ms = benchmark_streaming(
        model,
        inputs,
        warmup_repeats=args.warmup_repeats,
        benchmark_repeats=args.benchmark_repeats,
    )
    print("⏱️ 真正逐幀 streaming latency")
    print(f"   First frame (無 flow): {first_ms:.2f} ms")
    print(f"   Steady state (t>=1):  {steady_ms:.2f} ± {steady_std_ms:.2f} ms")
    print(f"   Steady-state FPS:     {1000.0 / steady_ms:.2f}")
    print(f"📐 輸出 shape: {tuple(outputs.shape)}")

    save_outputs(outputs, args.output_dir)
    print(f"💾 已儲存輸出至: {args.output_dir}")

    try:
        gt_sequence, gt_paths = read_rgb_sequence(args.gt_dir)
    except FileNotFoundError:
        print("⚠️ 找不到 GT，略過 PSNR/SSIM/LPIPS。")
        print(
            "Spatial output std（只代表對比/紋理統計）: "
            f"{outputs.std().item():.4f}"
        )
        return

    if len(gt_paths) != len(input_paths):
        raise ValueError(
            f"LR 有 {len(input_paths)} 幀，但 GT 有 {len(gt_paths)} 幀"
        )
    evaluate(outputs, gt_sequence.to(device))


if __name__ == "__main__":
    main()

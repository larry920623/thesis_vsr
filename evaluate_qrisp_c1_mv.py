"""Unified QRISP evaluation for BasicVSR++, C0, and Renderer-MV C1.

All models use the same test sequences and the same metric implementation.
The script writes per-sequence results, aggregate results, and a Markdown table.
"""

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# OpenCV disables EXR unless this flag is set before importing cv2.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import lpips
import numpy as np
import torch
from thop import profile
from torchmetrics.functional.image import structural_similarity_index_measure

from archs.basicvsrpp.basicvsr_pp import BasicVSRPlusPlus
from archs.basicvsrpp.causal_basicvsr_pp import CausalBasicVSRPlusPlus
from archs.basicvsrpp.causal_basicvsrpp_mv import CausalBasicVSRPlusPlusMV


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class SequenceSample:
    scene: str
    angle: str
    lr_paths: List[Path]
    gt_paths: List[Path]
    motion_paths: List[Path]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified C1 evaluation for BasicVSR++, C0, and Renderer-MV C1"
    )
    parser.add_argument(
        "--qrisp-root",
        default="/home/larry/ssd_data/sr_project/datasets/QRISP",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument(
        "--bidirectional-checkpoint",
        default=(
            "/home/larry/ssd_data/sr_project/checkpoints/"
            "basicvsrpp/best_model_2.pth"
        ),
    )
    parser.add_argument(
        "--causal-checkpoint",
        default=(
            "/home/larry/ssd_data/sr_project/checkpoints/"
            "causal_basicvsrpp/best_model.pth"
        ),
    )
    parser.add_argument(
        "--renderer-mv-checkpoint",
        default=(
            "/home/larry/ssd_data/sr_project/checkpoints/"
            "causal_basicvsrpp_mv/best_model.pth"
        ),
    )
    parser.add_argument(
        "--motion-modality",
        default="MotionVectorsMipBiasMinus2",
        help="QRISP 270p motion-vector directory name.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/larry/ssd_data/sr_project/experiments/"
            "c1_renderer_mv_evaluation"
        ),
    )
    parser.add_argument(
        "--models",
        choices=("all", "bidirectional", "causal", "renderer_mv", "causal_vs_mv"),
        default="renderer_mv",
        help=(
            "renderer_mv: only C1; causal_vs_mv: C0 and C1; "
            "all: BasicVSR++, C0, and C1."
        ),
    )
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--runtime-repeats", type=int, default=3)
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Only evaluate the first N sequences. Useful for a smoke test.",
    )
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--skip-flops", action="store_true")
    return parser.parse_args()


def list_files(directory: Path, extensions) -> Dict[str, Path]:
    if not directory.is_dir():
        return {}
    images = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            images[path.stem] = path
    return images


def discover_sequences(
    qrisp_root: Path,
    split: str,
    num_frames: int,
    max_sequences: Optional[int],
    motion_modality: str,
) -> List[SequenceSample]:
    list_path = qrisp_root / f"{split}_list.txt"
    if not list_path.is_file():
        raise FileNotFoundError(f"找不到 split 清單: {list_path}")

    scenes = [
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples: List[SequenceSample] = []

    for scene in scenes:
        lr_native = qrisp_root / scene / "270p" / "Native"
        gt_native = qrisp_root / scene / "1080p" / "Native"
        motion_root = qrisp_root / scene / "270p" / motion_modality
        if not lr_native.is_dir() or not gt_native.is_dir() or not motion_root.is_dir():
            print(
                f"⚠️ 跳過 {scene}: 找不到 270p/Native、1080p/Native "
                f"或 270p/{motion_modality}"
            )
            continue

        angles = sorted(
            set(path.name for path in lr_native.iterdir() if path.is_dir())
            & set(path.name for path in gt_native.iterdir() if path.is_dir())
            & set(path.name for path in motion_root.iterdir() if path.is_dir())
        )
        for angle in angles:
            lr_by_stem = list_files(lr_native / angle, VALID_IMAGE_EXTENSIONS)
            gt_by_stem = list_files(gt_native / angle, VALID_IMAGE_EXTENSIONS)
            motion_by_stem = list_files(motion_root / angle, {".exr"})
            common_stems = sorted(
                set(lr_by_stem) & set(gt_by_stem) & set(motion_by_stem)
            )
            if len(common_stems) < num_frames:
                print(
                    f"⚠️ 跳過 {scene}/{angle}: 只有 {len(common_stems)} 組配對影格，"
                    f"需要 {num_frames} 組"
                )
                continue

            # Every model receives the same centered, continuous clip.
            start = (len(common_stems) - num_frames) // 2
            selected = common_stems[start : start + num_frames]
            samples.append(
                SequenceSample(
                    scene=scene,
                    angle=angle,
                    lr_paths=[lr_by_stem[stem] for stem in selected],
                    gt_paths=[gt_by_stem[stem] for stem in selected],
                    motion_paths=[motion_by_stem[stem] for stem in selected],
                )
            )
            if max_sequences is not None and len(samples) >= max_sequences:
                return samples

    if not samples:
        raise RuntimeError("沒有找到任何可評估的 QRISP sequence")
    return samples


def read_rgb(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"無法讀取影像: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1)


def read_renderer_motion(path: Path) -> torch.Tensor:
    """Decode QRISP EXR as current-to-previous LR displacement in pixels."""
    motion = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if motion is None:
        raise RuntimeError(f"無法讀取 EXR Motion Vector: {path}")
    if motion.ndim != 3 or motion.shape[2] < 3:
        raise ValueError(f"預期至少 3-channel EXR，得到 {motion.shape}: {path}")
    motion = motion.astype(np.float32)
    height, width = motion.shape[:2]

    # Verified on SpaceShipDemo and Flooded_Grounds:
    # current frame, x = -OpenCV channel 2 * W, y = +channel 1 * H.
    flow_x = -motion[..., 2] * width
    flow_y = motion[..., 1] * height
    flow = np.stack((flow_x, flow_y), axis=0)
    if not np.isfinite(flow).all():
        raise ValueError(f"Motion Vector 包含 NaN/Inf: {path}")
    return torch.from_numpy(flow)


def load_sequence(sample: SequenceSample, device: torch.device):
    lr = torch.stack([read_rgb(path) for path in sample.lr_paths]).unsqueeze(0)
    gt = torch.stack([read_rgb(path) for path in sample.gt_paths]).unsqueeze(0)
    motion = torch.stack(
        [read_renderer_motion(path) for path in sample.motion_paths]
    ).unsqueeze(0)
    if motion.shape[-2:] != lr.shape[-2:]:
        raise ValueError(
            f"RGB/MV 尺寸不一致: RGB={tuple(lr.shape)}, MV={tuple(motion.shape)}"
        )
    return lr.to(device), gt.to(device), motion.to(device)


def clean_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
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


def build_model(model_key: str, checkpoint_path: str, device: torch.device):
    if model_key == "bidirectional":
        model = BasicVSRPlusPlus(
            mid_channels=64,
            num_blocks=7,
            is_low_res_input=True,
        )
    elif model_key == "causal":
        model = CausalBasicVSRPlusPlus(
            mid_channels=64,
            num_blocks=7,
            is_low_res_input=True,
        )
    elif model_key == "renderer_mv":
        model = CausalBasicVSRPlusPlusMV(
            mid_channels=64,
            num_blocks=7,
            is_low_res_input=True,
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint), strict=True)
    return model.to(device).eval()


@torch.no_grad()
def infer_causal(model, sequence, motions=None):
    state = None
    outputs = []
    for frame_index in range(sequence.size(1)):
        if motions is None:
            output, state = model.forward_step(sequence[:, frame_index], state)
        else:
            output, state = model.forward_step(
                sequence[:, frame_index], motions[:, frame_index], state
            )
        outputs.append(output)
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def infer(model_key: str, model, sequence, motions):
    if model_key == "bidirectional":
        # The original implementation only ever sets this flag to True, so it
        # must be reset when a new, independent sequence starts.
        model.is_mirror_extended = False
        return model(sequence)
    if model_key == "renderer_mv":
        return infer_causal(model, sequence, motions)
    return infer_causal(model, sequence)


@torch.no_grad()
def calculate_metrics(outputs, targets, lpips_model):
    if outputs.shape != targets.shape:
        raise ValueError(
            f"Output/GT shape 不一致: {tuple(outputs.shape)} vs "
            f"{tuple(targets.shape)}"
        )

    values = {"psnr_db": [], "ssim": [], "lpips": [], "std_diff": []}
    for frame_index in range(outputs.size(1)):
        output = outputs[:, frame_index]
        target = targets[:, frame_index]

        mse = torch.mean((output - target) ** 2)
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        ssim = structural_similarity_index_measure(
            output, target, data_range=1.0
        )
        perceptual = lpips_model(
            output * 2.0 - 1.0,
            target * 2.0 - 1.0,
        )
        std_diff = torch.abs(output.std() - target.std())

        values["psnr_db"].append(psnr.item())
        values["ssim"].append(ssim.item())
        values["lpips"].append(perceptual.item())
        values["std_diff"].append(std_diff.item())

    return {key: float(np.mean(items)) for key, items in values.items()}


@torch.no_grad()
def benchmark_bidirectional(model, sequence, warmup_repeats, runtime_repeats):
    for _ in range(warmup_repeats):
        model.is_mirror_extended = False
        model(sequence)
    torch.cuda.synchronize()

    per_frame_times = []
    for _ in range(runtime_repeats):
        model.is_mirror_extended = False
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(sequence)
        end.record()
        torch.cuda.synchronize()
        per_frame_times.append(start.elapsed_time(end) / sequence.size(1))

    time_ms = float(np.mean(per_frame_times))
    return {
        "latency_mode": "clip_average_compute",
        "first_frame_ms": "",
        "time_ms_per_frame": time_ms,
        "runtime_std_ms": float(np.std(per_frame_times)),
        "fps": 1000.0 / time_ms,
    }


@torch.no_grad()
def benchmark_causal(
    model, sequence, motions, warmup_repeats, runtime_repeats
):
    for _ in range(warmup_repeats):
        infer_causal(model, sequence, motions)
    torch.cuda.synchronize()

    all_times = []
    for _ in range(runtime_repeats):
        state = None
        events = []
        for frame_index in range(sequence.size(1)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if motions is None:
                _, state = model.forward_step(sequence[:, frame_index], state)
            else:
                _, state = model.forward_step(
                    sequence[:, frame_index], motions[:, frame_index], state
                )
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        all_times.append([start.elapsed_time(end) for start, end in events])

    timings = np.asarray(all_times, dtype=np.float64)
    first_frame_ms = float(timings[:, 0].mean())
    if timings.shape[1] > 1:
        steady = timings[:, 1:].reshape(-1)
    else:
        steady = timings.reshape(-1)
    time_ms = float(steady.mean())
    return {
        "latency_mode": "streaming_steady_state",
        "first_frame_ms": first_frame_ms,
        "time_ms_per_frame": time_ms,
        "runtime_std_ms": float(steady.std()),
        "fps": 1000.0 / time_ms,
    }


def benchmark(
    model_key,
    model,
    sequence,
    motions,
    warmup_repeats,
    runtime_repeats,
):
    if model_key == "bidirectional":
        return benchmark_bidirectional(
            model, sequence, warmup_repeats, runtime_repeats
        )
    renderer_motions = motions if model_key == "renderer_mv" else None
    return benchmark_causal(
        model,
        sequence,
        renderer_motions,
        warmup_repeats,
        runtime_repeats,
    )


@torch.no_grad()
def calculate_flops_per_frame(model_key, model, sequence, motions):
    if hasattr(model, "is_mirror_extended"):
        model.is_mirror_extended = False
    inputs = (
        (sequence, motions)
        if model_key == "renderer_mv"
        else (sequence,)
    )
    macs, _ = profile(model, inputs=inputs, verbose=False)
    return float(macs * 2.0 / 1e9 / sequence.size(1))


def mean_and_std(rows: Sequence[Dict], key: str) -> Tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std())


def aggregate_model_rows(
    model_key: str,
    rows: Sequence[Dict],
    params_m: float,
    flops_g: Optional[float],
):
    summary = {
        "model": {
            "bidirectional": "BasicVSR++",
            "causal": "C0 Causal BasicVSR++ (SPyNet)",
            "renderer_mv": "C1 Causal BasicVSR++ (Renderer MV)",
        }[model_key],
        "num_sequences": len(rows),
        "num_frames": sum(int(row["num_frames"]) for row in rows),
        "params_m": params_m,
        "flops_g_per_frame": "" if flops_g is None else flops_g,
        "lookahead": "Future frames" if model_key == "bidirectional" else "0 frame",
        "latency_mode": rows[0]["latency_mode"],
    }
    for key in ("psnr_db", "ssim", "lpips", "std_diff"):
        mean, std = mean_and_std(rows, key)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_sequence_std"] = std

    if rows[0]["time_ms_per_frame"] == "":
        summary["first_frame_ms"] = ""
        summary["time_ms_per_frame"] = ""
        summary["runtime_std_ms"] = ""
        summary["fps"] = ""
    else:
        time_mean, _ = mean_and_std(rows, "time_ms_per_frame")
        runtime_std_mean, _ = mean_and_std(rows, "runtime_std_ms")
        summary["time_ms_per_frame"] = time_mean
        summary["runtime_std_ms"] = runtime_std_mean
        summary["fps"] = 1000.0 / time_mean
        if model_key != "bidirectional":
            first_mean, _ = mean_and_std(rows, "first_frame_ms")
            summary["first_frame_ms"] = first_mean
        else:
            summary["first_frame_ms"] = ""
    return summary


def write_csv(path: Path, rows: Sequence[Dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_number(value, digits):
    if value == "" or value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def write_markdown(path: Path, summaries: Sequence[Dict]):
    lines = [
        "# C1 Renderer-MV Unified Evaluation\n",
        "| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Std Diff ↓ | Params (M) ↓ | FLOPs (G/frame) ↓ | Time (ms/frame) ↓ | FPS ↑ | Lookahead |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {model} | {psnr} | {ssim} | {lpips} | {std} | {params} | "
            "{flops} | {time} | {fps} | {lookahead} |".format(
                model=row["model"],
                psnr=format_number(row["psnr_db_mean"], 2),
                ssim=format_number(row["ssim_mean"], 4),
                lpips=format_number(row["lpips_mean"], 4),
                std=format_number(row["std_diff_mean"], 4),
                params=format_number(row["params_m"], 2),
                flops=format_number(row["flops_g_per_frame"], 2),
                time=format_number(row["time_ms_per_frame"], 2),
                fps=format_number(row["fps"], 2),
                lookahead=row["lookahead"],
            )
        )
    lines.extend(
        [
            "",
            "> Bidirectional 的時間為整段 clip compute time / frame；Causal 的時間為排除第一幀後的 streaming steady-state latency。",
            "> FLOPs 為 THOP 估算值，可能未完整計入 MMCV deformable convolution。",
            "> Std Diff 為空間對比／紋理統計，不代表 temporal consistency。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("此評估需要 CUDA 與可用的 MMCV DCN operator")
    if args.num_frames < 2:
        raise ValueError("--num-frames 至少必須為 2")
    if args.runtime_repeats < 1:
        raise ValueError("--runtime-repeats 至少必須為 1")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_sequences(
        Path(args.qrisp_root),
        args.split,
        args.num_frames,
        args.max_sequences,
        args.motion_modality,
    )
    print(f"📦 找到 {len(samples)} 個有效 sequence，每個取 {args.num_frames} 幀")

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    model_configs = []
    if args.models in {"all", "bidirectional"}:
        model_configs.append(
            ("bidirectional", args.bidirectional_checkpoint)
        )
    if args.models in {"all", "causal", "causal_vs_mv"}:
        model_configs.append(("causal", args.causal_checkpoint))
    if args.models in {"all", "renderer_mv", "causal_vs_mv"}:
        model_configs.append(("renderer_mv", args.renderer_mv_checkpoint))

    all_rows: List[Dict] = []
    summaries: List[Dict] = []
    for model_key, checkpoint_path in model_configs:
        display_name = {
            "bidirectional": "BasicVSR++",
            "causal": "C0 Causal BasicVSR++ (SPyNet)",
            "renderer_mv": "C1 Causal BasicVSR++ (Renderer MV)",
        }[model_key]
        print(f"\n{'=' * 72}\n🚀 評估 {display_name}\nCheckpoint: {checkpoint_path}")
        model = build_model(model_key, checkpoint_path, device)
        params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
        model_rows: List[Dict] = []
        flops_g = None

        for sequence_index, sample in enumerate(samples):
            lr_sequence, gt_sequence, motion_sequence = load_sequence(sample, device)
            if flops_g is None and not args.skip_flops:
                print("🧮 使用第一個 sequence 估算 FLOPs...")
                flops_g = calculate_flops_per_frame(
                    model_key, model, lr_sequence, motion_sequence
                )

            outputs = infer(model_key, model, lr_sequence, motion_sequence)
            metrics = calculate_metrics(outputs, gt_sequence, lpips_model)

            if args.skip_runtime:
                runtime = {
                    "latency_mode": (
                        "clip_average_compute"
                        if model_key == "bidirectional"
                        else "streaming_steady_state"
                    ),
                    "first_frame_ms": "",
                    "time_ms_per_frame": "",
                    "runtime_std_ms": "",
                    "fps": "",
                }
            else:
                runtime = benchmark(
                    model_key,
                    model,
                    lr_sequence,
                    motion_sequence,
                    args.warmup_repeats,
                    args.runtime_repeats,
                )

            row = {
                "model": display_name,
                "scene": sample.scene,
                "angle": sample.angle,
                "num_frames": args.num_frames,
                **metrics,
                **runtime,
            }
            model_rows.append(row)
            all_rows.append(row)
            print(
                f"[{sequence_index + 1:03d}/{len(samples):03d}] "
                f"{sample.scene}/{sample.angle} | "
                f"PSNR {metrics['psnr_db']:.2f} | "
                f"SSIM {metrics['ssim']:.4f} | "
                f"LPIPS {metrics['lpips']:.4f}"
            )

            del lr_sequence, gt_sequence, motion_sequence, outputs
            torch.cuda.empty_cache()

        summary = aggregate_model_rows(
            model_key=model_key,
            rows=model_rows,
            params_m=params_m,
            flops_g=flops_g,
        )
        summaries.append(summary)
        print(
            f"✅ {display_name}: PSNR {summary['psnr_db_mean']:.2f} dB, "
            f"SSIM {summary['ssim_mean']:.4f}, "
            f"LPIPS {summary['lpips_mean']:.4f}"
        )

        del model
        torch.cuda.empty_cache()

    per_sequence_path = output_dir / "per_sequence_results.csv"
    summary_path = output_dir / "summary_results.csv"
    markdown_path = output_dir / "comparison_table.md"
    write_csv(per_sequence_path, all_rows)
    write_csv(summary_path, summaries)
    write_markdown(markdown_path, summaries)

    print("\n🏁 評估完成")
    print(f"逐序列結果: {per_sequence_path}")
    print(f"總結結果:   {summary_path}")
    print(f"比較表格:   {markdown_path}")


if __name__ == "__main__":
    main()

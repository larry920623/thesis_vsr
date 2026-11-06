"""Compare C2 controls and C3 on the same QRISP frames and metrics."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import torch
import lpips

from evaluate_qrisp_c1_mv import (
    discover_sequences,
    load_sequence,
    clean_state_dict,
    calculate_metrics,
    write_csv,
    format_number,
)
from archs.basicvsrpp.causal_basicvsrpp_mv_depth import (
    CausalBasicVSRPlusPlusMVDepth,
)
from archs.basicvsrpp.causal_basicvsrpp_mv_depth_gate import (
    CausalBasicVSRPlusPlusMVDepthGate,
)
from qrisp_depth import depth_tensor, matching_depth_path, check_contiguous


CHECKPOINT_ROOT = "/home/larry/ssd_data/sr_project/checkpoints"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--qrisp-root",
        default="/home/larry/ssd_data/sr_project/datasets/QRISP",
    )
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--num-frames", type=int, default=10)
    p.add_argument("--max-sequences", type=int, default=None)
    p.add_argument("--expected-sequences", type=int, default=None)
    p.add_argument(
        "--control-checkpoint",
        default=f"{CHECKPOINT_ROOT}/c2_gate0_control/best_model.pth",
    )
    p.add_argument(
        "--c2-checkpoint",
        default=f"{CHECKPOINT_ROOT}/causal_basicvsrpp_mv_depth/best_model.pth",
    )
    p.add_argument(
        "--c3-checkpoint",
        default=f"{CHECKPOINT_ROOT}/c3_learned_history_gate/best_model.pth",
    )
    p.add_argument(
        "--output-dir",
        default="/home/larry/ssd_data/sr_project/experiments/c3_gate_val_quality",
    )
    p.add_argument("--skip-runtime", action="store_true")
    p.add_argument("--skip-flops", action="store_true")
    p.add_argument("--warmup-repeats", type=int, default=2)
    p.add_argument("--runtime-repeats", type=int, default=3)
    return p.parse_args()


def load_model(kind, checkpoint_path, expected_strength=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("confidence_config")
    if not isinstance(cfg, dict):
        raise ValueError(f"Missing confidence_config: {checkpoint_path}")

    if kind == "c3":
        if cfg.get("type") != "encoded_depth_feature_gate_v2":
            raise ValueError(
                "C3 requires the corrected encoded-depth v2 checkpoint"
            )
        if cfg.get("gate_mode") != "learned":
            raise ValueError("C3 checkpoint must use learned gate mode")

        model = CausalBasicVSRPlusPlusMVDepthGate(
            depth_tau=cfg["depth_tau"],
            gate_mode=cfg["gate_mode"],
            gate_hidden_channels=cfg["gate_hidden_channels"],
        )
    else:
        if cfg.get("gate_strength") != expected_strength:
            raise ValueError(
                f"Expected gate_strength={expected_strength}: "
                f"{checkpoint_path}\nSaved config: {cfg}"
            )

        model = CausalBasicVSRPlusPlusMVDepth(
            depth_tau=cfg["depth_tau"],
            gate_strength=cfg["gate_strength"],
        )

    current_cfg = model.confidence_config()
    if cfg != current_cfg:
        raise ValueError(
            f"Checkpoint/model config mismatch: {checkpoint_path}\n"
            f"Saved: {cfg}\nCurrent: {current_cfg}"
        )

    model.load_state_dict(clean_state_dict(checkpoint), strict=True)
    info = {
        "checkpoint": str(checkpoint_path),
        "epoch_index": checkpoint.get("epoch"),
        "confidence_config": cfg,
        "training_protocol": checkpoint.get("training_protocol"),
    }
    return model.float().eval(), info


@torch.no_grad()
def infer(model, lr, mv, depth):
    state = None
    outputs = []

    for i in range(lr.size(1)):
        output, state = model.forward_step(
            lr[:, i], mv[:, i], depth[:, i], state
        )
        outputs.append(output)

    return torch.stack(outputs, dim=1)


@torch.no_grad()
def benchmark(model, lr, mv, depth, warmup=2, repeats=3):
    for _ in range(warmup):
        infer(model, lr, mv, depth)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        state = None
        events = []

        for i in range(lr.size(1)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _, state = model.forward_step(
                lr[:, i], mv[:, i], depth[:, i], state
            )
            end.record()

            events.append((start, end))

        torch.cuda.synchronize()
        times.append([
            start.elapsed_time(end) for start, end in events
        ])

    times = np.asarray(times)
    steady = times[:, 1:].ravel()
    mean = float(steady.mean())

    return {
        "first_frame_ms": float(times[:, 0].mean()),
        "time_ms_per_frame": mean,
        "runtime_std_ms": float(steady.std()),
        "p95_ms": float(np.percentile(steady, 95)),
        "fps": 1000.0 / mean,
        "latency_mode": "streaming_steady_state",
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Requires CUDA and MMCV DCN")
    if args.num_frames < 2:
        raise ValueError("num-frames must be >= 2")
    if args.runtime_repeats < 1 or args.warmup_repeats < 0:
        raise ValueError(
            "runtime-repeats >= 1; warmup-repeats >= 0"
        )
    if args.max_sequences is not None and args.max_sequences < 1:
        raise ValueError("max-sequences must be positive")

    device = torch.device("cuda")

    samples = discover_sequences(
        Path(args.qrisp_root),
        args.split,
        args.num_frames,
        args.max_sequences,
        "MotionVectorsMipBiasMinus2",
    )
    if not samples:
        raise RuntimeError("No evaluation sequences found")

    expected = args.expected_sequences
    if expected is None and args.split == "test":
        expected = 44

    if args.max_sequences is None and expected is not None:
        if len(samples) != expected:
            raise ValueError(
                f"Expected {expected} sequences, got {len(samples)}; "
                "inspect split"
            )

    manifest = []
    for sample in samples:
        check_contiguous(sample.lr_paths)
        depths = [
            matching_depth_path(path) for path in sample.lr_paths
        ]
        manifest.append({
            "scene": sample.scene,
            "angle": sample.angle,
            "rgb": [str(path) for path in sample.lr_paths],
            "gt": [str(path) for path in sample.gt_paths],
            "mv": [str(path) for path in sample.motion_paths],
            "depth": [str(path) for path in depths],
        })

    print(
        f"Sequences: {len(samples)}, "
        f"frames/sequence: {args.num_frames}; "
        "depth pairing complete"
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    specs = [
        (
            "C2 gate-strength=0 control",
            "c2",
            args.control_checkpoint,
            0.0,
        ),
        (
            "C2 gate-strength=1",
            "c2",
            args.c2_checkpoint,
            1.0,
        ),
        (
            "C3 learned history gate",
            "c3",
            args.c3_checkpoint,
            None,
        ),
    ]

    evaluation_config = {
        **vars(args),
        "models": [],
        "metric_source": "evaluate_qrisp_c1_mv.calculate_metrics",
        "timing": (
            "FP32 CUDA streaming compute including geometry/gates; "
            "excludes file IO/H2D; "
            "steady-state timing excludes first frame"
        ),
    }

    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    all_rows, summaries = [], []

    timing_keys = (
        "first_frame_ms",
        "time_ms_per_frame",
        "runtime_std_ms",
        "p95_ms",
    )

    for label, kind, checkpoint_path, strength in specs:
        model, info = load_model(
            kind, checkpoint_path, strength
        )
        model = model.to(device).eval()

        evaluation_config["models"].append({
            "label": label,
            **info,
        })
        (out / "evaluation_config.json").write_text(
            json.dumps(evaluation_config, indent=2),
            encoding="utf-8",
        )

        print(f"\nEvaluating {label}: {checkpoint_path}")

        params = sum(
            parameter.numel()
            for parameter in model.parameters()
        ) / 1e6

        rows = []
        flops = None

        for index, sample in enumerate(samples):
            lr, gt, mv = load_sequence(sample, device)

            depth = torch.stack([
                depth_tensor(path)
                for path in manifest[index]["depth"]
            ])[None].to(
                device=device,
                dtype=torch.float32,
            )

            if depth.shape[-2:] != lr.shape[-2:]:
                raise ValueError("Depth/RGB resolution mismatch")

            if flops is None and not args.skip_flops:
                from thop import profile

                with torch.no_grad():
                    macs, _ = profile(
                        model,
                        inputs=(lr, mv, depth),
                        verbose=False,
                    )

                flops = macs * 2 / 1e9 / lr.size(1)

            prediction = infer(model, lr, mv, depth)

            if not torch.isfinite(prediction).all():
                raise RuntimeError(
                    f"Nonfinite prediction: {label}"
                )

            metrics = calculate_metrics(
                prediction, gt, perceptual
            )

            if args.skip_runtime:
                timing = {
                    key: "" for key in timing_keys
                }
                timing.update(
                    fps="",
                    latency_mode="streaming_steady_state",
                )
            else:
                timing = benchmark(
                    model,
                    lr,
                    mv,
                    depth,
                    args.warmup_repeats,
                    args.runtime_repeats,
                )

            row = {
                "model": label,
                "scene": sample.scene,
                "angle": sample.angle,
                "num_frames": lr.size(1),
                **metrics,
                **timing,
            }

            rows.append(row)
            all_rows.append(row)

            print(
                f"[{index + 1}/{len(samples)}] {label}: "
                f"{sample.scene}/{sample.angle} "
                f"PSNR={metrics['psnr_db']:.3f}"
            )

            del lr, gt, mv, depth, prediction

        summary = {
            "model": label,
            "num_sequences": len(rows),
            "num_frames": sum(
                row["num_frames"] for row in rows
            ),
            "params_m": params,
            "flops_g_per_frame": (
                "" if flops is None else flops
            ),
            "lookahead": "0 frame (architecture)",
        }

        for key in ("psnr_db", "ssim", "lpips", "std_diff"):
            values = [row[key] for row in rows]
            summary[key] = float(np.mean(values))
            summary[key + "_sequence_std"] = float(
                np.std(values)
            )

        for key in timing_keys:
            summary[key] = (
                ""
                if args.skip_runtime
                else float(np.mean([
                    row[key] for row in rows
                ]))
            )

        summary["fps"] = (
            ""
            if args.skip_runtime
            else 1000.0 / summary["time_ms_per_frame"]
        )

        summaries.append(summary)
        del model
        torch.cuda.empty_cache()

    write_csv(
        out / "per_sequence_results.csv",
        all_rows,
    )
    write_csv(
        out / "summary_results.csv",
        summaries,
    )

    lines = [
        "# C3 Learned History Gate Evaluation",
        "",
        "| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Std Diff ↓ | Params M | Approx. FLOPs G/frame | Time ms/frame | FPS | Lookahead |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in summaries:
        fields = [row["model"]] + [
            format_number(row[key], digits)
            for key, digits in (
                ("psnr_db", 2),
                ("ssim", 4),
                ("lpips", 4),
                ("std_diff", 4),
                ("params_m", 2),
                ("flops_g_per_frame", 2),
                ("time_ms_per_frame", 2),
                ("fps", 2),
            )
        ] + [row["lookahead"]]

        lines.append(
            "| " + " | ".join(fields) + " |"
        )

    lines += [
        "",
        "> Metrics are sequence-averaged using the existing C1 metric functions.",
        "> Std Diff is spatial, not a temporal consistency metric.",
        "> THOP may omit DCN and functional operations; FLOPs are incomplete estimates.",
        "> Timing excludes decoding/H2D. Steady-state timing excludes the first frame.",
        "> Summary p95_ms is the mean of per-sequence p95, not pooled global p95.",
        "> Zero lookahead is the architectural design; trained-weight causality tests are separate.",
    ]

    report = out / "comparison_table.md"
    report.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved: {report}")


if __name__ == "__main__":
    main()
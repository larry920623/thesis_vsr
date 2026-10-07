"""C1 vs C2 on EXACT C1 frame selection and metrics. No missing-depth filtering."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import lpips
from thop import profile
from evaluate_qrisp_c1_mv import (
    discover_sequences, load_sequence, clean_state_dict, calculate_metrics,
    write_csv, format_number,
)
from archs.basicvsrpp.causal_basicvsrpp_mv import CausalBasicVSRPlusPlusMV
from archs.basicvsrpp.causal_basicvsrpp_mv_depth import CausalBasicVSRPlusPlusMVDepth
from qrisp_depth import depth_tensor, matching_depth_path, check_contiguous


@torch.no_grad()
def infer_step(model, lr, mv, depth, index, state, is_c2):
    if is_c2:
        return model.forward_step(lr[:, index], mv[:, index], depth[:, index], state)
    return model.forward_step(lr[:, index], mv[:, index], state)


@torch.no_grad()
def infer(model, lr, mv, depth, is_c2):
    state = None; outputs = []
    for i in range(lr.size(1)):
        y, state = infer_step(model, lr, mv, depth, i, state, is_c2)
        outputs.append(y)
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def benchmark(model, lr, mv, depth, is_c2, warmup=2, repeats=3):
    for _ in range(warmup):
        infer(model, lr, mv, depth, is_c2)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        state = None; events = []
        for i in range(lr.size(1)):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            _, state = infer_step(model, lr, mv, depth, i, state, is_c2)
            end.record(); events.append((start, end))
        torch.cuda.synchronize()
        times.append([s.elapsed_time(e) for s, e in events])
    times = np.asarray(times)
    steady = times[:, 1:].ravel()
    mean = float(steady.mean())
    return {"first_frame_ms": float(times[:, 0].mean()), "time_ms_per_frame": mean,
            "runtime_std_ms": float(steady.std()), "p95_ms": float(np.percentile(steady, 95)),
            "fps": 1000 / mean, "latency_mode": "streaming_steady_state"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qrisp-root", default="/home/larry/ssd_data/sr_project/datasets/QRISP")
    p.add_argument("--split", default="test")
    p.add_argument("--num-frames", type=int, default=10)
    p.add_argument("--max-sequences", type=int, default=None)
    p.add_argument("--c1-checkpoint", default="/home/larry/ssd_data/sr_project/checkpoints/causal_basicvsrpp_mv/best_model.pth")
    p.add_argument("--c2-checkpoint", default="/home/larry/ssd_data/sr_project/checkpoints/causal_basicvsrpp_mv_depth/best_model.pth")
    p.add_argument("--init-from-c1", action="store_true", help="Zero-fine-tune C2 diagnostic using C1 weights")
    p.add_argument("--depth-tau", type=float, default=None, help="Override checkpoint tau; validation experiments only")
    p.add_argument("--gate-strength", type=float, default=None, help="Override checkpoint gate; 0 = exact C1 architecture")
    p.add_argument("--output-dir", default="/home/larry/ssd_data/sr_project/experiments/c2_depth_evaluation")
    p.add_argument("--skip-runtime", action="store_true")
    p.add_argument("--skip-flops", action="store_true")
    p.add_argument("--warmup-repeats", type=int, default=2)
    p.add_argument("--runtime-repeats", type=int, default=3)
    p.add_argument("--expected-sequences", type=int, default=44)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Requires CUDA and MMCV DCN")
    if args.num_frames < 2 or args.runtime_repeats < 1 or args.warmup_repeats < 0:
        raise ValueError("num-frames >= 2, runtime-repeats >= 1, warmup-repeats >= 0")
    device = torch.device("cuda")
    samples = discover_sequences(Path(args.qrisp_root), args.split, args.num_frames,
                                 args.max_sequences, "MotionVectorsMipBiasMinus2")
    if args.split == "test" and args.max_sequences is None and len(samples) != args.expected_sequences:
        raise ValueError(f"Expected {args.expected_sequences} C1 test sequences, got {len(samples)}; inspect split")
    manifest = []
    for sample in samples:
        check_contiguous(sample.lr_paths)
        depths = [matching_depth_path(path) for path in sample.lr_paths]
        manifest.append({"scene": sample.scene, "angle": sample.angle,
                         "rgb": [str(x) for x in sample.lr_paths],
                         "gt": [str(x) for x in sample.gt_paths],
                         "mv": [str(x) for x in sample.motion_paths],
                         "depth": [str(x) for x in depths]})
    print(f"Sequences: {len(samples)}, frames/sequence: {args.num_frames}; depth pairing complete")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    c2_path = args.c1_checkpoint if args.init_from_c1 else args.c2_checkpoint
    c2_checkpoint = torch.load(c2_path, map_location="cpu")
    if not args.init_from_c1 and "confidence_config" not in c2_checkpoint:
        raise ValueError("C2 checkpoint missing confidence_config; use --init-from-c1 for C1 weights")
    cfg = c2_checkpoint.get("confidence_config", {})
    tau = args.depth_tau if args.depth_tau is not None else cfg.get("depth_tau", 0.001)
    strength = args.gate_strength if args.gate_strength is not None else cfg.get("gate_strength", 1.0)
    (out / "evaluation_config.json").write_text(json.dumps({**vars(args), "effective_tau": tau,
        "effective_gate_strength": strength, "actual_c2_checkpoint": c2_path,
        "timing": "FP32 CUDA compute including depth confidence; excludes file IO/H2D; first frame excluded"}, indent=2))
    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    all_rows, summaries = [], []
    for label, is_c2, checkpoint_path in (("C1 Renderer MV", False, args.c1_checkpoint),
                                        ("C2 Renderer MV + Depth confidence", True, c2_path)):
        model = (CausalBasicVSRPlusPlusMVDepth(depth_tau=tau, gate_strength=strength)
                 if is_c2 else CausalBasicVSRPlusPlusMV())
        checkpoint = c2_checkpoint if is_c2 else torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint), strict=True)
        model = model.to(device).eval()
        params = sum(x.numel() for x in model.parameters()) / 1e6
        rows = []; flops = None
        for index, sample in enumerate(samples):
            lr, gt, mv = load_sequence(sample, device)
            depth = torch.stack([depth_tensor(path) for path in manifest[index]["depth"]])[None].to(device)
            if depth.shape[-2:] != lr.shape[-2:]:
                raise ValueError("Depth/RGB resolution mismatch")
            if flops is None and not args.skip_flops:
                inputs = (lr, mv, depth) if is_c2 else (lr, mv)
                with torch.no_grad():
                    macs, _ = profile(model, inputs=inputs, verbose=False)
                flops = macs * 2 / 1e9 / lr.size(1)
            prediction = infer(model, lr, mv, depth, is_c2)
            metrics = calculate_metrics(prediction, gt, perceptual)
            if args.skip_runtime:
                timing = {"first_frame_ms": "", "time_ms_per_frame": "", "runtime_std_ms": "",
                          "p95_ms": "", "fps": "", "latency_mode": "streaming_steady_state"}
            else:
                timing = benchmark(model, lr, mv, depth, is_c2, args.warmup_repeats, args.runtime_repeats)
            row = {"model": label, "scene": sample.scene, "angle": sample.angle,
                   "num_frames": lr.size(1), **metrics, **timing}
            rows.append(row); all_rows.append(row)
            print(f"[{index+1}/{len(samples)}] {label}: {sample.scene}/{sample.angle} PSNR={metrics['psnr_db']:.3f}")
            del lr, gt, mv, depth, prediction
        summary = {"model": label, "num_sequences": len(rows), "num_frames": sum(r["num_frames"] for r in rows),
                   "params_m": params, "flops_g_per_frame": "" if flops is None else flops, "lookahead": "0 frame"}
        for key in ("psnr_db", "ssim", "lpips", "std_diff"):
            values = [r[key] for r in rows]
            summary[key] = float(np.mean(values)); summary[key + "_sequence_std"] = float(np.std(values))
        for key in ("first_frame_ms", "time_ms_per_frame", "runtime_std_ms", "p95_ms"):
            summary[key] = "" if args.skip_runtime else float(np.mean([r[key] for r in rows]))
        summary["fps"] = "" if args.skip_runtime else 1000 / summary["time_ms_per_frame"]
        summaries.append(summary)
        del model; torch.cuda.empty_cache()
    write_csv(out / "per_sequence_results.csv", all_rows)
    write_csv(out / "summary_results.csv", summaries)
    lines = ["# C2 Depth Confidence Evaluation", "",
             "| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Std Diff ↓ | Params M | Approx. FLOPs G/frame | Time ms/frame | FPS | Lookahead |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summaries:
        fields = [row["model"]] + [format_number(row[key], digits) for key, digits in
                  (("psnr_db", 2), ("ssim", 4), ("lpips", 4), ("std_diff", 4), ("params_m", 2),
                   ("flops_g_per_frame", 2), ("time_ms_per_frame", 2), ("fps", 2))] + [row["lookahead"]]
        lines.append("| " + " | ".join(fields) + " |")
    lines += ["", "> THOP may omit DCN and functional confidence operations; FLOPs are incomplete estimates.",
              "> Timing includes confidence, excludes decoding/H2D; Std Diff is spatial, not temporal consistency.",
              "> p95_ms in CSV is mean of per-sequence p95, not pooled global p95."]
    (out / "comparison_table.md").write_text("\n".join(lines) + "\n")
    print(f"Saved: {out / 'comparison_table.md'}")


if __name__ == "__main__":
    main()

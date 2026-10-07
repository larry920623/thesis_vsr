"""Read/visualize packed depth and causal confidence before training C2."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
import torch
from qrisp_depth import read_qrisp_depth, check_contiguous
from archs.basicvsrpp.depth_confidence import depth_motion_confidence, sampling_grid, sample_history


def stats(array):
    values = np.asarray(array)
    return {"min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()),
            "percentiles": np.percentile(values, [0, 1, 10, 50, 90, 99, 100]).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qrisp-root", default="/home/larry/ssd_data/sr_project/datasets/QRISP")
    p.add_argument("--scene", default="Flooded_Grounds")
    p.add_argument("--angle", default="0013")
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--depth-tau", type=float, default=0.001)
    p.add_argument("--output-dir", default="/home/larry/ssd_data/sr_project/experiments/c2_depth_inspection")
    args = p.parse_args()
    roots = [Path(args.qrisp_root) / args.scene,
             Path(args.qrisp_root) / "TrainSet" / args.scene,
             Path(args.qrisp_root) / "TestSet" / args.scene]
    root = next((x for x in roots if x.is_dir()), None)
    if root is None:
        raise FileNotFoundError(f"Scene missing: {args.scene}")
    base = root / "270p"
    rgb_dir = base / "Native" / args.angle
    depth_dir = base / "DepthMipBiasMinus2" / args.angle
    motion_dir = base / "MotionVectorsMipBiasMinus2" / args.angle
    paths = sorted(rgb_dir.glob("*.png"))
    if len(paths) < 2 or args.pairs < 1:
        raise ValueError("Need at least two frames and pairs >= 1")
    check_contiguous(paths)
    out = Path(args.output_dir) / args.scene / args.angle / f"tau_{args.depth_tau:g}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, min(len(paths), args.pairs + 1)):
        prev_path, cur_path = paths[i - 1], paths[i]
        prev = cv2.imread(str(prev_path), cv2.IMREAD_COLOR)
        cur = cv2.imread(str(cur_path), cv2.IMREAD_COLOR)
        if prev is None or cur is None:
            raise FileNotFoundError("Unreadable RGB")
        dp = read_qrisp_depth(depth_dir / (prev_path.stem + ".png"))
        dc = read_qrisp_depth(depth_dir / (cur_path.stem + ".png"))
        exr = cv2.imread(str(motion_dir / (cur_path.stem + ".exr")), cv2.IMREAD_UNCHANGED)
        if exr is None or exr.ndim != 3 or exr.shape[2] < 3:
            raise ValueError("Unreadable/malformed current-frame EXR")
        h, w = cur.shape[:2]
        if dp.shape != (h, w) or dc.shape != (h, w) or exr.shape[:2] != (h, w) or prev.shape != cur.shape:
            raise ValueError("RGB/depth/MV size mismatch")
        motion = torch.from_numpy(np.stack((-exr[..., 2] * w, exr[..., 1] * h), axis=0)).float()[None]
        if not torch.isfinite(motion).all():
            raise ValueError("Nonfinite EXR flow")
        confidence, diag = depth_motion_confidence(
            torch.from_numpy(dc)[None, None], torch.from_numpy(dp)[None, None], motion, args.depth_tau
        )
        grid, _ = sampling_grid(motion)
        prev_t = torch.from_numpy(prev.astype(np.float32) / 255).permute(2, 0, 1)[None]
        warped_rgb = sample_history(prev_t, grid)[0].permute(1, 2, 0).numpy()
        c = confidence[0, 0].numpy()
        valid = diag["valid"][0, 0].numpy() > 0.5
        delta = diag["depth_delta"][0, 0].numpy()
        dw = diag["warped_depth"][0, 0].numpy()
        error = np.abs(cur.astype(np.float32) / 255 - warped_rgb).mean(axis=-1)
        def masked_mean(mask):
            return float(error[mask].mean()) if mask.any() else None
        row = {"previous": prev_path.name, "current": cur_path.name,
               "depth_current": stats(dc), "depth_previous": stats(dp),
               "confidence": stats(c), "valid_fraction": float(valid.mean()),
               "low_confidence_valid_fraction": float(((c < 0.1) & valid).sum() / max(valid.sum(), 1)),
               "warped_rgb_mae_valid": masked_mean(valid),
               "warped_rgb_mae_conf_ge_05": masked_mean(valid & (c >= 0.5)),
               "warped_rgb_mae_conf_lt_01": masked_mean(valid & (c < 0.1))}
        rows.append(row)
        pair_out = out / cur_path.stem
        pair_out.mkdir(exist_ok=True)
        # SHARED pair scale for visualization only. Model uses untouched values.
        low, high = np.percentile(np.concatenate([dc.ravel(), dp.ravel()]), [1, 99])
        high = max(high, low + 1e-8)
        def save(name, value):
            if not cv2.imwrite(str(pair_out / name), value):
                raise IOError(f"Could not save {name}")
        for name, d in (("depth_current.png", dc), ("depth_previous.png", dp), ("depth_warped.png", dw)):
            save(name, (np.clip((d - low) / (high - low), 0, 1) * 255).astype(np.uint8))
        save("current_rgb.png", cur)
        save("warped_previous_rgb.png", (np.clip(warped_rgb, 0, 1) * 255).astype(np.uint8))
        save("confidence.png", (c * 255).round().astype(np.uint8))
        save("valid.png", valid.astype(np.uint8) * 255)
        save("depth_delta.png", (np.clip(delta / args.depth_tau, 0, 5) / 5 * 255).astype(np.uint8))
        save("rgb_error.png", cv2.applyColorMap((np.clip(error * 5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
        np.save(pair_out / "depth_current.npy", dc)
        np.save(pair_out / "confidence.npy", c)
        print(f"{cur_path.stem}: valid={valid.mean():.3f}, mean confidence={c.mean():.3f}, "
              f"depth=[{dc.min():.8f},{dc.max():.8f}], std={dc.std():.8g}")
    camera_root = base / "CameraData"
    camera_files = sorted(camera_root.rglob("*.json")) if camera_root.is_dir() else []
    candidates = [x for x in camera_files if args.angle in x.parts or x.stem == args.angle]
    camera_info = []
    for path in candidates[:2]:
        data = json.loads(path.read_text())
        camera_info.append({"path": str(path), "top_level_type": type(data).__name__,
                            "keys": list(data)[:30] if isinstance(data, dict) else None})
    report = {"scene": args.scene, "angle": args.angle, "depth_tau": args.depth_tau,
              "encoding": "R/255 + G/255^2 + B/255^3 + A/255^4 (OpenCV BGRA reordered)",
              "interpretation": "encoded-z consistency proxy, not occlusion ground truth",
              "camera_candidates": camera_info, "pairs": rows}
    (out / "depth_confidence_inspection.json").write_text(json.dumps(report, indent=2))
    print(f"Saved: {out / 'depth_confidence_inspection.json'}")


if __name__ == "__main__":
    main()

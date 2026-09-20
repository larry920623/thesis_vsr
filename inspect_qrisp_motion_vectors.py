"""Inspect QRISP renderer motion-vector storage and infer warp conventions.

Run this before replacing SPyNet.  The script validates paths, EXR support,
channel statistics, channel order, signs, scale, and frame association by
measuring how well each candidate convention warps I_(t-1) toward I_t.
"""

import argparse
import itertools
import json
import os
import re
from pathlib import Path

# This must be set before importing cv2 on OpenCV builds that gate EXR I/O.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


VALID_RGB_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect and validate QRISP renderer motion vectors"
    )
    parser.add_argument(
        "--qrisp-root",
        default="/home/larry/ssd_data/sr_project/datasets/QRISP",
    )
    parser.add_argument("--scene", default="SpaceShipDemo")
    parser.add_argument("--angle", default="0000")
    parser.add_argument("--resolution", default="270p")
    parser.add_argument(
        "--motion-modality",
        default="MotionVectorsMipBiasMinus2",
        help="Non-jittered 270p motion-vector modality from the QRISP paper.",
    )
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/larry/ssd_data/sr_project/experiments/"
            "motion_vector_inspection"
        ),
    )
    return parser.parse_args()


def natural_key(value):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]


def frame_id(path: Path):
    numbers = re.findall(r"\d+", path.stem)
    return numbers[-1] if numbers else path.stem


def resolve_scene_root(qrisp_root: Path, scene: str) -> Path:
    candidates = [
        qrisp_root / scene,
        qrisp_root / "TestSet" / scene,
        qrisp_root / "TrainSet" / scene,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "找不到 scene。已檢查:\n" + "\n".join(str(path) for path in candidates)
    )


def find_named_directory(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_dir():
        return direct
    direct_matches = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.lower() == name.lower()
    ]
    if not direct_matches:
        raise FileNotFoundError(f"在 {root} 下找不到資料夾 {name}")
    if len(direct_matches) > 1:
        print(f"⚠️ 找到多個 {name}，使用第一個:")
        for path in direct_matches:
            print(f"   {path}")
    return sorted(direct_matches, key=lambda path: natural_key(str(path)))[0]


def resolve_angle_directory(modality_root: Path, angle: str) -> Path:
    direct = modality_root / angle
    if direct.is_dir():
        return direct
    matches = [
        path
        for path in modality_root.rglob("*")
        if path.is_dir() and path.name == angle
    ]
    if matches:
        return sorted(matches, key=lambda path: natural_key(str(path)))[0]
    # Some archives place files directly in the modality folder.
    return modality_root


def list_files(directory: Path, extensions):
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(paths, key=lambda path: natural_key(path.name))


def read_rgb(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"無法讀取 RGB: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def read_motion(path: Path):
    motion = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if motion is None:
        raise RuntimeError(
            f"無法讀取 EXR: {path}\n"
            "請確認 OpenCV 具有 OpenEXR 支援，並在 import cv2 前設定 "
            "OPENCV_IO_ENABLE_OPENEXR=1。"
        )
    if motion.ndim == 2:
        motion = motion[..., None]
    return motion.astype(np.float32)


def print_tree_summary(root: Path, max_depth=3):
    print(f"\n📁 Scene 結構摘要: {root}")
    root_depth = len(root.parts)
    for directory, subdirs, files in os.walk(root):
        path = Path(directory)
        depth = len(path.parts) - root_depth
        if depth > max_depth:
            subdirs[:] = []
            continue
        indent = "  " * depth
        suffix_counts = {}
        for filename in files:
            suffix = Path(filename).suffix.lower() or "<no_ext>"
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        details = ""
        if suffix_counts:
            details = " | " + ", ".join(
                f"{suffix}:{count}" for suffix, count in sorted(suffix_counts.items())
            )
        print(f"{indent}{path.name}/{details}")


def channel_statistics(motion):
    stats = []
    for channel_index in range(motion.shape[2]):
        values = motion[..., channel_index]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            stats.append({"channel": channel_index, "finite": 0})
            continue
        stats.append(
            {
                "channel": channel_index,
                "finite": int(finite.size),
                "min": float(finite.min()),
                "p01": float(np.percentile(finite, 1)),
                "median": float(np.median(finite)),
                "p99": float(np.percentile(finite, 99)),
                "max": float(finite.max()),
                "nonzero_fraction": float(np.mean(np.abs(finite) > 1e-8)),
            }
        )
    return stats


def resize_motion(motion, height, width):
    if motion.shape[:2] == (height, width):
        return motion
    resized_channels = [
        cv2.resize(
            motion[..., channel],
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        for channel in range(motion.shape[2])
    ]
    return np.stack(resized_channels, axis=2)


def make_flow(motion, width, height, config):
    motion = resize_motion(motion, height, width)
    x = (
        motion[..., config["x_channel"]]
        * width
        * config["pixel_scale"]
        * config["x_sign"]
    )
    y = (
        motion[..., config["y_channel"]]
        * height
        * config["pixel_scale"]
        * config["y_sign"]
    )
    return np.stack([x, y], axis=2).astype(np.float32)


def warp_previous_to_current(previous, flow):
    height, width = previous.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    finite = np.isfinite(map_x) & np.isfinite(map_y)
    valid = (
        finite
        & (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    map_x = np.nan_to_num(map_x, nan=-1.0, posinf=-1.0, neginf=-1.0)
    map_y = np.nan_to_num(map_y, nan=-1.0, posinf=-1.0, neginf=-1.0)
    warped = cv2.remap(
        previous,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, valid


def evaluate_config(records, config):
    warp_errors = []
    baseline_errors = []
    valid_fractions = []
    for record in records:
        motion = (
            record["motion_current"]
            if config["motion_frame"] == "current"
            else record["motion_previous"]
        )
        current = record["current"]
        previous = record["previous"]
        height, width = current.shape[:2]
        flow = make_flow(motion, width, height, config)
        warped, valid = warp_previous_to_current(previous, flow)
        if not np.any(valid):
            return None
        warp_error = np.abs(warped - current).mean(axis=2)
        baseline_error = np.abs(previous - current).mean(axis=2)
        warp_errors.append(float(warp_error[valid].mean()))
        baseline_errors.append(float(baseline_error[valid].mean()))
        valid_fractions.append(float(valid.mean()))

    warp_mae = float(np.mean(warp_errors))
    baseline_mae = float(np.mean(baseline_errors))
    result = dict(config)
    result.update(
        {
            "warp_mae": warp_mae,
            "baseline_mae_on_same_valid_pixels": baseline_mae,
            "improvement_fraction": (
                (baseline_mae - warp_mae) / max(baseline_mae, 1e-12)
            ),
            "valid_fraction": float(np.mean(valid_fractions)),
        }
    )
    return result


def save_preview(output_dir: Path, record, best):
    current = record["current"]
    previous = record["previous"]
    motion = (
        record["motion_current"]
        if best["motion_frame"] == "current"
        else record["motion_previous"]
    )
    height, width = current.shape[:2]
    flow = make_flow(motion, width, height, best)
    warped, valid = warp_previous_to_current(previous, flow)
    error = np.abs(warped - current).mean(axis=2)
    error[~valid] = 0
    magnitude = np.linalg.norm(flow, axis=2)
    magnitude[~np.isfinite(magnitude)] = 0

    def save_rgb(filename, image):
        uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / filename), cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR))

    save_rgb("previous.png", previous)
    save_rgb("current.png", current)
    save_rgb("warped_best.png", warped)

    error_scale = max(float(np.percentile(error[valid], 99)) if np.any(valid) else 0.0, 1e-6)
    error_image = np.clip(error / error_scale, 0, 1)
    cv2.imwrite(str(output_dir / "warp_error_best.png"), (error_image * 255).astype(np.uint8))

    magnitude_scale = max(float(np.percentile(magnitude, 99)), 1e-6)
    magnitude_image = np.clip(magnitude / magnitude_scale, 0, 1)
    magnitude_color = cv2.applyColorMap(
        (magnitude_image * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    cv2.imwrite(str(output_dir / "motion_magnitude.png"), magnitude_color)


def main():
    args = parse_args()
    if args.pairs < 1:
        raise ValueError("--pairs 至少必須為 1")

    qrisp_root = Path(args.qrisp_root)
    scene_root = resolve_scene_root(qrisp_root, args.scene)
    resolution_root = scene_root / args.resolution
    if not resolution_root.is_dir():
        raise FileNotFoundError(f"找不到解析度資料夾: {resolution_root}")

    print_tree_summary(resolution_root, max_depth=2)

    rgb_root = find_named_directory(resolution_root, "Native")
    motion_root = find_named_directory(resolution_root, args.motion_modality)
    rgb_dir = resolve_angle_directory(rgb_root, args.angle)
    motion_dir = resolve_angle_directory(motion_root, args.angle)
    print(f"\nRGB directory:    {rgb_dir}")
    print(f"Motion directory: {motion_dir}")

    rgb_paths = list_files(rgb_dir, VALID_RGB_EXTENSIONS)
    motion_paths = list_files(motion_dir, {".exr"})
    if len(rgb_paths) < 2:
        raise RuntimeError(f"RGB 影格不足: {rgb_dir}")
    if not motion_paths:
        raise RuntimeError(f"找不到 EXR motion vectors: {motion_dir}")

    rgb_by_id = {frame_id(path): path for path in rgb_paths}
    motion_by_id = {frame_id(path): path for path in motion_paths}
    common_ids = sorted(
        set(rgb_by_id).intersection(motion_by_id), key=natural_key
    )
    if len(common_ids) < 2:
        raise RuntimeError(
            "RGB 與 Motion Vector 無法依 frame ID 配對。\n"
            f"RGB samples: {[path.name for path in rgb_paths[:3]]}\n"
            f"MV samples: {[path.name for path in motion_paths[:3]]}"
        )

    sample_motion = read_motion(motion_by_id[common_ids[0]])
    stats = channel_statistics(sample_motion)
    print(
        f"\n🔎 EXR sample: {motion_by_id[common_ids[0]]}\n"
        f"shape={sample_motion.shape}, dtype={sample_motion.dtype}"
    )
    for item in stats:
        print(f"   channel {item['channel']}: {item}")

    pair_count = min(args.pairs, len(common_ids) - 1)
    start = max(0, (len(common_ids) - pair_count - 1) // 2)
    selected_ids = common_ids[start : start + pair_count + 1]
    records = []
    for previous_id, current_id in zip(selected_ids[:-1], selected_ids[1:]):
        records.append(
            {
                "previous_id": previous_id,
                "current_id": current_id,
                "previous": read_rgb(rgb_by_id[previous_id]),
                "current": read_rgb(rgb_by_id[current_id]),
                "motion_previous": read_motion(motion_by_id[previous_id]),
                "motion_current": read_motion(motion_by_id[current_id]),
            }
        )

    channel_count = min(sample_motion.shape[2], 4)
    configs = []
    for motion_frame in ("current", "previous"):
        for x_channel, y_channel in itertools.permutations(range(channel_count), 2):
            for x_sign in (-1, 1):
                for y_sign in (-1, 1):
                    for pixel_scale in (1.0, 0.5):
                        configs.append(
                            {
                                "motion_frame": motion_frame,
                                "x_channel": x_channel,
                                "y_channel": y_channel,
                                "x_sign": x_sign,
                                "y_sign": y_sign,
                                "pixel_scale": pixel_scale,
                            }
                        )

    results = []
    for config in configs:
        result = evaluate_config(records, config)
        if result is not None:
            results.append(result)
    if not results:
        raise RuntimeError("所有 Motion Vector convention 測試皆失敗")
    results.sort(key=lambda item: item["warp_mae"])

    print("\n🏆 Photometric warp error 最低的 conventions")
    for rank, result in enumerate(results[:12], start=1):
        print(
            f"{rank:02d}. MAE={result['warp_mae']:.6f} | "
            f"baseline={result['baseline_mae_on_same_valid_pixels']:.6f} | "
            f"improvement={result['improvement_fraction'] * 100:+.2f}% | "
            f"valid={result['valid_fraction'] * 100:.1f}% | "
            f"frame={result['motion_frame']} "
            f"x=ch{result['x_channel']}*{result['x_sign']:+d} "
            f"y=ch{result['y_channel']}*{result['y_sign']:+d} "
            f"scale={result['pixel_scale']}"
        )

    best = results[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_preview(output_dir, records[0], best)

    report = {
        "scene": args.scene,
        "angle": args.angle,
        "resolution": args.resolution,
        "rgb_directory": str(rgb_dir),
        "motion_directory": str(motion_dir),
        "num_rgb_frames": len(rgb_paths),
        "num_motion_frames": len(motion_paths),
        "tested_pairs": pair_count,
        "motion_shape": list(sample_motion.shape),
        "motion_dtype_after_load": str(sample_motion.dtype),
        "channel_statistics": stats,
        "best_convention": best,
        "top_12_conventions": results[:12],
    }
    report_path = output_dir / "motion_vector_inspection.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n✅ 檢查完成")
    print(f"Best convention: {best}")
    print(f"JSON report:     {report_path}")
    print(f"Preview images:  {output_dir}")
    if best["improvement_fraction"] <= 0:
        print(
            "⚠️ 最佳 warp 仍未優於不做 warp。請先不要實作 C1，"
            "需要進一步確認 MV frame association 或 modality。"
        )


if __name__ == "__main__":
    main()

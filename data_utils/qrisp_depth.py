"""QRISP packed z-buffer decoder (ICCV 2023 supplemental, Eq. 4).

Returns encoded z-buffer depth, NOT metric/linear camera distance.
OpenCV returns BGRA; decoding requires R,G,B,A in that order.
"""
from pathlib import Path
import numpy as np


def decode_qrisp_depth_bgra(image):
    if image is None or image.ndim != 3 or image.shape[-1] != 4:
        raise ValueError("QRISP depth must be a four-channel PNG; preserve alpha")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 BGRA depth, got {image.dtype}")
    # Decode in float64 BEFORE converting to float32. No per-frame normalization.
    rgba = image[..., [2, 1, 0, 3]].astype(np.float64)
    weights = 1.0 / np.power(255.0, np.arange(1, 5, dtype=np.float64))
    depth = (rgba * weights).sum(axis=-1)
    if not np.isfinite(depth).all():
        raise ValueError("Decoded depth contains NaN/Inf")
    # Preserve raw depth; confidence excludes pixels outside [0,1].
    return depth.astype(np.float32)


def read_qrisp_depth(path):
    import cv2
    image = cv2.imread(str(Path(path)), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read depth: {path}")
    return decode_qrisp_depth_bgra(image)


def depth_tensor(path):
    import torch
    return torch.from_numpy(np.ascontiguousarray(read_qrisp_depth(path))).unsqueeze(0)


def matching_depth_path(rgb_path, modality="DepthMipBiasMinus2"):
    path = Path(rgb_path)
    # .../270p/Native/<segment>/<frame>.png
    if path.parent.parent.name != "Native":
        raise ValueError(f"Expected 270p/Native/segment/frame.png, got {path}")
    result = path.parent.parent.parent / modality / path.parent.name / (path.stem + ".png")
    if not result.is_file():
        raise FileNotFoundError(f"Missing paired depth (do not silently change split): {result}")
    return result


def check_contiguous(paths):
    """Fail on numeric frame gaps; renderer MV is for adjacent frames."""
    stems = [Path(p).stem for p in paths]
    if all(s.isdigit() for s in stems):
        indices = [int(s) for s in stems]
        if any(b != a + 1 for a, b in zip(indices, indices[1:])):
            raise ValueError(f"Nonconsecutive frame IDs: {stems}")

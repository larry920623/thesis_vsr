import argparse
import os
import random
from pathlib import Path

# Must be set before importing cv2.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from archs.basicvsrpp.causal_basicvsrpp_mv import CausalBasicVSRPlusPlusMV


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target):
        difference = prediction - target
        return torch.mean(torch.sqrt(difference * difference + self.eps))


def files_by_stem(directory: Path, extensions):
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    }


def resolve_scene_root(qrisp_root: Path, scene: str) -> Path:
    for path in (
        qrisp_root / scene,
        qrisp_root / "TrainSet" / scene,
        qrisp_root / "TestSet" / scene,
    ):
        if path.is_dir():
            return path
    raise FileNotFoundError(f"找不到 scene: {scene}")


class QRISPRendererMVDataset(Dataset):
    """QRISP Native RGB + non-jittered renderer MV sequence dataset."""

    def __init__(
        self,
        qrisp_root,
        split="train",
        num_frames=15,
        lr_patch_size=64,
        scale=4,
        motion_modality="MotionVectorsMipBiasMinus2",
    ):
        self.qrisp_root = Path(qrisp_root)
        self.split = split
        self.num_frames = num_frames
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.scale = scale
        self.motion_modality = motion_modality

        list_path = self.qrisp_root / f"{split}_list.txt"
        if not list_path.is_file():
            raise FileNotFoundError(f"找不到資料清單: {list_path}")
        scenes = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.samples = []
        for scene in scenes:
            scene_root = resolve_scene_root(self.qrisp_root, scene)
            lr_root = scene_root / "270p" / "Native"
            hr_root = scene_root / "1080p" / "Native"
            motion_root = scene_root / "270p" / motion_modality
            if not lr_root.is_dir() or not hr_root.is_dir() or not motion_root.is_dir():
                print(f"⚠️ 跳過 {scene}: RGB/HR/MV modality 不完整")
                continue

            segments = sorted(
                set(path.name for path in lr_root.iterdir() if path.is_dir())
                & set(path.name for path in hr_root.iterdir() if path.is_dir())
                & set(path.name for path in motion_root.iterdir() if path.is_dir())
            )
            for segment in segments:
                lr_by_stem = files_by_stem(
                    lr_root / segment, VALID_IMAGE_EXTENSIONS
                )
                hr_by_stem = files_by_stem(
                    hr_root / segment, VALID_IMAGE_EXTENSIONS
                )
                motion_by_stem = files_by_stem(
                    motion_root / segment, {".exr"}
                )
                common_stems = sorted(
                    set(lr_by_stem) & set(hr_by_stem) & set(motion_by_stem)
                )
                if len(common_stems) < num_frames:
                    print(
                        f"⚠️ 跳過 {scene}/{segment}: 只有 "
                        f"{len(common_stems)} 組 RGB/HR/MV"
                    )
                    continue
                self.samples.append(
                    {
                        "scene": scene,
                        "segment": segment,
                        "lr": [lr_by_stem[stem] for stem in common_stems],
                        "hr": [hr_by_stem[stem] for stem in common_stems],
                        "motion": [motion_by_stem[stem] for stem in common_stems],
                    }
                )

        if not self.samples:
            raise RuntimeError(f"QRISP {split} split 沒有有效的 RGB/HR/MV sequence")
        print(
            f"📦 QRISP Renderer MV [{split}]：{len(self.samples)} 個 sequences，"
            f"MV={motion_modality}"
        )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def read_rgb(path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"無法讀取 RGB: {path}")
        return image

    @staticmethod
    def read_renderer_motion(path: Path):
        """Decode QRISP EXR into current->previous LR pixel displacement."""
        motion = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if motion is None:
            raise RuntimeError(f"無法讀取 EXR Motion Vector: {path}")
        if motion.ndim != 3 or motion.shape[2] < 3:
            raise ValueError(f"預期至少 3-channel EXR，得到 {motion.shape}: {path}")
        motion = motion.astype(np.float32)
        height, width = motion.shape[:2]

        # Empirically verified on SpaceShipDemo and Flooded_Grounds:
        # current frame, x = -OpenCV channel 2 * width,
        # y = +OpenCV channel 1 * height.
        flow_x = -motion[..., 2] * width
        flow_y = motion[..., 1] * height
        flow = np.stack([flow_x, flow_y], axis=2)
        if not np.isfinite(flow).all():
            raise ValueError(f"Motion Vector 包含 NaN/Inf: {path}")
        return flow.astype(np.float32)

    def __getitem__(self, index):
        sample = self.samples[index]
        total_frames = len(sample["lr"])
        if self.split == "train":
            start = random.randint(0, total_frames - self.num_frames)
        else:
            start = (total_frames - self.num_frames) // 2
        stop = start + self.num_frames

        lr_images = [self.read_rgb(path) for path in sample["lr"][start:stop]]
        hr_images = [self.read_rgb(path) for path in sample["hr"][start:stop]]
        motions = [
            self.read_renderer_motion(path)
            for path in sample["motion"][start:stop]
        ]

        lr_height, lr_width = lr_images[0].shape[:2]
        for motion in motions:
            if motion.shape[:2] != (lr_height, lr_width):
                raise ValueError(
                    f"RGB/MV 尺寸不一致: RGB {(lr_height, lr_width)}, "
                    f"MV {motion.shape[:2]}"
                )

        if self.split == "train":
            if lr_height < self.lr_patch_size or lr_width < self.lr_patch_size:
                raise ValueError("LR frame 小於指定 patch size")
            top = random.randint(0, lr_height - self.lr_patch_size)
            left = random.randint(0, lr_width - self.lr_patch_size)
            hr_top, hr_left = top * self.scale, left * self.scale

            lr_images = [
                image[
                    top : top + self.lr_patch_size,
                    left : left + self.lr_patch_size,
                ]
                for image in lr_images
            ]
            motions = [
                motion[
                    top : top + self.lr_patch_size,
                    left : left + self.lr_patch_size,
                ]
                for motion in motions
            ]
            hr_images = [
                image[
                    hr_top : hr_top + self.hr_patch_size,
                    hr_left : hr_left + self.hr_patch_size,
                ]
                for image in hr_images
            ]

            if random.random() < 0.5:
                lr_images = [cv2.flip(image, 1) for image in lr_images]
                hr_images = [cv2.flip(image, 1) for image in hr_images]
                motions = [cv2.flip(motion, 1) for motion in motions]
                for motion in motions:
                    motion[..., 0] *= -1.0

            if random.random() < 0.5:
                lr_images = [cv2.flip(image, 0) for image in lr_images]
                hr_images = [cv2.flip(image, 0) for image in hr_images]
                motions = [cv2.flip(motion, 0) for motion in motions]
                for motion in motions:
                    motion[..., 1] *= -1.0

            if random.random() < 0.5:
                lr_images = [
                    cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    for image in lr_images
                ]
                hr_images = [
                    cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    for image in hr_images
                ]
                rotated_motions = []
                for motion in motions:
                    rotated = cv2.rotate(motion, cv2.ROTATE_90_CLOCKWISE)
                    old_x = rotated[..., 0].copy()
                    old_y = rotated[..., 1].copy()
                    # Clockwise image rotation: (dx, dy) -> (-dy, dx).
                    rotated[..., 0] = -old_y
                    rotated[..., 1] = old_x
                    rotated_motions.append(rotated)
                motions = rotated_motions

        def image_to_tensor(image):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32) / 255.0
            return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)

        def motion_to_tensor(motion):
            motion = np.ascontiguousarray(motion, dtype=np.float32)
            return torch.from_numpy(motion).permute(2, 0, 1)

        return (
            torch.stack([image_to_tensor(image) for image in lr_images]),
            torch.stack([motion_to_tensor(motion) for motion in motions]),
            torch.stack([image_to_tensor(image) for image in hr_images]),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train C1 Causal BasicVSR++ with QRISP renderer MV"
    )
    parser.add_argument(
        "--qrisp-root",
        default="/home/larry/ssd_data/sr_project/datasets/QRISP",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=(
            "/home/larry/ssd_data/sr_project/checkpoints/"
            "causal_basicvsrpp_mv"
        ),
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--num-frames", type=int, default=15)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "architecture": "CausalBasicVSRPlusPlusMV",
            "motion_convention": {
                "motion_frame": "current",
                "x": "-exr_channel_2 * width",
                "y": "+exr_channel_1 * height",
                "units": "LR pixels",
            },
        },
        path,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("此訓練需要 CUDA 與 MMCV DCN operator")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print("🚀 初始化 C1：Causal BasicVSR++ + Renderer MV")
    model = CausalBasicVSRPlusPlusMV(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
    ).to(device)
    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(f"🧠 Parameters: {params_m:.2f} M（SPyNet 已完全移除）")

    train_dataset = QRISPRendererMVDataset(
        args.qrisp_root,
        split="train",
        num_frames=args.num_frames,
        lr_patch_size=args.patch_size,
    )
    val_dataset = QRISPRendererMVDataset(
        args.qrisp_root,
        split="val",
        num_frames=args.num_frames,
        lr_patch_size=args.patch_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_workers = max(0, args.num_workers // 2)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=val_workers > 0,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CharbonnierLoss().to(device)
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"♻️ 從 epoch {start_epoch} 繼續訓練: {args.resume}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    printed_motion_stats = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_sum = 0.0
        for batch_index, (lr_sequence, motions, hr_sequence) in enumerate(train_loader):
            lr_sequence = lr_sequence.to(device, non_blocking=True)
            motions = motions.to(device, non_blocking=True)
            hr_sequence = hr_sequence.to(device, non_blocking=True)

            if not printed_motion_stats:
                print(
                    "🔎 First batch MV: "
                    f"shape={tuple(motions.shape)}, "
                    f"min={motions.min().item():.3f}, "
                    f"max={motions.max().item():.3f}, "
                    f"mean_abs={motions.abs().mean().item():.3f} pixels"
                )
                printed_motion_stats = True

            optimizer.zero_grad(set_to_none=True)
            outputs = model(lr_sequence, motions)
            loss = criterion(outputs, hr_sequence)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)
            optimizer.step()
            train_loss_sum += loss.item()

            if batch_index % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Batch [{batch_index}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        scheduler.step()
        average_train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for lr_sequence, motions, hr_sequence in val_loader:
                lr_sequence = lr_sequence.to(device, non_blocking=True)
                motions = motions.to(device, non_blocking=True)
                hr_sequence = hr_sequence.to(device, non_blocking=True)
                outputs = model(lr_sequence, motions)
                val_loss_sum += criterion(outputs, hr_sequence).item()
        average_val_loss = val_loss_sum / len(val_loader)

        print(
            f"🏁 Epoch {epoch + 1}: train={average_train_loss:.6f}, "
            f"val={average_val_loss:.6f}, "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )
        latest_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            epoch,
            min(best_val_loss, average_val_loss),
        )
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
            )
            print(f"🏆 已更新最佳 C1 模型: {best_path}")
        print("-" * 72)


if __name__ == "__main__":
    main()

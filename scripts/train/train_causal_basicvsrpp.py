import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from archs.basicvsrpp.causal_basicvsr_pp import CausalBasicVSRPlusPlus


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target):
        diff = prediction - target
        return torch.mean(torch.sqrt(diff * diff + self.eps))


class QRISPDataset(Dataset):
    """QRISP RGB sequence dataset used by the original baseline script."""

    def __init__(
        self,
        qrisp_root,
        split="train",
        num_frames=15,
        lr_patch_size=64,
        scale=4,
    ):
        self.num_frames = num_frames
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.scale = scale
        self.split = split

        list_path = os.path.join(qrisp_root, f"{split}_list.txt")
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"找不到資料清單: {list_path}")
        with open(list_path, "r", encoding="utf-8") as file:
            scenes = [line.strip() for line in file if line.strip()]

        self.samples = []
        valid_exts = (".png", ".jpg", ".jpeg", ".exr")
        for scene in scenes:
            scene_dir = os.path.join(qrisp_root, scene)
            lr_native_dir = os.path.join(scene_dir, "270p", "Native")
            hr_native_dir = os.path.join(scene_dir, "1080p", "Native")
            if not os.path.isdir(lr_native_dir) or not os.path.isdir(hr_native_dir):
                print(f"⚠️ 跳過 {scene}: 找不到 270p/Native 或 1080p/Native")
                continue

            angles = sorted(
                directory
                for directory in os.listdir(lr_native_dir)
                if os.path.isdir(os.path.join(lr_native_dir, directory))
            )
            for angle in angles:
                lr_angle_dir = os.path.join(lr_native_dir, angle)
                hr_angle_dir = os.path.join(hr_native_dir, angle)
                if not os.path.isdir(hr_angle_dir):
                    print(f"⚠️ 跳過 {scene}/{angle}: 找不到 HR 角度資料夾")
                    continue

                lr_frames = sorted(
                    os.path.join(lr_angle_dir, name)
                    for name in os.listdir(lr_angle_dir)
                    if name.lower().endswith(valid_exts)
                )
                hr_frames = sorted(
                    os.path.join(hr_angle_dir, name)
                    for name in os.listdir(hr_angle_dir)
                    if name.lower().endswith(valid_exts)
                )
                if len(lr_frames) < num_frames:
                    print(
                        f"⚠️ 跳過 {scene}/{angle}: 影格數太少 "
                        f"(LR 有 {len(lr_frames)} 張)"
                    )
                    continue
                if len(lr_frames) != len(hr_frames):
                    print(f"⚠️ 跳過 {scene}/{angle}: LR/HR 影格數不一致")
                    continue
                self.samples.append(
                    {
                        "scene_angle": f"{scene}/{angle}",
                        "lr_frames": lr_frames,
                        "hr_frames": hr_frames,
                    }
                )

        if not self.samples:
            raise RuntimeError(f"QRISP {split} split 沒有任何有效 sequence")
        print(
            f"📦 QRISP [{split}] 初始化完成，共 {len(self.samples)} 個 sequences。"
        )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_image(path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"無法讀取影像: {path}")
        return image

    def __getitem__(self, index):
        sample = self.samples[index]
        total_frames = len(sample["lr_frames"])
        if self.split == "train":
            start = random.randint(0, total_frames - self.num_frames)
        else:
            start = (total_frames - self.num_frames) // 2

        stop = start + self.num_frames
        lr_images = [
            self._read_image(path) for path in sample["lr_frames"][start:stop]
        ]
        hr_images = [
            self._read_image(path) for path in sample["hr_frames"][start:stop]
        ]

        if self.split == "train":
            lr_h, lr_w = lr_images[0].shape[:2]
            if lr_h < self.lr_patch_size or lr_w < self.lr_patch_size:
                raise ValueError(
                    f"LR frame {lr_w}x{lr_h} 小於 patch "
                    f"{self.lr_patch_size}x{self.lr_patch_size}"
                )
            top = random.randint(0, lr_h - self.lr_patch_size)
            left = random.randint(0, lr_w - self.lr_patch_size)
            hr_top, hr_left = top * self.scale, left * self.scale

            lr_images = [
                image[
                    top : top + self.lr_patch_size,
                    left : left + self.lr_patch_size,
                ]
                for image in lr_images
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
            if random.random() < 0.5:
                lr_images = [cv2.flip(image, 0) for image in lr_images]
                hr_images = [cv2.flip(image, 0) for image in hr_images]
            if random.random() < 0.5:
                lr_images = [
                    cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    for image in lr_images
                ]
                hr_images = [
                    cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    for image in hr_images
                ]

        def to_tensor(image):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32) / 255.0
            return torch.from_numpy(rgb).permute(2, 0, 1)

        return (
            torch.stack([to_tensor(image) for image in lr_images]),
            torch.stack([to_tensor(image) for image in hr_images]),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train RGB-only Causal BasicVSR++")
    parser.add_argument(
        "--qrisp-root",
        default="/home/larry/ssd_data/sr_project/datasets/QRISP",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/home/larry/ssd_data/sr_project/checkpoints/causal_basicvsrpp",
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
            "architecture": "CausalBasicVSRPlusPlus",
            "mid_channels": model.mid_channels,
            "is_low_res_input": model.is_low_res_input,
        },
        path,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("此訓練設定需要 CUDA 與可用的 MMCV DCN operator")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print("🚀 初始化 RGB-only Causal BasicVSR++ 訓練程式...")
    model = CausalBasicVSRPlusPlus(
        mid_channels=64, num_blocks=7, is_low_res_input=True
    ).to(device)

    train_dataset = QRISPDataset(
        args.qrisp_root,
        split="train",
        num_frames=args.num_frames,
        lr_patch_size=args.patch_size,
    )
    val_dataset = QRISPDataset(
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
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
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_sum = 0.0
        for batch_idx, (lr_sequence, hr_sequence) in enumerate(train_loader):
            lr_sequence = lr_sequence.to(device, non_blocking=True)
            hr_sequence = hr_sequence.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(lr_sequence)
            loss = criterion(output, hr_sequence)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)
            optimizer.step()
            train_loss_sum += loss.item()

            if batch_idx % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        scheduler.step()
        average_train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for lr_sequence, hr_sequence in val_loader:
                lr_sequence = lr_sequence.to(device, non_blocking=True)
                hr_sequence = hr_sequence.to(device, non_blocking=True)
                output = model(lr_sequence)
                val_loss_sum += criterion(output, hr_sequence).item()
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
            print(f"🏆 已更新最佳模型: {best_path}")
        print("-" * 60)


if __name__ == "__main__":
    main()

"""C3 training: joint fine-tuning or a gate-only diagnostic.

Uses the existing C2 dataset and Charbonnier loss without changing their
sampling, augmentation, depth decoding, or loss definition.
Requires the corrected encoded_depth_feature_gate_v2 C3 model.
"""

import argparse
import math
import os
import random
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# Import torch first, matching the successful environment diagnostic.
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from archs.basicvsrpp.causal_basicvsrpp_mv_depth_gate import (
    CausalBasicVSRPlusPlusMVDepthGate,
)
from scripts.train.train_causal_basicvsrpp_mv_depth import (
    QRISPRendererMVDepthDataset,
    CharbonnierLoss,
)


CHECKPOINT_ROOT = "/home/larry/ssd_data/sr_project/checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--qrisp-root",
        default="/home/larry/ssd_data/sr_project/datasets/QRISP",
    )
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init-c1",
        default=f"{CHECKPOINT_ROOT}/causal_basicvsrpp_mv/best_model.pth",
    )
    parser.add_argument("--depth-tau", type=float, default=0.001)
    parser.add_argument(
        "--gate-mode",
        choices=("learned", "zero", "one"),
        default="learned",
    )
    parser.add_argument(
        "--gate-hidden-channels",
        type=int,
        default=16,
    )
    parser.add_argument("--initial-gate", type=float, default=0.99)
    parser.add_argument("--gate-only", action="store_true")

    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-sequences", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=15)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.checkpoint_dir is None:
        folder = (
            "c3_gate_only_diagnostic"
            if args.gate_only
            else "c3_learned_history_gate"
        )
        args.checkpoint_dir = str(Path(CHECKPOINT_ROOT) / folder)

    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_args(args):
    if args.epochs < 1 or args.num_frames < 3:
        raise ValueError("epochs >= 1; num-frames >= 3")

    if args.batch_size < 1 or args.patch_size < 2:
        raise ValueError("batch-size >= 1; patch-size >= 2")

    if args.num_workers < 0:
        raise ValueError("num-workers must be >= 0")

    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")

    if not math.isfinite(args.depth_tau) or args.depth_tau <= 0:
        raise ValueError("depth-tau must be finite and positive")

    if not 0 < args.initial_gate < 1:
        raise ValueError("initial-gate must be between 0 and 1")

    if args.gate_hidden_channels < 1:
        raise ValueError("gate-hidden-channels must be positive")

    if args.gate_only and args.gate_mode != "learned":
        raise ValueError("--gate-only requires --gate-mode learned")

    for name in ("max_train_batches", "max_val_sequences"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
):
    torch.save(
        {
            "model": model.state_dict(),
            "confidence_config": model.confidence_config(),
            "training_protocol": model.training_protocol,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "architecture": "CausalBasicVSRPlusPlusMVDepthGate",
            "motion_convention": {
                "motion_frame": "current",
                "x": "-exr_channel_2 * width",
                "y": "+exr_channel_1 * height",
                "units": "LR pixels",
            },
        },
        path,
    )


def register_gate_observers(model, records):
    def make_hook(name):
        def hook(module, inputs, output):
            gate = output.detach().float()

            records[name].append({
                "mean": gate.mean().item(),
                "min": gate.min().item(),
                "max": gate.max().item(),
            })

        return hook

    return [
        gate.register_forward_hook(make_hook(name))
        for name, gate in model.history_gates.items()
    ]


def print_gate_statistics(records):
    print(
        "Gate probe: first validation sequence, "
        "history frames only"
    )

    for name, values in records.items():
        if not values:
            print(f"  {name}: gate network bypassed")
            continue

        mean = sum(
            value["mean"] for value in values
        ) / len(values)
        minimum = min(value["min"] for value in values)
        maximum = max(value["max"] for value in values)

        print(
            f"  {name}: mean={mean:.6f}, "
            f"min={minimum:.6f}, max={maximum:.6f}"
        )


def main():
    args = parse_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This training requires CUDA and MMCV DCN"
        )

    seed_everything(args.seed)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(
        "Initializing C3: "
        "Geometry-Conditioned Learned History Gate"
    )

    model = CausalBasicVSRPlusPlusMVDepthGate(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        depth_tau=args.depth_tau,
        gate_mode=args.gate_mode,
        gate_hidden_channels=args.gate_hidden_channels,
        initial_gate=args.initial_gate,
    ).to(device)

    if (
        model.confidence_config().get("type")
        != "encoded_depth_feature_gate_v2"
    ):
        raise ValueError(
            "Update the C3 model to encoded-depth v2 first"
        )

    if args.gate_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("history_gates.")
            )
        print("Gate-only diagnostic: backbone frozen")
    else:
        print(
            "Joint fine-tuning: "
            "backbone and gate parameters trainable"
        )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    total_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    if trainable_count == 0:
        raise RuntimeError("No trainable parameters")

    print(f"Total parameters: {total_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print("Confidence config:", model.confidence_config())

    model.training_protocol = vars(args).copy()

    train_dataset = QRISPRendererMVDepthDataset(
        args.qrisp_root,
        split="train",
        num_frames=args.num_frames,
        lr_patch_size=args.patch_size,
    )
    val_dataset = QRISPRendererMVDepthDataset(
        args.qrisp_root,
        split="val",
        num_frames=args.num_frames,
        lr_patch_size=args.patch_size,
    )

    if args.max_val_sequences is not None:
        val_dataset.samples = val_dataset.samples[
            :args.max_val_sequences
        ]

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

    optimizer = optim.Adam(
        trainable_parameters,
        lr=args.lr,
        betas=(0.9, 0.99),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    criterion = CharbonnierLoss().to(device)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
        )

        if (
            checkpoint.get("confidence_config")
            != model.confidence_config()
        ):
            raise ValueError(
                "Resume confidence/gate configuration mismatch"
            )

        saved_protocol = checkpoint.get(
            "training_protocol", {}
        )

        if (
            bool(saved_protocol.get("gate_only", False))
            != args.gate_only
        ):
            raise ValueError(
                "Resume must use the original gate-only setting"
            )

        for name in (
            "epochs",
            "num_frames",
            "patch_size",
            "batch_size",
            "lr",
            "seed",
            "max_train_batches",
            "max_val_sequences",
        ):
            if (
                name in saved_protocol
                and saved_protocol[name] != getattr(args, name)
            ):
                raise ValueError(
                    f"Resume setting mismatch: {name}; "
                    f"saved={saved_protocol[name]}, "
                    f"current={getattr(args, name)}"
                )

        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        if scheduler.T_max != args.epochs:
            raise ValueError(
                "Resume epochs must match "
                "the original schedule budget"
            )

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        model.training_protocol = {
            **vars(args),
            **saved_protocol,
            "resume": args.resume,
            "checkpoint_dir": args.checkpoint_dir,
        }

        print(
            f"Resuming from epoch {start_epoch + 1}: "
            f"{args.resume}"
        )

        del checkpoint

    else:
        checkpoint = torch.load(
            args.init_c1,
            map_location="cpu",
        )

        weights = checkpoint.get(
            "model",
            checkpoint.get("state_dict", checkpoint),
        )
        weights = {
            key.removeprefix("module.").removeprefix(
                "generator."
            ): value
            for key, value in weights.items()
        }

        result = model.load_c1_state_dict(weights)

        print("New gate keys:", result.missing_keys)
        print(f"Loaded C1 weights: {args.init_c1}")

        del weights, checkpoint

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    printed_input_stats = False

    for epoch in range(start_epoch, args.epochs):
        model.train()

        train_loss_sum = 0.0
        train_batches = 0
        epoch_lr = optimizer.param_groups[0]["lr"]

        for batch_index, batch in enumerate(train_loader):
            (
                lr_sequence,
                motions,
                depths,
                hr_sequence,
            ) = [
                tensor.to(device, non_blocking=True)
                for tensor in batch
            ]

            if not printed_input_stats:
                print(
                    "First batch MV: "
                    f"shape={tuple(motions.shape)}, "
                    f"min={motions.min().item():.3f}, "
                    f"max={motions.max().item():.3f}, "
                    "mean_abs="
                    f"{motions.abs().mean().item():.3f} pixels"
                )
                print(
                    "First batch depth: "
                    f"shape={tuple(depths.shape)}, "
                    f"min={depths.min().item():.6f}, "
                    f"max={depths.max().item():.6f}; "
                    "encoded z-buffer"
                )
                printed_input_stats = True

            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                lr_sequence,
                motions,
                depths,
            )
            loss = criterion(outputs, hr_sequence)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Nonfinite training loss; run not saved"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=50,
                error_if_nonfinite=True,
            )

            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

            if batch_index % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Batch [{batch_index}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

            del (
                lr_sequence,
                motions,
                depths,
                hr_sequence,
                outputs,
                loss,
            )

            if (
                args.max_train_batches is not None
                and train_batches >= args.max_train_batches
            ):
                break

        if train_batches == 0:
            raise RuntimeError("No training batches")

        scheduler.step()
        average_train_loss = train_loss_sum / train_batches

        model.eval()

        val_loss_sum = 0.0
        val_batches = 0

        gate_records = {
            name: []
            for name in model.history_gates
        }
        handles = register_gate_observers(
            model,
            gate_records,
        )

        try:
            with torch.no_grad():
                for batch_index, batch in enumerate(val_loader):
                    (
                        lr_sequence,
                        motions,
                        depths,
                        hr_sequence,
                    ) = [
                        tensor.to(device, non_blocking=True)
                        for tensor in batch
                    ]

                    outputs = model(
                        lr_sequence,
                        motions,
                        depths,
                    )

                    # Record only the first validation sequence.
                    if batch_index == 0:
                        for handle in handles:
                            handle.remove()
                        handles = []

                    val_loss = criterion(
                        outputs,
                        hr_sequence,
                    )

                    if not torch.isfinite(val_loss):
                        raise RuntimeError(
                            "Nonfinite validation loss; "
                            "run not saved"
                        )

                    val_loss_sum += val_loss.item()
                    val_batches += 1

                    del (
                        lr_sequence,
                        motions,
                        depths,
                        hr_sequence,
                        outputs,
                        val_loss,
                    )
        finally:
            for handle in handles:
                handle.remove()

        if val_batches == 0:
            raise RuntimeError("No validation batches")

        average_val_loss = val_loss_sum / val_batches

        print(
            f"Epoch {epoch + 1}: "
            f"train={average_train_loss:.6f}, "
            f"val={average_val_loss:.6f}, "
            f"lr_used={epoch_lr:.3e}, "
            f"next_lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        print_gate_statistics(gate_records)

        improved = average_val_loss < best_val_loss
        if improved:
            best_val_loss = average_val_loss

        save_checkpoint(
            checkpoint_dir / "latest_checkpoint.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
        )

        if improved:
            best_path = checkpoint_dir / "best_model.pth"

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
            )

            print(
                f"Updated best C3 model: {best_path}"
            )

        print("-" * 72)

    print(
        "Training finished. "
        f"Best validation loss: {best_val_loss:.6f}"
    )
    print(f"Checkpoint directory: {checkpoint_dir}")


if __name__ == "__main__":
    main()
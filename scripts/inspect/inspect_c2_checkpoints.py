"""Read the user's own C2 checkpoints on CPU; never rewrite the originals."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def inspect(path):
    result = {"path": str(path.resolve()), "ok": False}
    try:
        result["size_bytes"] = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint is not a dictionary")
        weights = checkpoint.get("model", checkpoint.get("state_dict"))
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Missing nonempty model/state_dict")
        tensors = [value for value in weights.values() if torch.is_tensor(value)]
        if not tensors:
            raise ValueError("No model tensors")
        invalid = [key for key, value in weights.items()
                   if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())
                   and not bool(torch.isfinite(value).all())]
        epoch = checkpoint.get("epoch")
        loss = checkpoint.get("best_val_loss")
        loss = float(loss) if loss is not None else None
        result.update({
            "epoch_zero_based": epoch,
            "saved_epoch_display": epoch + 1 if isinstance(epoch, int) else None,
            "best_val_loss": loss if loss is None or math.isfinite(loss) else str(loss),
            "architecture": checkpoint.get("architecture"),
            "confidence_config": checkpoint.get("confidence_config"),
            "training_protocol": checkpoint.get("training_protocol"),
            "has_optimizer": "optimizer" in checkpoint,
            "has_scheduler": "scheduler" in checkpoint,
            "num_model_tensors": len(tensors),
            "nonfinite_model_keys": invalid,
        })
        if invalid:
            raise ValueError("Model has NaN/Inf tensors")
        result["ok"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(
        "/home/larry/ssd_data/sr_project/checkpoints/causal_basicvsrpp_mv_depth"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [inspect(args.checkpoint_dir / name)
               for name in ("best_model.pth", "latest_checkpoint.pth")]
    payload = {"torch_version": str(torch.__version__), "checkpoints": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in results:
        print(Path(row["path"]).name, "OK" if row["ok"] else "FAILED",
              "saved_epoch=", row.get("saved_epoch_display"),
              "best_val_loss=", row.get("best_val_loss"))
        print("confidence_config:", row.get("confidence_config"))
        if not row["ok"]:
            print(row.get("error"))
    print("Saved:", args.output)
    if not all(row["ok"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

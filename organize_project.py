"""Group thesis_vsr entry scripts without rewriting their contents.

Place this script and run.py in the project root. Preview by default; --apply
moves files. Does not touch architectures, datasets, checkpoints or results.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


GROUPS = {
    "train": ("train_",),
    "eval": ("evaluate_",),
    "infer": ("run_",),
    "inspect": ("inspect_",),
    "tests": ("smoke_test_",),
}


def destination(root, path):
    if path.name == "qrisp_depth.py":
        return root / "data_utils" / path.name
    for group, prefixes in GROUPS.items():
        if path.name.startswith(prefixes):
            return root / "scripts" / group / path.name
    return None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true", help="Move files; default only prints plan")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    root = args.root.resolve()
    if not (root / "archs").is_dir():
        raise SystemExit(f"Expected project root containing archs/: {root}")
    if not (root / "run.py").is_file():
        raise SystemExit("Copy the supplied run.py into project root first")
    plan = []
    for src in sorted(root.glob("*.py")):
        if not src.is_file() or src.is_symlink():
            continue
        dst = destination(root, src)
        if dst is not None:
            # Any duplicate is a conflict; never overwrite or guess its version.
            if dst.exists():
                raise SystemExit(f"Destination already exists; resolve duplicates before moving: {dst}")
            plan.append((src, dst))
    if not plan:
        print("No root-level entry scripts to move. Already organized, or no matching scripts.")
        return
    for src, dst in plan:
        print(f"{src.relative_to(root)} -> {dst.relative_to(root)}")
    print(f"\nFiles: {len(plan)}")
    if not args.apply:
        print("Preview only. Execute: python organize_project.py --apply")
        return
    backup = root / "backups" / ("folder_cleanup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    backup.mkdir(parents=True, exist_ok=False)
    manifest = []
    # Back up every source before moving the first source.
    for src, dst in plan:
        shutil.copy2(src, backup / src.name)
        manifest.append({"source": str(src.relative_to(root)),
                         "destination": str(dst.relative_to(root)),
                         "sha256": digest(src)})
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    moved = []
    try:
        for (src, dst), item in zip(plan, manifest):
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                raise RuntimeError(f"Destination appeared during move: {dst}")
            shutil.move(str(src), str(dst))
            moved.append((src, dst))
            if digest(dst) != item["sha256"]:
                raise RuntimeError(f"Content verification failed: {dst}")
    except Exception:
        for src, dst in reversed(moved):
            if dst.is_file() and not src.exists():
                shutil.move(str(dst), str(src))
        raise
    for group in GROUPS:
        (root / "scripts" / group).mkdir(parents=True, exist_ok=True)
    (root / "data_utils").mkdir(exist_ok=True)
    print(f"\nComplete. Original contents preserved: {len(moved)} files")
    print(f"Backup: {backup.relative_to(root)}")
    print("List commands: python run.py --list")


if __name__ == "__main__":
    main()

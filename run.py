"""Stable entrypoint: python run.py SCRIPT_NAME [original script arguments]."""
import os
import runpy
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    groups = ("train", "eval", "infer", "inspect", "tests")
    directories = [root / "scripts" / group for group in groups]
    scripts = sorted(path for directory in directories for path in directory.glob("*.py")
                     if not path.name.startswith("_") and path.is_file())
    if len(sys.argv) < 2 or sys.argv[1] in ("--list", "-h", "--help"):
        print("Usage: python run.py SCRIPT_NAME [original arguments]")
        print("Examples: python run.py evaluate_qrisp_c1_mv --max-sequences 1")
        print("Available scripts:")
        for path in scripts:
            print(f"  {path.stem:<42} {path.relative_to(root)}")
        return
    name = sys.argv[1]
    candidates = [path for path in scripts if name in
                  (path.stem, path.name, path.relative_to(root).as_posix(),
                   path.relative_to(root / "scripts").as_posix())]
    if len(candidates) != 1:
        message = "Unknown script" if not candidates else "Ambiguous script name"
        raise SystemExit(f"{message}: {name}. Use python run.py --list")
    target = candidates[0]
    # Keep working-directory paths identical to the previous root execution.
    os.chdir(root)
    # Preserve legacy bare imports both within and across script categories.
    # The target directory takes precedence, then root and other categories.
    paths = [target.parent, root, root / "data_utils", *directories]
    seen = set()
    import_paths = []
    for path in paths:
        value = str(path)
        if value not in seen:
            import_paths.append(value)
            seen.add(value)
    sys.path[:0] = import_paths
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

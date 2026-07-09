"""Create an isolated epi-diff-abm runtime with shared large data assets.

The epi-diff-abm runner mutates ``covid_abm/yamls/config.yaml`` while building
an Executor. A copied runtime gives each worker its own config file and lock,
while symlinking large immutable assets such as ``data``.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


SKIP_NAMES = {
    ".git",
    "__pycache__",
}

SYMLINK_NAMES = {
    "data",
    "populations",
    "result_graphs",
}


def _copy_tree(src: Path, dst: Path, *, force: bool) -> None:
    if dst.exists():
        if not force:
            raise FileExistsError(f"Destination already exists: {dst}")
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        name = item.name
        if name in SKIP_NAMES or name.startswith("._"):
            continue
        out = dst / name
        if name in SYMLINK_NAMES:
            target = item.resolve()
            if out.exists() or out.is_symlink():
                out.unlink()
            os.symlink(target, out)
            continue
        if item.is_dir():
            shutil.copytree(
                item,
                out,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"),
                symlinks=True,
            )
        else:
            shutil.copy2(item, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="external_repos/epi-diff-abm")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd()
    src = Path(args.source)
    if not src.is_absolute():
        src = project_root / src
    dst = Path(args.dest)
    if not dst.is_absolute():
        dst = project_root / dst
    src = src.resolve()

    if not (src / "covid_abm" / "yamls" / "config.yaml").exists():
        raise FileNotFoundError(f"Source does not look like epi-diff-abm: {src}")

    _copy_tree(src, dst, force=bool(args.force))

    config_path = dst / "covid_abm" / "yamls" / "config.yaml"
    lock_path = config_path.with_name(f"{config_path.name}.lock")
    print(f"runtime={dst}")
    print(f"config={config_path}")
    print(f"lock={lock_path}")
    for name in sorted(SYMLINK_NAMES):
        path = dst / name
        print(f"{name} -> {os.readlink(path) if path.is_symlink() else 'not_symlink'}")


if __name__ == "__main__":
    main()

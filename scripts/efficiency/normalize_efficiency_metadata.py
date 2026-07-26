#!/usr/bin/env python3
"""Normalize provenance fields without changing measured efficiency values."""

from pathlib import Path


RUN_ROOT = Path(
    "/home/liam/pythonProject/VCIP-ICML-main/"
    "results/efficiency_kdd26/formal_20260726"
)
CODE_COMMIT = "b29df9d36edb1e391472f3f13dcf86e29e8ac3a9"
HARNESS_COMMIT = "5a72cd64d09a2d1a0f93503575e927f699384c7d"


def normalize(path: Path) -> None:
    entries = []
    seen = set()
    for line in path.read_text(errors="replace").splitlines():
        if "\t" not in line:
            entries.append((None, line))
            continue
        key, value = line.split("\t", 1)
        if key == "git_commit":
            value = CODE_COMMIT
        elif key == "harness_commit":
            value = HARNESS_COMMIT
        entries.append((key, value))
        seen.add(key)
    if "git_commit" not in seen:
        entries.append(("git_commit", CODE_COMMIT))
    if "harness_commit" not in seen:
        entries.append(("harness_commit", HARNESS_COMMIT))
    text = "\n".join(
        value if key is None else f"{key}\t{value}" for key, value in entries
    )
    path.write_text(text + "\n")


def main() -> None:
    paths = sorted(RUN_ROOT.glob("*/*/seed_*/metadata.tsv"))
    for path in paths:
        normalize(path)
    print(f"normalized_metadata={len(paths)}")


if __name__ == "__main__":
    main()

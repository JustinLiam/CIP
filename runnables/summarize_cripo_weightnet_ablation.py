"""Summarize the canonical Tumor WeightNet ablation outputs."""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


EXPECTED_SEEDS = {10, 101, 1010, 10101, 101010}
EXPECTED_TAUS = set(range(1, 7))
WEIGHT_PATTERN = re.compile(
    r"Weight diagnostics outer=(?P<outer>\d+) \| mode=(?P<mode>\S+) "
    r"ess=(?P<ess>[0-9.eE+-]+) max=(?P<max>[0-9.eE+-]+) "
    r"var=(?P<var>[0-9.eE+-]+) p50=(?P<p50>[0-9.eE+-]+) "
    r"p90=(?P<p90>[0-9.eE+-]+) p95=(?P<p95>[0-9.eE+-]+) "
    r"p99=(?P<p99>[0-9.eE+-]+)"
)


def _condition_roots(main_root: Path, ablation_root: Path) -> Iterable[Tuple[str, int, Path]]:
    for gamma in range(1, 5):
        yield "sinkhorn", gamma, main_root / f"gamma_{gamma}" / "sinkhorn"
        yield "mmd", gamma, ablation_root / "mmd" / f"gamma_{gamma}"
        yield "uniform", gamma, ablation_root / "uniform" / f"gamma_{gamma}"


def _read_condition_rows(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(root.glob("gpu*/summary.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _deduplicate(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    latest: Dict[Tuple[int, int], Dict[str, str]] = {}
    for row in rows:
        if row.get("split") != "test" or row.get("rmse_uns") in {None, "", "NA"}:
            continue
        key = (int(row["seed"]), int(row["eval_tau"]))
        latest[key] = row
    return list(latest.values())


def _weight_diagnostics(train_log: Path, best_outer: int) -> Dict[str, float]:
    matches: Dict[int, Dict[str, float]] = {}
    with train_log.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = WEIGHT_PATTERN.search(line)
            if not match:
                continue
            values = match.groupdict()
            outer = int(values.pop("outer"))
            values.pop("mode")
            matches[outer] = {key: float(value) for key, value in values.items()}
    if best_outer not in matches:
        raise ValueError(f"Missing Weight diagnostics outer={best_outer} in {train_log}")
    return matches[best_outer]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    args = parser.parse_args()

    analysis_dir = args.ablation_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: List[Dict[str, object]] = []
    weight_rows: List[Dict[str, object]] = []

    for variant, gamma, root in _condition_roots(args.main_root, args.ablation_root):
        rows = _deduplicate(_read_condition_rows(root))
        observed = {(int(row["seed"]), int(row["eval_tau"])) for row in rows}
        expected = {(seed, tau) for seed in EXPECTED_SEEDS for tau in EXPECTED_TAUS}
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"Incomplete {variant} gamma={gamma}: missing={missing}, extra={extra}, root={root}"
            )
        for row in rows:
            raw_rows.append(
                {
                    "variant": variant,
                    "gamma": gamma,
                    "seed": int(row["seed"]),
                    "dataset_seed": int(row["dataset_seed"]),
                    "tau": int(row["eval_tau"]),
                    "rmse_uns": float(row["rmse_uns"]),
                    "best_outer": int(float(row["best_outer"])),
                    "best_val_metric": row["best_val_metric"],
                    "best_val_score": float(row["best_val_score"]),
                }
            )

        seed_rows: Dict[int, Dict[str, str]] = {}
        for row in rows:
            seed_rows[int(row["seed"])] = row
        for seed, row in sorted(seed_rows.items()):
            best_outer = int(float(row["best_outer"]))
            diagnostics = _weight_diagnostics(Path(row["train_log"]), best_outer)
            weight_rows.append(
                {
                    "variant": variant,
                    "gamma": gamma,
                    "seed": seed,
                    "dataset_seed": int(row["dataset_seed"]),
                    "best_outer": best_outer,
                    **diagnostics,
                }
            )

    raw_path = analysis_dir / "test_rmse_raw.csv"
    raw_fields = list(raw_rows[0])
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(sorted(raw_rows, key=lambda row: (row["variant"], row["gamma"], row["tau"], row["seed"])))

    grouped: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["variant"]), int(row["gamma"]), int(row["tau"]))].append(float(row["rmse_uns"]))
    summary_path = analysis_dir / "test_rmse_mean_std.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "gamma", "tau", "n", "mean_rmse", "std_rmse"])
        for key, values in sorted(grouped.items()):
            writer.writerow([*key, len(values), statistics.mean(values), statistics.stdev(values)])

    weight_path = analysis_dir / "weight_diagnostics_best_checkpoint.csv"
    weight_fields = list(weight_rows[0])
    with weight_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=weight_fields)
        writer.writeheader()
        writer.writerows(sorted(weight_rows, key=lambda row: (row["variant"], row["gamma"], row["seed"])))

    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {weight_path}")


if __name__ == "__main__":
    main()

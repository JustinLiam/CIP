"""Strictly merge and analyze the nine-action EpiABM mini-oracle experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


METHODS = ("full_cripo", "conv1d_only", "attention_only")
SEEDS = (10, 101, 1010, 10101, 101010)
TAUS = (7, 14, 21)
METRICS = (
    "training_return_regret",
    "q_training_return_regret",
    "in_support_training_return_regret",
    "in_support_q_training_return_regret",
    "terminal_regret_per_10k",
    "trajectory_regret_per_10k",
    "selected_terminal_error_per_10k",
    "selected_trajectory_rmse_per_10k",
)


def _read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _strict_unique(rows: Iterable[dict], fields: Sequence[str], name: str) -> Dict[tuple, dict]:
    out = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in out:
            raise ValueError(f"Duplicate {name} key: {key}")
        out[key] = row
    return out


def _two_way_bootstrap(
    matrix: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict:
    if matrix.shape != (len(SEEDS), 23):
        raise ValueError(f"Expected a 5x23 paired matrix, got {matrix.shape}")
    rng = np.random.default_rng(seed)
    values = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        seed_index = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        county_index = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        values[index] = matrix[np.ix_(seed_index, county_index)].mean()
    return {
        "estimate": float(matrix.mean()),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
        "bootstrap_samples": int(samples),
    }


def _paired_matrix(
    model_rows: Dict[tuple, dict],
    *,
    split: str,
    tau: int,
    metric: str,
) -> np.ndarray:
    counties = sorted({
        key[2] for key in model_rows
        if key[0] == split and key[3] == int(tau)
    })
    if len(counties) != 23:
        raise ValueError(f"Expected 23 counties for {split}/tau{tau}, got {len(counties)}")
    matrix = np.empty((len(SEEDS), len(counties)), dtype=np.float64)
    for seed_index, seed in enumerate(SEEDS):
        for county_index, county in enumerate(counties):
            conv = model_rows[(split, "conv1d_only", county, int(tau), seed)]
            attention = model_rows[(split, "attention_only", county, int(tau), seed)]
            matrix[seed_index, county_index] = float(conv[metric]) - float(attention[metric])
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--worker-ids", nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    split_dir = run_dir / args.split
    output_dir = split_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    oracle_raw: List[dict] = []
    model_raw: List[dict] = []
    for worker_id in args.worker_ids:
        shard = split_dir / "shards" / worker_id
        manifest = json.loads((shard / "manifest.json").read_text())
        summary = json.loads((shard / "summary.json").read_text())
        if not summary.get("complete"):
            raise ValueError(f"Incomplete shard: {shard}")
        manifests.append(manifest)
        oracle_raw.extend(_read_jsonl(shard / "oracle_rows.jsonl"))
        model_raw.extend(_read_jsonl(shard / "model_rows.jsonl"))
    family_hashes = {manifest["protocol_family_hash"] for manifest in manifests}
    if len(family_hashes) != 1:
        raise ValueError(f"Protocol family mismatch: {family_hashes}")

    oracle = _strict_unique(
        oracle_raw,
        ("split", "county", "tau", "action_index"),
        "oracle",
    )
    model = _strict_unique(
        model_raw,
        ("split", "method", "county", "tau", "seed"),
        "model",
    )
    expected_oracle = 23 * len(TAUS) * 9
    expected_model = 23 * len(TAUS) * len(METHODS) * len(SEEDS)
    if len(oracle) != expected_oracle:
        raise ValueError(f"Expected {expected_oracle} oracle rows, got {len(oracle)}")
    if len(model) != expected_model:
        raise ValueError(f"Expected {expected_model} model rows, got {len(model)}")

    support_indices = sorted({
        int(row["action_index"])
        for row in oracle.values()
        if bool(row["in_observed_support"])
    })
    if not support_indices:
        raise ValueError("Observed training action support is empty")
    for key, row in model.items():
        split, _method, county, tau, _seed = key
        candidates = {
            action_index: oracle[(split, county, tau, action_index)]
            for action_index in support_indices
        }
        oracle_index = min(
            support_indices,
            key=lambda index: (
                float(candidates[index]["training_return_loss"]), index
            ),
        )
        selected_index = max(
            support_indices,
            key=lambda index: (float(row["regularized_scores"][index]), -index),
        )
        q_selected_index = max(
            support_indices,
            key=lambda index: (float(row["q_values"][index]), -index),
        )
        oracle_loss = float(candidates[oracle_index]["training_return_loss"])
        row["in_support_training_return_regret"] = (
            float(candidates[selected_index]["training_return_loss"]) - oracle_loss
        )
        row["in_support_q_training_return_regret"] = (
            float(candidates[q_selected_index]["training_return_loss"]) - oracle_loss
        )
        row["in_support_oracle_action_index"] = oracle_index
        row["in_support_selected_action_index"] = selected_index
        row["in_support_q_selected_action_index"] = q_selected_index

    seed_rows = []
    for method in METHODS:
        for seed in SEEDS:
            for tau in TAUS:
                group = [
                    row for key, row in model.items()
                    if key[0] == args.split and key[1] == method and key[3] == tau and key[4] == seed
                ]
                if len(group) != 23:
                    raise ValueError(f"Expected 23 rows for {method}/seed{seed}/tau{tau}, got {len(group)}")
                row = {
                    "split": args.split,
                    "method": method,
                    "seed": seed,
                    "tau": tau,
                    "counties": len(group),
                }
                for metric in METRICS:
                    row[metric] = float(np.mean([float(item[metric]) for item in group]))
                row["train_oracle_top1"] = float(np.mean([bool(item["train_oracle_agreement"]) for item in group]))
                row["q_train_oracle_top1"] = float(np.mean([bool(item["q_train_oracle_agreement"]) for item in group]))
                row["selected_off_support_rate"] = float(np.mean([not bool(item["selected_in_observed_support"]) for item in group]))
                seed_rows.append(row)

    aggregate_rows = []
    numeric = list(METRICS) + [
        "train_oracle_top1",
        "q_train_oracle_top1",
        "selected_off_support_rate",
    ]
    for method in METHODS:
        for tau in TAUS:
            group = [row for row in seed_rows if row["method"] == method and row["tau"] == tau]
            aggregate = {"split": args.split, "method": method, "tau": tau, "seeds": len(group)}
            for field in numeric:
                values = [float(row[field]) for row in group]
                aggregate[f"{field}_mean"] = statistics.mean(values)
                aggregate[f"{field}_sample_std"] = statistics.stdev(values)
            aggregate_rows.append(aggregate)

    contrast_metrics = {}
    contrast_fields = (
        "training_return_regret",
        "q_training_return_regret",
        "in_support_training_return_regret",
        "in_support_q_training_return_regret",
        "terminal_regret_per_10k",
        "trajectory_regret_per_10k",
    )
    for metric_index, metric in enumerate(contrast_fields):
        metric_contrasts = {}
        matrices = {}
        for tau in TAUS:
            matrix = _paired_matrix(
                model,
                split=args.split,
                tau=tau,
                metric=metric,
            )
            matrices[tau] = matrix
            metric_contrasts[f"conv_minus_attention_tau{tau}"] = _two_way_bootstrap(
                matrix,
                samples=args.bootstrap_samples,
                seed=20260721 + 100 * metric_index + tau,
            )
        metric_contrasts["horizon_interaction_tau21_minus_tau7"] = _two_way_bootstrap(
            matrices[21] - matrices[7],
            samples=args.bootstrap_samples,
            seed=20260749 + 100 * metric_index,
        )
        contrast_metrics[metric] = metric_contrasts
    contrasts = contrast_metrics["training_return_regret"]
    short = contrasts["conv_minus_attention_tau7"]
    long = contrasts["conv_minus_attention_tau21"]
    horizon_interaction = contrasts["horizon_interaction_tau21_minus_tau7"]
    hypothesis = {
        "short_term_conv_better": short["ci95_high"] < 0.0,
        "long_term_attention_better": long["ci95_low"] > 0.0,
        "relative_advantage_shifts_conv_to_attention": horizon_interaction["ci95_low"] > 0.0,
        "fully_supported": (
            short["ci95_high"] < 0.0
            and long["ci95_low"] > 0.0
            and horizon_interaction["ci95_low"] > 0.0
        ),
    }

    oracle_choices = defaultdict(Counter)
    terminal_choices = defaultdict(Counter)
    oracle_groups = defaultdict(list)
    for row in oracle.values():
        oracle_groups[(row["split"], row["county"], int(row["tau"]))].append(row)
    for (split, _county, tau), group in oracle_groups.items():
        train_choice = min(
            group,
            key=lambda row: (float(row["training_return_loss"]), int(row["action_index"])),
        )
        terminal_choice = min(
            group,
            key=lambda row: (float(row["terminal_error_per_10k"]), int(row["action_index"])),
        )
        oracle_choices[(split, tau)][int(train_choice["action_index"])] += 1
        terminal_choices[(split, tau)][int(terminal_choice["action_index"])] += 1

    _write_csv(output_dir / "seed_metrics.csv", seed_rows)
    _write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    summary = {
        "schema": "epi_abm_nine_action_mini_oracle_aggregate_v1",
        "split": args.split,
        "protocol_family_hash": next(iter(family_hashes)),
        "counts": {
            "oracle_rows": len(oracle),
            "model_rows": len(model),
            "counties": 23,
            "methods": len(METHODS),
            "seeds": len(SEEDS),
            "taus": list(TAUS),
        },
        "contrast_definition": "mean training_return_regret(conv1d_only) - mean training_return_regret(attention_only)",
        "contrasts": contrasts,
        "contrast_metrics": contrast_metrics,
        "observed_support_action_indices": support_indices,
        "hypothesis": hypothesis,
        "train_oracle_action_counts": {
            f"{split}_tau{tau}": dict(sorted(counter.items()))
            for (split, tau), counter in oracle_choices.items()
        },
        "terminal_oracle_action_counts": {
            f"{split}_tau{tau}": dict(sorted(counter.items()))
            for (split, tau), counter in terminal_choices.items()
        },
        "aggregate_metrics": aggregate_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# Nine-action mini ABM oracle: {args.split}",
        "",
        "Primary contrast is Conv1d-only regret minus Attention-only regret; negative favors Conv1d-only.",
        "",
        "| Contrast | Estimate | 95% two-way cluster bootstrap CI |",
        "|---|---:|---:|",
    ]
    for name, value in contrasts.items():
        lines.append(
            f"| {name} | {value['estimate']:.6f} | [{value['ci95_low']:.6f}, {value['ci95_high']:.6f}] |"
        )
    lines.extend(["", f"Hypothesis fully supported: **{hypothesis['fully_supported']}**", ""])
    (output_dir / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({
        "event": "mini_oracle_aggregate_complete",
        "split": args.split,
        "output_dir": str(output_dir),
        "hypothesis": hypothesis,
        "contrasts": contrasts,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

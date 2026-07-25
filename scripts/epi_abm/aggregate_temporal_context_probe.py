"""Strict aggregation for the EpiABM temporal context probe."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


METHODS = ("full_cripo", "conv1d_only", "attention_only")
SEEDS = (10, 101, 1010, 10101, 101010)
TAUS = (7, 14, 21)
LENGTHS = (7, 14, 28, 56, 112, 161)
RECENT_BLOCKS = ("recent_1_7", "local_8_28")
REMOTE_BLOCKS = ("long_57_112", "remote_113_161")


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Centered linear CKA between two sample-by-feature matrices."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Invalid CKA shapes: {x.shape}, {y.shape}")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = float(np.square(x.T @ y).sum())
    denominator = float(
        np.sqrt(np.square(x.T @ x).sum() * np.square(y.T @ y).sum())
    )
    return numerator / max(denominator, 1e-12)


def _read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _strict_unique(rows: Iterable[dict], fields: Sequence[str], name: str) -> Dict[tuple, dict]:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"Duplicate {name} key: {key}")
        result[key] = row
    return result


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _two_way_bootstrap(matrix: np.ndarray, *, samples: int, seed: int) -> dict:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (len(SEEDS), 23):
        raise ValueError(f"Expected 5x23 paired matrix, got {matrix.shape}")
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


def _matrix(counties: Sequence[str], fn) -> np.ndarray:
    result = np.empty((len(SEEDS), len(counties)), dtype=np.float64)
    for seed_index, seed in enumerate(SEEDS):
        for county_index, county in enumerate(counties):
            result[seed_index, county_index] = float(fn(seed, county))
    return result


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


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
    context_raw: List[dict] = []
    occlusion_raw: List[dict] = []
    max_parity = 0.0
    for worker_id in args.worker_ids:
        shard = split_dir / "shards" / worker_id
        manifest = json.loads((shard / "manifest.json").read_text())
        summary = json.loads((shard / "summary.json").read_text())
        if not summary.get("complete"):
            raise ValueError(f"Incomplete shard: {shard}")
        manifests.append(manifest)
        context_raw.extend(_read_jsonl(shard / "context_rows.jsonl"))
        occlusion_raw.extend(_read_jsonl(shard / "occlusion_rows.jsonl"))
        max_parity = max(max_parity, float(summary["full_context_parity_max_abs"]))
    family_hashes = {manifest["protocol_family_hash"] for manifest in manifests}
    if len(family_hashes) != 1:
        raise ValueError(f"Protocol family mismatch: {family_hashes}")

    context = _strict_unique(
        context_raw,
        ("split", "method", "seed", "county", "tau", "history_length"),
        "context",
    )
    occlusion = _strict_unique(
        occlusion_raw,
        ("split", "method", "seed", "county", "tau", "block"),
        "occlusion",
    )
    expected_context = 23 * len(METHODS) * len(SEEDS) * len(TAUS) * len(LENGTHS)
    expected_occlusion = 23 * len(METHODS) * len(SEEDS) * len(TAUS) * 5
    if len(context) != expected_context:
        raise ValueError(f"Expected {expected_context} context rows, got {len(context)}")
    if len(occlusion) != expected_occlusion:
        raise ValueError(f"Expected {expected_occlusion} occlusion rows, got {len(occlusion)}")
    counties = sorted({str(row["county"]) for row in context.values()})
    if len(counties) != 23:
        raise ValueError(f"Expected 23 counties, got {len(counties)}")

    primary = "in_support_training_return_regret"
    seed_rows = []
    for method in METHODS:
        for seed in SEEDS:
            for tau in TAUS:
                for length in LENGTHS:
                    group = [
                        context[(args.split, method, seed, county, tau, length)]
                        for county in counties
                    ]
                    seed_rows.append({
                        "split": args.split,
                        "method": method,
                        "seed": seed,
                        "tau": tau,
                        "history_length": length,
                        "counties": len(group),
                        "in_support_training_return_regret": float(np.mean([
                            float(row["in_support_training_return_regret"]) for row in group
                        ])),
                        "training_return_regret": float(np.mean([
                            float(row["training_return_regret"]) for row in group
                        ])),
                        "q_training_return_regret": float(np.mean([
                            float(row["q_training_return_regret"]) for row in group
                        ])),
                        "terminal_regret_per_10k": float(np.mean([
                            float(row["terminal_regret_per_10k"]) for row in group
                        ])),
                        "trajectory_regret_per_10k": float(np.mean([
                            float(row["trajectory_regret_per_10k"]) for row in group
                        ])),
                    })

    aggregate_rows = []
    fields = (
        "in_support_training_return_regret",
        "training_return_regret",
        "q_training_return_regret",
        "terminal_regret_per_10k",
        "trajectory_regret_per_10k",
    )
    for method in METHODS:
        for tau in TAUS:
            for length in LENGTHS:
                group = [
                    row for row in seed_rows
                    if row["method"] == method and row["tau"] == tau
                    and row["history_length"] == length
                ]
                result = {
                    "split": args.split,
                    "method": method,
                    "tau": tau,
                    "history_length": length,
                    "seeds": len(group),
                }
                for field in fields:
                    mean, std = _mean_std([float(row[field]) for row in group])
                    result[f"{field}_mean"] = mean
                    result[f"{field}_sample_std"] = std
                aggregate_rows.append(result)

    def c(method: str, seed: int, county: str, tau: int, length: int) -> dict:
        return context[(args.split, method, seed, county, tau, length)]

    context_gain_matrices = {}
    for method in METHODS:
        for tau in TAUS:
            context_gain_matrices[(method, tau)] = _matrix(
                counties,
                lambda seed, county, m=method, t=tau: (
                    float(c(m, seed, county, t, 28)[primary])
                    - float(c(m, seed, county, t, 161)[primary])
                ),
            )

    context_tests = {}
    for tau in TAUS:
        matrix = (
            context_gain_matrices[("attention_only", tau)]
            - context_gain_matrices[("conv1d_only", tau)]
        )
        context_tests[f"attention_minus_conv_context_gain_tau{tau}"] = _two_way_bootstrap(
            matrix,
            samples=args.bootstrap_samples,
            seed=20260731 + tau,
        )
    context_tests["horizon_interaction_tau21_minus_tau7"] = _two_way_bootstrap(
        (
            context_gain_matrices[("attention_only", 21)]
            - context_gain_matrices[("conv1d_only", 21)]
        ) - (
            context_gain_matrices[("attention_only", 7)]
            - context_gain_matrices[("conv1d_only", 7)]
        ),
        samples=args.bootstrap_samples,
        seed=20260763,
    )

    def o(method: str, seed: int, county: str, tau: int, block: str) -> dict:
        return occlusion[(args.split, method, seed, county, tau, block)]

    sensitivity = {}
    for method in METHODS:
        for tau in TAUS:
            for metric in ("q_centered_relative_change", "z_relative_change"):
                recent = _matrix(
                    counties,
                    lambda seed, county, m=method, t=tau, f=metric: np.mean([
                        float(o(m, seed, county, t, block)[f]) for block in RECENT_BLOCKS
                    ]),
                )
                remote = _matrix(
                    counties,
                    lambda seed, county, m=method, t=tau, f=metric: np.mean([
                        float(o(m, seed, county, t, block)[f]) for block in REMOTE_BLOCKS
                    ]),
                )
                sensitivity[(method, tau, metric, "recent")] = recent
                sensitivity[(method, tau, metric, "remote")] = remote
                sensitivity[(method, tau, metric, "recent_share")] = recent / np.maximum(
                    recent + remote, 1e-12
                )

    sensitivity_tests = {}
    for tau in TAUS:
        sensitivity_tests[f"conv_minus_attention_q_recent_share_tau{tau}"] = _two_way_bootstrap(
            sensitivity[("conv1d_only", tau, "q_centered_relative_change", "recent_share")]
            - sensitivity[("attention_only", tau, "q_centered_relative_change", "recent_share")],
            samples=args.bootstrap_samples,
            seed=20260800 + tau,
        )
        sensitivity_tests[f"attention_minus_conv_q_remote_tau{tau}"] = _two_way_bootstrap(
            sensitivity[("attention_only", tau, "q_centered_relative_change", "remote")]
            - sensitivity[("conv1d_only", tau, "q_centered_relative_change", "remote")],
            samples=args.bootstrap_samples,
            seed=20260830 + tau,
        )
        sensitivity_tests[f"conv_minus_attention_z_recent_share_tau{tau}"] = _two_way_bootstrap(
            sensitivity[("conv1d_only", tau, "z_relative_change", "recent_share")]
            - sensitivity[("attention_only", tau, "z_relative_change", "recent_share")],
            samples=args.bootstrap_samples,
            seed=20260860 + tau,
        )
        sensitivity_tests[f"full_minus_attention_q_recent_share_tau{tau}"] = _two_way_bootstrap(
            sensitivity[("full_cripo", tau, "q_centered_relative_change", "recent_share")]
            - sensitivity[("attention_only", tau, "q_centered_relative_change", "recent_share")],
            samples=args.bootstrap_samples,
            seed=20260890 + tau,
        )
        sensitivity_tests[f"conv_minus_full_q_recent_share_tau{tau}"] = _two_way_bootstrap(
            sensitivity[("conv1d_only", tau, "q_centered_relative_change", "recent_share")]
            - sensitivity[("full_cripo", tau, "q_centered_relative_change", "recent_share")],
            samples=args.bootstrap_samples,
            seed=20260920 + tau,
        )
        sensitivity_tests[f"full_q_remote_sensitivity_tau{tau}"] = _two_way_bootstrap(
            sensitivity[("full_cripo", tau, "q_centered_relative_change", "remote")],
            samples=args.bootstrap_samples,
            seed=20260950 + tau,
        )

    complementarity_tests = {}
    for tau in TAUS:
        full_minus_best = _matrix(
            counties,
            lambda seed, county, t=tau: (
                float(c("full_cripo", seed, county, t, 161)[primary])
                - min(
                    float(c("conv1d_only", seed, county, t, 161)[primary]),
                    float(c("attention_only", seed, county, t, 161)[primary]),
                )
            ),
        )
        full_minus_average = _matrix(
            counties,
            lambda seed, county, t=tau: (
                float(c("full_cripo", seed, county, t, 161)[primary])
                - 0.5 * (
                    float(c("conv1d_only", seed, county, t, 161)[primary])
                    + float(c("attention_only", seed, county, t, 161)[primary])
                )
            ),
        )
        complementarity_tests[f"full_minus_best_branch_tau{tau}"] = _two_way_bootstrap(
            full_minus_best,
            samples=args.bootstrap_samples,
            seed=20260900 + tau,
        )
        complementarity_tests[f"full_minus_average_branch_tau{tau}"] = _two_way_bootstrap(
            full_minus_average,
            samples=args.bootstrap_samples,
            seed=20260930 + tau,
        )

    cka_rows = []
    for seed in SEEDS:
        for length in LENGTHS:
            representations = {}
            for method in METHODS:
                representations[method] = np.asarray([
                    c(method, seed, county, 7, length)["z"] for county in counties
                ], dtype=np.float64)
            cka_rows.append({
                "split": args.split,
                "seed": seed,
                "history_length": length,
                "full_conv_cka": linear_cka(
                    representations["full_cripo"], representations["conv1d_only"]
                ),
                "full_attention_cka": linear_cka(
                    representations["full_cripo"], representations["attention_only"]
                ),
                "conv_attention_cka": linear_cka(
                    representations["conv1d_only"], representations["attention_only"]
                ),
            })

    cka_aggregate = []
    for length in LENGTHS:
        group = [row for row in cka_rows if row["history_length"] == length]
        result = {"split": args.split, "history_length": length, "seeds": len(group)}
        for field in ("full_conv_cka", "full_attention_cka", "conv_attention_cka"):
            mean, std = _mean_std([float(row[field]) for row in group])
            result[f"{field}_mean"] = mean
            result[f"{field}_sample_std"] = std
        cka_aggregate.append(result)

    conv_q_invariance_max_abs = 0.0
    for seed in SEEDS:
        for county in counties:
            for tau in TAUS:
                reference = np.asarray(
                    c("conv1d_only", seed, county, tau, 161)["q_values"], dtype=np.float64
                )
                for length in LENGTHS:
                    candidate = np.asarray(
                        c("conv1d_only", seed, county, tau, length)["q_values"], dtype=np.float64
                    )
                    conv_q_invariance_max_abs = max(
                        conv_q_invariance_max_abs,
                        float(np.max(np.abs(candidate - reference))),
                    )
    conv_remote_sensitivity_max = max(
        float(o("conv1d_only", seed, county, tau, block)["q_centered_relative_change"])
        for seed in SEEDS
        for county in counties
        for tau in TAUS
        for block in REMOTE_BLOCKS
    )

    local_test = sensitivity_tests["conv_minus_attention_q_recent_share_tau7"]
    long_context_test = context_tests["attention_minus_conv_context_gain_tau21"]
    long_sensitivity_test = sensitivity_tests["attention_minus_conv_q_remote_tau21"]
    full_above_attention = sensitivity_tests["full_minus_attention_q_recent_share_tau21"]
    conv_above_full = sensitivity_tests["conv_minus_full_q_recent_share_tau21"]
    full_remote = sensitivity_tests["full_q_remote_sensitivity_tau21"]
    complement_test = complementarity_tests["full_minus_average_branch_tau21"]
    claims = {
        "conv_local_inductive_bias_supported": local_test["ci95_low"] > 0.0,
        "conv_numerically_invariant_beyond_local_receptive_field": (
            conv_q_invariance_max_abs <= 1e-4 and conv_remote_sensitivity_max == 0.0
        ),
        "attention_uses_remote_context_supported": long_sensitivity_test["ci95_low"] > 0.0,
        "full_temporal_profile_between_branches_supported": (
            full_above_attention["ci95_low"] > 0.0
            and conv_above_full["ci95_low"] > 0.0
            and full_remote["ci95_low"] > 0.0
        ),
        "attention_long_context_improves_oracle_regret": long_context_test["ci95_low"] > 0.0,
        "full_performance_beats_average_branches_at_tau21": complement_test["ci95_high"] < 0.0,
    }
    claims["mechanistic_statement_supported"] = all((
        claims["conv_local_inductive_bias_supported"],
        claims["conv_numerically_invariant_beyond_local_receptive_field"],
        claims["attention_uses_remote_context_supported"],
        claims["full_temporal_profile_between_branches_supported"],
    ))

    _write_csv(output_dir / "seed_context_metrics.csv", seed_rows)
    _write_csv(output_dir / "aggregate_context_metrics.csv", aggregate_rows)
    _write_csv(output_dir / "representation_cka_by_seed.csv", cka_rows)
    _write_csv(output_dir / "representation_cka_aggregate.csv", cka_aggregate)
    summary = {
        "schema": "epi_abm_temporal_context_probe_aggregate_v1",
        "split": args.split,
        "protocol_family_hash": next(iter(family_hashes)),
        "counts": {
            "context_rows": len(context),
            "occlusion_rows": len(occlusion),
            "counties": len(counties),
            "methods": len(METHODS),
            "seeds": len(SEEDS),
            "taus": list(TAUS),
            "history_lengths": list(LENGTHS),
        },
        "primary_metric": primary,
        "context_gain_definition": "regret(L=28)-regret(L=161); positive means long context helps",
        "full_context_parity_max_abs": max_parity,
        "conv_q_invariance_max_abs_across_history_lengths": conv_q_invariance_max_abs,
        "conv_q_invariance_tolerance": 1e-4,
        "conv_remote_q_sensitivity_max": conv_remote_sensitivity_max,
        "context_tests": context_tests,
        "sensitivity_tests": sensitivity_tests,
        "complementarity_tests": complementarity_tests,
        "claims": claims,
        "aggregate_context_metrics": aggregate_rows,
        "representation_cka": cka_aggregate,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# Temporal context probe: {args.split}",
        "",
        "Primary metric: in-support nine-action training-return oracle regret.",
        "",
        "## Claim tests",
        "",
        "| Test | Estimate | 95% two-way cluster bootstrap CI |",
        "|---|---:|---:|",
    ]
    selected = {
        "Conv minus Attention recent-sensitivity share, tau7": local_test,
        "Attention minus Conv remote sensitivity, tau21": long_sensitivity_test,
        "Full minus Attention recent-sensitivity share, tau21": full_above_attention,
        "Conv minus Full recent-sensitivity share, tau21": conv_above_full,
        "Attention minus Conv long-context regret gain, tau21": long_context_test,
        "Full minus average branch regret, tau21": complement_test,
    }
    for label, value in selected.items():
        lines.append(
            f"| {label} | {value['estimate']:.6f} | "
            f"[{value['ci95_low']:.6f}, {value['ci95_high']:.6f}] |"
        )
    lines.extend([
        "",
        f"Mechanistic statement supported: **{claims['mechanistic_statement_supported']}**",
        "",
        f"Beneficial long-context effect supported: **{claims['attention_long_context_improves_oracle_regret']}**",
        "",
        f"Performance complementarity supported: **{claims['full_performance_beats_average_branches_at_tau21']}**",
        "",
    ])
    (output_dir / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({
        "event": "temporal_context_aggregate_complete",
        "split": args.split,
        "output_dir": str(output_dir),
        "claims": claims,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

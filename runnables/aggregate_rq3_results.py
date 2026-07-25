"""Aggregate five-seed Tumor RQ3 diagnostics and emit paper-ready artifacts."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPECTED_SEEDS = [10, 101, 1010, 10101, 101010]
METRICS = ["low_support_rate", "q_over", "rmse_uns", "q_mae", "q_bias"]


def main(root_raw: str, reference_csv_raw: str) -> None:
    root = Path(root_raw).resolve()
    reference_csv = Path(reference_csv_raw).resolve()
    records = []
    payloads = {}
    for seed in EXPECTED_SEEDS:
        path = root / f"seed_{seed}" / "result.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[seed] = payload
        if float(payload["checkpoint"]["beta"]) != 2.0:
            raise ValueError(f"seed={seed}: expected beta=2.0")
        for method in ("actor", "qgrid"):
            row = {
                "seed": seed,
                "method": method,
                "n_samples": payload["test"][method]["n_samples"],
                "n_decisions": payload["test"][method]["n_decisions"],
                "threshold": payload["test"][method]["low_support_threshold"],
            }
            row.update({key: payload["test"][method][key] for key in METRICS})
            records.append(row)

    reference = {
        int(row["seed"]): float(row["rmse_uns"])
        for row in csv.DictReader(reference_csv.open(encoding="utf-8"))
        if row["variant"] == "sinkhorn"
        and row["kappa"] == "4"
        and row["split"] == "test"
        and row["eval_tau"] == "12"
    }
    reproduction = {}
    for seed in EXPECTED_SEEDS:
        observed = float(payloads[seed]["test"]["actor"]["rmse_uns"])
        expected = reference[seed]
        reproduction[seed] = {
            "rq3_actor_rmse": observed,
            "reference_main_rmse": expected,
            "absolute_difference": abs(observed - expected),
        }
    max_difference = max(item["absolute_difference"] for item in reproduction.values())
    if max_difference > 1e-6:
        raise ValueError(f"Main-evaluation reproduction failed: max difference={max_difference}")

    with (root / "per_seed.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    aggregate_rows = []
    aggregate = {}
    for method in ("actor", "qgrid"):
        aggregate[method] = {}
        method_rows = [row for row in records if row["method"] == method]
        for metric in METRICS:
            values = np.asarray([row[metric] for row in method_rows], dtype=np.float64)
            summary = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "values": values.tolist(),
            }
            aggregate[method][metric] = summary
            aggregate_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": summary["mean"],
                    "std": summary["std"],
                }
            )
    with (root / "aggregate.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "metric", "mean", "std"])
        writer.writeheader()
        writer.writerows(aggregate_rows)

    comparisons = {}
    for metric in ("low_support_rate", "q_over", "rmse_uns"):
        actor = np.asarray(aggregate["actor"][metric]["values"])
        qgrid = np.asarray(aggregate["qgrid"][metric]["values"])
        comparisons[metric] = {
            "actor_lower_in_all_seeds": bool(np.all(actor < qgrid)),
            "paired_absolute_reduction_mean": float((qgrid - actor).mean()),
            "paired_absolute_reduction_std": float((qgrid - actor).std(ddof=1)),
            "paired_relative_reduction_percent_mean": float(
                np.mean((qgrid - actor) / qgrid * 100.0)
            ),
            "paired_relative_reduction_percent_std": float(
                np.std((qgrid - actor) / qgrid * 100.0, ddof=1)
            ),
        }

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": EXPECTED_SEEDS,
        "n_seeds": len(EXPECTED_SEEDS),
        "protocol": payloads[EXPECTED_SEEDS[0]]["protocol"],
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "main_rmse_reproduction": {
            "reference_csv": str(reference_csv),
            "max_absolute_difference": max_difference,
            "per_seed": reproduction,
        },
    }
    (root / "aggregate.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    actor = aggregate["actor"]
    qgrid = aggregate["qgrid"]

    def pm(method, metric, mean_digits, std_digits, scale=1.0):
        item = method[metric]
        return (
            f"{item['mean'] * scale:.{mean_digits}f}"
            f"$\\pm${item['std'] * scale:.{std_digits}f}"
        )

    table = rf"""\begin{{table}}[t]
    \caption{{
        Behavioral-support diagnostics on Tumor
        ($\kappa=4$, $\tau=12$; mean$\pm$std over five seeds).
        The low-support threshold is calibrated on validation data;
        Q-over is measured in normalized return units.
    }}
    \label{{tab:action_support}}
    \centering
    \small
    \setlength{{\tabcolsep}}{{3pt}}
    \begin{{tabular}}{{@{{}}lccc@{{}}}}
        \toprule
        Action selection
        & Low-supp. (\%) $\downarrow$
        & Q-over. $\downarrow$
        & RMSE $\downarrow$ \\
        \midrule
        CRIPO actor
        & \textbf{{{pm(actor, "low_support_rate", 1, 1, 100.0)}}}
        & \textbf{{{pm(actor, "q_over", 3, 3)}}}
        & \textbf{{{pm(actor, "rmse_uns", 2, 2)}}} \\
        Q-grid maximizer
        & {pm(qgrid, "low_support_rate", 1, 1, 100.0)}
        & {pm(qgrid, "q_over", 3, 3)}
        & {pm(qgrid, "rmse_uns", 2, 2)} \\
        \bottomrule
    \end{{tabular}}
\end{{table}}
"""
    (root / "table.tex").write_text(table, encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(table)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: aggregate_rq3_results.py ROOT REFERENCE_CSV")
    main(sys.argv[1], sys.argv[2])

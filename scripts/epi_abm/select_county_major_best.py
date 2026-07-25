"""Select validation checkpoints and build a county-major test manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


def _absolute(path: str, base: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (base / value).resolve()


def _load_rows(path: Path) -> List[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    seen = set()
    for row in rows:
        key = (
            row["split"],
            int(row["row_idx"]),
            int(row["tau"]),
            row["label"],
        )
        if key in seen:
            raise ValueError(f"Duplicate validation key in {path}: {key}")
        seen.add(key)
    return rows


def _rmse(values: List[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate RMSE for an empty group")
    return math.sqrt(sum(value * value for value in values) / len(values))


def _distance_per_10k(row: dict, target_reference: str) -> float:
    if target_reference in {
        "cached_target",
        "external_target_file",
        "external_gpu_factual_replay",
    }:
        return float(row["target_distance_per_10k"])
    if target_reference == "factual_replay":
        return float(row["policy_vs_factual_final_improvement_per_10k"])
    raise ValueError(f"Unsupported target reference: {target_reference}")


def _select_job(
    *,
    job: dict,
    manifest_dir: Path,
    merged_name: str,
    selection_taus: List[int],
    expected_counties: int,
    target_reference: str,
) -> dict:
    out_dir = _absolute(job["out_dir"], manifest_dir)
    metrics_path = out_dir / merged_name / "county_metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing merged validation metrics: {metrics_path}")
    rows = _load_rows(metrics_path)
    labels = sorted(job["ckpts"])
    expected = expected_counties * len(selection_taus)
    scores: Dict[str, dict] = {}
    for label in labels:
        rmse_by_tau = {}
        for tau in selection_taus:
            group = [
                _distance_per_10k(row, target_reference)
                for row in rows
                if (
                    row["split"] == "val"
                    and row["label"] == label
                    and int(row["tau"]) == int(tau)
                )
            ]
            if len(group) != expected_counties:
                raise ValueError(
                    f"Job {job['id']} label={label} tau={tau} has "
                    f"{len(group)} counties, expected {expected_counties}"
                )
            rmse_by_tau[str(tau)] = _rmse(group)
        selection_rows = [
            row for row in rows
            if row["split"] == "val"
            and row["label"] == label
            and int(row["tau"]) in selection_taus
        ]
        if len(selection_rows) != expected:
            raise ValueError(f"Unexpected selection row count for {job['id']} {label}")
        score = {
            "RMSE_per_10k_by_tau": rmse_by_tau,
            "mean_RMSE_per_10k": (
                sum(rmse_by_tau.values()) / len(selection_taus)
            ),
        }
        if target_reference in {
            "cached_target",
            "external_target_file",
            "external_gpu_factual_replay",
        }:
            score.update({
                "target_RMSE_per_10k_by_tau": rmse_by_tau,
                "mean_target_RMSE_per_10k": score["mean_RMSE_per_10k"],
            })
        else:
            score.update({
                "factual_reference_RMSE_per_10k_by_tau": rmse_by_tau,
                "mean_factual_reference_RMSE_per_10k": score[
                    "mean_RMSE_per_10k"
                ],
            })
        scores[label] = score

    best_label = min(
        labels,
        key=lambda label: (scores[label]["mean_RMSE_per_10k"], label),
    )
    return {
        "job_id": job["id"],
        "method": job.get("method"),
        "seed": job.get("seed"),
        "target_reference": target_reference,
        "selection_taus": selection_taus,
        "expected_counties": expected_counties,
        "validation_metrics": str(metrics_path),
        "best_label": best_label,
        "best_checkpoint": job["ckpts"][best_label],
        "best_score": scores[best_label],
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--selection-taus", nargs="+", type=int, default=[7, 14, 21])
    parser.add_argument("--expected-counties", type=int, required=True)
    parser.add_argument("--merged-name", default="parallel_merged")
    parser.add_argument(
        "--target-reference",
        choices=[
            "cached_target",
            "factual_replay",
            "external_target_file",
            "external_gpu_factual_replay",
        ],
        default="cached_target",
        help=(
            "Reference used for validation RMSE. external_target_file uses the "
            "target_distance_per_10k written from an immutable target file."
        ),
    )
    parser.add_argument("--test-dir-name", required=True)
    parser.add_argument("--selection-output", required=True)
    parser.add_argument("--test-manifest-output", required=True)
    args = parser.parse_args()

    val_manifest_path = Path(args.val_manifest).expanduser().resolve()
    val_manifest = json.loads(val_manifest_path.read_text())
    selection_taus = [int(tau) for tau in args.selection_taus]
    selections = []
    test_jobs = []
    for job in val_manifest["jobs"]:
        selection = _select_job(
            job=job,
            manifest_dir=val_manifest_path.parent,
            merged_name=args.merged_name,
            selection_taus=selection_taus,
            expected_counties=int(args.expected_counties),
            target_reference=args.target_reference,
        )
        selections.append(selection)
        val_out_dir = _absolute(job["out_dir"], val_manifest_path.parent)
        method_root = val_out_dir.parent.parent
        test_out_dir = method_root / args.test_dir_name / val_out_dir.name
        best_label = selection["best_label"]
        test_jobs.append({
            **{key: value for key, value in job.items() if key not in {"out_dir", "ckpts"}},
            "out_dir": str(test_out_dir.resolve()),
            "ckpts": {best_label: job["ckpts"][best_label]},
            "validation_selection": {
                "best_label": best_label,
                "selection_taus": selection_taus,
                "target_reference": args.target_reference,
                "mean_RMSE_per_10k": selection["best_score"]["mean_RMSE_per_10k"],
            },
        })

    selection_output = Path(args.selection_output).expanduser().resolve()
    selection_output.parent.mkdir(parents=True, exist_ok=True)
    selection_output.write_text(json.dumps({
        "schema": "epi_abm_county_major_best_v2",
        "validation_manifest": str(val_manifest_path),
        "target_reference": args.target_reference,
        "selection_taus": selection_taus,
        "expected_counties": int(args.expected_counties),
        "jobs": selections,
    }, indent=2, sort_keys=True) + "\n")

    test_manifest_output = Path(args.test_manifest_output).expanduser().resolve()
    test_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    test_manifest_output.write_text(json.dumps({
        "schema": "epi_abm_county_major_jobs_v2",
        "source_validation_manifest": str(val_manifest_path),
        "source_selection": str(selection_output),
        "target_reference": args.target_reference,
        "jobs": test_jobs,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "county_major_best_selected",
        "jobs": len(selections),
        "target_reference": args.target_reference,
        "selection_output": str(selection_output),
        "test_manifest_output": str(test_manifest_output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

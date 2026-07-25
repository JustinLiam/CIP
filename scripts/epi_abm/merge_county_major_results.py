"""Strictly merge disjoint county-major GPU evaluation shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.epi_abm.evaluate_county_last_window_iql import _summarize  # noqa: E402


def _absolute(path: str, base: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (base / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--worker-ids", nargs="+", required=True)
    parser.add_argument("--output-name", default="parallel_merged")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    top = json.loads(manifest_path.read_text())
    jobs = top["jobs"]
    for job in jobs:
        job_id = str(job["id"])
        out_dir = _absolute(job["out_dir"], manifest_path.parent)
        rows = []
        seen = set()
        family_hash = None
        expected_rows = set()
        checkpoint_labels = None
        shard_manifests = []
        for worker_id in args.worker_ids:
            shard_dir = out_dir / "county_major_shards" / worker_id
            shard_manifest_path = shard_dir / "manifest.json"
            metrics_path = shard_dir / "county_metrics.jsonl"
            if not shard_manifest_path.is_file() or not metrics_path.is_file():
                raise FileNotFoundError(
                    f"Missing shard for job={job_id}, worker={worker_id}: {shard_dir}"
                )
            shard_manifest = json.loads(shard_manifest_path.read_text())
            shard_manifests.append(shard_manifest)
            if shard_manifest["job_id"] != job_id:
                raise ValueError(
                    f"Shard job mismatch in {shard_manifest_path}: "
                    f"{shard_manifest['job_id']!r} != {job_id!r}"
                )
            if family_hash is None:
                family_hash = shard_manifest["protocol_family_hash"]
                checkpoint_labels = sorted(shard_manifest["ckpts"])
            elif shard_manifest["protocol_family_hash"] != family_hash:
                raise ValueError(
                    f"Protocol family mismatch in {shard_manifest_path}"
                )
            if sorted(shard_manifest["ckpts"]) != checkpoint_labels:
                raise ValueError(
                    f"Checkpoint labels mismatch in {shard_manifest_path}"
                )
            split = shard_manifest["split"]
            taus = [int(tau) for tau in shard_manifest["taus"]]
            for row_idx in shard_manifest["row_indices"]:
                for tau in taus:
                    for label in checkpoint_labels:
                        expected_rows.add((split, int(row_idx), tau, label))

            for line_number, line in enumerate(
                metrics_path.read_text().splitlines(), start=1
            ):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    row["split"],
                    int(row["row_idx"]),
                    int(row["tau"]),
                    row["label"],
                )
                if key in seen:
                    raise ValueError(
                        f"Duplicate key across shards for job={job_id}: {key}"
                    )
                seen.add(key)
                rows.append(row)

        missing = sorted(expected_rows - seen)
        extra = sorted(seen - expected_rows)
        if missing or extra:
            raise ValueError(
                f"Strict merge failed for job={job_id}: "
                f"missing={len(missing)} first={missing[:3]}, "
                f"extra={len(extra)} first={extra[:3]}"
            )

        rows.sort(
            key=lambda row: (
                int(row["row_idx"]),
                int(row["tau"]),
                row["label"],
            )
        )
        merged_dir = out_dir / args.output_name
        merged_dir.mkdir(parents=True, exist_ok=True)
        (merged_dir / "county_metrics.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        summary = _summarize(rows)
        (merged_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        (merged_dir / "manifest.json").write_text(json.dumps({
            "schema": "epi_abm_county_major_merge_v1",
            "job_id": job_id,
            "protocol_family_hash": family_hash,
            "worker_ids": args.worker_ids,
            "shard_manifests": shard_manifests,
            "row_count": len(rows),
            "expected_row_count": len(expected_rows),
            "unique_counties": len({row["county"] for row in rows}),
            "taus": sorted({int(row["tau"]) for row in rows}),
            "labels": sorted({row["label"] for row in rows}),
        }, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "event": "county_major_merge_done",
            "job_id": job_id,
            "output": str(merged_dir),
            "rows": len(rows),
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

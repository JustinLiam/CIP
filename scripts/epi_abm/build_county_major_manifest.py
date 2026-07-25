"""Build a county-major evaluation manifest from trained method directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _method_spec(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Method must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Method must be NAME=PATH")
    return name.strip(), Path(raw_path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", type=_method_spec, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config-name",
        default="configs/epi_abm_multi_daily_seed100.yaml",
    )
    parser.add_argument("--eval-dir-name", default="gpu_county_major_val")
    parser.add_argument("--outer-start", type=int, default=1)
    parser.add_argument("--outer-end", type=int, default=12)
    args = parser.parse_args()

    jobs = []
    for method_name, method_root in args.method:
        config = method_root / args.config_name
        if not config.is_file():
            raise FileNotFoundError(f"Missing config for {method_name}: {config}")
        for seed in args.seeds:
            checkpoint_dir = method_root / "train" / f"seed_{seed}" / "em_ckpt"
            checkpoints = {}
            for outer in range(int(args.outer_start), int(args.outer_end) + 1):
                label = f"outer{outer:04d}"
                path = checkpoint_dir / f"ct_iql_em_{label}.pt"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing checkpoint for {method_name} seed={seed}: {path}"
                    )
                checkpoints[label] = str(path.resolve())
            jobs.append({
                "id": f"{method_name}_seed_{seed}",
                "method": method_name,
                "seed": int(seed),
                "config": str(config.resolve()),
                "out_dir": str(
                    (method_root / args.eval_dir_name / f"seed_{seed}").resolve()
                ),
                "ckpts": checkpoints,
            })

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema": "epi_abm_county_major_jobs_v1",
        "jobs": jobs,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "county_major_manifest_created",
        "output": str(output),
        "jobs": len(jobs),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

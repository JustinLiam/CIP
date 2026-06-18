#!/usr/bin/env python3
"""
Aggregate em_iql_lr_ablation (or compatible grid) results into summary.csv.

Reads logs/{tag}/train.log + eval.log and done/{tag}.done metadata.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

RE_TAG = re.compile(r"^(runA_sym3e4|runB_alr1e3_qlr3e4_vf1e3)_seed(\d+)$")
RE_EVAL_MAE = re.compile(
    r"MAE normalized:\s*([\d.eE+-]+)\s*\|\s*MAE unscaled:\s*([\d.eE+-]+)"
    r"(?:\s*\|\s*RMSE unscaled:\s*([\d.eE+-]+))?"
)
RE_TRAIN_DONE = re.compile(
    r"EM training done\.\s+best_outer=(\d+)\s+best_\w+=([\d.eE+-]+)"
)
RE_OUTER = re.compile(r"EM outer (\d+)/(\d+)")


def _read_kv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _parse_train(train_log: Path) -> Dict[str, Optional[str]]:
    text = train_log.read_text(errors="replace") if train_log.is_file() else ""
    done = RE_TRAIN_DONE.search(text)
    outers = RE_OUTER.findall(text)
    last_outer = outers[-1] if outers else None
    return {
        "train_done": "1" if "EM training done" in text else "0",
        "segfault": "1" if re.search(r"段错误|segfault", text, re.I) else "0",
        "best_outer": done.group(1) if done else (last_outer[0] if last_outer else ""),
        "best_val_mae_uns": done.group(2) if done else "",
        "last_outer": last_outer[0] if last_outer else "",
        "em_outer_iters": last_outer[1] if last_outer else "",
    }


def _parse_eval(eval_log: Path) -> Dict[str, str]:
    text = eval_log.read_text(errors="replace") if eval_log.is_file() else ""
    m = None
    for match in RE_EVAL_MAE.finditer(text):
        m = match
    if not m:
        return {
            "eval_mae_norm": "",
            "eval_mae_uns": "",
            "eval_rmse_uns": "",
            "has_eval": "0",
        }
    rmse = m.group(3) if m.lastindex and m.lastindex >= 3 and m.group(3) else ""
    return {
        "eval_mae_norm": m.group(1),
        "eval_mae_uns": m.group(2),
        "eval_rmse_uns": rmse,
        "has_eval": "1",
    }


def collect(grid_root: Path) -> List[Dict[str, str]]:
    logs_dir = grid_root / "logs"
    done_dir = grid_root / "done"
    rows: List[Dict[str, str]] = []

    tags = sorted({p.name for p in logs_dir.iterdir() if p.is_dir()} | {f.stem for f in done_dir.glob("*.done")})
    for tag in tags:
        m = RE_TAG.match(tag)
        if not m:
            continue
        run_label, seed = m.group(1), m.group(2)
        meta = _read_kv(done_dir / f"{tag}.done")
        train_info = _parse_train(logs_dir / tag / "train.log")
        eval_info = _parse_eval(logs_dir / tag / "eval.log")

        actor_lr = meta.get("iql_actor_lr", "")
        qf_lr = meta.get("iql_qf_lr", "")
        vf_lr = meta.get("iql_vf_lr", "")
        row = {
            "tag": tag,
            "run_label": run_label,
            "seed": seed,
            "iql_actor_lr": actor_lr,
            "iql_qf_lr": qf_lr,
            "iql_vf_lr": vf_lr,
            "iql_beta": meta.get("iql_beta", ""),
            "iql_tau": meta.get("iql_tau", ""),
            "iql_max_grad_norm": meta.get("iql_max_grad_norm", ""),
            "em_m_steps_per_outer": meta.get("em_m_steps_per_outer", ""),
            "train_done": train_info["train_done"] or "0",
            "segfault": train_info["segfault"] or "0",
            "train_exit": meta.get("train_exit", ""),
            "best_outer": train_info["best_outer"] or "",
            "best_val_mae_uns": train_info["best_val_mae_uns"] or "",
            "last_outer": train_info["last_outer"] or "",
            "has_eval": eval_info["has_eval"],
            "eval_mae_norm": eval_info["eval_mae_norm"],
            "eval_mae_uns": eval_info["eval_mae_uns"],
            "eval_rmse_uns": eval_info["eval_rmse_uns"],
            "em_ckpt": meta.get("em_ckpt", ""),
            "finished_at": meta.get("finished_at", ""),
            "mlflow_experiment": meta.get("mlflow_experiment", ""),
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["run_label"], int(r["seed"]) if r["seed"].isdigit() else r["seed"]))
    return rows


FIELDNAMES = [
    "tag",
    "run_label",
    "seed",
    "iql_actor_lr",
    "iql_qf_lr",
    "iql_vf_lr",
    "iql_beta",
    "iql_tau",
    "iql_max_grad_norm",
    "em_m_steps_per_outer",
    "train_done",
    "segfault",
    "train_exit",
    "best_outer",
    "best_val_mae_uns",
    "last_outer",
    "has_eval",
    "eval_mae_norm",
    "eval_mae_uns",
    "eval_rmse_uns",
    "em_ckpt",
    "finished_at",
    "mlflow_experiment",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect LR ablation summary.csv")
    ap.add_argument("--grid-root", type=Path, required=True)
    ap.add_argument("--gamma", type=int, default=None, help="unused; for CLI symmetry with bash script")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    grid_root = args.grid_root.resolve()
    out = args.output or (grid_root / "summary.csv")
    rows = collect(grid_root)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    # Print quick aggregate for terminal
    by_run: Dict[str, List[float]] = {}
    for r in rows:
        if r["eval_mae_uns"]:
            by_run.setdefault(r["run_label"], []).append(float(r["eval_mae_uns"]))
    print(f"Wrote {len(rows)} rows -> {out}")
    for label, maes in sorted(by_run.items()):
        mean = sum(maes) / len(maes)
        print(f"  {label}: n={len(maes)} eval_mae_uns mean={mean:.4f}")


if __name__ == "__main__":
    main()

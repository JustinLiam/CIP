#!/usr/bin/env python3
"""
Analyze em_iql_lr_ablation summary.csv and write a short report.

Usage:
  python scripts/cancer/train/analyze_lr_ablation_results.py \\
    --grid-root grid_results/em_iql_lr_ablation/gamma_4
"""
from __future__ import annotations

import argparse
import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def _f(x: str) -> Optional[float]:
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def analyze(rows: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("EM IQL LR Ablation Analysis")
    lines.append("=" * 60)
    lines.append(f"Total rows in summary: {len(rows)}")

    eval_rows = [r for r in rows if _f(r.get("eval_mae_uns", "")) is not None]
    lines.append(f"Rows with eval_mae_uns: {len(eval_rows)}")
    lines.append("")

    by_run: Dict[str, List[float]] = defaultdict(list)
    by_run_seed: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in eval_rows:
        mae = float(r["eval_mae_uns"])
        by_run[r["run_label"]].append(mae)
        by_run_seed[r["run_label"]][r["seed"]] = mae

    if not by_run:
        lines.append("No eval results yet. Wait for jobs to finish.")
        return "\n".join(lines) + "\n"

    lines.append("--- Per-run aggregate (eval MAE unscaled) ---")
    run_stats = []
    for label, maes in sorted(by_run.items()):
        mean = stats.mean(maes)
        sd = stats.pstdev(maes) if len(maes) > 1 else 0.0
        run_stats.append((mean, sd, len(maes), label, min(maes), max(maes)))
        lines.append(
            f"  {label}: mean={mean:.4f} sd={sd:.4f} n={len(maes)} "
            f"min={min(maes):.4f} max={max(maes):.4f}"
        )
    lines.append("")

    if len(run_stats) >= 2:
        run_stats.sort()
        best = run_stats[0]
        second = run_stats[1]
        delta = second[0] - best[0]
        lines.append("--- A vs B ---")
        lines.append(f"  Lower mean is better.")
        lines.append(f"  Best:  {best[3]} (mean={best[0]:.4f})")
        lines.append(f"  Other: {second[3]} (mean={second[0]:.4f})")
        lines.append(f"  Delta (other - best): {delta:+.4f}")
        if delta > 0.01:
            lines.append(f"  => Prefer {best[3]} on average.")
        elif delta < -0.01:
            lines.append(f"  => Prefer {second[3]} on average.")
        else:
            lines.append("  => No clear winner (<0.01 gap); check per-seed and stability.")
        lines.append("")

    # Paired by seed where both runs exist
    seeds = sorted(set(r["seed"] for r in eval_rows), key=lambda s: int(s) if s.isdigit() else s)
    labels = sorted(by_run.keys())
    if len(labels) == 2:
        a, b = labels[0], labels[1]
        # ensure A is runA if present
        if "runA" in a or "sym3e4" in a:
            pass
        elif "runA" in b or "sym3e4" in b:
            a, b = b, a
        paired = []
        for s in seeds:
            if s in by_run_seed[a] and s in by_run_seed[b]:
                paired.append((s, by_run_seed[a][s], by_run_seed[b][s]))
        if paired:
            lines.append("--- Paired comparison by seed ---")
            wins_a = wins_b = 0
            for s, m_a, m_b in paired:
                winner = a if m_a < m_b else b
                if m_a < m_b:
                    wins_a += 1
                elif m_b < m_a:
                    wins_b += 1
                lines.append(f"  seed={s:>6}: {a}={m_a:.4f}  {b}={m_b:.4f}  -> {winner}")
            lines.append(f"  Wins: {a}={wins_a}, {b}={wins_b}, ties={len(paired)-wins_a-wins_b}")
            diffs = [mb - ma for _, ma, mb in paired]
            lines.append(f"  Mean paired diff ({b}-{a}): {stats.mean(diffs):+.4f}")
            lines.append("")

    lines.append("--- Per-seed marginal (all runs) ---")
    by_seed: Dict[str, List[float]] = defaultdict(list)
    for r in eval_rows:
        by_seed[r["seed"]].append(float(r["eval_mae_uns"]))
    for s in seeds:
        v = by_seed[s]
        lines.append(f"  seed={s:>6}: mean={stats.mean(v):.4f} n={len(v)}")
    lines.append("")

    incomplete = [r for r in rows if r.get("has_eval") != "1" or not r.get("eval_mae_uns")]
    if incomplete:
        lines.append("--- Incomplete / missing eval ---")
        for r in incomplete:
            lines.append(
                f"  {r.get('tag','?')}: train_done={r.get('train_done')} "
                f"segfault={r.get('segfault')} has_eval={r.get('has_eval')}"
            )
        lines.append("")

    lines.append("--- Recommendation draft (for vcip.yaml) ---")
    if run_stats:
        winner = min(run_stats, key=lambda x: x[0])
        wl = winner[3]
        if "sym3e4" in wl or "runA" in wl:
            lines.append("  iql_actor_lr: 3.0e-4")
            lines.append("  iql_qf_lr:    3.0e-4")
            lines.append("  iql_vf_lr:    3.0e-4")
        else:
            lines.append("  iql_actor_lr: 1.0e-3")
            lines.append("  iql_qf_lr:    3.0e-4")
            lines.append("  iql_vf_lr:    1.0e-3")
        lines.append(f"  (based on lower mean eval MAE: {wl})")
        if winner[2] < 6:
            lines.append("  NOTE: fewer than 6 seeds per run; treat as provisional.")
    lines.append("=" * 60)
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    grid_root = args.grid_root.resolve()
    summary = grid_root / "summary.csv"
    out = args.output or (grid_root / "analysis_report.txt")
    report = analyze(load_rows(summary))
    out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()

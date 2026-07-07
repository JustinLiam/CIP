from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


STATE_BY_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "72": "PR",
}


def normalize_county(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def counties_from_csv(path: Path) -> List[str]:
    counties = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "county" not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain a 'county' column.")
        for row in reader:
            counties.add(normalize_county(row["county"]))
    return sorted(counties)


def counties_from_arg(values: Sequence[str]) -> List[str]:
    counties = []
    for value in values:
        path = Path(value)
        if path.exists():
            counties.extend(normalize_county(line) for line in path.read_text().splitlines() if line.strip())
        else:
            counties.extend(normalize_county(x) for x in value.replace(",", " ").split() if x.strip())
    return sorted(set(counties))


def has_networks(epi_root: Path, county: str, num_steps: int) -> bool:
    net_root = epi_root / "data" / "networks" / "covid_output_causal" / county / "mobility_networks"
    required = [net_root / "HOUSEHOLD_NETWORK.pkl"]
    for subdir in ("schoolnets", "occnets", "randnets"):
        required.append(net_root / subdir / "0.pkl")
        required.append(net_root / subdir / f"{num_steps - 1}.pkl")
    return all(path.exists() for path in required)


def calibrated_param_paths(epi_root: Path, county: str, date_tag: str) -> List[Path]:
    root = epi_root / "result_graphs" / county / date_tag
    if not root.exists():
        return []
    return sorted(root.glob("*/calibrated_params.txt"))


def audit_county(epi_root: Path, county: str, date_tag: str, num_steps: int) -> Dict[str, object]:
    state = STATE_BY_FIPS.get(county[:2], "")
    population_dir = epi_root / "populations" / f"pop{county}"
    processed_dir = epi_root / "data" / "processed_data" / county / date_tag
    param_paths = calibrated_param_paths(epi_root, county, date_tag)
    status = {
        "county": county,
        "state": state,
        "delphi": (epi_root / "data" / "delphi_county_data" / f"{county}_data.csv").exists(),
        "state_data": all(
            (epi_root / "data" / "state_data" / state / county / name).exists()
            for name in ("agents_ages.csv", "agents_household_sizes.csv", "agents_occupations.csv")
        ) if state else False,
        "population_csv": (epi_root / "data" / "population_data" / f"{state}_population_data" / f"{county}_population.csv").exists() if state else False,
        "population_package": all(
            (population_dir / name).exists()
            for name in ("__init__.py", "age.pickle", "disease_stages.csv", "intervention.csv")
        ),
        "processed_data": all((processed_dir / name).exists() for name in ("daily_data.csv", "weekly_data.csv")),
        "networks": has_networks(epi_root, county, num_steps),
        "calibrated_params": bool(param_paths),
        "calibrated_param_paths": [str(path) for path in param_paths],
    }
    status["ready_for_cache"] = bool(
        status["population_package"]
        and status["processed_data"]
        and status["networks"]
        and status["calibrated_params"]
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit per-county upstream epi-diff-abm assets.")
    parser.add_argument("--epi-root", default="external_repos/epi-diff-abm")
    parser.add_argument("--date-tag", default="202010-202104")
    parser.add_argument("--num-steps", type=int, default=182)
    parser.add_argument("--from-epicf-csv", default="external_repos/epi-diff-abm/data/multi_policy_data.csv")
    parser.add_argument("--counties", nargs="*", default=None, help="County FIPS values or files containing one county per line.")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--ready-out", default=None)
    args = parser.parse_args()

    epi_root = Path(args.epi_root)
    if args.counties:
        counties = counties_from_arg(args.counties)
    else:
        counties = counties_from_csv(Path(args.from_epicf_csv))

    rows = [audit_county(epi_root, county, args.date_tag, args.num_steps) for county in counties]
    ready = [row["county"] for row in rows if row["ready_for_cache"]]
    missing = [row["county"] for row in rows if not row["ready_for_cache"]]
    summary = {
        "total_counties": len(counties),
        "ready_for_cache": len(ready),
        "missing_or_incomplete": len(missing),
        "ready_counties": ready,
        "missing_counties": missing,
        "by_state": {},
    }
    for row in rows:
        summary["by_state"].setdefault(row["state"], {"total": 0, "ready": 0})
        summary["by_state"][row["state"]]["total"] += 1
        summary["by_state"][row["state"]]["ready"] += int(bool(row["ready_for_cache"]))

    payload = {"summary": summary, "counties": rows}
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.ready_out:
        path = Path(args.ready_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(ready) + ("\n" if ready else ""), encoding="utf-8")


if __name__ == "__main__":
    main()

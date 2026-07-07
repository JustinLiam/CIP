from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import math


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_epi_abm_root() -> Path:
    return project_root() / "external_repos" / "epi-diff-abm"


@dataclass(frozen=True)
class EpiABMRegistryEntry:
    county: str
    state_abbrev: str
    date_tag: str
    num_steps: int
    num_weeks: int
    num_weeks_to_eval: int
    initial_infection_rate: str = "0.0005"
    exposed_to_infected: str = "3"
    infected_to_recovered: str = "5"
    with_k: str = "True"
    with_vacc: str = "False"
    population_size: int = 44933
    calibration_rmse: Optional[float] = None

    def param_subdir(self) -> str:
        return (
            f"{self.initial_infection_rate}_{self.exposed_to_infected}_"
            f"{self.infected_to_recovered}_{self.with_k}_{self.with_vacc}"
        )

    def calibrated_params_path(self, epi_root: Path) -> Path:
        return (
            epi_root
            / "result_graphs"
            / self.county
            / self.date_tag
            / self.param_subdir()
            / "calibrated_params.txt"
        )

    def population_dir(self, epi_root: Path) -> Path:
        return epi_root / "populations" / f"pop{self.county}"

    def processed_daily_path(self, epi_root: Path) -> Path:
        return epi_root / "data" / "processed_data" / self.county / self.date_tag / "daily_data.csv"


DEFAULT_REGISTRY: Dict[str, EpiABMRegistryEntry] = {
    "01045": EpiABMRegistryEntry(
        county="01045",
        state_abbrev="AL",
        date_tag="202010-202104",
        num_steps=182,
        num_weeks=26,
        num_weeks_to_eval=24,
        population_size=44933,
    )
}


def _parse_param_subdir(name: str) -> Dict[str, str]:
    parts = name.split("_")
    if len(parts) < 5:
        return {}
    return {
        "initial_infection_rate": parts[0],
        "exposed_to_infected": parts[1],
        "infected_to_recovered": parts[2],
        "with_k": parts[3],
        "with_vacc": parts[4],
    }


def _infer_num_steps(epi_root: Path, county: str, date_tag: str) -> int:
    intervention_path = epi_root / "populations" / f"pop{county}" / "intervention.csv"
    if intervention_path.exists():
        return max(1, sum(1 for _ in intervention_path.open()) - 1)
    daily_path = epi_root / "data" / "processed_data" / county / date_tag / "daily_data.csv"
    if daily_path.exists():
        return max(1, sum(1 for _ in daily_path.open()) - 1)
    return 182


def discover_registry_entry(county: str, *, date_tag: Optional[str] = None, epi_root: Optional[Path] = None) -> Optional[EpiABMRegistryEntry]:
    county = str(county).zfill(5)
    root = default_epi_abm_root() if epi_root is None else Path(epi_root)
    county_root = root / "result_graphs" / county
    if not county_root.exists():
        return None

    date_dirs = [p for p in county_root.iterdir() if p.is_dir()]
    if date_tag is not None:
        date_dirs = [p for p in date_dirs if p.name == str(date_tag)]
    for date_dir in sorted(date_dirs):
        for param_file in sorted(date_dir.glob("*/calibrated_params.txt")):
            params = _parse_param_subdir(param_file.parent.name)
            num_steps = _infer_num_steps(root, county, date_dir.name)
            num_weeks = int(math.ceil(num_steps / 7.0))
            return EpiABMRegistryEntry(
                county=county,
                state_abbrev="",
                date_tag=date_dir.name,
                num_steps=num_steps,
                num_weeks=num_weeks,
                num_weeks_to_eval=max(num_weeks - 2, 1),
                population_size=0,
                **params,
            )
    return None


def get_registry_entry(county: str, *, date_tag: Optional[str] = None) -> EpiABMRegistryEntry:
    county = str(county).zfill(5)
    if county not in DEFAULT_REGISTRY:
        discovered = discover_registry_entry(county, date_tag=date_tag)
        if discovered is not None:
            return discovered
        raise KeyError(f"No EpiABM calibration registry entry for county {county!r}.")
    entry = DEFAULT_REGISTRY[county]
    if date_tag is None or str(date_tag) == entry.date_tag:
        return entry
    discovered = discover_registry_entry(county, date_tag=date_tag)
    if discovered is not None:
        return discovered
    raise KeyError(f"No EpiABM registry entry for county={county!r}, date_tag={date_tag!r}.")

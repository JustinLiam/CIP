#!/usr/bin/env bash
# Reproduce the bundled epi-diff-abm single-county 01045 pipeline.
#
# This wrapper only changes runtime parameters, file/package placement, and logging.
# It does not change the data-generation, network-generation, calibration, or
# counterfactual mechanisms in the upstream Python code.

set -euo pipefail

MODE="${1:-prep}"  # prep | calibrate | counterfactual | all | verify

COUNTY="${COUNTY:-01045}"
STATE_ABBREV="${STATE_ABBREV:-AL}"
CONDA_ENV="${CONDA_ENV:-torch_309}"
NUM_STEPS="${NUM_STEPS:-182}"
NUM_WEEKS="${NUM_WEEKS:-26}"
NUM_WEEKS_TO_EVAL="${NUM_WEEKS_TO_EVAL:-24}"
DATE_TAG="${DATE_TAG:-202010-202104}"
DEVICE="${DEVICE:-cuda}"
DELPHI_FETCH_MODE="${DELPHI_FETCH_MODE:-direct_http}"  # direct_http | upstream

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/epi_abm/upstream_reproduction/${COUNTY}_${RUN_ID}}"
LOG_DIR="${RUN_ROOT}/logs"
CONFIG_PATH="${ROOT}/covid_abm/yamls/config.yaml"
CONFIG_BACKUP="${RUN_ROOT}/config.yaml.before"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*"
}

run_logged() {
  local name="$1"
  shift
  log "running ${name}: $*"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

activate_env() {
  local had_nounset=0
  case "$-" in
    *u*)
      had_nounset=1
      set +u
      ;;
  esac
  if [[ -n "${CONDA_EXE:-}" && -f "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  else
    eval "$(conda shell.bash hook)"
  fi
  conda activate "${CONDA_ENV}"
  if [[ "${had_nounset}" == "1" ]]; then
    set -u
  fi
  export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
}

require_keys() {
  : "${COVIDCAST_API_KEY:?Set COVIDCAST_API_KEY in the environment.}"
  : "${CENSUS_API_KEY:?Set CENSUS_API_KEY in the environment.}"
}

write_effective_config() {
  python - "${CONFIG_PATH}" "${COUNTY}" "${DATE_TAG}" "${NUM_STEPS}" "${NUM_WEEKS}" "${NUM_WEEKS_TO_EVAL}" "${DEVICE}" "$1" <<'PY'
import sys
from pathlib import Path
import yaml

path, county, date_tag, num_steps, num_weeks, num_weeks_to_eval, device, generating_cf = sys.argv[1:]
cfg_path = Path(path)
cfg = yaml.safe_load(cfg_path.read_text())
meta = cfg["simulation_metadata"]
meta["POPULATION"] = str(county)
meta["DATE"] = str(date_tag)
meta["num_steps_per_episode"] = int(num_steps)
meta["NUM_WEEKS"] = int(num_weeks)
meta["NUM_WEEKS_TO_EVAL"] = int(num_weeks_to_eval)
meta["device"] = str(device)
meta["GENERATING_COUNTERFACTUAL"] = str(generating_cf).lower() in {"1", "true", "yes"}
meta["calibration"] = True
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

restore_config() {
  if [[ -f "${CONFIG_BACKUP}" && "${KEEP_CONFIG:-0}" != "1" ]]; then
    cp "${CONFIG_BACKUP}" "${CONFIG_PATH}"
    log "restored ${CONFIG_PATH} from ${CONFIG_BACKUP}"
  fi
}

prepare_config_backup() {
  if [[ ! -f "${CONFIG_BACKUP}" ]]; then
    cp "${CONFIG_PATH}" "${CONFIG_BACKUP}"
  fi
  trap restore_config EXIT
}

prepare_population_package() {
  python - "${ROOT}" "${COUNTY}" <<'PY'
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1])
county = sys.argv[2]
pop_root = root / "populations"
pop_dir = pop_root / f"pop{county}"
pop_dir.mkdir(parents=True, exist_ok=True)
(pop_root / "__init__.py").touch()
(pop_dir / "__init__.py").touch()

mapping = pop_dir / "mapping.json"
population_mapping = pop_dir / "population_mapping.json"
if mapping.exists() and not population_mapping.exists():
    shutil.copyfile(mapping, population_mapping)
PY
}

verify_assets() {
  python - "${ROOT}" "${COUNTY}" "${DATE_TAG}" "${NUM_STEPS}" "${STATE_ABBREV}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
county = sys.argv[2]
date_tag = sys.argv[3]
num_steps = int(sys.argv[4])
state_abbrev = sys.argv[5]
checks = [
    root / "data" / "delphi_county_data" / f"{county}_data.csv",
    root / "data" / "state_data" / state_abbrev / county / "agents_ages.csv",
    root / "data" / "state_data" / state_abbrev / county / "agents_household_sizes.csv",
    root / "data" / "state_data" / state_abbrev / county / "agents_occupations.csv",
    root / "data" / "population_data" / f"{state_abbrev}_population_data" / f"{county}_population.csv",
    root / "data" / "processed_data" / county / date_tag / "daily_data.csv",
    root / "populations" / f"pop{county}" / "__init__.py",
    root / "populations" / f"pop{county}" / "age.pickle",
    root / "populations" / f"pop{county}" / "disease_stages.csv",
    root / "populations" / f"pop{county}" / "intervention.csv",
    root / "data" / "networks" / "covid_output_causal" / county / "mobility_networks" / "HOUSEHOLD_NETWORK.pkl",
]
for t in range(min(num_steps, 3)):
    checks.extend([
        root / "data" / "networks" / "covid_output_causal" / county / "mobility_networks" / "schoolnets" / f"{t}.pkl",
        root / "data" / "networks" / "covid_output_causal" / county / "mobility_networks" / "occnets" / f"{t}.pkl",
        root / "data" / "networks" / "covid_output_causal" / county / "mobility_networks" / "randnets" / f"{t}.pkl",
    ])

missing = [str(p.relative_to(root)) for p in checks if not p.exists()]
if missing:
    print("MISSING_ASSETS")
    for item in missing:
        print(item)
    raise SystemExit(1)
print("asset verification passed")
PY
}

run_prep() {
  require_keys
  export EPI_STATE_ABBREV="${STATE_ABBREV}"
  export EPI_COUNTIES="${COUNTY}"
  if [[ "${DELPHI_FETCH_MODE}" == "upstream" ]]; then
    run_logged 01_delphi python scripts/delphi_api.py
  elif [[ "${DELPHI_FETCH_MODE}" == "direct_http" ]]; then
    run_logged 01_delphi_direct_http python - "${ROOT}" "${COUNTY}" <<'PY'
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

root = Path(sys.argv[1])
county = sys.argv[2]
out_dir = root / "data" / "delphi_county_data"
out_dir.mkdir(parents=True, exist_ok=True)

def fetch(signal: str) -> pd.DataFrame:
    params = {
        "source": "covidcast",
        "data_source": "indicator-combination",
        "signal": signal,
        "time_type": "day",
        "geo_type": "county",
        "time_values": "20200601-20210731",
        "geo_values": county,
    }
    url = "https://api.delphi.cmu.edu/epidata/api.php"
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("result") != 1:
        raise RuntimeError(f"Delphi API failed for {signal}: {payload.get('message')}")
    rows = payload.get("epidata", [])
    if not rows:
        raise RuntimeError(f"Delphi API returned no rows for {signal}")
    df = pd.DataFrame(rows)
    df = df[["time_value", "geo_value", "value"]].copy()
    df["time_value"] = df["time_value"].map(
        lambda x: datetime.strptime(str(int(x)), "%Y%m%d").strftime("%Y-%m-%d")
    )
    return df.rename(columns={"value": signal})

cases = fetch("confirmed_incidence_num")
deaths = fetch("deaths_incidence_num")
data = cases.merge(deaths, on=["time_value", "geo_value"], how="outer")
data = data.rename(
    columns={
        "confirmed_incidence_num": "cases",
        "deaths_incidence_num": "deaths",
    }
).sort_values(["geo_value", "time_value"])
out_path = out_dir / f"{county}_data.csv"
data[["time_value", "geo_value", "cases", "deaths"]].to_csv(out_path, index=False)
print(data[["time_value", "geo_value", "cases", "deaths"]].head(20).to_string(index=False))
print(f"wrote {out_path} rows={len(data)}")
PY
  else
    echo "Unknown DELPHI_FETCH_MODE=${DELPHI_FETCH_MODE}" >&2
    exit 2
  fi
  run_logged 02_census python scripts/census.py
  log "running 03_networks: full experiment, ${NUM_STEPS} timesteps, pickle output"
  (
    cd "${ROOT}/networks"
    printf 'yes\nfull\n%s\nyes\n' "${NUM_STEPS}" | python initialize_experiment.py
  ) 2>&1 | tee "${LOG_DIR}/03_networks.log"
  run_logged 04_process_counties python scripts/process_counties.py
  prepare_population_package
  verify_assets 2>&1 | tee "${LOG_DIR}/05_verify_assets.log"
}

run_calibrate() {
  prepare_config_backup
  prepare_population_package
  write_effective_config false
  verify_assets 2>&1 | tee "${LOG_DIR}/10_verify_before_calibration.log"
  run_logged 11_calibration python main.py
}

run_counterfactual() {
  prepare_config_backup
  prepare_population_package
  write_effective_config true
  run_logged 21_counterfactual python main.py
}

main() {
  cd "${ROOT}"
  activate_env
  log "root=${ROOT}"
  log "mode=${MODE} county=${COUNTY} state=${STATE_ABBREV} conda_env=${CONDA_ENV}"
  log "run_root=${RUN_ROOT}"
  python - <<'PY' | tee "${LOG_DIR}/00_environment.log"
import sys
import pandas as pd
import torch
print("python", sys.version)
print("pandas", pd.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

  case "${MODE}" in
    prep)
      run_prep
      ;;
    calibrate)
      run_calibrate
      ;;
    counterfactual)
      run_counterfactual
      ;;
    all)
      run_prep
      run_calibrate
      run_counterfactual
      ;;
    verify)
      prepare_population_package
      verify_assets
      ;;
    *)
      echo "Unknown mode: ${MODE}" >&2
      echo "Usage: $0 [prep|calibrate|counterfactual|all|verify]" >&2
      exit 2
      ;;
  esac
  log "completed mode=${MODE}"
}

main "$@"

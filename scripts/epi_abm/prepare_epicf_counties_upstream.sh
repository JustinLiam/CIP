#!/usr/bin/env bash
# Prepare upstream epi-diff-abm assets for EpiCF counties.
#
# Modes:
#   audit      report which counties already have population/network/calibrated assets
#   prep       fetch Delphi/Census data, generate populations/networks/intervention files
#   calibrate  run upstream county-level calibration for counties with prep assets
#   all        prep then calibrate
#
# This script parameterizes the existing upstream scripts; it does not change
# their data generation or calibration mechanisms.

set -euo pipefail

MODE="${1:-audit}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EPI_ROOT="${EPI_ROOT:-${ROOT}/data_generation/epi_diff_abm}"
COUNTY_CSV="${COUNTY_CSV:-${EPI_ROOT}/data/multi_policy_data.csv}"
COUNTY_FILE="${COUNTY_FILE:-}"
CONDA_ENV="${CONDA_ENV:-torch_309}"
NUM_STEPS="${NUM_STEPS:-182}"
DATE_TAG="${DATE_TAG:-202010-202104}"
BATCH_RUN_ROOT="${BATCH_RUN_ROOT:-${ROOT}/results/epi_abm/upstream_prep/epicf_counties_$(date +%Y%m%d_%H%M%S)}"
MAX_COUNTIES="${MAX_COUNTIES:-0}"
DELPHI_DIRECT_RANGE="${DELPHI_DIRECT_RANGE:-1}"
DELPHI_COUNTY_BATCH_SIZE="${DELPHI_COUNTY_BATCH_SIZE:-1}"
DELPHI_CHUNK_DAYS="${DELPHI_CHUNK_DAYS:-21}"
DELPHI_TIMEOUT_SECONDS="${DELPHI_TIMEOUT_SECONDS:-120}"
DELPHI_RETRIES="${DELPHI_RETRIES:-3}"
CENSUS_TIMEOUT_SECONDS="${CENSUS_TIMEOUT_SECONDS:-60}"
CENSUS_RETRIES="${CENSUS_RETRIES:-5}"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*"
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
  export PYTHONPATH="${ROOT}:${EPI_ROOT}:${PYTHONPATH:-}"
}

county_state_table() {
  python - "${COUNTY_CSV}" "${COUNTY_FILE}" "${MAX_COUNTIES}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
county_file = sys.argv[2]
max_counties = int(sys.argv[3])
state_by_fips = {
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

def norm(x):
    x = str(x).strip()
    if x.endswith(".0"):
        x = x[:-2]
    return x.zfill(5)

if county_file:
    counties = [norm(line) for line in Path(county_file).read_text().splitlines() if line.strip()]
else:
    counties = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counties.add(norm(row["county"]))
    counties = sorted(counties)
if max_counties > 0:
    counties = counties[:max_counties]
for county in counties:
    print(f"{state_by_fips[county[:2]]}\t{county}")
PY
}

run_audit() {
  mkdir -p "${BATCH_RUN_ROOT}"
  python "${ROOT}/scripts/epi_abm/audit_epi_abm_assets.py" \
    --epi-root "${EPI_ROOT}" \
    --date-tag "${DATE_TAG}" \
    --num-steps "${NUM_STEPS}" \
    --from-epicf-csv "${COUNTY_CSV}" \
    --json-out "${BATCH_RUN_ROOT}/asset_audit.json" \
    --ready-out "${BATCH_RUN_ROOT}/ready_counties.txt"
}

run_prep() {
  : "${COVIDCAST_API_KEY:?Set COVIDCAST_API_KEY for Delphi/covidcast access.}"
  : "${CENSUS_API_KEY:?Set CENSUS_API_KEY for Census access.}"
  mkdir -p "${BATCH_RUN_ROOT}/logs"
  county_state_table | awk -F'\t' '{ by_state[$1]=by_state[$1] " " $2 } END { for (s in by_state) print s "\t" by_state[s] }' | sort |
  while IFS=$'\t' read -r state counties; do
    counties="${counties# }"
    log "prep state=${state} counties=${counties}"
    (
      cd "${EPI_ROOT}"
      export EPI_STATE_ABBREV="${state}"
      export EPI_COUNTIES="${counties}"
      export EPI_DELPHI_DIRECT_RANGE="${DELPHI_DIRECT_RANGE}"
      export EPI_DELPHI_COUNTY_BATCH_SIZE="${DELPHI_COUNTY_BATCH_SIZE}"
      export EPI_DELPHI_CHUNK_DAYS="${DELPHI_CHUNK_DAYS}"
      export EPI_DELPHI_TIMEOUT_SECONDS="${DELPHI_TIMEOUT_SECONDS}"
      export EPI_DELPHI_RETRIES="${DELPHI_RETRIES}"
      export EPI_CENSUS_TIMEOUT_SECONDS="${CENSUS_TIMEOUT_SECONDS}"
      export EPI_CENSUS_RETRIES="${CENSUS_RETRIES}"
      python scripts/delphi_api.py
      python scripts/census.py
      cd networks
      printf 'yes\nfull\n%s\nyes\n' "${NUM_STEPS}" | python initialize_experiment.py
      cd ..
      python scripts/process_counties.py
    ) 2>&1 | tee "${BATCH_RUN_ROOT}/logs/prep_${state}.log"
  done
}

run_calibrate() {
  mkdir -p "${BATCH_RUN_ROOT}/logs"
  county_state_table |
  while IFS=$'\t' read -r state county; do
    log "calibrate county=${county} state=${state}"
    (
      cd "${EPI_ROOT}"
      COUNTY="${county}" \
      STATE_ABBREV="${state}" \
      CONDA_ENV="${CONDA_ENV}" \
      NUM_STEPS="${NUM_STEPS}" \
      DATE_TAG="${DATE_TAG}" \
      RUN_ROOT="${BATCH_RUN_ROOT}/${county}" \
      bash scripts/reproduce_01045_torch309.sh calibrate
    ) 2>&1 | tee "${BATCH_RUN_ROOT}/logs/calibrate_${county}.log"
  done
}

main() {
  activate_env
  log "mode=${MODE} epi_root=${EPI_ROOT} batch_run_root=${BATCH_RUN_ROOT}"
  case "${MODE}" in
    audit)
      run_audit
      ;;
    prep)
      run_prep
      run_audit
      ;;
    calibrate)
      run_calibrate
      run_audit
      ;;
    all)
      run_prep
      run_calibrate
      run_audit
      ;;
    *)
      echo "Usage: $0 [audit|prep|calibrate|all]" >&2
      exit 2
      ;;
  esac
}

main "$@"

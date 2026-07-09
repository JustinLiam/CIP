# CRIPO EpiABM Data Generation

This directory vendors the modified EpiABM simulator used by the CRIPO
experiments. It is included directly in this repository so reviewers can run the
data-generation pipeline without cloning a separate upstream repository or
applying a patch.

The code is based on `complex-ai-lab/epi-diff-abm` at upstream commit
`824ca2a9785038eaec4e277903856d796ac4adb3`, with the CRIPO adapter changes
already applied.

## What This Directory Contains

- `abm_nets.py` and `main.py`: upstream calibration and counterfactual entry
  points.
- `agent_torch/` and `covid_abm/`: the minimal ABM runtime needed by
  calibration and online rollout.
- `scripts/`: Delphi, Census, county processing, and single-county reproduction
  helpers.
- `networks/`: population and contact-network generation code.
- `covid_abm/yamls/config.yaml`: mutable runtime config template.
- `THIRD_PARTY_NOTICE.md` and `LICENSE.AGENTTORCH.md`: attribution and license
  information for bundled third-party code.

## CRIPO Adapter Changes

The vendored simulator keeps the original EpiABM calibration/data-generation
mechanism and adds only the integration hooks needed by CRIPO:

- dynamic population-module loading from `covid_abm/yamls/config.yaml`;
- optional online intervention injection in `NewTransmission`;
- continuous freezing-interval actions for school/workplace interventions;
- retry/direct-fetch support for Delphi and Census data preparation;
- environment-variable county/state selection for multi-county prep;
- duplicate OmegaConf resolver handling for repeated runner initialization.

## Data Layout

Generated or licensed assets are not committed. The release uses the following
canonical paths:

```text
data_generation/epi_diff_abm/data/
  multi_policy_data.csv                         EpiCF county/policy metadata
  full_google_mob_data/                         Google mobility CSV files
  delphi_county_data/                           Delphi/COVIDcast county cases
  state_data/                                   Census-derived county inputs
  population_data/                              generated demographic tables
  processed_data/<county>/<date_tag>/           upstream daily/weekly ABM inputs
  networks/covid_output_causal/<county>/        network pickle assets

data_generation/epi_diff_abm/populations/
  pop<county>/                                  generated population packages

data_generation/epi_diff_abm/result_graphs/
  <county>/<date_tag>/<calibration_setting>/    calibrated_params.txt and
                                                upstream calibration diagnostics

data_generation/epi_diff_abm/results/           upstream generated factual/CF CSVs
data/processed/epi_abm/                         CRIPO-ready dataset caches
results/epi_abm/                                run logs, smoke tests, evaluation
```

This mirrors the existing project convention: Tumor and MIMIC processed datasets
also live under `data/processed/`, while simulator-specific raw assets stay next
to their generator code.

## Reproduction Entry Points

Run these commands from the repository root unless noted otherwise.

Prepare upstream county assets:

```bash
bash scripts/epi_abm/prepare_epicf_counties_upstream.sh
```

Run calibration in parallel:

```bash
python scripts/epi_abm/run_parallel_calibration_pool.py
```

This writes logs and worker manifests under `results/epi_abm/calibration/` by
default. Calibrated parameters remain under
`data_generation/epi_diff_abm/result_graphs/`.

Build the CRIPO dataset cache:

```bash
python scripts/epi_abm/build_multi_county_cache.py
```

This writes the model-ready cache under `data/processed/epi_abm/multi_county/`
by default.

Create an isolated EpiABM runtime when running concurrent rollouts:

```bash
python scripts/epi_abm/create_isolated_runtime.py \
  --source data_generation/epi_diff_abm \
  --dest runtime/epi_abm/run
```

Run the release smoke test after placing or generating the assets:

```bash
python scripts/epi_abm/smoke_test_release.py --device cuda
```

The smoke test writes logs under `results/epi_abm/smoke/` and a temporary cache
under `data/processed/epi_abm/smoke/`.

Use `EPI_DIFF_ABM_ROOT` to point the CRIPO adapter to another local copy of this
directory when needed:

```bash
export EPI_DIFF_ABM_ROOT=/path/to/epi_diff_abm
```

## Required External Inputs

- Delphi COVIDcast API data, fetched by `scripts/delphi_api.py`.
- US Census API data, fetched by `scripts/census.py`.
- Google COVID-19 Community Mobility Reports, placed under
  `data/full_google_mob_data/`.
- EpiCF benchmark county/policy metadata, including `multi_policy_data.csv`.
- Generated EpiABM networks, population modules, and calibrated parameter files.

The API keys should be provided through local environment variables or a local
`.env` file. Do not commit API keys or generated data.

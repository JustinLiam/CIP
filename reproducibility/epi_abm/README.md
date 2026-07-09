# EpiABM Reproducibility Notes

This directory preserves the code-level bridge between CRIPO and the upstream
`epi-diff-abm` simulator without vendoring upstream code, generated data, or
local runtime artifacts.

## Upstream

- Repository: `https://github.com/complex-ai-lab/epi-diff-abm`
- Base commit used for the adapter patch: `824ca2a9785038eaec4e277903856d796ac4adb3`
- Local checkout convention: `external_repos/epi-diff-abm`

The upstream repository is not included in this repo. Clone it locally:

```bash
git clone https://github.com/complex-ai-lab/epi-diff-abm external_repos/epi-diff-abm
cd external_repos/epi-diff-abm
git checkout 824ca2a9785038eaec4e277903856d796ac4adb3
git apply ../../reproducibility/epi_abm/patches/epi-diff-abm-vcip-adapter-824ca2a.patch
```

## Patch Scope

`patches/epi-diff-abm-vcip-adapter-824ca2a.patch` contains only code changes
needed by the CRIPO EpiABM integration:

- dynamic population-module loading from `covid_abm/yamls/config.yaml`;
- optional online intervention injection in `NewTransmission`;
- continuous freezing interval actions;
- retry/direct-fetch support for Delphi and Census data preparation;
- environment-variable county/state selection for multi-county prep;
- duplicate OmegaConf resolver handling for repeated executor initialization.

The patch intentionally excludes:

- `covid_abm/yamls/config.yaml`, because it is a mutable runtime file and can
  contain machine-specific paths;
- `data/`, `populations/`, `result_graphs/`, `results/`, and
  `online_rollout_runs/`;
- generated CRIPO caches and training/evaluation outputs.

## Data Policy

Large or licensed datasets are not versioned here. Place them under the
upstream local checkout following the EpiCF/EpiABM data-preparation README:

```text
external_repos/epi-diff-abm/data/
external_repos/epi-diff-abm/populations/
external_repos/epi-diff-abm/result_graphs/
```

CRIPO-ready processed caches should stay ignored under:

```text
data/processed/epi_abm/
```

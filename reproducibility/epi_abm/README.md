# EpiABM Reproducibility Notes

This directory preserves provenance for the CRIPO changes to the upstream
`epi-diff-abm` simulator. The modified simulator code used by the paper is
vendored under `data_generation/epi_diff_abm` so reviewers do not need to clone
or patch a separate repository.

## Upstream

- Repository: `https://github.com/complex-ai-lab/epi-diff-abm`
- Base commit used for the adapter patch: `824ca2a9785038eaec4e277903856d796ac4adb3`
- Bundled CRIPO simulator path: `data_generation/epi_diff_abm`

The patch remains useful for auditing the difference from upstream. To verify
the vendored code against the upstream base, clone the original repository in a
temporary directory and apply the patch:

```bash
git clone https://github.com/complex-ai-lab/epi-diff-abm /tmp/epi-diff-abm
cd /tmp/epi-diff-abm
git checkout 824ca2a9785038eaec4e277903856d796ac4adb3
git apply /path/to/CRIPO/reproducibility/epi_abm/patches/epi-diff-abm-vcip-adapter-824ca2a.patch
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
bundled simulator checkout following the EpiCF/EpiABM data-preparation README:

```text
data_generation/epi_diff_abm/data/
data_generation/epi_diff_abm/populations/
data_generation/epi_diff_abm/result_graphs/
```

CRIPO-ready processed caches should stay ignored under:

```text
data/processed/epi_abm/
```

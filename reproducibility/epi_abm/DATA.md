# EpiABM Data Placement

The code release vendors the modified EpiABM simulator under:

```text
data_generation/epi_diff_abm/
```

Generated data are not committed. Before reproducing the EpiCF/CRIPO dataset,
place or generate assets in the following locations:

```text
data_generation/epi_diff_abm/data/multi_policy_data.csv
data_generation/epi_diff_abm/data/full_google_mob_data/
data_generation/epi_diff_abm/data/delphi_county_data/
data_generation/epi_diff_abm/data/state_data/
data_generation/epi_diff_abm/data/population_data/
data_generation/epi_diff_abm/data/processed_data/
data_generation/epi_diff_abm/data/networks/
data_generation/epi_diff_abm/populations/
data_generation/epi_diff_abm/result_graphs/
```

The final CRIPO-ready cache is generated under:

```text
data/processed/epi_abm/
```

This processed cache is also ignored by git. Use `EPI_DIFF_ABM_ROOT` to point the
CRIPO adapter to a different local simulator/data root if needed:

```bash
export EPI_DIFF_ABM_ROOT=/path/to/epi_diff_abm
```

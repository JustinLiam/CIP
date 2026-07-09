# Scripts README

This folder contains the data retrieval and preprocessing scripts for the COVID ABM simulation.

## Running the Data Preparation

To execute the entire data preparation pipeline, run the following command from the project root directory:

```bash
bash scripts/data_prep.sh
```

The script will automatically perform the following steps:

1. **Fetch Delphi Data:** Retrieves daily COVID-19 cases and deaths incidence from the COVIDcast API.
2. **Fetch Census Data:** Obtains household, age, and occupation distributions from the US Census Bureau.
3. **Initialize Experiment:** Triggers the network generation process (located in the `networks/` folder).
4. **Process Counties:** Performs final data cleaning, aggregates stats, and prepares the simulation environment folders.

## Individual Scripts

You can also run scripts individually if needed:
- `delphi_api.py`: Fetches epidemiological data.
- `census.py`: Fetches demographic distributions.
- `process_counties.py`: Prepares final simulation input files.

Ensure your `.env` file is configured with the necessary API keys before running these scripts.

## Configuration

Specific **counties** and **states** must be manually configured within each script. You can find the relevant variables (e.g., `target_counties`, `state_abbreviation`, or `fips_dict`) at the beginning of each `.py` file.

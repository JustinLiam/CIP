# Networks README

This folder contains the network generation scripts for the COVID ABM simulation.

## Running the Networks

To generate mobility networks, run the following command from the project root directory:

```bash
python3 networks/initialize_experiment.py
```

The script will prompt you with several configuration options:

1. **Customize population?** (yes/no): Choose whether to create a new custom population or use existing cached data.
2. **Run full experiment or test?** (full/test): Select the scope of the simulation.
3. **Number of timesteps**: Specify the duration of the simulation (up to 366).
4. **Output format** (pickle/csv): Choose your preferred file format for the generated networks.

## Configuration

Specific **counties** and **states** must be manually configured within `initialize_experiment.py`. You can find the relevant variables (`state_abbrev`, `target_counties`) at the beginning of the file. Ensure these match the data generated in the `scripts/` phase.

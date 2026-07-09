# Epidemic Differentiable ABM

This repository provides the implementation of an Agent-Based Model (ABM) benchmark for COVID-19 transmission and intervention modeling.

## Papers based on this project
* [KDD 2026] [Benchmarking Counterfactual Prediction in Epidemic Time Series with Time-Varying Interventions](https://github.com/complex-ai-lab/epi-cf-benchmark)

## Overview
The goal of this benchmark is to provide a standardized environment for simulating COVID-19 dynamics and testing various intervention strategies (counterfactuals). The pipeline follows a structured workflow:
1.  **Synthetic Population & Network Generation**: We generate digital representations of agents and their interaction environments (digital twins) for specific counties.
2.  **Simulation & Calibration**: We use differentiable simulations to calibrate model parameters to historical COVID-19 data.
3.  **Counterfactual Analysis**: We evaluate the impact of hypothetical policy changes (e.g., toggling school or workplace restrictions) compared to factual history.

## Requirements & API Keys

### Dependencies
The project requires a Python environment (>=3.9). Install all requirements using:
```bash
pip install -r requirements.txt
```

### API Keys
The data preparation scripts require access to external APIs. Create a `.env` file in the root directory with the following keys:
*   `COVIDCAST_API_KEY`: Required for the [Delphi COVIDcast API](https://cmu-delphi.github.io/delphi-epidata/api/covidcast_clients.html) to fetch epidemiological data.
*   `CENSUS_API_KEY`: Required for the [US Census Bureau API](https://api.census.gov/data/key_signup.html) to fetch demographic distributions.

## How to run

### Step 1: Data Preparation
In this step, we generate synthetic populations and mobility networks by combining real-world time series demographic and epidemiological data.

**What we are generating:**
*   **Synthetic Populations**: We use US Census data (household size, age, and occupation distributions) to create a set of digital agents that statistically match the demographic profile of a target county.
*   **Interaction Networks**: We construct mobility and contact networks (Household, School, and Occupational) that define how agents interact. This procedure maps agents to specific interaction "venues" based on their demographic attributes (e.g., assigning children to schools and adults to specific industry sectors).

**Execution:**
To execute the entire data preparation pipeline (fetching data and generating networks), run:
```bash
bash scripts/data_prep.sh
```
While `data_prep.sh` automates the process, you can manually run scripts in the `scripts/` and `networks/` folders for more control. Both directories contain their own respective `README.md` files with detailed instructions.

**Important**: Ensure that you run the network generation scripts (located in the `networks/` folder) to create the interaction networks required for the simulation, and make sure to run the script in the `scripts/` folder first.

*Note: Before running, ensure your `.env` file is set up and that the target counties/states are correctly configured in `scripts/delphi_api.py` and `scripts/census.py`.*

### Step 2: Main Experiment
The core simulation and calibration logic is managed by `abm_nets.py`, which is executed through `main.py`. The experiment is divided into two primary phases: Calibration and Counterfactual Generation.

#### A. Calibration (Training)
In this phase, the model's learnable parameters (e.g., transmission rates, susceptibility) are calibrated to match historical COVID-19 data (cases and deaths).
*   **Goal**: Find the parameter set that minimizes the loss between simulated and actual data.
*   **Configuration**: Ensure `GENERATING_COUNTERFACTUAL` is set to `false` in `covid_abm/yamls/config.yaml`.
*   **Epochs**: The number of training iterations can be configured in `abm_nets.py` by modifying the `epochs` variable (default is typically `251`).

#### B. Counterfactual Generation
Once the model is calibrated, you can run counterfactual scenarios to evaluate "what if" different policies (like school closures or work restrictions) had been implemented.
*   **Goal**: Evaluate the impact of hypothetical interventions using calibrated parameters.
*   **Configuration**: Set `GENERATING_COUNTERFACTUAL` to `true` in `covid_abm/yamls/config.yaml`.
*   **Selecting Scenarios & Iterations**: You can select which counterfactual types to run and how many stochastic iterations to perform for each by modifying the following block in `abm_nets.py`:

```python
if generating_counterfactual:
    cf_types_to_run = range(7, 11)  # Select the range of Counterfactual Types

    for cf_type in cf_types_to_run:
        # ...
        num_iterations = 30  # Number of stochastic runs per scenario
        for num in range(num_iterations):
            # ...
```

#### Configuration Requirements
Before running either phase, ensure the following configurations are correctly set:

**1. `abm_nets.py` Modifications:**
*   **Population Import**: Update the population import statement (e.g., `from populations import pop01045`).
*   **Population Loader**: Update the `Executor` initialization in `eval_net` to use the correct population module (e.g., `sim = Executor(covid_abm, pop_loader=LoadPopulation(pop01045))`).
*   **Scenario Selection**: Update `cf_types_to_run` to specify which intervention types to execute.

#### Counterfactual Types
The simulation supports 10 distinct intervention scenarios, defined by combinations of School and Occupational (Occ) restrictions:

| Type | School | Occ | Description |
| :--- | :--- | :--- | :--- |
| **1** | 0 | 0 | No restrictions (Both fixed to 0) |
| **2** | 1 | 0 | School closed (1), Occ open (0) |
| **3** | 0 | 1 | School open (0), Occ closed (1) |
| **4** | 1 | 1 | Full restrictions (Both fixed to 1) |
| **5** | 0 | Factual | School open (0), Occ remains as it was historically |
| **6** | 0 | Counterfactual | School open (0), Occ is toggled (0<->1) |
| **7** | Factual | Factual | **Baseline Factual scenario** (Matches history) |
| **8** | Counterfactual | Factual | School is toggled, Occ remains factual |
| **9** | Factual | Counterfactual | School remains factual, Occ is toggled |
| **10** | Counterfactual | Counterfactual | **Both School and Occ are toggled** |

To run the experiment:
```bash
python main.py
```

## Credits and Attributions
This project includes code from the following sources. We are grateful for their contributions that made this work possible.

**agenttorch**
*   **Description**: Our `agent_torch` and `covid_abm` folders are based on the [agenttorch](https://github.com/agenttorch/agenttorch) repository.

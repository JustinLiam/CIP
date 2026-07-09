import os
import importlib
import numpy as np
from constants import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import pandas as pd
import yaml

import covid_abm
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation


class LearnableParams(nn.Module):
    def __init__(self, num_params, context_dim=3, device=DEVICE):
        super().__init__()
        self.device = device
        self.num_params = num_params

        self.fc1 = nn.Linear(context_dim, 64).to(self.device)
        self.fc2 = nn.Linear(64, 32).to(self.device)
        self.fc3 = nn.Linear(32, self.num_params + 2).to(self.device)
        self.ReLU = nn.ReLU()
        self.learnable_params = nn.Parameter(torch.rand(num_params + 2, device=self.device))

        min_values_list = [0] * num_params + [0.0] + [0.2]
        max_values_list = [5] * num_params + [0.0005] + [1]

        self.min_values = torch.tensor(min_values_list, device=self.device)
        self.max_values = torch.tensor(max_values_list, device=self.device)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(self.device)
        out = self.ReLU(self.fc1(x))
        out = self.ReLU(self.fc2(out))
        out = self.fc3(out)
        out = self.min_values + (self.max_values -
                                 self.min_values) * self.sigmoid(out)
        return out

def map_and_replace_tensor(input_string):
    parts = input_string.split('.')
    function = parts[1]
    index = parts[2]
    sub_func = parts[3]
    var_name = parts[5]

    def getter_and_setter(runner, counterfactual=False, new_value=None, mode_calibrate=True):
        substep_type = getattr(runner.initializer, function)
        substep_function = getattr(substep_type[str(index)], sub_func)

        if mode_calibrate:
            current_tensor = getattr(substep_function, 'calibrate_' + var_name)
        else:
            current_tensor = getattr(getattr(substep_function, 'learnable_args'), var_name)

        if new_value is not None:
            if not counterfactual:
                assert new_value.requires_grad == current_tensor.requires_grad
            if mode_calibrate:
                setvar_name = 'calibrate_' + var_name
                setattr(substep_function, setvar_name, new_value)
                current_tensor = getattr(substep_function, setvar_name)
            else:
                setvar_name = var_name
                subfunc_param = getattr(substep_function, 'learnable_args')
                setattr(subfunc_param, setvar_name, new_value)
                current_tensor = getattr(subfunc_param, setvar_name)

            return current_tensor
        else:
            return current_tensor

    return getter_and_setter

def execute(sim, runner, Y_actual, epoch, epochs, n_steps=28, loss_cutoff_step=None):
    runner.step(n_steps)

    labels = runner.state_trajectory[-1][-1]['environment']['daily_infected']

    weeks_to_eval = sim.config['simulation_metadata']['NUM_WEEKS_TO_EVAL']

    start = n_steps - weeks_to_eval * 7
    labels = labels[start:]
    Y_sched = labels.to(Y_actual.device)

    date = sim.config['simulation_metadata']['DATE']
    initial_rate = sim.config['simulation_metadata']['INITIAL_INFECTION_RATE']
    exposed_to_infected = sim.config['simulation_metadata']['EXPOSED_TO_INFECTED_TIME']
    infected_to_recovered = sim.config['simulation_metadata']['INFECTED_TO_RECOVERED_TIME']
    population = sim.config['simulation_metadata']['POPULATION']
    generating_counterfactual = sim.config['simulation_metadata']['GENERATING_COUNTERFACTUAL']
    cf_type = sim.config['simulation_metadata']['COUNTERFACTUAL_TYPE']
    with_k = sim.config['simulation_metadata']['WITH_K']
    with_vacc = sim.config['simulation_metadata']['WITH_VACC']

    if generating_counterfactual:
        labels_np = labels.cpu().detach().numpy()
        output_dir = f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}'
        os.makedirs(output_dir, exist_ok=True)

        # Plot comparison if possible
        try:
            county_data = pd.read_csv(f"data/processed_data/{population}/{date}/daily_data.csv", parse_dates=['date'])
            county_cases = torch.tensor(county_data['cases'].values[:n_steps], dtype=torch.float)
            county_cases_np = county_cases.cpu().detach().numpy()

            plt.figure(figsize=(8, 6))
            plt.plot(labels_np, marker='o', label='Simulation Cases')
            plt.plot(county_cases_np, marker='x', label='Actual Cases')
            plt.xlabel('Days')
            plt.ylabel('Number of Cases')
            plt.title(f'Factual vs Counterfactual Data (Type {cf_type}) for {population}')
            plt.legend()
            plt.savefig(f'{output_dir}/counterfactual_results{cf_type}.png')
            plt.close()

            counterfactual_df = pd.DataFrame({
                'day': range(len(labels_np)),
                'counterfactual_cases': labels_np,
                'actual_cases': county_cases_np[:len(labels_np)]
            })
        except:
            print("Warning: Could not load factual data for comparison plotting.")
            counterfactual_df = pd.DataFrame({
                'day': range(len(labels_np)),
                'counterfactual_cases': labels_np
            })

        counterfactual_df.to_csv(f'{output_dir}/counterfactual_data{cf_type}.csv', index=False)

        # Append mode for raw infected labels
        file_path_raw = os.path.join(output_dir, f'cf_raw_{cf_type}.csv')
        with open(file_path_raw, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(labels_np.tolist())

        return 0.0

    if epoch % 50 == 0:
        county_data = pd.read_csv(f"data/processed_data/{population}/{date}/daily_data.csv", parse_dates=['date'])
        county_cases = torch.tensor(county_data['cases'].values[:n_steps], dtype=torch.float)

        labels_np = labels.cpu().detach().numpy()
        county_cases_np = county_cases.cpu().detach().numpy()

        plt.figure(figsize=(8, 6))
        plt.plot(labels_np, marker='o', label='Simulation Cases')
        plt.plot(county_cases_np, marker='x', label='Actual Cases')
        plt.xlabel('Days')
        plt.ylabel('Number of Cases')
        plt.title(f'Simulation vs Actual Cases in {population}')
        plt.legend()

        os.makedirs(f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}/{epoch}', exist_ok=True)
        plt.savefig(f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}/{epoch}/simulation_results.png')
        plt.close()

        trans_substep = runner.initializer.transition_function['0'].new_transmission
        history_data = trans_substep.proportion_history
        if history_data:
            df = pd.DataFrame(history_data, columns=["t", "susceptible", "exposed", "infected", "recovered", "dead"])
            stages = ["susceptible", "exposed", "infected", "recovered", "dead"]
            for stage in stages:
                plt.figure()
                plt.plot(df["t"], df[stage], marker='o', label=stage)
                plt.xlabel("Time (t)")
                plt.ylabel(f"{stage.capitalize()} Proportion")
                plt.title(f"{stage.capitalize()} Proportion Over Time")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}/{epoch}/{stage}_proportion.png")
                plt.close()

    if epoch == epochs - 1:
        output_dir = f'results/{population}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}'
        os.makedirs(output_dir, exist_ok=True)
        generated_df = pd.DataFrame({
            "day": list(range(len(Y_sched))),
            "generated_factual_cases": Y_sched.cpu().detach().numpy()
        })
        generated_df.to_csv(f"{output_dir}/generated_factual.csv", index=False)

    if loss_cutoff_step is not None:
        Y_sched_eval = Y_sched[:loss_cutoff_step]
        Y_actual_eval = Y_actual[:loss_cutoff_step]
    else:
        Y_sched_eval = Y_sched
        Y_actual_eval = Y_actual

    under_loss_squared = torch.clamp((Y_actual_eval - Y_sched_eval)**2, min=0)
    total_loss = (under_loss_squared).mean() * len(under_loss_squared)

    return total_loss


def _load_population_from_config():
    config_path = os.path.join(covid_abm.__path__[0], "yamls", "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    population = str(cfg["simulation_metadata"]["POPULATION"]).strip()
    if population.endswith(".0"):
        population = population[:-2]
    population = population.zfill(5)
    return importlib.import_module(f"populations.pop{population}")


def eval_net():
    population_module = _load_population_from_config()
    sim = Executor(covid_abm, pop_loader=LoadPopulation(population_module))
    test = sim._get_runner
    runner = test(sim.config)
    runner.init()
    learnable_params = [(name, param) for (name, param) in runner.named_parameters()]

    date = sim.config['simulation_metadata']['DATE']
    num_steps = sim.config['simulation_metadata']['num_steps_per_episode']
    num_weeks = sim.config['simulation_metadata']['NUM_WEEKS']
    population = sim.config['simulation_metadata']['POPULATION']
    generating_counterfactual = sim.config['simulation_metadata']['GENERATING_COUNTERFACTUAL']
    initial_rate = sim.config['simulation_metadata']['INITIAL_INFECTION_RATE']
    exposed_to_infected = sim.config['simulation_metadata']['EXPOSED_TO_INFECTED_TIME']
    infected_to_recovered = sim.config['simulation_metadata']['INFECTED_TO_RECOVERED_TIME']
    with_k = sim.config['simulation_metadata']['WITH_K']
    with_vacc = sim.config['simulation_metadata']['WITH_VACC']

    df = pd.read_csv(f"data/processed_data/{population}/{date}/daily_data.csv", parse_dates=["date"])
    case_numbers = torch.tensor(df['cases'].values, dtype=torch.float, device=DEVICE)

    context_tensor = torch.tensor([initial_rate, exposed_to_infected, infected_to_recovered], dtype=torch.float, device=DEVICE)
    learn_model = LearnableParams(num_weeks, context_dim=3, device=DEVICE)
    learn_model = torch.compile(learn_model)

    def deep_clone_state(state):
        new_state = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                new_state[k] = v.clone()
            elif isinstance(v, dict):
                new_state[k] = deep_clone_state(v)
            else:
                new_state[k] = v
        return new_state

    initial_state = deep_clone_state(runner.state)

    if generating_counterfactual:
        cf_types_to_run = range(7, 11)

        for cf_type in cf_types_to_run:
            sim.config['simulation_metadata']['COUNTERFACTUAL_TYPE'] = cf_type
            all_age_proportions = []

            num_iterations = 30

            for num in range(num_iterations):
                runner.state = deep_clone_state(initial_state)
                runner.state_trajectory = []

                base_dir = f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}'
                param_array = np.loadtxt(os.path.join(base_dir, "calibrated_params.txt"))
                param_tensor = torch.tensor(param_array, dtype=torch.float, device=DEVICE)
                param_tensor = param_tensor[:, None] if param_tensor.ndim == 1 else param_tensor

                tensorfunc_R = map_and_replace_tensor(learnable_params[1][0])
                tensorfunc_R(runner, True, param_tensor[:num_weeks], mode_calibrate=True)

                tensorfunc_inf = map_and_replace_tensor(learnable_params[3][0])
                tensorfunc_inf(runner, True, param_tensor[-2], mode_calibrate=True)

                tensorfunc_k = map_and_replace_tensor(learnable_params[4][0])
                tensorfunc_k(runner, True, param_tensor[-1], mode_calibrate=True)

                _ = execute(sim, runner, case_numbers, epoch=0, epochs=1, n_steps=num_steps)

                substep = runner.initializer.transition_function['0'].new_transmission
                output_dir = f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}'
                path = f'{output_dir}/counterfactual_data{cf_type}_proportions.csv'

                if num_iterations > 1:
                    substep.save_proportions_to_disk2(path, num)
                else:
                    substep.save_proportions_to_disk(path)

                all_age_proportions.append(substep.age_proportion_history)

            if num_iterations > 1 and all_age_proportions:
                all_runs_np = np.array(all_age_proportions)
                avg_age_proportions = np.mean(all_runs_np, axis=0)
                age_path = f'{output_dir}/counterfactual_data{cf_type}_age_proportions.csv'
                cols = [
                    "t",
                    "child_susceptible", "child_exposed", "child_infected", "child_recovered", "child_dead",
                    "adult_susceptible", "adult_exposed", "adult_infected", "adult_recovered", "adult_dead",
                    "elderly_susceptible", "elderly_exposed", "elderly_infected", "elderly_recovered", "elderly_dead"
                ]
                avg_df = pd.DataFrame(avg_age_proportions, columns=cols)
                avg_df["t"] = avg_df["t"].astype(int)
                avg_df.to_csv(age_path, index=False)

        return

    opt = optim.Adam(learn_model.parameters(), lr=0.0005)
    loss_array = np.array([])
    epochs = 251

    for epoch in range(epochs):
        torch.autograd.set_detect_anomaly(True)
        opt.zero_grad()

        runner.state = deep_clone_state(initial_state)
        runner.state_trajectory = []

        debug_tensor = learn_model(context_tensor)
        debug_tensor = debug_tensor[:, None]

        map_and_replace_tensor(learnable_params[1][0])(runner, False, debug_tensor[:num_weeks], mode_calibrate=True)
        map_and_replace_tensor(learnable_params[3][0])(runner, False, debug_tensor[-2], mode_calibrate=True)
        map_and_replace_tensor(learnable_params[4][0])(runner, False, debug_tensor[-1], mode_calibrate=True)

        loss = execute(sim, runner, case_numbers, epoch, epochs, num_steps)
        loss.backward()
        opt.step()

        loss_np = loss.detach().cpu().numpy()
        loss_array = np.append(loss_array, loss_np)

        if epoch == epochs - 1:
            substep = runner.initializer.transition_function['0'].new_transmission
            output_dir = f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}'
            path = f'{output_dir}/training_proportions.csv'
            substep.save_proportions_to_disk(path)
            np.savetxt(os.path.join(output_dir, "calibrated_params.txt"), debug_tensor.detach().cpu().numpy(), fmt="%.6f")

    iters = np.arange(1, epochs + 1)
    data = np.column_stack((iters, loss_array))

    output_dir = f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}'
    with open(os.path.join(output_dir, "training_loss.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Task Loss'])
        writer.writerows(data)

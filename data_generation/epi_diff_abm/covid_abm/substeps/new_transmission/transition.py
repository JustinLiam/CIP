import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import re
import math
import pandas as pd
import numpy as np
import os
import json
import time

from agent_torch.core.substep import SubstepTransitionMessagePassing
from agent_torch.core.helpers import get_by_path
from agent_torch.core.helpers import set_by_path
from agent_torch.core.distributions import StraightThroughBernoulli

class NewTransmission(SubstepTransitionMessagePassing):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.SUSCEPTIBLE_VAR = self.config["simulation_metadata"]["SUSCEPTIBLE_VAR"]
        self.EXPOSED_VAR = self.config["simulation_metadata"]["EXPOSED_VAR"]
        self.RECOVERED_VAR = self.config["simulation_metadata"]["RECOVERED_VAR"]
        self.INFECTED_VAR = self.config["simulation_metadata"]["INFECTED_VAR"]
        self.MORTALITY_VAR = self.config["simulation_metadata"]["MORTALITY_VAR"]

        self.num_timesteps = self.config["simulation_metadata"]["num_steps_per_episode"]
        self.num_weeks = self.config["simulation_metadata"]["NUM_WEEKS"]

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.data_dir = os.path.join(project_root, 'data')

        archive_payload = self._load_network_archive()
        if archive_payload is None:
            self.networks = self._preload_all_networks()
            self.household_net = self._load_single_net(
                f"{self.data_dir}/networks/covid_output_causal/"
                f"{self.config['simulation_metadata']['POPULATION']}/"
                "mobility_networks/HOUSEHOLD_NETWORK.pkl"
            )
        else:
            self.networks, self.household_net = archive_payload
        self.proportion_history = []
        self.age_proportion_history = []

        self.STAGE_UPDATE_VAR = 1
        self.INFINITY_TIME = self.config["simulation_metadata"]["INFINITY_TIME"]
        self.EXPOSED_TO_INFECTED_TIME = self.config["simulation_metadata"][
            "EXPOSED_TO_INFECTED_TIME"
        ]
        self.INFECTED_TO_RECOVERED_TIME = self.config["simulation_metadata"][
            "INFECTED_TO_RECOVERED_TIME"
        ]

        self.mode = self.config["simulation_metadata"]["EXECUTION_MODE"]
        self.st_bernoulli = StraightThroughBernoulli.apply

        self.calibration_mode = self.config['simulation_metadata']['calibration']

    def _lam(
        self,
        x_i,
        x_j,
        edge_attr,
        t,
        R,
        SFSusceptibility,
        SFInfector,
        lam_gamma_integrals,
    ):
        S_A_s = SFSusceptibility[x_i[:, 0].long()]
        A_s_i = SFInfector[x_j[:, 1].long()]
        B_n = edge_attr[1, :]
        integrals = torch.zeros_like(B_n)
        infected_idx = x_j[:, 2].bool()
        infected_times = t - x_j[infected_idx, 3] - 1
        infected_times = infected_times.clamp(min=0, max=lam_gamma_integrals.size(0) - 1)

        integrals[infected_idx] = lam_gamma_integrals[infected_times.long()]
        edge_network_numbers = edge_attr[0, :]

        I_bar = torch.gather(x_i[:, 4], 0, edge_network_numbers.long()).view(-1)

        will_isolate = x_i[:, 6]
        not_isolated = 1 - will_isolate

        if self.mode == "llm":
            res = (
                R * S_A_s * A_s_i * B_n * integrals / I_bar
            )
        else:
            res = R * S_A_s * A_s_i * B_n * integrals / I_bar

        return res.view(-1, 1)

    def message(
        self,
        x_i,
        x_j,
        edge_attr,
        t,
        R,
        SFSusceptibility,
        SFInfector,
        lam_gamma_integrals,
    ):
        return self._lam(
            x_i, x_j, edge_attr, t, R, SFSusceptibility, SFInfector, lam_gamma_integrals
        )

    def update_stages(self, t, current_stages, agents_next_stage_times, newly_exposed_today):
        transition_to_infected = self.INFECTED_VAR*(agents_next_stage_times <= t) + self.EXPOSED_VAR*(agents_next_stage_times > t)
        transition_to_mortality_or_recovered = self.RECOVERED_VAR*(agents_next_stage_times <= t) + self.INFECTED_VAR*(agents_next_stage_times > t)

        stage_progression = (current_stages == self.SUSCEPTIBLE_VAR)*self.SUSCEPTIBLE_VAR \
            + (current_stages == self.RECOVERED_VAR)*self.RECOVERED_VAR + (current_stages == self.MORTALITY_VAR)*self.MORTALITY_VAR \
            + (current_stages == self.EXPOSED_VAR)*transition_to_infected \
            + (current_stages == self.INFECTED_VAR)*transition_to_mortality_or_recovered

        current_stages = newly_exposed_today*self.EXPOSED_VAR + stage_progression
        return current_stages

    def update_transition_times(self, t, agents_next_stage_times, newly_exposed_today, current_stages):
        """Note: not differentiable"""
        exposed_to_infected_time = self.EXPOSED_TO_INFECTED_TIME
        new_transition_times = torch.clone(agents_next_stage_times)
        curr_stages = torch.clone(current_stages).long()
        new_transition_times[(curr_stages==self.INFECTED_VAR)*(agents_next_stage_times == t)] = self.INFINITY_TIME
        new_transition_times[(curr_stages==self.EXPOSED_VAR)*(agents_next_stage_times == t)] = t+self.INFECTED_TO_RECOVERED_TIME
        return newly_exposed_today*(t+1+exposed_to_infected_time) + (1 - newly_exposed_today)*new_transition_times

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        timestep_tensor = torch.tensor([timestep])
        one_hot_tensor = F.one_hot(timestep_tensor, num_classes=num_timesteps)
        one_hot_tensor = one_hot_tensor.view(1, -1)
        return one_hot_tensor.to(self.device)[0]

    def update_infected_times(self, t, agents_infected_times, newly_exposed_today):
        """Note: not differentiable"""
        updated_infected_times = torch.clone(agents_infected_times).to(
            agents_infected_times.device
        )

        updated_infected_times[newly_exposed_today.bool()] = t

        return updated_infected_times

    def get_stage_proportions(self, t, current_stages):
        total_people = current_stages.shape[0]
        counts = torch.stack([
            (current_stages == self.SUSCEPTIBLE_VAR).sum(),
            (current_stages == self.EXPOSED_VAR).sum(),
            (current_stages == 2).sum(),
            (current_stages == self.RECOVERED_VAR).sum(),
            (current_stages == 4).sum(),
        ])
        proportions = counts.float() / total_people

        self.proportion_history.append([t] + proportions.detach().cpu().tolist())

    def save_proportions_to_disk(self, output_path):
        if self.proportion_history:
            df = pd.DataFrame(self.proportion_history,
                            columns=["t", "susceptible", "exposed", "infected", "recovered", "dead"])
            df.to_csv(output_path, index=False)

    def save_proportions_to_disk2(self, output_path, epoch_num):
        if self.proportion_history:
            cols = [
                "t",
                f"susceptible_{epoch_num}",
                f"exposed_{epoch_num}",
                f"infected_{epoch_num}",
                f"recovered_{epoch_num}",
                f"dead_{epoch_num}"
            ]

            new_df = pd.DataFrame(self.proportion_history, columns=cols)

            if os.path.exists(output_path):
                existing_df = pd.read_csv(output_path)

                if "t" in existing_df.columns:
                    final_df = pd.merge(existing_df, new_df, on="t", how="outer")
                else:
                    final_df = pd.concat([existing_df, new_df.drop(columns=["t"])], axis=1)
            else:
                final_df = new_df

            final_df.to_csv(output_path, index=False)

    def modify_initial_infected(self, current_stages, proportion, tau=0.5, hard=True):
        device = current_stages.device
        N = int(self.config['simulation_metadata']['num_agents'])

        new_stages = current_stages.clone()
        new_stages[:, 0] = 0.0

        logits = torch.zeros(N, 2, device=device)
        logits[:, 1] = torch.log(torch.tensor(proportion / (1 - proportion), device=device))

        samples = F.gumbel_softmax(logits, tau=tau, hard=hard)
        infected_mask = samples[:, 1]

        new_stages[:, 0] = 2.0 * infected_mask

        return new_stages

    def get_mean_agent_interactions(self, agents_ages):
        ADULT_LOWER_INDEX, ADULT_UPPER_INDEX = (
            1,
            4,
        )

        agents_mean_interactions = 0 * torch.ones(size=agents_ages.shape)
        mean_int_ran_mu = torch.tensor([2, 3, 4]).float()

        child_agents = (agents_ages < ADULT_LOWER_INDEX).view(-1)
        adult_agents = torch.logical_and(
            agents_ages >= ADULT_LOWER_INDEX, agents_ages <= ADULT_UPPER_INDEX
        ).view(-1)
        elderly_agents = (agents_ages > ADULT_UPPER_INDEX).view(-1)

        agents_mean_interactions[child_agents.bool(), 0] = mean_int_ran_mu[0]
        agents_mean_interactions[adult_agents.bool(), 0] = mean_int_ran_mu[1]
        agents_mean_interactions[elderly_agents.bool(), 0] = mean_int_ran_mu[2]

        return agents_mean_interactions

    def update_initial_times(self, agents_next_stage_times, agents_infected_time, agents_stages):
        infected_to_recovered_time = self.INFECTED_TO_RECOVERED_TIME
        exposed_to_infected_time = self.EXPOSED_TO_INFECTED_TIME

        agents_infected_time[agents_stages==self.EXPOSED_VAR] = -1
        agents_infected_time[agents_stages==self.INFECTED_VAR] = -1*self.EXPOSED_TO_INFECTED_TIME
        agents_next_stage_times[agents_stages==self.EXPOSED_VAR] = exposed_to_infected_time
        agents_next_stage_times[agents_stages==self.INFECTED_VAR] = infected_to_recovered_time

        return agents_infected_time, agents_next_stage_times

    def soft_eq(self, x, target, temperature=0.1):
        diff = torch.abs(x - target)
        return torch.exp(-diff / temperature)

    def combined(self, x, target, t, agents_next_stage_times):
        soft = self.soft_eq(x, target)
        hard = (x == target).float() * (agents_next_stage_times <= t).float()

        return hard.detach() + soft - soft.detach()

    def update_number_of_dead(self, daily_dead, current_stages, agents_next_stage_times, t, newly_exposed_today):
        if self.calibration_mode:
            mortality_rate = self.calibrate_M.to(self.device)
        else:
            mortality_rate = self.learnable_args["M"]

        mask = self.combined(current_stages, self.INFECTED_VAR, t, agents_next_stage_times)

        new_death_recovered_today = (current_stages * mask) / self.INFECTED_VAR


        NEW_DEATHS_TODAY = mortality_rate * new_death_recovered_today.sum()

        daily_dead = (
            daily_dead
            + self._generate_one_hot_tensor(t, self.num_timesteps) * NEW_DEATHS_TODAY.squeeze()
        )

        return daily_dead

    def update_adjacency_matrix(self, state, combined_net):
        source_nodes = combined_net[:, 0]
        target_nodes = combined_net[:, 1]

        edge_list = torch.stack((source_nodes, target_nodes), dim=0).to(self.device)

        edge_attr = torch.ones(2, edge_list.size(1)).to(self.device)

        adjacency_matrix_path = ["network", "agent_agent", "infection_network", "adjacency_matrix"]
        adjacency_matrix = (edge_list, edge_attr)

        return set_by_path(state, adjacency_matrix_path, adjacency_matrix)

    def _load_single_net(self, path):
        df = pd.read_pickle(path)
        if hasattr(df, 'edges'):
            df = pd.DataFrame(df.edges(), columns=["node1", "node2"])
        return torch.tensor(df.values, device=self.device, dtype=torch.long)

    def _load_network_archive(self):
        """Load contiguous, pre-converted county networks when available."""
        population = str(self.config['simulation_metadata']['POPULATION'])
        mobility_dir = os.path.join(
            self.data_dir,
            "networks",
            "covid_output_causal",
            population,
            "mobility_networks",
        )
        archive_path = os.environ.get(
            "EPI_ABM_NETWORK_ARCHIVE",
            os.path.join(mobility_dir, "tensor_archive_v1.pt"),
        )
        if not os.path.exists(archive_path):
            if os.environ.get("EPI_ABM_REQUIRE_NETWORK_ARCHIVE", "0") == "1":
                raise FileNotFoundError(
                    f"EPI_ABM_REQUIRE_NETWORK_ARCHIVE=1 but archive is missing: "
                    f"{archive_path}"
                )
            return None

        started = time.time()
        payload = torch.load(archive_path, map_location="cpu")
        if int(payload.get("format_version", -1)) != 1:
            raise ValueError(
                f"Unsupported EpiABM network archive format in {archive_path}: "
                f"{payload.get('format_version')!r}"
            )
        if str(payload.get("county", "")).zfill(5) != population.zfill(5):
            raise ValueError(
                f"Network archive county mismatch: archive={payload.get('county')!r}, "
                f"config={population!r}"
            )
        if int(payload.get("num_steps", -1)) != int(self.num_timesteps):
            raise ValueError(
                f"Network archive step mismatch: archive={payload.get('num_steps')!r}, "
                f"config={self.num_timesteps!r}"
            )

        networks = {}
        for kind in ("school", "occ", "rand"):
            edges = payload[f"{kind}_edges"]
            offsets = payload[f"{kind}_offsets"]
            if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[1] != 2:
                raise ValueError(
                    f"Invalid {kind} edge tensor in {archive_path}: "
                    f"shape={tuple(edges.shape)}, dtype={edges.dtype}"
                )
            offsets_list = [int(value) for value in offsets.tolist()]
            if len(offsets_list) != self.num_timesteps + 1:
                raise ValueError(
                    f"Invalid {kind} offsets in {archive_path}: "
                    f"expected {self.num_timesteps + 1}, got {len(offsets_list)}"
                )
            edges = edges.to(self.device)
            networks[kind] = [
                edges[offsets_list[idx]:offsets_list[idx + 1]]
                for idx in range(self.num_timesteps)
            ]

        household = payload["household_edges"]
        if household.dtype != torch.long or household.ndim != 2 or household.shape[1] != 2:
            raise ValueError(
                f"Invalid household edge tensor in {archive_path}: "
                f"shape={tuple(household.shape)}, dtype={household.dtype}"
            )
        household = household.to(self.device)
        print(json.dumps({
            "event": "epi_abm_network_archive_loaded",
            "county": population.zfill(5),
            "archive": archive_path,
            "device": str(self.device),
            "elapsed_sec": round(time.time() - started, 3),
            "edge_counts": {
                kind: int(payload[f"{kind}_edges"].shape[0])
                for kind in ("school", "occ", "rand")
            },
            "household_edges": int(payload["household_edges"].shape[0]),
        }), flush=True)
        return networks, household

    def _preload_all_networks(self):
        """Load all time-step networks into a dictionary of tensors on the GPU."""
        nets = {'school': [], 'occ': [], 'rand': []}
        population = self.config['simulation_metadata']['POPULATION']
        for t in range(self.num_timesteps):
            nets['school'].append(self._load_single_net(f"{self.data_dir}/networks/covid_output_causal/{population}/mobility_networks/schoolnets/{t}.pkl"))
            nets['occ'].append(self._load_single_net(f"{self.data_dir}/networks/covid_output_causal/{population}/mobility_networks/occnets/{t}.pkl"))
            nets['rand'].append(self._load_single_net(f"{self.data_dir}/networks/covid_output_causal/{population}/mobility_networks/randnets/{t}.pkl"))
        return nets

    def apply_intervention_fast(self, intervention, edges):
        if edges.size(0) == 0: return edges

        if isinstance(intervention, dict) and intervention.get("mode") == "continuous_freeze":
            value = intervention.get("value", 0.0)
            if isinstance(value, torch.Tensor):
                action_value = value.detach().flatten()[0].to(self.device)
            else:
                action_value = torch.tensor(float(value), dtype=torch.float, device=self.device)
            action_value = torch.clamp(action_value, 0.0, 1.0)
            low = 0.75 * action_value
            high = 0.25 + 0.75 * action_value
            sample = low + torch.rand((1,), device=self.device) * (high - low)
        elif intervention == 0:
            sample = torch.clamp(torch.normal(0.5, 0.15, (1,)), 0, 0.25).to(self.device)
        else:
            sample = torch.clamp(torch.normal(0.5, 0.15, (1,)), 0.75, 1).to(self.device)

        keep_frac = (1.0 - sample).item()
        k = int(edges.size(0) * keep_frac)

        idx = torch.randperm(edges.size(0), device=self.device)[:k]
        return edges[idx]

    def get_age_stage_proportions(self, t, current_stages, agents_ages):
            ADULT_LOWER_INDEX, ADULT_UPPER_INDEX = 1, 4

            child_agents = (agents_ages < ADULT_LOWER_INDEX).view(-1)
            adult_agents = torch.logical_and(
                agents_ages >= ADULT_LOWER_INDEX, agents_ages <= ADULT_UPPER_INDEX
            ).view(-1)
            elderly_agents = (agents_ages > ADULT_UPPER_INDEX).view(-1)

            masks = [child_agents, adult_agents, elderly_agents]
            row = [t]

            for mask in masks:
                total_people_in_group = mask.sum()
                if total_people_in_group > 0:
                    counts = torch.stack([
                        (current_stages[mask] == self.SUSCEPTIBLE_VAR).sum(),
                        (current_stages[mask] == self.EXPOSED_VAR).sum(),
                        (current_stages[mask] == 2).sum(),
                        (current_stages[mask] == self.RECOVERED_VAR).sum(),
                        (current_stages[mask] == 4).sum(),
                    ])
                    proportions = counts.float() / total_people_in_group
                    row.extend(proportions.detach().cpu().tolist())
                else:
                    row.extend([0.0] * 5)

            self.age_proportion_history.append(row)

    def _coerce_online_intervention(self, online_intervention):
        """Return school/occupation intervention if an online action is set."""
        if online_intervention is None:
            return None

        mode = "binary_threshold"
        if isinstance(online_intervention, dict):
            mode = online_intervention.get("mode", mode)
            values = online_intervention.get("values")
            if values is None:
                school = online_intervention.get("school_intervention", online_intervention.get("school"))
                occ = online_intervention.get("occ_intervention", online_intervention.get("workplace"))
                if school is None or occ is None:
                    return None
                values = [school, occ]
        else:
            values = online_intervention

        if isinstance(values, torch.Tensor):
            values = values.detach().flatten().to(self.device)
            if values.numel() < 2:
                return None
            school, occ = values[0], values[1]
            if mode in {"continuous_freeze", "continuous", "freezing_interval"}:
                return (
                    {"mode": "continuous_freeze", "value": school},
                    {"mode": "continuous_freeze", "value": occ},
                )
            return int((school >= 0.5).item()), int((occ >= 0.5).item())

        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size < 2:
            return None
        if mode in {"continuous_freeze", "continuous", "freezing_interval"}:
            return (
                {"mode": "continuous_freeze", "value": float(arr[0])},
                {"mode": "continuous_freeze", "value": float(arr[1])},
            )
        return int(arr[0] >= 0.5), int(arr[1] >= 0.5)

    def forward(self, state, action=None):
        input_variables = self.input_variables
        t = int(state["current_step"])

        generating_counterfactual = self.config['simulation_metadata']['GENERATING_COUNTERFACTUAL']
        cf_type = self.config['simulation_metadata']['COUNTERFACTUAL_TYPE']
        with_k = self.config['simulation_metadata']['WITH_K']

        intervention_df = pd.read_csv(f"{self.config['simulation_metadata']['population_dir']}/intervention.csv")
        curr_intervention = intervention_df[intervention_df['t'] == t]

        school_intervention = curr_intervention.iloc[0]['school_intervention']
        occ_intervention = curr_intervention.iloc[0]['occ_intervention']

        online_values = self._coerce_online_intervention(
            state.get("environment", {}).get("online_intervention", None)
        )
        if online_values is not None:
            school_intervention, occ_intervention = online_values
        elif generating_counterfactual:
            if (t == 0):
                print(f"Counterfactual Type: {cf_type}")

            logic_map = {
                1: (0, 0),
                2: (1, 0),
                3: (0, 1),
                4: (1, 1),
                5: (0, "F"),
                6: (0, "CF"),
                7: ("F", "F"),
                8: ("CF", "F"),
                9: ("F", "CF"),
                10: ("CF", "CF")
            }

            if cf_type in logic_map:
                s_logic, o_logic = logic_map[cf_type]

                if s_logic == "CF":
                    school_intervention = 1 - school_intervention
                elif s_logic != "F":
                    school_intervention = s_logic

                if o_logic == "CF":
                    occ_intervention = 1 - occ_intervention
                elif o_logic != "F":
                    occ_intervention = o_logic

        school_net = self.apply_intervention_fast(school_intervention, self.networks['school'][0])
        occ_net = self.apply_intervention_fast(occ_intervention, self.networks['occ'][t])

        combined_net = torch.cat([
                    school_net,
                    occ_net,
                    self.networks['rand'][t],
                    self.household_net
                ], dim=0)

        state = self.update_adjacency_matrix(state, combined_net)

        time_step_one_hot = self._generate_one_hot_tensor(t, self.num_timesteps)

        week_id = int(t / 7)
        week_one_hot = self._generate_one_hot_tensor(week_id, self.num_weeks)

        if self.calibration_mode:
            R_tensor = self.calibrate_R2.to(self.device)
            initial_infection_rate = self.calibrate_infected_proportion.to(self.device)
            k = self.calibrate_k.to(self.device)
        else:
            R_tensor = self.learnable_args["R2"]
            initial_infection_rate = self.learnable_args["infected_proportion"]
            k = self.learnable_args["k"]

        if (t == 0):
            self.proportion_history = []
            self.age_proportion_history = []

        R = (R_tensor.T * week_one_hot).sum()

        current_stages = state['agents']['citizens']['disease_stage']
        agents_ages = get_by_path(state, re.split("/", input_variables["age"]))
        agents_next_stage_times = state['agents']['citizens']['next_stage_time']
        agents_infected_time = state['agents']['citizens']['infected_time']

        daily_deaths = get_by_path(
            state, re.split("/", input_variables["daily_deaths"])
        )

        if (t == 0):
            current_stages = self.modify_initial_infected(current_stages, initial_infection_rate)
            agents_infected_time, agents_next_stage_times = self.update_initial_times(agents_next_stage_times, agents_infected_time, current_stages)

        SFSusceptibility = get_by_path(
            state, re.split("/", input_variables["SFSusceptibility"])
        )
        SFInfector = get_by_path(state, re.split("/", input_variables["SFInfector"]))
        all_lam_gamma = get_by_path(
            state, re.split("/", input_variables["lam_gamma_integrals"])
        )

        agents_mean_interactions_split = self.get_mean_agent_interactions(agents_ages)

        all_edgelist, all_edgeattr = get_by_path(
            state, re.split("/", input_variables["adjacency_matrix"])
        )

        daily_infected = get_by_path(
            state, re.split("/", input_variables["daily_infected"])
        )

        agents_infected_index = torch.logical_and(
            current_stages > self.SUSCEPTIBLE_VAR, current_stages < self.RECOVERED_VAR
        )

        will_isolate = action["citizens"]["isolation_decision"]

        all_node_attr = (
            torch.stack(
                (
                    agents_ages.to(self.device),
                    current_stages.detach(),
                    agents_infected_index,
                    agents_infected_time,
                    agents_mean_interactions_split.to(self.device),
                    torch.unsqueeze(
                        torch.arange(self.config["simulation_metadata"]["num_agents"]),
                        1,
                    ).to(
                        self.device
                    ),
                    will_isolate,
                )
            )
            .transpose(0, 1)
            .squeeze()
        )

        agents_data = Data(
            all_node_attr, edge_index=all_edgelist, edge_attr=all_edgeattr, t=t
        )

        new_transmission = self.propagate(
            agents_data.edge_index,
            x=agents_data.x,
            edge_attr=agents_data.edge_attr,
            t=agents_data.t,
            R=R,
            SFSusceptibility=SFSusceptibility,
            SFInfector=SFInfector,
            lam_gamma_integrals=all_lam_gamma.squeeze(),
        )

        prob_not_infected = torch.exp(-1 * new_transmission)
        probs = torch.hstack((1 - prob_not_infected, prob_not_infected))

        potentially_exposed_today = self.st_bernoulli(probs)[:, 0].to(
            self.device
        )

        newly_exposed_today = (
            current_stages == self.SUSCEPTIBLE_VAR
        ).squeeze() * potentially_exposed_today

        daily_infected = daily_infected + (newly_exposed_today.sum() + 0) * time_step_one_hot

        if with_k:
            k_mask = torch.ones_like(daily_infected)
            k_mask[t] = k
            daily_infected = daily_infected * k_mask

        daily_infected = daily_infected.squeeze(0)

        newly_exposed_today = newly_exposed_today.unsqueeze(1)

        daily_deaths = self.update_number_of_dead(daily_deaths, current_stages, agents_next_stage_times, t, newly_exposed_today)

        updated_stages = self.update_stages(t, current_stages, agents_next_stage_times, newly_exposed_today)
        updated_next_stage_times = self.update_transition_times(
            t, agents_next_stage_times, newly_exposed_today, current_stages
        )

        updated_infected_times = self.update_infected_times(
            t, agents_infected_time, newly_exposed_today
        )

        self.get_stage_proportions(t, updated_stages)
        self.get_age_stage_proportions(t, updated_stages, agents_ages)

        return {
            self.output_variables[0]: updated_stages,
            self.output_variables[1]: updated_next_stage_times,
            self.output_variables[2]: updated_infected_times,
            self.output_variables[3]: daily_infected,
            self.output_variables[4]: daily_deaths,
        }

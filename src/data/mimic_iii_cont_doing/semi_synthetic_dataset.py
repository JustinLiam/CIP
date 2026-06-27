import pandas as pd
import numpy as np
import logging
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from joblib import Parallel, delayed
import multiprocessing
import torch
from tqdm import tqdm
from typing import List
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from hydra.utils import instantiate
from copy import deepcopy

from src import ROOT_PATH
from src.data.mimic_iii_cont_doing.load_data import load_mimic3_data_raw
from src.data.mimic_iii_cont_doing.utils import sigmoid, SplineTrendsMixture
from src.data.mimic_iii_cont_doing.real_dataset import MIMIC3RealDataset
from src.data.dataset_collection import SyntheticDatasetCollection

import scipy.stats as stats
from scipy.stats import beta as scipy_beta

from scipy.interpolate import CubicSpline
t_control = np.array([0, 0.4, 0.7, 1])
f_control = np.array([0, 0.2, 0.85, 1])
cs = CubicSpline(t_control, f_control, bc_type='clamped')

logger = logging.getLogger(__name__)
max_sequence_length = 100  # Larger than Maximum sequence length 

class SyntheticOutcomeGenerator:
    """
    Generator of synthetic outcome
    """

    def __init__(self,
                 exogeneous_vars: List[str],
                 exog_dependency: callable,
                 exog_weight: float,
                 endo_dependency: callable,
                 endo_rand_weight: float,
                 endo_spline_weight: float,
                 outcome_name: str):
        """
        Args:
            exogeneous_vars: List of time-varying covariates
            exog_dependency: Callable function of exogeneous_vars (f_Z)
            exog_weight: alpha_f
            endo_dependency: Callable function of endogenous variables (g)
            endo_rand_weight: alpha_g
            endo_spline_weight: alpha_S
            outcome_name: Name of the outcome variable j
        """
        self.exogeneous_vars = exogeneous_vars
        self.exog_dependency = exog_dependency
        self.exog_weight = exog_weight
        self.endo_rand_weight = endo_rand_weight
        self.endo_spline_weight = endo_spline_weight
        self.endo_dependency = endo_dependency
        self.outcome_name = outcome_name

    def simulate_untreated(self, all_vitals: pd.DataFrame, static_features: pd.DataFrame, max_number):
        """
        Simulate untreated outcomes (Z)
        Args:
            all_vitals: Time-varying covariates (as exogeneous vars)
            static_features: Static covariates (as exogeneous vars)
        """
        logger.info(f'Simulating untreated outcome {self.outcome_name}')
        user_sizes = all_vitals.groupby(level='subject_id', sort=False).size()

        # Exogeneous dependency
        all_vitals[f'{self.outcome_name}_exog'] = self.exog_weight * self.exog_dependency(all_vitals[self.exogeneous_vars].values)

        # Endogeneous dependency + B-spline trend
        time_range = np.arange(0, max_sequence_length)
        y_endo_rand = self.endo_dependency(time_range, max_number)
        y_endo_splines = SplineTrendsMixture(n_patients=max_number , max_time=max_sequence_length)(time_range)
        y_endo_full = self.endo_rand_weight * y_endo_rand + self.endo_spline_weight * y_endo_splines

        all_vitals[f'{self.outcome_name}_endo'] = \
            np.array([value for (i, l) in enumerate(user_sizes) for value in y_endo_full[i, :l]]).reshape(-1, 1)

        # Untreated outcome
        all_vitals[f'{self.outcome_name}_untreated'] = \
            all_vitals[f'{self.outcome_name}_exog'] + all_vitals[f'{self.outcome_name}_endo']

        # Placeholder for treated outcome
        all_vitals[f'{self.outcome_name}'] = all_vitals[f'{self.outcome_name}_untreated'].copy()

class SyntheticTreatment:
    """
    Generator of synthetic treatment
    """

    def __init__(self,
                 confounding_vars: List[str],
                 confounder_outcomes: List[str],
                 confounding_dependency: callable,
                 window: float,
                 conf_outcome_weight: float,
                 conf_vars_weight: float,
                 bias: float,
                 full_effect: float,
                 effect_window: float,
                 treatment_name: str,
                 base_concentration: float = 2.0,
                 post_nonlinearity: callable = None):
        """
        Args:
            confounding_vars: Confounding time-varying covariates (from all_vitals)
            confounder_outcomes: Confounding previous outcomes
            confounding_dependency: Callable function of confounding_vars (f_Y)
            window: Window of averaging of confounding previous outcomes (T_l)
            conf_outcome_weight: gamma_Y
            conf_vars_weight: gamma_X
            bias: constant bias
            full_effect: beta
            effect_window: w_l
            treatment_name: Name of treatment l
            post_nonlinearity: Post non-linearity after sigmoid
        """
        self.confounding_vars = confounding_vars
        self.confounder_outcomes = confounder_outcomes
        self.confounding_dependency = confounding_dependency
        self.treatment_name = treatment_name
        self.post_nonlinearity = post_nonlinearity

        # Parameters
        self.window = window
        self.conf_outcome_weight = conf_outcome_weight
        self.conf_vars_weight = conf_vars_weight
        self.bias = bias

        self.full_effect = full_effect
        self.effect_window = effect_window

        self.base_concentration = base_concentration

    def treatment_proba(self, patient_df, t, rng):
        """
        Probability of treatment sampled using beta distribution, see chemo_prob form
        
        Args:
            patient_df: DataFrame of patient
            t: Time-step
            rng: Random number generator
        Returns: 
            therapeutic probability of beta distribution sampling (1-dimensional array)
        """
        #1. Calculate the impact of historical results (contribution of y)
        t_start = max(0, t - self.window)
        agr_range = np.arange(t_start, t + 1)
        avg_y = patient_df.loc[agr_range, self.confounder_outcomes].values.mean()
        
        #2. Calculate the current covariate impact (contribution of x)
        x = patient_df.loc[t, self.confounding_vars].values.reshape(1, -1) #
        # x = patient_df.loc[agr_range, self.confounding_vars].values.mean(axis=0).reshape(1, -1)
        f_x = self.confounding_dependency(x)
        
        #3. Calculate the base probability (chemo_prob-like role)
        treatment_prob = sigmoid(
            self.bias + 
            self.conf_outcome_weight * avg_y + 
            self.conf_vars_weight * f_x
        ).flatten()[0]
        
        #4. Calculate alpha and beta parameters with reference to your form
        treatment_alpha = self.base_concentration * treatment_prob
        treatment_beta = self.base_concentration - treatment_alpha
        
        treat_proba = scipy_beta.rvs(treatment_alpha, treatment_beta, 
                                   random_state=rng)
        
        #6. Post-processing (if required)
        if self.post_nonlinearity is not None:
            treat_proba = self.post_nonlinearity(treat_proba)
            
        return np.array([treat_proba])

    def get_treated_outcome(self, patient_df, t, outcome_name, treat_proba=1.0):
        """
        Calculate future outcome under treatment, applied at the time-step t
        Args:
            patient_df: DataFrame of patient
            t: Time-step
            outcome_name: Name of the outcome variable j
            treat_proba: Propensity scores of treatment

        Returns: Effect window, treated outcome
        """
        scaled_effect = self.full_effect * cs(treat_proba)

        t_stop = min(max(patient_df.index.get_level_values('hours_in')), t + self.effect_window)
        treatment_range = np.arange(t + 1, t_stop + 1)
        treatment_range_rel = treatment_range - t

        future_outcome = patient_df.loc[treatment_range, outcome_name]
        future_outcome += scaled_effect / treatment_range_rel ** 0.5
        return treatment_range, future_outcome

    @staticmethod
    def combine_treatments(treatment_ranges, treated_future_outcomes):
        """
        Min combining of different treatment effects
        Args:
            treatment_ranges: List of effect windows w_l
            treated_future_outcomes: Future outcomes under each individual treatment

        Returns: Combined effect window, combined treated outcome
        """
        treated_future_outcomes = pd.concat(treated_future_outcomes, axis=1)
        common_treatment_range = [set(treatment_range) for i, treatment_range in enumerate(treatment_ranges)]
        common_treatment_range = set.union(*common_treatment_range)
        common_treatment_range = sorted(list(common_treatment_range))
        treated_future_outcomes = treated_future_outcomes.loc[common_treatment_range]
        treated_future_outcomes['agg'] = np.nanmin(treated_future_outcomes.values, axis=1)
        return common_treatment_range, treated_future_outcomes['agg']


class MIMIC3SyntheticDataset(MIMIC3RealDataset):
    """
    Pytorch-style semi-synthetic MIMIC-III dataset
    """
    def __init__(self,
                 all_vitals: pd.DataFrame,
                 static_features: pd.DataFrame,
                 vital_list: list,
                 synthetic_outcomes: List[SyntheticOutcomeGenerator],
                 synthetic_treatments: List[SyntheticTreatment],
                 treatment_outcomes_influence: dict,
                 subset_name: str,
                 mode='factual',
                 projection_horizon: int = None,
                 treatments_seq: np.array = None,
                 n_treatments_seq: int = None,
                 max_number=200):
        """
        Args:
            all_vitals: DataFrame with vitals (time-varying covariates); multiindex by (patient_id, timestep)
            static_features: DataFrame with static features
            synthetic_outcomes: List of SyntheticOutcomeGenerator
            synthetic_treatments: List of SyntheticTreatment
            treatment_outcomes_influence: dict with treatment-outcomes influences
            subset_name: train / val / test
            mode: factual
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            treatments_seq: Fixed (non-random) treatment sequecne for multiple-step-ahead prediction
            n_treatments_seq: Number of random trajectories after rolling origin in test subset
        """

        self.subset_name = subset_name
        self.all_vitals = all_vitals.copy().sort_index()
        vital_cols = all_vitals.columns
        self.synthetic_outcomes = synthetic_outcomes
        self.synthetic_treatments = synthetic_treatments
        self.treatment_outcomes_influence = treatment_outcomes_influence
        self.vital_name_to_idx = {name: idx for idx, name in enumerate(vital_list)}
        self.max_number = max_number

        prev_treatment_cols = [f'{treatment.treatment_name}_prev' for treatment in self.synthetic_treatments]
        outcome_cols = [outcome.outcome_name for outcome in self.synthetic_outcomes]

        # Sampling untreated outcomes
        for outcome in self.synthetic_outcomes:
            outcome.simulate_untreated(self.all_vitals, static_features, max_number)
        # Placeholders
        for treatment in self.synthetic_treatments:
            self.all_vitals[f'{treatment.treatment_name}_prev'] = 0.0
        self.all_vitals['fact'] = np.nan
        self.all_vitals.loc[(slice(None), 0), 'fact'] = 1.0  # First observation is always factual
        user_sizes = self.all_vitals.groupby(level='subject_id', sort=False).size()

        # Treatment application
        seeds = np.random.randint(np.iinfo(np.int32).max, size=len(static_features))
        par = Parallel(n_jobs=max(1, multiprocessing.cpu_count() - 10), backend='loky')
        # par = Parallel(n_jobs=4, backend='loky')
        logger.info(f'Simulating {mode} treatments and applying them to outcomes.')
        if mode == 'factual':
            self.all_vitals = par(delayed(self.treat_patient_factually)(patient_ix, seed)
                                  for patient_ix, seed in tqdm(zip(static_features.index, seeds), total=len(static_features)))
        else:
            raise NotImplementedError(f"Mode {mode} not supported in simplified version")

        self.all_vitals = pd.concat(self.all_vitals, keys=static_features.index)

        if mode == 'factual':
            # Padding with nans
            self.all_vitals = self.all_vitals.unstack(fill_value=np.nan, level=0).stack(dropna=False).swaplevel(0, 1).sort_index()
            static_features = static_features.sort_index()
            static_features = static_features.values

        # Conversion to np arrays
        treatments = self.all_vitals[prev_treatment_cols].fillna(0.0).values.reshape((-1, max(user_sizes),
                                                                                      len(prev_treatment_cols)))
        vitals = self.all_vitals[vital_cols].fillna(0.0).values.reshape((-1, max(user_sizes), len(vital_cols)))
        outcomes_unscaled = self.all_vitals[outcome_cols].fillna(0.0).values.reshape((-1, max(user_sizes), len(outcome_cols)))
        active_entries = (~self.all_vitals.isna().all(1)).astype(float)
        active_entries = active_entries.values.reshape((-1, max(user_sizes), 1))
        user_sizes = np.squeeze(active_entries.sum(1))

        logger.info(f'Shape of exploded vitals: {vitals.shape}.')

        self.data = {
            'sequence_lengths': user_sizes - 1,
            'prev_treatments': treatments[:, :-1, :],
            'vitals': vitals[:, 1:, :],
            'next_vitals': vitals[:, 2:, :],
            'current_treatments': treatments[:, 1:, :],
            'static_features': static_features,
            'active_entries': active_entries[:, 1:, :],
            'unscaled_outputs': outcomes_unscaled[:, 1:, :],
            'prev_unscaled_outputs': outcomes_unscaled[:, :-1, :],
            'sample_indices': np.arange(len(outcomes_unscaled)),
        }
        self.data['current_covariates'] = self.data['vitals']
        self.data['next_covariates'] = self.data['next_vitals']

        self.processed = False  # Need for normalisation of newly generated outcomes
        self.processed_sequential = False
        self.processed_autoregressive = False

        self.norm_const = 1.0

    def plot_timeseries(self, n_patients=5, mode='factual'):
        """
        Plotting patient trajectories
        Args:
            n_patients: Number of trajectories
            mode: factual
        """
        fig, ax = plt.subplots(nrows=4 * len(self.synthetic_outcomes) + len(self.synthetic_treatments), ncols=1, figsize=(15, 30))
        for i, patient_ix in enumerate(self.all_vitals.index.levels[0][:n_patients]):
            ax_ind = 0
            factuals = self.all_vitals.fillna(0.0).fact.astype(bool)
            for outcome in self.synthetic_outcomes:
                outcome_name = outcome.outcome_name
                ax[ax_ind].plot(self.all_vitals[factuals].loc[patient_ix, f'{outcome_name}_exog'].
                                groupby('hours_in').head(1).values)
                ax[ax_ind + 1].plot(self.all_vitals[factuals].loc[patient_ix, f'{outcome_name}_endo'].
                                    groupby('hours_in').head(1).values)
                ax[ax_ind + 2].plot(self.all_vitals[factuals].loc[patient_ix, f'{outcome_name}_untreated'].
                                    groupby('hours_in').head(1).values)
                if mode == 'factual':
                    ax[ax_ind + 3].plot(self.all_vitals.loc[patient_ix, outcome_name].values)

                ax[ax_ind].set_title(f'{outcome_name}_exog')
                ax[ax_ind + 1].set_title(f'{outcome_name}_endo')
                ax[ax_ind + 2].set_title(f'{outcome_name}_untreated')
                ax[ax_ind + 3].set_title(f'{outcome_name}')
                ax_ind += 4

            for treatment in self.synthetic_treatments:
                treatment_name = treatment.treatment_name
                ax[ax_ind].plot(self.all_vitals[factuals].loc[patient_ix, f'{treatment_name}_prev'].
                                groupby('hours_in').head(1).values + 2 * i)
                ax[ax_ind].set_title(f'{treatment_name}')
                ax_ind += 1

        fig.suptitle(f'Time series from {self.subset_name}', fontsize=16)
        plt.show()

    def _sample_treatments_from_factuals(self, patient_df, t, rng=np.random.RandomState(None)):
        """
        Sample treatment for patient_df and time-step t
        Args:
            patient_df: DataFrame of patient
            t: Time-step
            rng: Random numbers generator (for parallelizing)

        Returns: Propensity scores
        """
        factual_patient_df = patient_df[patient_df.fact.astype(bool)]
        treat_probas = {treatment.treatment_name: treatment.treatment_proba(factual_patient_df, t, rng) for treatment in
                        self.synthetic_treatments}
        return treat_probas

    def _combined_treating(self, patient_df, t, outcome: SyntheticOutcomeGenerator, treat_probas: dict):
        """
        Combing application of treatments
        Args:
            patient_df: DataFrame of patient
            t: Time-step
            outcome: Outcome to treat
            treat_probas: Propensity scores

        Returns: Combined effect window, combined treated outcome
        """
        treatment_ranges, treated_future_outcomes = [], []
        influencing_treatments = self.treatment_outcomes_influence[outcome.outcome_name]
        influencing_treatments = \
            [treatment for treatment in self.synthetic_treatments if treatment.treatment_name in influencing_treatments]

        for treatment in influencing_treatments:
            treatment_range, treated_future_outcome = \
                treatment.get_treated_outcome(patient_df, t, outcome.outcome_name, treat_probas[treatment.treatment_name])

            treatment_ranges.append(treatment_range)
            treated_future_outcomes.append(treated_future_outcome)

        common_treatment_range, future_outcomes = SyntheticTreatment.combine_treatments(
            treatment_ranges,
            treated_future_outcomes
        )
        return common_treatment_range, future_outcomes

    def treat_patient_factually(self, patient_ix: int, seed: int = None):
        """
        Generate factually treated outcomes
        Args:
            patient_ix: Index of patient
            seed: Random seed

        Returns: DataFrame of patient
        """
        patient_df = self.all_vitals.loc[patient_ix].copy()
        rng = np.random.RandomState(seed)
        prev_treatment_cols = [f'{treatment.treatment_name}_prev' for treatment in self.synthetic_treatments]

        for t in range(len(patient_df)):

            # Sampling treatments, based on previous factual outcomes
            treat_probas = self._sample_treatments_from_factuals(patient_df, t, rng)

            if t < max(patient_df.index.get_level_values('hours_in')):
                # Setting factuality flags
                patient_df.loc[t + 1, 'fact'] = 1.0

                # Setting factual sampled treatments
                patient_df.loc[t + 1, prev_treatment_cols] = {f'{t}_prev': v[0] for t, v in treat_probas.items()}

                # Treatments applications

                # Treating each outcome separately
                for outcome in self.synthetic_outcomes:
                    common_treatment_range, future_outcomes = self._combined_treating(patient_df, t, outcome, treat_probas)
                    patient_df.loc[common_treatment_range, f'{outcome.outcome_name}'] = future_outcomes
                        
        return patient_df

    def get_scaling_params(self):
        outcome_cols = [outcome.outcome_name for outcome in self.synthetic_outcomes]
        logger.info('Performing normalisation.')
        scaling_params = {
            'output_means': self.all_vitals[outcome_cols].mean(0).to_numpy(),
            'output_stds': self.all_vitals[outcome_cols].std(0).to_numpy(),
        }
        return scaling_params

    def process_data(self, scaling_params):
        """
        Pre-process dataset for one-step-ahead prediction
        Args:
            scaling_params: dict of standard normalization parameters (calculated with train subset)
        """
        if not self.processed:
            logger.info(f'Processing {self.subset_name} dataset before training')

            self.data['outputs'] = (self.data['unscaled_outputs'] - scaling_params['output_means']) / \
                scaling_params['output_stds']
            self.data['prev_outputs'] = (self.data['prev_unscaled_outputs'] - scaling_params['output_means']) / \
                scaling_params['output_stds']

            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            self.scaling_params = scaling_params
            self.processed = True
        else:
            logger.info(f'{self.subset_name} Dataset already processed')

        return self.data
    
    def _to_tensor(self, arr, device):
        if isinstance(arr, torch.Tensor):
            return arr.to(device)
        return torch.tensor(arr, device=device)
    
    def simulate_output_after_actions(self, H_t, actions, scaling_params=None):
        """
        Batch counterfactual outcome simulation
        
        Args:
            H_t: Single batch data dictionary (from self.data field of MIMIC3SyntheticDataset)
                - sequence_lengths: (batch_size,) Effective sequence lengths for each patient
                - prev_treatments: (batch_size, seq_len-1, n_treatments) Previous time-step treatments
                - vitals: (batch_size, seq_len-1, n_vitals) Current vital signs
                - current_treatments: (batch_size, seq_len-1, n_treatments) Current time-step treatments
                - unscaled_outputs: (batch_size, seq_len-1, n_outcomes) Unscaled outcomes
                - static_features: (batch_size, n_static) Static features
                - active_entries: (batch_size, seq_len-1, 1) Valid time-step masks
                - sample_indices: (batch_size,) Sample indices for each patient
            
        
        Returns:
            (batch_size, n_outcomes) Counterfactual outcomes at the final time step
        """
        
        #Determine the device of the original data first
        original_device = None
        for key, value in H_t.items():
            if isinstance(value, torch.Tensor):
                original_device = value.device
                break
        if original_device is None:
            original_device = 'cpu'
        
        device = 'cpu'
        Ht = {}
        for key, value in H_t.items():
            if isinstance(value, np.ndarray):
                Ht[key] = torch.from_numpy(value).to(device)
            else:
                Ht[key] = value.to(device)

        synthetic_outcomes = self.synthetic_outcomes
        synthetic_treatments = self.synthetic_treatments
        treatment_outcomes_influence = self.treatment_outcomes_influence
        vital_name_to_idx = self.vital_name_to_idx

        actions = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions

        # Extract batch metadata
        batch_size = Ht['sequence_lengths'].shape[0]
        sample_indices = Ht['sample_indices'].int()  # (batch_size,)
        historical_time = Ht['vitals'].shape[1]
        projection_horizon = actions.shape[1]
        max_time = historical_time + projection_horizon  
        global_time_range = np.arange(0, max_sequence_length)

        # Initialize result storage
        if isinstance(Ht['unscaled_outputs'], torch.Tensor):
            unscaled_outputs = Ht['unscaled_outputs'].clone()  
        else:
            unscaled_outputs = self._to_tensor(Ht['unscaled_outputs'], device=device).clone()

        counterfactual_outputs = torch.zeros((batch_size, max_time, len(synthetic_outcomes)), device=device, dtype=torch.double)  # (batch_size, max_time, n_outcomes)
        
        counterfactual_outputs[:, :historical_time, :] = unscaled_outputs  # Copy historical outcomes
        
        # Treatment/outcome name-to-index mappings
        treat_name_to_idx = {treat.treatment_name: idx for idx, treat in enumerate(synthetic_treatments)}
        outcome_name_to_idx = {outcome.outcome_name: idx for idx, outcome in enumerate(synthetic_outcomes)}

        # Calculate untreated outcomes independently for each outcome variable
        for outcome_idx, outcome in enumerate(synthetic_outcomes):
            # 1. Exogenous contribution calculation
            exog_var_indices = [vital_name_to_idx[var] for var in outcome.exogeneous_vars]
            exog_vars_historical = Ht['vitals'][:, :, exog_var_indices]  # (batch_size, , n_exog_vars)
            exog_vars_future = Ht['future_vitals'][:, :projection_horizon, exog_var_indices]  # (batch_size, projection_horizon, n_exog_vars)
            exog_vars = torch.cat((exog_vars_historical, exog_vars_future), dim=1)  # (batch_size, his_time, n_exog_vars)
            exog_vars_flat = exog_vars.reshape(batch_size * max_time, -1)  # (batch_size * max_time, n_exog_vars)
            exog_contribution = outcome.exog_weight * outcome.exog_dependency(exog_vars_flat)
            exog_contribution = exog_contribution.reshape(batch_size, max_time)  
            
            # 2. Endogenous contribution calculation
            # Random fluctuation term (batch_size patients, each with a time series)
            endo_rand = outcome.endo_dependency(global_time_range, self.max_number)  
            endo_rand_contribution = outcome.endo_rand_weight * endo_rand  # (max_sequence_length, max_number)
            # Spline trend term
            spline_trends = SplineTrendsMixture(n_patients=self.max_number, max_time=max_sequence_length)(global_time_range)
            endo_spline_contribution = outcome.endo_spline_weight * spline_trends  # (max_sequence_length, max_number)
            
            # Combined endogenous contribution
            endo_contribution = endo_rand_contribution + endo_spline_contribution  # (max_sequence_length, max_number)
            endo_contribution = endo_contribution[sample_indices, 1:(max_time+1)]   # (batch_size, max_time)

            # 3. Untreated outcome = exogenous contribution + endogenous contribution
            untreated_result = exog_contribution + endo_contribution  # (batch_size, max_time)
            
            # 4. For each patient, only overwrite future outcomes within their valid time steps
            for patient_idx in range(batch_size):
                valid_seq_len = int(Ht['sequence_lengths'][patient_idx])
                future_time_steps = torch.arange(valid_seq_len, valid_seq_len + projection_horizon , device=device)
                
                if len(future_time_steps) > 0:
                    counterfactual_outputs[patient_idx, future_time_steps, outcome_idx] = \
                        self._to_tensor(untreated_result[patient_idx, future_time_steps], device=device)

        # Iterate through each patient 
        for patient_idx in range(batch_size):
            # Get patient's valid sequence length 
            valid_seq_len = int(Ht['sequence_lengths'][patient_idx])
            future_time_steps = torch.arange(valid_seq_len, valid_seq_len + projection_horizon , device=device)
            patient_current_treat = np.concatenate(
                [Ht['current_treatments'][patient_idx, :valid_seq_len], 
                actions[patient_idx, :]], 
                axis=0
            )  # (valid_seq_len + future_time, n_treatments)
            patient_current_treat = self._to_tensor(patient_current_treat, device)  
                    
            # -------------------- Treatment effect calculation  --------------------
            for t in range(valid_seq_len + projection_horizon): 

                current_treat_vector = patient_current_treat[t]
                t_max = valid_seq_len + projection_horizon - 1  

                # Calculate treatment effects for each outcome variable
                for outcome in synthetic_outcomes:
                    outcome_name = outcome.outcome_name
                    outcome_idx = outcome_name_to_idx[outcome_name]
                    
                    influencing_treats = [
                        treat for treat in synthetic_treatments
                        if treat.treatment_name in treatment_outcomes_influence.get(outcome_name, [])
                    ]

                    # Collect effects from all relevant treatments (each treatment generates (effect_range, effect_values))
                    treatment_effects = []
                    for treat in influencing_treats:
                        treat_idx = treat_name_to_idx[treat.treatment_name]
                        treat_proba = current_treat_vector[treat_idx]  
                        # Calculate effect range
                        t_start_effect = t 
                        t_end_effect = min(t + int(treat.effect_window) - 1, t_max)
                        effect_range = torch.arange(t_start_effect, t_end_effect + 1, device=device)   # [t, ..., t_end-1]
                        rel_time = effect_range - t + 1  # [1, 2, ..., (t_end - t)]
                        
                        # Calculate effect values
                        scaled_effect = treat.full_effect * cs(treat_proba)
                        effect_values = scaled_effect / rel_time ** 0.5  

                        treatment_effects.append((effect_range, effect_values))
                        
                    # Combine effects from multiple treatments 
                    
                    # Find union of all treatment effect time steps 
                    all_effect_times = torch.cat([er for er, _ in treatment_effects])
                    unique_effect_times = torch.unique(all_effect_times)
                    
                    # Filter future time steps
                    future_effect_times = unique_effect_times[unique_effect_times >= valid_seq_len]
                    
                    if future_effect_times.numel() > 0:
                        # For each time step, take the minimum effect value 
                        for time in future_effect_times:
                            # Collect all treatment effects on this time step
                            values = []
                            for er, ev in treatment_effects:
                                mask = (er == time)  
                                if mask.any():
                                    values.append(ev[mask])
                                
                            values = torch.cat(values)
                            min_effect = values.min()
                            counterfactual_outputs[patient_idx, time, outcome_idx] += min_effect
                    
        # Extract final time step outcomes for each patient (valid_seq_len-1)
        final_outcomes = []
        for i in range(batch_size):
            valid_seq_len = int(Ht['sequence_lengths'][i])
            final_outcome = counterfactual_outputs[i, valid_seq_len + projection_horizon - 1]
            final_outcomes.append(final_outcome)
        final_outcomes = torch.stack(final_outcomes)  # (batch_size, n_outcomes)

        mean = self._to_tensor(scaling_params['output_means'], device=device)
        std = self._to_tensor(scaling_params['output_stds'], device=device)
        outputs = (final_outcomes - mean) / std

        return outputs.cpu().numpy()

class MIMIC3SyntheticDatasetCollection(SyntheticDatasetCollection):
    """
    Dataset collection (train_f, val_f, test_f)
    """
    def __init__(self,
                 path,
                 synth_outcomes_list: list,
                 synth_treatments_list: list,
                 vital_list: list,
                 treatment_outcomes_influence: dict,
                 min_seq_length: int = None,
                 max_seq_length: int = None,
                 max_number: int = None,
                 seed: int = 100,
                 data_seed: int = 100,
                 split: dict = {'val': 0.2, 'test': 0.2},
                 projection_horizon: int = 4,
                 autoregressive=True,
                 n_treatments_seq: int = None,
                 **kwargs):
        """
        Args:
            path: Path with MIMIC-3 dataset (HDFStore)
            synth_outcomes_list: List of SyntheticOutcomeGenerator
            synth_treatments_list: List of SyntheticTreatment
            treatment_outcomes_influence: dict with treatment-outcomes influences
            min_seq_length: Min sequence lenght in cohort
            max_seq_length: Max sequence lenght in cohort
            max_number: Maximum number of patients in cohort
            seed: Seed for sampling random functions
            data_seed: Seed for random cohort patient selection
            split: Ratio of train / val / test split
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            n_treatments_seq: Number of random trajectories after rolling origin in test subset
        """
        super(MIMIC3SyntheticDatasetCollection, self).__init__()
        self.seed = seed
        np.random.seed(seed)
        all_vitals, static_features = load_mimic3_data_raw(ROOT_PATH + '/' + path, min_seq_length=min_seq_length,
                                                           max_seq_length=max_seq_length, max_number=max_number,
                                                           data_seed=data_seed, **kwargs)

        # Train/val/test random_split
        static_features, static_features_test = train_test_split(static_features, test_size=split['test'], random_state=seed)
        all_vitals, all_vitals_test = all_vitals.loc[static_features.index], all_vitals.loc[static_features_test.index]

        if split['val'] > 0.0:
            static_features_train, static_features_val = train_test_split(static_features,
                                                                          test_size=split['val'] / (1 - split['test']),
                                                                          random_state=2 * seed)
            all_vitals_train, all_vitals_val = all_vitals.loc[static_features_train.index], \
                all_vitals.loc[static_features_val.index]
        else:
            static_features_train = static_features
            all_vitals_train = all_vitals

        self.train_f = \
            MIMIC3SyntheticDataset(all_vitals_train, static_features_train, vital_list, synth_outcomes_list, synth_treatments_list,
                                   treatment_outcomes_influence, 'train', max_number=max_number)

        if split['val'] > 0.0:
            self.val_f = MIMIC3SyntheticDataset(all_vitals_val, static_features_val, vital_list, synth_outcomes_list, synth_treatments_list,
                                                treatment_outcomes_influence, 'val', max_number=max_number)
        
        self.test_f = MIMIC3SyntheticDataset(all_vitals_test, static_features_test, vital_list, synth_outcomes_list, synth_treatments_list,
                                           treatment_outcomes_influence, 'test', max_number=max_number)

        self.projection_horizon = projection_horizon
        self.autoregressive = autoregressive
        self.has_vitals = True
        self.train_scaling_params = self.train_f.get_scaling_params()

    def process_data_multi(self):
        """
        CT/IQL path for the GIFT-aligned semi-synthetic MIMIC benchmark.

        The Tumor SyntheticDatasetCollection expects counterfactual one-step and
        treatment-sequence subsets. GIFT's MIMIC environment uses train/val/test
        factual splits plus the oracle ``simulate_output_after_actions`` instead,
        so we keep this override local to the MIMIC collection.
        """
        self.train_f.process_data(self.train_scaling_params)
        if hasattr(self, 'val_f') and self.val_f is not None:
            self.val_f.process_data(self.train_scaling_params)
        if hasattr(self, 'test_f') and self.test_f is not None:
            self.test_f.process_data(self.train_scaling_params)
        self.processed_data_multi = True

import torch.optim as optim
from pytorch_lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
import torch
from typing import Union
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
import ray
from ray import tune
from ray import ray_constants
from copy import deepcopy
from pytorch_lightning import Trainer
from torch_ema import ExponentialMovingAverage
from typing import List
from tqdm import tqdm

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.baselines.utils import grad_reverse, BRTreatmentOutcomeHead, AlphaRise, bce
from src.data.cip_dataset import CIPDataset, get_dataloader
from src.utils.utils import set_seed
from scipy.stats import spearmanr
import pandas as pd
from scipy.stats import rankdata
from src.utils.helper_functions import generate_perturbed_sequences, enhanced_analyze_case
import warnings
warnings.filterwarnings('ignore')
import pickle
import os


logger = logging.getLogger(__name__)
ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD = 10**8  # ~ 100Mb


def train_eval_factual(args: dict, train_f: Dataset, val_f: Dataset, orig_hparams: DictConfig, input_size: int, model_cls,
                       tuning_criterion='rmse', **kwargs):
    """
    Globally defined method, used for ray tuning
    :param args: Hyperparameter configuration
    :param train_f: Factual train dataset
    :param val_f: Factual val dataset
    :param orig_hparams: DictConfig of original hyperparameters
    :param input_size: Input size of model, infuences concrete hyperparameter configuration
    :param model_cls: class of model
    :param kwargs: Other args
    """
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    new_params = deepcopy(orig_hparams)
    model_cls.set_hparams(new_params.model, args, input_size, model_cls.model_type)
    if model_cls.model_type == 'decoder':
        # Passing encoder takes too much memory
        encoder_r_size = new_params.model.encoder.br_size if 'br_size' in new_params.model.encoder \
            else new_params.model.encoder.seq_hidden_units  # Using either br_size or Memory adapter
        model = model_cls(new_params, encoder_r_size=encoder_r_size, **kwargs)
    else:
        model = model_cls(new_params, **kwargs)

    train_loader = DataLoader(train_f, shuffle=True, batch_size=new_params.model[model_cls.model_type]['batch_size'],
                              drop_last=True)
    trainer = Trainer(gpus=eval(str(new_params.exp.gpus))[:1],
                      logger=None,
                      max_epochs=new_params.exp.max_epochs,
                      progress_bar_refresh_rate=0,
                      gradient_clip_val=new_params.model[model_cls.model_type]['max_grad_norm']
                      if 'max_grad_norm' in new_params.model[model_cls.model_type] else None,
                      callbacks=[AlphaRise(rate=new_params.exp.alpha_rate)])
    trainer.fit(model, train_dataloader=train_loader)

    if tuning_criterion == 'rmse':
        val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(val_f)
        tune.report(val_rmse_orig=val_rmse_orig, val_rmse_all=val_rmse_all)
    elif tuning_criterion == 'bce':
        val_bce_orig, val_bce_all = model.get_masked_bce(val_f)
        tune.report(val_bce_orig=val_bce_orig, val_bce_all=val_bce_all)
    else:
        raise NotImplementedError()

class TimeVaryingCausalModel(LightningModule):
    """
    Abstract class for baselines, estimating counterfactual outcomes over time
    """

    model_type = None  # Will be defined in subclasses
    possible_model_types = None  # Will be defined in subclasses
    tuning_criterion = None

    def __init__(self,
                 args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__()
        self.dataset_collection = dataset_collection
        if dataset_collection is not None:
            self.autoregressive = self.dataset_collection.autoregressive
            self.has_vitals = self.dataset_collection.has_vitals
            self.bce_weights = None  # Will be calculated, when calling preparing data
        else:
            self.autoregressive = autoregressive
            self.has_vitals = has_vitals
            self.bce_weights = bce_weights
            print(self.bce_weights)

        # General datasets parameters
        self.dim_treatments = args.model.dim_treatments
        self.dim_vitals = args.model.dim_vitals
        self.dim_static_features = args.model.dim_static_features
        self.dim_outcome = args.model.dim_outcomes
        self.tau = args.exp.tau
        self.config = args

        self.input_size = None  # Will be defined in subclasses

        self.save_hyperparameters(args)  # Will be logged to mlflow

        try:
            self.std = self.dataset_collection.train_scaling_params[1]['cancer_volume']
        except:
            self.std = 1

    def _get_optimizer(self, param_optimizer: list):
        no_decay = ['bias', 'layer_norm']
        sub_args = self.hparams.model[self.model_type]
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': sub_args['optimizer']['weight_decay'],
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        lr = sub_args['optimizer']['learning_rate']
        optimizer_cls = sub_args['optimizer']['optimizer_cls']
        if optimizer_cls.lower() == 'adamw':
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == 'adam':
            optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == 'sgd':
            optimizer = optim.SGD(optimizer_grouped_parameters, lr=lr,
                                  momentum=sub_args['optimizer']['momentum'])
        else:
            raise NotImplementedError()

        return optimizer

    def _get_lr_schedulers(self, optimizer):
        if not isinstance(optimizer, list):
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            return [optimizer], [lr_scheduler]
        else:
            lr_schedulers = []
            for opt in optimizer:
                lr_schedulers.append(optim.lr_scheduler.ExponentialLR(opt, gamma=0.99))
            return optimizer, lr_schedulers

    def configure_optimizers(self):
        optimizer = self._get_optimizer(list(self.named_parameters()))
        if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
            return self._get_lr_schedulers(optimizer)
        return optimizer

    def train_dataloader(self) -> DataLoader:
        sub_args = self.hparams.model[self.model_type]
        return DataLoader(self.dataset_collection.train_f, shuffle=True, batch_size=sub_args['batch_size'], drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_collection.val_f, batch_size=self.hparams.dataset.val_batch_size)

    def get_predictions(self, dataset: Dataset) -> np.array:
        raise NotImplementedError()

    def get_propensity_scores(self, dataset: Dataset) -> np.array:
        raise NotImplementedError()

    def get_representations(self, dataset: Dataset) -> np.array:
        raise NotImplementedError()

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Autoregressive Prediction for {dataset.subset_name}.')
        if self.model_type == 'decoder':  # CRNDecoder / EDCTDecoder / RMSN Decoder

            predicted_outputs = np.zeros((len(dataset), self.hparams.dataset.projection_horizon, self.dim_outcome))
            for t in range(self.hparams.dataset.projection_horizon):
                logger.info(f't = {t + 2}')

                outputs_scaled = self.get_predictions(dataset)
                predicted_outputs[:, t] = outputs_scaled[:, t]

                if t < (self.hparams.dataset.projection_horizon - 1):
                    dataset.data['prev_outputs'][:, t + 1, :] = outputs_scaled[:, t, :]
        else:
            raise NotImplementedError()

        return predicted_outputs

    def get_masked_bce(self, dataset: Dataset):
        logger.info(f'BCE calculation for {dataset.subset_name}.')
        treatment_pred = torch.tensor(self.get_propensity_scores(dataset))
        current_treatments = torch.tensor(dataset.data['current_treatments'])

        bce = (self.bce_loss(treatment_pred, current_treatments, kind='predict')).unsqueeze(-1).numpy()
        bce = bce * dataset.data['active_entries']

        # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
        bce_orig = bce.sum(0).sum(-1) / dataset.data['active_entries'].sum(0).sum(-1)
        bce_orig = bce_orig.mean()

        # Masked averaging over all dimensions at once
        bce_all = bce.sum() / dataset.data['active_entries'].sum()

        return bce_orig, bce_all

    def get_normalised_masked_rmse(self, dataset: Dataset, one_step_counterfactual=False):
        logger.info(f'RMSE calculation for {dataset.subset_name}.')
        outputs_scaled = self.get_predictions(dataset)
        unscale = self.hparams.exp.unscale_rmse
        percentage = self.hparams.exp.percentage_rmse

        if unscale:
            output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
            outputs_unscaled = outputs_scaled * output_stds + output_means

            # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
            mse = ((outputs_unscaled - dataset.data['unscaled_outputs']) ** 2) * dataset.data['active_entries']
        else:
            # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
            mse = ((outputs_scaled - dataset.data['outputs']) ** 2) * dataset.data['active_entries']

        # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
        mse_orig = mse.sum(0).sum(-1) / dataset.data['active_entries'].sum(0).sum(-1)
        mse_orig = mse_orig.mean()
        rmse_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const

        # Masked averaging over all dimensions at once
        mse_all = mse.sum() / dataset.data['active_entries'].sum()
        rmse_normalised_all = np.sqrt(mse_all) / dataset.norm_const

        if percentage:
            rmse_normalised_orig *= 100.0
            rmse_normalised_all *= 100.0

        if one_step_counterfactual:
            # Only considering last active entry with actual counterfactuals
            num_samples, time_dim, output_dim = dataset.data['active_entries'].shape
            last_entries = dataset.data['active_entries'] - np.concatenate([dataset.data['active_entries'][:, 1:, :],
                                                                            np.zeros((num_samples, 1, output_dim))], axis=1)
            if unscale:
                mse_last = ((outputs_unscaled - dataset.data['unscaled_outputs']) ** 2) * last_entries
            else:
                mse_last = ((outputs_scaled - dataset.data['outputs']) ** 2) * last_entries

            mse_last = mse_last.sum() / last_entries.sum()
            rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const

            if percentage:
                rmse_normalised_last *= 100.0

            return rmse_normalised_orig, rmse_normalised_all, rmse_normalised_last

        return rmse_normalised_orig, rmse_normalised_all

    def get_normalised_n_step_rmses(self, dataset: Dataset, datasets_mc: List[Dataset] = None):
        logger.info(f'RMSE calculation for {dataset.subset_name}.')
        assert self.model_type == 'decoder' or self.model_type == 'multi' or self.model_type == 'g_net' or \
               self.model_type == 'msm_regressor'
        assert hasattr(dataset, 'data_processed_seq')

        unscale = self.hparams.exp.unscale_rmse
        percentage = self.hparams.exp.percentage_rmse
        outputs_scaled = self.get_autoregressive_predictions(dataset if datasets_mc is None else datasets_mc)

        if unscale:
            output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
            outputs_unscaled = outputs_scaled * output_stds + output_means

            mse = ((outputs_unscaled - dataset.data_processed_seq['unscaled_outputs']) ** 2) \
                * dataset.data_processed_seq['active_entries']
        else:
            mse = ((outputs_scaled - dataset.data_processed_seq['outputs']) ** 2) * dataset.data_processed_seq['active_entries']

        nan_idx = np.unique(np.where(np.isnan(dataset.data_processed_seq['outputs']))[0])
        not_nan = np.array([i for i in range(outputs_scaled.shape[0]) if i not in nan_idx])

        # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
        mse_orig = mse[not_nan].sum(0).sum(-1) / dataset.data_processed_seq['active_entries'][not_nan].sum(0).sum(-1)
        rmses_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const

        if percentage:
            rmses_normalised_orig *= 100.0

        return rmses_normalised_orig

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        raise NotImplementedError()

    def finetune(self, resources_per_trial: dict):
        """
        Hyperparameter tuning with ray[tune]
        """
        self.prepare_data()
        sub_args = self.hparams.model[self.model_type]
        logger.info(f"Running hyperparameters selection with {sub_args['tune_range']} trials")
        ray.init(num_gpus=len(eval(str(self.hparams.exp.gpus))), num_cpus=4, include_dashboard=False,
                 _redis_max_memory=ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD)

        hparams_grid = {k: tune.choice(v) for k, v in sub_args['hparams_grid'].items()}
        analysis = tune.run(tune.with_parameters(train_eval_factual,
                                                 input_size=self.input_size,
                                                 model_cls=self.__class__,
                                                 tuning_criterion=self.tuning_criterion,
                                                 train_f=deepcopy(self.dataset_collection.train_f),
                                                 val_f=deepcopy(self.dataset_collection.val_f),
                                                 orig_hparams=self.hparams,
                                                 autoregressive=self.autoregressive,
                                                 has_vitals=self.has_vitals,
                                                 bce_weights=self.bce_weights,
                                                 projection_horizon=self.projection_horizon
                                                 if hasattr(self, 'projection_horizon') else None),
                            resources_per_trial=resources_per_trial,
                            metric=f"val_{self.tuning_criterion}_all",
                            mode="min",
                            config=hparams_grid,
                            num_samples=sub_args['tune_range'],
                            name=f"{self.__class__.__name__}{self.model_type}",
                            max_failures=3)
        ray.shutdown()

        logger.info(f"Best hyperparameters found: {analysis.best_config}.")
        logger.info("Resetting current hyperparameters to best values.")
        self.set_hparams(self.hparams.model, analysis.best_config, self.input_size, self.model_type)

        self.__init__(self.hparams,
                      dataset_collection=self.dataset_collection,
                      encoder=self.encoder if hasattr(self, 'encoder') else None,
                      propensity_treatment=self.propensity_treatment if hasattr(self, 'propensity_treatment') else None,
                      propensity_history=self.propensity_history if hasattr(self, 'propensity_history') else None)
        return self

    def visualize(self, dataset: Dataset, index=0, artifacts_path=None):
        pass

    def bce_loss(self, treatment_pred, current_treatments, kind='predict'):
        mode = self.hparams.dataset.treatment_mode
        bce_weights = torch.tensor(self.bce_weights).type_as(current_treatments) if self.hparams.exp.bce_weight else None

        if kind == 'predict':
            bce_loss = bce(treatment_pred, current_treatments, mode, bce_weights)
        elif kind == 'confuse':
            uniform_treatments = torch.ones_like(current_treatments)
            if mode == 'multiclass':
                uniform_treatments *= 1 / current_treatments.shape[-1]
            elif mode == 'multilabel':
                uniform_treatments *= 0.5
            bce_loss = bce(treatment_pred, uniform_treatments, mode)
        else:
            raise NotImplementedError()
        return bce_loss

    def cauculate_re_loss(self, batch, update_D=True):
        br = self.build_br_mine(batch)
        if update_D:
            br = br.detach()
        current_treatments = batch['current_treatments']
        active_entries = batch['active_entries'] if 'active_entries' in batch else torch.ones_like(current_treatments)
        logits = self.br_treatment_outcome_head.build_RE(br)
        loss = F.mse_loss(F.sigmoid(logits), current_treatments, reduce=False)
        loss = torch.sum(loss * active_entries) / torch.sum(active_entries)

        return loss

    def on_fit_start(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = list(self.possible_model_types - {self.model_type})

    def on_fit_end(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = list(self.possible_model_types)

    def optimize_interventions(self, encoder=None, num_iterations=100, learning_rate=0.01, batch_size=64):
        results = ['\n' + '-' * 50]
        tau = self.tau
        for i in range(1, 13):
        # for i in [2, 6, 9]:
            print(f'start predicting results for tau={i} ...')
            self.tau = i
            self.config.exp.tau = i
            print(self.tau, self.config.exp.tau)
            _, loss = self.optimize_interventions_onetime(encoder=encoder, num_iterations=num_iterations, learning_rate=learning_rate, batch_size=batch_size)
            results.append(f"Optimized interventions for tau={i}, {num_iterations} iterations, lr={learning_rate}: {loss}")
        return '\n'.join(results)

    def optimize_interventions_onetime(self, encoder=None, num_iterations=100, learning_rate=0.01, batch_size=64):
        set_seed(self.config['exp']['seed'])
        try:
            if not self.config.exp.test:
                data = self.dataset_collection.val_f.data_original
            else:
                data = self.dataset_collection.test_f.data_original
            print('load data_original')
        except:
            if not self.config.exp.test:
                data = self.dataset_collection.val_f.data
            else:
                data = self.dataset_collection.test_f.data
        dataloader = get_dataloader(CIPDataset(data, self.config), batch_size=batch_size, shuffle=False)
        
        device = "cuda"
        self.to(device)

        results = []
        losses = []
        losses_2 = []
        ture_output_list = []
        output_after_actions_list = []
        ture_output_actions_list = []
        elbos = []
        
        for (i, batch) in enumerate(dataloader):
            # if i not in [46, 99, 53, 36, 83]:
            # if i not in [99]:
            #     continue
            H_t, targets = batch
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)
            Y_targets = targets['outputs']
            # print(f"shape of Y_targets: {Y_targets.shape}")
            batch_size = Y_targets.size(0)
            
            # Initialize best tracking variables
            # best_a_seq = None
            best_elbo = float('inf')
            best_step = 0
            
            a_seq = torch.rand(batch_size, self.tau, self.config.dataset.treatment_size, device=device, requires_grad=True)
            # a_seq = torch.normal(mean=3, std=0.1, 
            #         size=(batch_size, self.tau, self.config.dataset.treatment_size),
            #         device=device, requires_grad=True)

            best_a_seq = torch.sigmoid(a_seq).clone().detach()
            optimizer = torch.optim.Adam([a_seq], lr=learning_rate)
            # use a different optimizer 
            # optimizer = torch.optim.SGD([a_seq], lr=learning_rate)

            print(f"-" * 50)
            for iteration in range(num_iterations):
                # print(f"Iteration {iteration}")
                # print(f"shape of current_treatments: {H_t['current_treatments'].shape}")
                a_seq_sigmoid = torch.sigmoid(a_seq)
                optimizer.zero_grad()
                loss = self.get_predictions_after_tau_steps_loss(H_t, Y_targets, a_seq_sigmoid, encoder=encoder)
                loss.backward(retain_graph=False)
                optimizer.step()
                # print(f"Iteration {iteration}: ELBO = {elbo.item()}, reg_loss = {reg_loss.item()}")
                # print(f"targets['current_treatments']: {targets['current_treatments']}")
                # print(f"a_seq_sigmoid: {a_seq_sigmoid}")

                
            optimized_a_seq = torch.sigmoid(a_seq).detach()
            
            output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(
                H_t, optimized_a_seq, self.dataset_collection.train_scaling_params)
            ture_output = targets['outputs'][:, -1, :].detach().cpu().numpy()

            true_actions = targets['current_treatments']
            ture_output_actions = self.dataset_collection.val_f.simulate_output_after_actions(
                H_t, true_actions, self.dataset_collection.train_scaling_params)

            loss = np.sqrt(((output_after_actions-ture_output) ** 2).mean())
            losses.append(loss)

            loss_2 = np.sqrt(((ture_output_actions-ture_output) ** 2).mean())
            losses_2.append(loss_2)

            ture_output_list.append(ture_output)
            output_after_actions_list.append(output_after_actions)
            ture_output_actions_list.append(ture_output_actions)

            results.append(optimized_a_seq)
        
        # print(f"Mean elbo: {sum(elbos) / len(elbos)}")

        true_actions = targets['current_treatments']
        ture_output_actions = self.dataset_collection.val_f.simulate_output_after_actions(
            H_t, true_actions, self.dataset_collection.train_scaling_params)

        print(f"Mean loss v1: {sum(losses) / len(losses)}")
        print(f"Mean loss_2 v2: {sum(losses_2) / len(losses_2)}")
        print(f"losses: {losses}")
        
        ture_output_list = np.concatenate(ture_output_list, axis=0)
        output_after_actions_list = np.concatenate(output_after_actions_list, axis=0)
        ture_output_actions_list = np.concatenate(ture_output_actions_list, axis=0)
        print(f"ture_output_list shape: {ture_output_list.shape}")
        print(f"first 10 ture_output_list: {ture_output_list[:10]}")
        
        loss = np.sqrt(((output_after_actions_list-ture_output_list) ** 2).mean())
        loss_2 = np.sqrt(((ture_output_actions_list-ture_output_list) ** 2).mean()) 
        
        print(f"Mean loss v2: {loss}")
        print(f"Mean loss_2 v2: {loss_2}")

        return torch.cat(results, dim=0), loss * self.std

    def get_predictions_after_tau_steps(self, H_t, Y_targets, a_seq, encoder=None):
        # Deep copy H_t to ensure complete independence
        H_t_copy = {k: v.clone().detach().float() for k, v in H_t.items()}
        
        if encoder is not None:
            with torch.no_grad():  # Don't track encoder computation
                # print(f"shape of current_treatments: {H_t['current_treatments'].shape}")
                if 'RMSN' in self.config.model.name:
                    _, br = encoder(H_t)
                else:
                    _, _, br = encoder(H_t)
                H_t_copy['init_state'] = br[:, -1, :].clone()

        length = H_t_copy['current_treatments'].shape[1]
        # Create new tensor for modifications
        current_treatments = H_t_copy['current_treatments'].clone()
        current_treatments[:, -1, :] = a_seq[:, 0, :]
        H_t_copy['current_treatments'] = current_treatments
        
        prev_outputs = H_t_copy['prev_outputs']
        prev_treatments = H_t_copy['prev_treatments']
        active_entries = H_t_copy['active_entries']
        if self.has_vitals:
            vitals = H_t_copy['vitals']
        
        for t in range(self.tau):
            if 'RMSN' in self.config.model.name:
                outcome_pred = self(H_t_copy)
            else:
                _, outcome_pred, _ = self(H_t_copy)
            
            # Create new tensors for each concatenation
            prev_outputs = torch.cat((prev_outputs, outcome_pred[:, -1, :].unsqueeze(1)), dim=1)
            prev_treatments = torch.cat((prev_treatments, a_seq[:, t, :].unsqueeze(1)), dim=1)
            active_entries = torch.cat((active_entries, torch.ones_like(active_entries[:, -1, :]).unsqueeze(1)), dim=1)
            if self.has_vitals:
                vitals = torch.cat((vitals, torch.zeros_like(vitals[:, -1, :]).unsqueeze(1)), dim=1)
            
            H_t_copy['prev_outputs'] = prev_outputs
            H_t_copy['prev_treatments'] = prev_treatments
            H_t_copy['active_entries'] = active_entries
            if self.has_vitals:
                H_t_copy['vitals'] = vitals
            
            if t < self.tau - 1:
                current_treatments = torch.cat((current_treatments, a_seq[:, t + 1, :].unsqueeze(1)), dim=1)
                H_t_copy['current_treatments'] = current_treatments
        last_outcome_pred = outcome_pred[:, -1, :]
        return last_outcome_pred

    def get_predictions_after_tau_steps_loss(self, H_t, Y_targets, a_seq, encoder=None):
        last_outcome_pred = self.get_predictions_after_tau_steps(H_t, Y_targets, a_seq, encoder=encoder)

        output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, a_seq.detach(), self.dataset_collection.train_scaling_params)
        ture_output = Y_targets[:, -1, :].detach().cpu().numpy()

        loss_1 = np.sqrt(((output_after_actions-ture_output) ** 2).mean())
        loss_2 = np.sqrt(((last_outcome_pred.detach().cpu().numpy()-ture_output) ** 2).mean())
        loss_3 = np.sqrt(((output_after_actions-last_outcome_pred.detach().cpu().numpy()) ** 2).mean())

        
        # Trim tensors back to original length
        # for key in ['prev_outputs', 'prev_treatments', 'active_entries', 'current_treatments']:
        #     H_t_copy[key] = H_t_copy[key][:, :length, :]

        Y_last = Y_targets[:, -1, :]
        loss = F.mse_loss(last_outcome_pred, Y_last)

        # print(f"loss_1: {loss_1}, loss_2: {loss_2}, loss_3: {loss_3}, loss: {loss}")
        # print(f"first one output_after_actions: {output_after_actions[0]}, ture_output: {ture_output[0]}")
        # print(f"*" * 50)

        return loss

    def get_loss_outcome_after_actions(self, H_t, Y_targets, a_seq):
        output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, a_seq, self.dataset_collection.train_scaling_params)
        ture_output = Y_targets[:, -1, :].detach().cpu().numpy()
        loss = np.sqrt(((output_after_actions-ture_output) ** 2).mean())
        loss = F.mse_loss(torch.tensor(output_after_actions, device=Y_targets.device), Y_targets[:, -1, :])
        return loss, output_after_actions

    def optimize_interventions_discrete(self, encoder=None, num_iterations=100, learning_rate=0.01, batch_size=64):
        model = self.config.model.name.split('/')[0]
        results = {}
        tau = self.tau
        for i in [2, 4, 6, 8]:
            print(f'start predicting results for tau={i} ...')
            self.tau = i
            self.config.exp.tau = i
            print(self.tau, self.config.exp.tau)
            infos = self.optimize_interventions_discrete_onetime(encoder=encoder)
            results[i] = infos
        final = {model:results}
        save_dir = f'case_infos/{self.config.exp.seed}/{self.config.exp.test}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + f'/case_infos_{model}.pkl', 'wb') as f:
            pickle.dump(final, f)

        write_str = []
        for key in results:
            rank = np.mean([results[key][i]['true_sequence_rank'] for i in range(len(results[key]))])
            write_str.append(f"{key}: Average rank of true sequences across all individuals: {rank}")
            corr = np.mean([results[key][i]['correlations']['model_true'] for i in range(len(results[key]))])
            write_str.append(f"{key}: Average corrs of true sequences across all individuals: {corr}")
        with open(save_dir + f'/case_infos_{model}.txt', 'w') as f:
            f.write('\n'.join(write_str))

        return final

    def optimize_interventions_discrete_onetime(self, k=100, encoder=None):
        set_seed(self.config['exp']['seed'])
        self.eval()
        try:
            if not self.config.exp.test:
                data = self.dataset_collection.val_f.data_original
            else:
                data = self.dataset_collection.test_f.data_original
            print('load data_original')
        except:
            if not self.config.exp.test:
                data = self.dataset_collection.val_f.data
            else:
                data = self.dataset_collection.test_f.data
        dataloader = get_dataloader(CIPDataset(data, self.config), batch_size=1, shuffle=False)
        
        device = "cuda"
        self.to(device)
        
        all_ranks = []
        all_correlation = [] 
        case_infos = []
        
        for (i, batch) in enumerate(dataloader):
            if i > 99:
                break
            print(f"-" * 50 + f"Individual {i}" + "-" * 50)
            H_t, targets = batch
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)
            Y_targets = targets['outputs']
            true_actions = targets['current_treatments']
            
            all_sequences = generate_perturbed_sequences(true_actions, k, self.tau, self.config.dataset.treatment_size, device, treatment_mode=self.config.dataset.treatment_mode)

            # is_duplicate = len(all_sequences) != len(set(map(tuple, all_sequences)))
            # print(f"存在重复: {is_duplicate}")
            # print(all_sequences.shape)
            # print(true_actions.shape)

            
            # 计算两种loss
            model_losses = []  
            true_losses = []   
            pred_losses = []
            
            with torch.no_grad():
                for seq in all_sequences:
                    model_loss = self.get_predictions_after_tau_steps_loss(H_t, Y_targets, seq, encoder=encoder)
                    model_losses.append(np.sqrt(model_loss.item()))
                    # print(f"seq:{seq}, loss:{model_losses[-1]}")
                    
                    output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(
                        H_t, seq, self.dataset_collection.train_scaling_params)
                    true_output = targets['outputs'][:, -1, :].detach().cpu().numpy()
                    true_loss = np.sqrt(((output_after_actions - true_output) ** 2).mean())
                    true_losses.append(true_loss)

                    model_output = self.get_predictions_after_tau_steps(H_t, Y_targets, seq, encoder=encoder).detach().cpu().numpy()
                    pred_loss = np.sqrt(((model_output - output_after_actions) ** 2).mean())
                    pred_losses.append(pred_loss)
            
            # 转换为numpy数组
            model_losses = np.array(model_losses)
            true_losses = np.array(true_losses)
            pred_losses = np.array(pred_losses)

            corr_model_true, _ = spearmanr(model_losses, true_losses)
            corr_model_pred, _ = spearmanr(model_losses, pred_losses)
            corr_pred_true, _ = spearmanr(pred_losses, true_losses)
            
            case_info = {
                'individual_id': i,
                'model_losses': model_losses,
                'true_losses': true_losses,
                'pred_losses': pred_losses,
                'correlations': {
                    'model_true': corr_model_true,
                    'model_pred': corr_model_pred,
                    'pred_true': corr_pred_true
                },
                'true_sequence': true_actions.cpu().numpy(),
                'true_sequence_rank': np.sum(model_losses < model_losses[-1]) + 1
            }
            case_infos.append(case_info)
            
            # if i == 1 or i == 2:
            # enhanced_analyze_case(case_info)
            
            correlation, p_value = spearmanr(model_losses, true_losses)
            all_correlation.append(correlation)
            
            # 计算真实序列的排名（基于模型loss）
            true_seq_rank = np.sum(model_losses < model_losses[-1]) + 1
            all_ranks.append(true_seq_rank)
            
            print(f"Individual {i} - True sequence rank: {true_seq_rank} out of {k}")
            print(f"Model Loss for true sequence: {model_losses[-1]}")
            print(f"True Loss for true sequence: {true_losses[-1]}")
            print(f"Rank correlation for this individual: {correlation:.3f} (p-value: {p_value:.3f})")
            
        # 计算平均结果
        avg_rank = sum(all_ranks) / len(all_ranks) / k * 100
        all_correlation = [c for c in all_correlation if not np.isnan(c)]
        
        if len(all_correlation) > 0:
            avg_correlation = sum(all_correlation) / len(all_correlation) / k * 100
        else:
            avg_correlation = 0
        print(f"Average rank of true sequences across all individuals: {avg_rank:.2f} out of {k}")
        print(f"Average rank correlation across all individuals: {avg_correlation:.3f}")
        
        return case_infos
class BRCausalModel(TimeVaryingCausalModel):
    """
    Abstract class for baselines, estimating counterfactual outcomes over time with balanced representations
    """

    model_type = None  # Will be defined in subclasses
    possible_model_types = None   # Will be defined in subclasses
    tuning_criterion = 'rmse'

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        # Balancing representation training parameters
        self.balancing = args.exp.balancing
        self.alpha = args.exp.alpha  # Used for gradient-reversal
        self.update_alpha = args.exp.update_alpha
        self.tau = args.exp.tau
        self.config = args

    def configure_optimizers(self):
        if self.balancing == 'grad_reverse' and not self.hparams.exp.weights_ema:  # one optimizer
            optimizer = self._get_optimizer(list(self.named_parameters()))

            if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
                return self._get_lr_schedulers(optimizer)

            return optimizer

        else:  # two optimizers - simultaneous gradient descent update
            treatment_head_params = \
                ['br_treatment_outcome_head.' + s for s in self.br_treatment_outcome_head.treatment_head_params]
            treatment_head_params = \
                [k for k in dict(self.named_parameters()) for param in treatment_head_params if k.startswith(param)]
            non_treatment_head_params = [k for k in dict(self.named_parameters()) if k not in treatment_head_params]

            assert len(treatment_head_params + non_treatment_head_params) == len(list(self.named_parameters()))

            treatment_head_params = [(k, v) for k, v in dict(self.named_parameters()).items() if k in treatment_head_params]
            non_treatment_head_params = [(k, v) for k, v in dict(self.named_parameters()).items()
                                         if k in non_treatment_head_params]

            if self.hparams.exp.weights_ema:
                self.ema_treatment = ExponentialMovingAverage([par[1] for par in treatment_head_params],
                                                              decay=self.hparams.exp.beta)
                self.ema_non_treatment = ExponentialMovingAverage([par[1] for par in non_treatment_head_params],
                                                                  decay=self.hparams.exp.beta)

            treatment_head_optimizer = self._get_optimizer(treatment_head_params)
            non_treatment_head_optimizer = self._get_optimizer(non_treatment_head_params)

            if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
                return self._get_lr_schedulers([non_treatment_head_optimizer, treatment_head_optimizer])

            return [non_treatment_head_optimizer, treatment_head_optimizer]

    def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer=None, optimizer_idx: int = None, *args,
                       **kwargs) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs)
        if self.hparams.exp.weights_ema and optimizer_idx == 0:
            self.ema_non_treatment.update()
        elif self.hparams.exp.weights_ema and optimizer_idx == 1:
            self.ema_treatment.update()

    def _calculate_bce_weights(self) -> None:
        if self.hparams.dataset.treatment_mode == 'multiclass':
            current_treatments = self.dataset_collection.train_f.data['current_treatments']
            current_treatments = current_treatments.reshape(-1, current_treatments.shape[-1])
            current_treatments = current_treatments[self.dataset_collection.train_f.data['active_entries'].flatten().astype(bool)]
            current_treatments = np.argmax(current_treatments, axis=1)

            self.bce_weights = len(current_treatments) / np.bincount(current_treatments) / len(np.bincount(current_treatments))
        else:
            raise NotImplementedError()

    def on_fit_start(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        for name, param in self.named_parameters():
            if param.dtype != torch.float32:
                print(f"Warning: Parameter {name} has dtype {param.dtype}")
                param.data = param.data.float()

        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = ['decoder'] if self.model_type == 'encoder' else ['encoder']

    def on_fit_end(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = ['encoder', 'decoder']

    def training_step(self, batch, batch_ind, optimizer_idx=0):
        for par in self.parameters():
            par.requires_grad = True

        # batch['active_entries'] = batch['active_entries'].float()
        # print(f"dtype of current_treatments: {batch['current_treatments'].dtype}")

        if optimizer_idx == 0:  # grad reversal or domain confusion representation update
            if self.hparams.exp.weights_ema:
                with self.ema_treatment.average_parameters():
                    treatment_pred, outcome_pred, _ = self(batch)
            else:
                treatment_pred, outcome_pred, _ = self(batch)

            mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
            if self.balancing == 'grad_reverse':
                bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'], kind='predict')
            elif self.balancing == 'domain_confusion':
                bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'], kind='confuse')
                bce_loss = self.br_treatment_outcome_head.alpha * bce_loss
            elif self.balancing == 'regression':
                bce_loss = self.cauculate_re_loss(batch, update_D=False)
                bce_loss = self.br_treatment_outcome_head.alpha * bce_loss
            else:
                raise NotImplementedError()

            # Masking for shorter sequences
            # Attention! Averaging across all the active entries (= sequence masks) for full batch
            if self.balancing != 'regression':
                bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
            mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()

            loss = bce_loss + mse_loss

            self.log(f'{self.model_type}_train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_alpha', self.br_treatment_outcome_head.alpha, on_epoch=True, on_step=False,
                     sync_dist=True)

            return loss

        elif optimizer_idx == 1:  # domain classifier update
            if self.hparams.exp.weights_ema:
                with self.ema_non_treatment.average_parameters():
                    treatment_pred, _, _ = self(batch, detach_treatment=True)
            else:
                treatment_pred, _, _ = self(batch, detach_treatment=True)

            if self.balancing == 'regression':
                bce_loss = self.cauculate_re_loss(batch, update_D=True)
            else:
                bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'], kind='predict')

            if self.balancing == 'domain_confusion':
                bce_loss = self.br_treatment_outcome_head.alpha * bce_loss
            
            
            # Masking for shorter sequences
            # Attention! Averaging across all the active entries (= sequence masks) for full batch
            if self.balancing != 'regression':
                bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()

            self.log(f'{self.model_type}_train_bce_loss_cl', bce_loss, on_epoch=True, on_step=False, sync_dist=True)

            return bce_loss

    def test_step(self, batch, batch_ind, **kwargs):
        # batch['active_entries'] = batch['active_entries'].float()
        # batch['current_treatments'] = batch['current_treatments'].float()

        if self.hparams.exp.weights_ema:
            with self.ema_non_treatment.average_parameters():
                with self.ema_treatment.average_parameters():
                    treatment_pred, outcome_pred, _ = self(batch)
        else:
            treatment_pred, outcome_pred, _ = self(batch)

        if self.balancing == 'grad_reverse':
            bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'], kind='predict')
        elif self.balancing == 'domain_confusion':
            bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'], kind='confuse')
        elif self.balancing == 'regression':
            bce_loss = self.cauculate_re_loss(batch, update_D=False)

        mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)

        # Masking for shorter sequences
        # Attention! Averaging across all the active entries (= sequence masks) for full batch
        if self.balancing != 'regression':
            bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
        mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()
        loss = bce_loss + mse_loss

        subset_name = self.test_dataloader().dataset.subset_name
        self.log(f'{self.model_type}_{subset_name}_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_{subset_name}_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_{subset_name}_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        """
        Generates normalised output predictions
        """
        # for key in batch:
        #     if isinstance(batch[key], torch.Tensor):
        #         batch[key] = batch[key].float()

        if self.hparams.exp.weights_ema:
            with self.ema_non_treatment.average_parameters():
                _, outcome_pred, br = self(batch)
        else:
            _, outcome_pred, br = self(batch)
        return outcome_pred.cpu(), br.cpu()

    def get_representations(self, dataset: Dataset) -> np.array:
        logger.info(f'Balanced representations inference for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        _, br = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return br.numpy()

    def get_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Predictions for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        outcome_pred, _ = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return outcome_pred.numpy()


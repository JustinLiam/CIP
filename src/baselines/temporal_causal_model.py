import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import logging
from typing import List
from thop import clever_format, profile
from fvcore.nn import FlopCountAnalysis
from src.baselines.utils_tcn import TemporalConvNet
from src.data.cip_dataset import CIPDataset, get_dataloader
from src.utils.utils import set_seed
from src.utils.helper_functions import generate_perturbed_sequences, enhanced_analyze_case
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
from scipy.stats import spearmanr
import pandas as pd
from scipy.stats import rankdata

class TemporalCausalInfModel(pl.LightningModule):
    def __init__(self, dataset_collection, config):
        super().__init__()
        self.dataset_collection = dataset_collection
        self.config = config
        self.init_params()
        self.init_model()
        self.init_ema()
        self.count_flops_processed = False
        self.automatic_optimization = False
        self.save_hyperparameters('config')
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_params(self):
        self.init_exp_params()
        self.init_dataset_params()
        self.init_model_params()

    def init_dataset_params(self):
        self.max_seq_length = self.config['dataset']['max_seq_length']
        self.treatment_size = self.config['dataset']['treatment_size']
        self.one_hot_treatment_size = self.config['dataset']['one_hot_treatment_size']
        self.static_size = self.config['dataset']['static_size']
        self.output_size = self.config['dataset']['output_size']
        self.input_size = self.config['dataset']['input_size']
        self.treatment_mode = self.config['dataset']['treatment_mode']
        self.autoregressive = self.config['dataset']['autoregressive']
        self.val_batch_size = self.config['dataset']['val_batch_size']
        self.projection_horizon = self.config['dataset']['projection_horizon']
        self.predict_X = self.config['dataset']['predict_X']
        
    def init_exp_params(self):
        self.lr = self.config['exp']['lr']
        self.lr_D = self.config['exp']['lr_D']
        self.weight_decay = self.config['exp']['weight_decay']
        self.weight_decay_D = self.config['exp']['weight_decay_D']
        self.patience = self.config['exp']['sch_patience']
        self.patience_D = self.config['exp']['sch_patience_D']
        if 'lr_X' in self.config['exp']:
            self.lr_X = self.config['exp']['lr_X']
            self.weight_decay_X = self.config['exp']['weight_decay_X']

        self.factor = self.config['exp']['factor']
        self.batch_size = self.config['exp']['batch_size']
        self.dropout = self.config['exp']['dropout']
        self.cooldown = self.config['exp']['cooldown']
        self.weights_ema = self.config['exp']['weights_ema']
        self.beta = self.config['exp']['beta']
        self.update_lambda_D = self.config['exp']['update_lambda_D']
        self.lambda_D = self.config['exp']['lambda_D'] if not self.update_lambda_D else 0.0
        self.lambda_D_max = self.config['exp']['lambda_D']
        self.lambda_X = self.config['exp']['lambda_X']
        self.lambda_Y = self.config['exp']['lambda_Y']
        self.loss_type_X = self.config['exp']['loss_type_X']
        self.epochs = self.config['exp']['epochs']

        self.tau = self.config['exp']['tau']

        try:
            self.std = self.dataset_collection.train_scaling_params[1]['cancer_volume']
        except:
            self.std = 1

    def init_model_params(self):
        pass

    def init_model_params_(self):
        self.transpose = self.config['model']['transpose']
        if self.transpose:
            self.transpose_size = self.config['model']['transpose_size']
        self.num_blocks = self.config['model']['num_blocks']
        # check is self.num_blocks equals to 1 or 2
        if self.num_blocks not in [1, 2]:
            raise ValueError('num_blocks should be 1 or 2')

        self.first_net = self.config['model']['first_net']
        # init parameters for the first net
        if self.first_net == 'lstm':
            self.hidden_size = self.config['model']['hidden_size']
            self.num_layers = self.config['model']['num_layers']
            # self.hidden_net = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size_1, num_layers=self.num_layers_1, batch_first=True)
        elif self.first_net == 'tcn':
            self.num_channels_hidden = self.config['model']['num_channels_hidden']
            self.kernel_size = self.config['model']['kernel_size']
            # self.hidden_net = TemporalConvNet(self.input_size, self.num_channels_hidden, self.kernel_size)
        else:
            raise ValueError('first_net should be one of lstm and tcn')
        self.br_size = self.config['model']['br_size']
        self.recursive = self.config['model']['recursive']
        # init parameters for the second net if self.num_blocks == 2
        if self.num_blocks == 2:
            self.init_second_net_params_()
        # init parameters for the G_y net to predict Y
        self.hiddens_G_y = self.config['model']['hiddens_G_y']
        # init parameters for the G_x net to predict X
        if self.predict_X:
            self.hiddens_G_x = self.config['model']['hiddens_G_x']
        self.ema_y = self.config['model']['ema_y']
        self.init = self.config['model']['init']

    def init_second_net_params_(self):
        self.second_net = self.config['model']['second_net']
        if self.second_net == 'lstm':
            if self.predict_X:
                self.hidden_size_x = self.config['model']['hidden_size_x']
                self.num_layers_x = self.config['model']['num_layers_x']
            self.hidden_size_y = self.config['model']['hidden_size_y']
            self.num_layers_y = self.config['model']['num_layers_y']
        elif self.second_net == 'tcn':
            if self.predict_X:
                self.num_channels_hidden_x = self.config['model']['num_channels_hidden_x']
                self.kernel_size_x = self.config['model']['kernel_size_x']
            self.num_channels_hidden_y = self.config['model']['num_channels_hidden_y']
            self.kernel_size_y = self.config['model']['kernel_size_y']
        else:
            raise ValueError('second_net should be one of lstm and tcn')

    def init_model(self):
        pass

    def init_model_(self):
        if self.static_size > 0:
            input_size = self.input_size + self.static_size + self.treatment_size
        if self.autoregressive:
            # we need to use the previous output as the input
            input_size += self.output_size
        # init the transpose net to transpose the input if needed
        if self.transpose:
            self.transpose_net = nn.Sequential()
            self.transpose_net.add_module('linear1', nn.Linear(input_size, self.transpose_size))
            # self.transpose_net.add_module('elu1', nn.ELU())
            input_size = self.transpose_size
        else:
            self.transpose_net = nn.Identity()

        # init the first net
        if self.first_net == 'lstm':
            self.hidden_net = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
            input_size = self.hidden_size
        elif self.first_net == 'tcn':
            self.hidden_net = TemporalConvNet(input_size, self.num_channels_hidden, self.kernel_size, dropout=self.dropout, init=self.init)
            input_size = self.num_channels_hidden[-1]
        else:
            raise ValueError('first_net should be one of lstm and tcn')
        # init the G_br net to learn the balancing representation
        self.G_br = nn.Sequential()
        self.G_br.add_module('linear1', nn.Linear(input_size, self.br_size))
        if self.config['model']['br_act']:
            self.G_br.add_module('elu1', nn.ELU())
        # init the second net if self.num_blocks == 2
        if self.num_blocks == 2:
            input_size_x, input_size_y = self.init_second_net_()
        else:
            input_size_x, input_size_y = self.br_size + self.treatment_size, self.br_size + self.treatment_size

        # init the G_y net to predict Y
        self.G_y = nn.Sequential()
        for i in range(len(self.hiddens_G_y)):
            if i == 0:
                self.G_y.add_module('fc{}'.format(i), nn.Linear(input_size_y, self.hiddens_G_y[i]))
            else:
                self.G_y.add_module('elu{}'.format(i), nn.ELU())
                self.G_y.add_module('fc{}'.format(i), nn.Linear(self.hiddens_G_y[i-1], self.hiddens_G_y[i]))
        self.G_y.add_module('elu{}'.format(len(self.hiddens_G_y)), nn.ELU())
        self.G_y.add_module('fc{}'.format(len(self.hiddens_G_y)), nn.Linear(self.hiddens_G_y[-1], self.output_size))
        # init the G_x net to predict X if needed
        if self.predict_X:
            self.G_x = nn.Sequential()
            for i in range(len(self.hiddens_G_x)):
                if i == 0:
                    self.G_x.add_module('fc{}'.format(i), nn.Linear(input_size_x, self.hiddens_G_x[i]))
                else:
                    self.G_x.add_module('elu{}'.format(i), nn.ELU())
                    self.G_x.add_module('fc{}'.format(i), nn.Linear(self.hiddens_G_x[i-1], self.hiddens_G_x[i]))
            self.G_x.add_module('elu{}'.format(len(self.hiddens_G_x)), nn.ELU())
            self.G_x.add_module('fc{}'.format(len(self.hiddens_G_x)), nn.Linear(self.hiddens_G_x[-1], self.input_size))
        else:
            self.G_x = nn.Identity()
        # init the ema_net_x to predict X if needed
        # the ema_net_x will be used to cauculate the ema of the predicted X
        if self.predict_X:
            self.ema_net_x = nn.Sequential()
            if 'hiddens_ema' in self.config['model']:
                hiddens_ema = self.config['model']['hiddens_ema']
                for i in range(len(hiddens_ema)):
                    if i == 0:
                        self.ema_net_x.add_module('fc{}'.format(i), nn.Linear(input_size_x, hiddens_ema[i]))
                    else:
                        self.ema_net_x.add_module('elu{}'.format(i), nn.ELU())
                        self.ema_net_x.add_module('fc{}'.format(i), nn.Linear(hiddens_ema[i-1], hiddens_ema[i]))
                self.ema_net_x.add_module('elu{}'.format(len(hiddens_ema)), nn.ELU())
                self.ema_net_x.add_module('fc{}'.format(len(hiddens_ema)), nn.Linear(hiddens_ema[-1], self.input_size))
                self.ema_net_x.add_module('sigmoid{}'.format(1), nn.Sigmoid())
            else:
                self.ema_net_x.add_module('fc{}'.format(1), nn.Linear(input_size_x, self.input_size))
                self.ema_net_x.add_module('sigmoid{}'.format(1), nn.Sigmoid())
        else:
            self.ema_net_x = nn.Identity()

        if self.config['model']['ema_y']:
            self.ema_net_y = nn.Sequential()
            self.ema_net_y.add_module('fc{}'.format(1), nn.Linear(input_size_y, self.output_size))
            self.ema_net_y.add_module('sigmoid{}'.format(1), nn.Sigmoid())
        else:
            self.ema_net_y = nn.Identity()

    def init_second_net_(self):
        # check self.recursive, if True, we will use the current A as the input of the second net, otherwise we will use the current A as the input of G_x and G_y
        if self.recursive:
            input_size = self.br_size + self.treatment_size 
        else:
            input_size = self.br_size
        input_size_x = 0
        if self.second_net == 'lstm':
            if self.predict_X:
                self.hidden_net_x = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size_x, num_layers=self.num_layers_x, batch_first=True)
                input_size_x = self.hidden_size_x
            else:
                self.hidden_net_x = nn.Identity()
            self.hidden_net_y = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size_y, num_layers=self.num_layers_y, batch_first=True)
            input_size_y = self.hidden_size_y
        elif self.second_net == 'tcn':
            if self.predict_X:
                if 'dropout_x' in self.config['exp']:
                    dropout = self.config['exp']['dropout_x']
                else:
                    dropout = self.dropout
                self.hidden_net_x = TemporalConvNet(input_size, self.num_channels_hidden_x, self.kernel_size_x, dropout=dropout, init=self.init)
                input_size_x = self.num_channels_hidden_x[-1]
            else:
                self.hidden_net_x = nn.Identity()
            self.hidden_net_y = TemporalConvNet(input_size, self.num_channels_hidden_y, self.kernel_size_y, dropout=self.dropout, init=self.init)
            input_size_y = self.num_channels_hidden_y[-1]
        else:
            raise ValueError('second_net should be one of lstm and tcn')

        if not self.recursive:
            if self.predict_X:
                input_size_x += self.treatment_size 
            input_size_y += self.treatment_size 
        return input_size_x, input_size_y

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_collection.train_f, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_collection.val_f, batch_size=self.val_batch_size)

    def count_flops(self, batch):
        flops = FlopCountAnalysis(self, batch)
        # print('FLOPs:', flops.total())
        mflops = flops.total() / 1e6 / self.batch_size
        print('FLOPs: {:.2f} MFLOPs'.format(mflops))
        for name, module_flops in flops.by_module().items():
            module_flops_per_sample = module_flops / 1e6 / self.batch_size
            # print(f'{name} : {module_flops_per_sample:.2f} MFLOPs')
        params = sum(p.numel() for p in self.parameters())
        print('Parameters:', params)
        return params, mflops

    def forward(self, x):
        # you should implement the forward pass specifically for your model.
        # this returns the concatenation of y_hat and x_hat
        pass

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward, but iis dependent on the strategy chosen for balancing.
        pass

    def validation_step(self, batch, batch_idx):
        # validation_step defined the train loop.
        # It is independent of forward, but iis dependent on the strategy chosen for balancing.
        pass

    def get_mse_at_follow_up_time(self, prediction, output, active_entries=None):
        # cauculate mse at follow up time
        mses = torch.sum(torch.sum((prediction - output) ** 2 * active_entries, dim=0), dim=-1) / torch.sum(torch.sum(active_entries, dim=0), dim=-1)
        return mses

    def get_mse_all(self, prediction, output, active_entries=None):
        mses = torch.sum((prediction - output) ** 2 * active_entries) / torch.sum(active_entries)
        return mses

    def get_l1_all(self, prediction, output, active_entries=None):
        l1 = torch.sum(torch.abs(prediction - output) * active_entries) / torch.sum(active_entries)
        return l1

    def get_predictions(self, dataset: Dataset, logger=None) -> np.array:
        if logger is not None:
            logger.info(f'Predictions for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.hparams['config']['dataset']['val_batch_size'], shuffle=False)
        outcome_pred, next_covariates_pred = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return outcome_pred.numpy(), next_covariates_pred.numpy()

    def get_normalised_masked_rmse(self, dataset: Dataset, one_step_counterfactual=False, logger=None):
        # adapted from https://github.com/Valentyn1997/CausalTransformer
        if logger is not None:
            logger.info(f'RMSE calculation for {dataset.subset_name}.')
        outputs_scaled, _ = self.get_predictions(dataset, logger=logger)
        
        unscale = self.hparams['config']['exp']['unscale_rmse']
        percentage = self.hparams['config']['exp']['percentage_rmse']

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
            last_entries = dataset.data['active_entries'] - np.concatenate([dataset.data['active_entries'][:, 1:, :], np.zeros((num_samples, 1, output_dim))], axis=1)
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

    def get_normalised_n_step_rmses(self, dataset: Dataset, datasets_mc: List[Dataset] = None, logger=None):
        # adapted from https://github.com/Valentyn1997/CausalTransformer
        if logger is not None:
            logger.info(f'RMSE calculation for {dataset.subset_name}.')
        assert hasattr(dataset, 'data_processed_seq')

        unscale = self.hparams['config']['exp']['unscale_rmse']
        percentage = self.hparams['config']['exp']['percentage_rmse']
        outputs_scaled = self.get_autoregressive_predictions(dataset if datasets_mc is None else datasets_mc, logger=logger)

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

    def get_autoregressive_predictions(self, dataset: Dataset, logger=None) -> np.array:
        # adapted from https://github.com/Valentyn1997/CausalTransformer
        if logger is not None:
            logger.info(f'Autoregressive Prediction for {dataset.subset_name}.')
        projection_horizon = self.hparams['config']['dataset']['projection_horizon']

        predicted_outputs = np.zeros((len(dataset), projection_horizon, self.output_size))

        for t in range(projection_horizon + 1):
            if logger is not None:
                logger.info(f't = {t + 1}')
            outputs_scaled, next_covariates_pred = self.get_predictions(dataset)

            for i in range(len(dataset)):
                split = int(dataset.data['future_past_split'][i])
                if t < projection_horizon:
                    if self.predict_X:
                        # replace the covariates in next step with the predicted covariates
                        dataset.data['vitals'][i, split + t, :] = next_covariates_pred[i, split - 1 + t, :]
                    dataset.data['prev_outputs'][i, split + t, :] = outputs_scaled[i, split - 1 + t, :]
                    pass

                if t > 0:
                    predicted_outputs[i, t - 1, :] = outputs_scaled[i, split - 1 + t, :]

        return predicted_outputs

    def optimize_interventions(self, num_iterations=100, learning_rate=0.01, batch_size=64):
        results = ['\n' + '-' * 50]
        tau = self.tau
        for i in range(1, 13):
            print(f'start predicting results for tau={i} ...')
            self.tau = i
            self.config.exp.tau = i
            print(self.tau, self.config.exp.tau)
            _, loss = self.optimize_interventions_onetime(num_iterations=num_iterations, learning_rate=learning_rate, batch_size=batch_size)
            results.append(f"Optimized interventions for tau={i}, {num_iterations} iterations, lr={learning_rate}: {loss}")
        return '\n'.join(results)

    def optimize_interventions_onetime(self, num_iterations=100, learning_rate=0.01, batch_size=64):
        set_seed(self.config['exp']['seed'])
        # try:
        #     data = self.dataset_collection.test_f.data_original
        #     data = self.dataset_collection.val_f.data_original
        # except:
        #     data = self.dataset_collection.test_f.data
        #     data = self.dataset_collection.val_f.data
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
                loss = self.get_predictions_after_tau_steps_loss(H_t, Y_targets, a_seq_sigmoid)
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

    def get_predictions_after_tau_steps(self, H_t, Y_targets, a_seq):
        # Deep copy H_t to ensure complete independence
        H_t_copy = {k: v.clone().detach().float() for k, v in H_t.items()}

        length = H_t_copy['current_treatments'].shape[1]
        # Create new tensor for modifications
        current_treatments = H_t_copy['current_treatments'].clone()
        current_treatments[:, -1, :] = a_seq[:, 0, :]
        H_t_copy['current_treatments'] = current_treatments
        
        prev_outputs = H_t_copy['prev_outputs']
        prev_treatments = H_t_copy['prev_treatments']
        active_entries = H_t_copy['active_entries']
        if self.predict_X:
            vitals = H_t_copy['vitals']
        
        for t in range(self.tau):
            x_y_pred = self(H_t_copy)
            outcome_pred = x_y_pred[:,:,:self.output_size]
            
            # Create new tensors for each concatenation
            prev_outputs = torch.cat((prev_outputs, outcome_pred[:, -1, :].unsqueeze(1)), dim=1)
            prev_treatments = torch.cat((prev_treatments, a_seq[:, t, :].unsqueeze(1)), dim=1)
            active_entries = torch.cat((active_entries, torch.ones_like(active_entries[:, -1, :]).unsqueeze(1)), dim=1)
            if self.predict_X:
                x_pred = x_y_pred[:,:,self.output_size:]
                vitals = torch.cat((vitals, x_pred[:, -1, :].unsqueeze(1)), dim=1)
            
            H_t_copy['prev_outputs'] = prev_outputs
            H_t_copy['prev_treatments'] = prev_treatments
            H_t_copy['active_entries'] = active_entries
            H_t_copy['static_features'] = torch.cat((H_t_copy['static_features'], H_t_copy['static_features'][:, 0, :].unsqueeze(1)), dim=1)
            if self.predict_X:
                H_t_copy['vitals'] = vitals
            
            if t < self.tau - 1:
                current_treatments = torch.cat((current_treatments, a_seq[:, t + 1, :].unsqueeze(1)), dim=1)
                H_t_copy['current_treatments'] = current_treatments
        
        last_outcome_pred = outcome_pred[:, -1, :]
        return last_outcome_pred
    
    def get_predictions_after_tau_steps_loss(self, H_t, Y_targets, a_seq):
        last_outcome_pred = self.get_predictions_after_tau_steps(H_t, Y_targets, a_seq)
        output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, a_seq.detach(), self.dataset_collection.train_scaling_params)
        ture_output = Y_targets[:, -1, :].detach().cpu().numpy()

        loss_1 = np.sqrt(((output_after_actions-ture_output) ** 2).mean())
        loss_2 = np.sqrt(((last_outcome_pred.detach().cpu().numpy()-ture_output) ** 2).mean())
        loss_3 = np.sqrt(((output_after_actions-last_outcome_pred.detach().cpu().numpy()) ** 2).mean())

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

    def optimize_interventions_discrete(self, num_iterations=100, learning_rate=0.01, batch_size=64):
        model = self.config.model.name.split('/')[0]
        results = {}
        tau = self.tau
        for i in [2, 4, 6, 8]:
            print(f'start predicting results for tau={i} ...')
            self.tau = i
            self.config.exp.tau = i
            print(self.tau, self.config.exp.tau)
            infos = self.optimize_interventions_discrete_onetime()
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
        with open(save_dir + f'/case_infos_{model}.txt', 'w') as f:
            f.write('\n'.join(write_str))

        return final

    def optimize_interventions_discrete_onetime(self, k=100):
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
        all_correlation = []  # 存储每个个体的排序相关性
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

            
            model_losses = []  
            true_losses = []   
            pred_losses = []
            
            with torch.no_grad():
                for seq in all_sequences:
                    model_loss = self.get_predictions_after_tau_steps_loss(H_t, Y_targets, seq)
                    model_losses.append(np.sqrt(model_loss.item()))
                    # print(f"seq:{seq}, loss:{model_losses[-1]}")
                    
                    output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(
                        H_t, seq, self.dataset_collection.train_scaling_params)
                    true_output = targets['outputs'][:, -1, :].detach().cpu().numpy()
                    true_loss = np.sqrt(((output_after_actions - true_output) ** 2).mean())
                    true_losses.append(true_loss)

                    model_output = self.get_predictions_after_tau_steps(H_t, Y_targets, seq).detach().cpu().numpy()
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

            # 使用scipy的spearmanr计算相关系数
            
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


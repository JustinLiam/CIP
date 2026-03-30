import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy

class AuxiliaryModel(pl.LightningModule):
    def __init__(self, config, dataset_collection):
        super().__init__()
        self.config = config
        self.dataset_collection = dataset_collection
        self.init_params()
        self.init_model()
        self.automatic_optimization = False
        # self.init_ema()
    
    def init_ema(self):
        if self.weights_ema:
            parameters = [par for par in self.parameters()]
            self.ema = ExponentialMovingAverage([par for par in parameters], decay=self.beta)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_params(self):
        self.init_dataset_params()
        self.init_exp_params()
        self.init_model_params()
    
    def init_dataset_params(self):
        self.max_seq_length = self.config['dataset']['max_seq_length']
        self.treatment_size = self.config['dataset']['treatment_size']
        self.static_size = self.config['dataset']['static_size']
        self.output_size = self.config['dataset']['output_size']
        self.input_size = self.config['dataset']['input_size']
        self.autoregressive = self.config['dataset']['autoregressive']
        self.val_batch_size = self.config['dataset']['val_batch_size']
        self.projection_horizon = self.config['dataset']['projection_horizon']
        self.predict_X = self.config['dataset']['predict_X']

    def init_exp_params(self):
        self.lr = self.config['exp']['lr']
        self.weight_decay = self.config['exp']['weight_decay']
        self.batch_size = self.config['exp']['batch_size']
        self.dropout = self.config['exp']['dropout']
        self.weights_ema = self.config['exp']['weights_ema']
        self.beta = self.config['exp']['beta']
        self.lambda_X = self.config['exp']['lambda_X']
        self.lambda_Y = self.config['exp']['lambda_Y']
        self.epochs = self.config['exp']['epochs']
    
    def init_model_params(self):
        self.hidden_dim = self.config['model']['auxiliary']['hidden_dim']
        self.num_layers = self.config['model']['auxiliary']['num_layers']
        self.hiddens_G_y = self.config['model']['auxiliary']['hiddens_G_y']
        self.hiddens_G_x = self.config['model']['auxiliary']['hiddens_G_x']

    def init_model(self):
        if self.static_size > 0:
            input_size = self.input_size + self.static_size + self.treatment_size
        else:
            input_size = self.input_size + self.treatment_size

        if self.autoregressive:
            input_size += self.output_size
        self.encoder = nn.LSTM(input_size, self.hidden_dim, self.num_layers, dropout=self.dropout, batch_first=True)

        # init the G_y net to predict Y
        input_size = self.hidden_dim + self.treatment_size
        self.G_y = nn.Sequential()
        if -1 not in self.hiddens_G_y:
            for i in range(len(self.hiddens_G_y)):
                if i == 0:
                    self.G_y.add_module('fc{}'.format(i), nn.Linear(input_size, self.hiddens_G_y[i]))
                else:
                    self.G_y.add_module('elu{}'.format(i), nn.ELU())
                    self.G_y.add_module('fc{}'.format(i), nn.Linear(self.hiddens_G_y[i-1], self.hiddens_G_y[i]))
            self.G_y.add_module('elu{}'.format(len(self.hiddens_G_y)), nn.ELU())
            self.G_y.add_module('fc{}'.format(len(self.hiddens_G_y)), nn.Linear(self.hiddens_G_y[-1], self.output_size))
        else:
            self.G_y.add_module('fc{}'.format(1), nn.Linear(input_size, self.output_size))
        
        # init the G_x net to predict X
        if self.predict_X:
            self.G_x = nn.Sequential()
            input_size = self.hidden_dim + self.treatment_size
            if -1 not in self.hiddens_G_x:
                for i in range(len(self.hiddens_G_x)):
                    if i == 0:
                        self.G_x.add_module('fc{}'.format(i), nn.Linear(input_size, self.hiddens_G_x[i]))
                    else:
                        self.G_x.add_module('elu{}'.format(i), nn.ELU())
                        self.G_x.add_module('fc{}'.format(i), nn.Linear(self.hiddens_G_x[i-1], self.hiddens_G_x[i]))
                self.G_x.add_module('elu{}'.format(len(self.hiddens_G_x)), nn.ELU())
                self.G_x.add_module('fc{}'.format(len(self.hiddens_G_x)), nn.Linear(self.hiddens_G_x[-1], self.input_size))
            else:
                self.G_x.add_module('fc{}'.format(1), nn.Linear(input_size, self.input_size))
        
        if self.predict_X:
            input_size = self.hidden_dim + self.treatment_size
            self.ema_net_x = nn.Sequential()
            self.ema_net_x.add_module('fc{}'.format(1), nn.Linear(input_size, self.input_size))
            self.ema_net_x.add_module('sigmoid{}'.format(1), nn.Sigmoid())
        else:
            self.ema_net_x = nn.Identity()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_collection.train_f, shuffle=True, batch_size=self.batch_size)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_collection.val_f, batch_size=self.val_batch_size)

    def build_representations(self, batch, only_last=False):
        # build representations of the history H_t for all t
        # if only_last is True, return only the last representation
        if self.static_size > 0:
            if self.predict_X:
                x = batch['vitals']
                x = torch.cat((x, batch['static_features']), dim=-1)
            # when we don't predict x, we use static features as the current_covariates
            else:
                # print(type(batch))
                x = batch['static_features']
        # if we use autoregressive, we need to use the previous output as the input
        
        if self.autoregressive:
            prev_outputs = batch['prev_outputs']
            x = torch.cat((x, prev_outputs), dim=-1)
        
        previous_treatments = batch['prev_treatments']
        x = torch.cat((x, previous_treatments), dim=-1) # (batch_size, seq_length, input_size)

        representations, _ = self.encoder(x)
        if only_last:
            return representations[:, -1, :]
        return representations
        
    def forward(self, batch):
        # build representations of the history H_t for all t
        representations = self.build_representations(batch)
        # build the treatments
        treatments = batch['current_treatments']
        # concatenate the representations with the treatments
        input = torch.cat((representations, treatments), dim=-1)
        # predict the output
        y_hat = self.G_y(input)
        n, T, _ = input.shape
        x_hat = torch.zeros(n, T, self.input_size).to(self.device)
        if self.predict_X:
            x_hat = self.G_x(input)
            ema_x = self.ema_net_x(input)
            x_hat = ema_x * batch['vitals'] + (1 - ema_x) * x_hat
        return torch.cat((y_hat, x_hat), dim=-1)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        active_entries = batch['active_entries'] if 'active_entries' in batch else torch.ones_like(output)
        y_x_hat = self(batch)
        y_hat, x_hat = y_x_hat[:, :, :self.output_size], y_x_hat[:, :, self.output_size:]

        output = batch['outputs']
        loss_y = self.get_mse_all(y_hat, output, active_entries)
        if self.predict_X:
            next_covariates = batch['next_vitals']
            x_hat = x_hat[:, :next_covariates.shape[1], :]
            loss_x = self.get_mse_all(x_hat, next_covariates, active_entries[:, 1:])
        else:
            loss_x = 0
        loss = self.lambda_Y * loss_y + self.lambda_X * loss_x 
        self.manual_backward(loss)
        
        optimizer.step()
        optimizer.zero_grad()
        self.log('train_loss', loss, on_epoch=True)
        return {'loss': loss, 'loss_x': loss_x}
    
    def validation_step(self, batch, batch_idx):
        y_x_hat = self(batch)
        y_hat, x_hat = y_x_hat[:, :, :self.output_size], y_x_hat[:, :, self.output_size:]
        output = batch['outputs']
        active_entries = batch['active_entries'] if 'active_entries' in batch else torch.ones_like(output)
        loss_y = self.get_mse_all(y_hat, output, active_entries)
        if self.predict_X:
            next_covariates = batch['next_vitals']
            x_hat = x_hat[:, :next_covariates.shape[1], :]
            loss_x = self.get_mse_all(x_hat, next_covariates, active_entries[:, 1:])
        else:
            loss_x = 0
        loss = self.lambda_Y * loss_y + self.lambda_X * loss_x
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_loss_x', loss_x, on_epoch=True)
        self.log('val_loss_y', loss_y, on_epoch=True)
        
        return {'val_loss': loss, 'val_loss_x': loss_x, 'val_loss_y': loss_y}

    def get_mse_all(self, prediction, output, active_entries=None, w=None):
        if w is not None:
            # active_entries to float
            mses = torch.sum((prediction - output) ** 2 * active_entries * w) / torch.sum(active_entries * w)
        else:
            mses = torch.sum((prediction - output) ** 2 * active_entries) / torch.sum(active_entries)
        return mses

    def get_predictions(self, batch, actions):
        # print(f'prev_outputs shape: {batch["prev_outputs"].shape}')
        
        history_length = batch['outputs'].shape[1]
        action_length = actions.shape[1]
        predictions = torch.zeros(actions.shape[0], action_length, self.output_size).to(self.device)
        for i in range(action_length):
            action = actions[:, i, :]
            if i == 0:
                # replace the last treatment with the new action
                batch['current_treatments'][:, -1, :] = action
            else:
                # append the new action to the treatments
                batch['current_treatments'] = torch.cat((batch['current_treatments'], action.unsqueeze(1)), dim=1) # (batch_size, history_length + i, treatment_size)

            y_x_hat = self(batch)
            y_hat = y_x_hat[:, :, :self.output_size]
            predictions[:, i, :] = y_hat[:, -1, :]
            # update other attributes
            batch['prev_outputs'] = torch.cat((batch['prev_outputs'], y_hat[:, -1, :].unsqueeze(1)), dim=1)
            if self.static_size > 0:
                batch['static_features'] = torch.cat((batch['static_features'], batch['static_features'][:, -1, :].unsqueeze(1)), dim=1)
            if self.predict_X:
                x_hat = y_x_hat[:, :, self.output_size:]
                batch['vitals'] = torch.cat((batch['vitals'], x_hat[:, -1, :].unsqueeze(1)), dim=1)
            batch['prev_treatments'] = torch.cat((batch['prev_treatments'], action.unsqueeze(1)), dim=1)

        # reset batch
        for key in ['prev_treatments', 'prev_outputs', 'current_treatments', 'static_features', 'vitals']:
            if key in batch:
                batch[key] = batch[key][:, :history_length, :]

        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
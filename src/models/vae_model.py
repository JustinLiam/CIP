import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np  
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.auxiliary_model import AuxiliaryModel
from src.models.generative_model import GenerativeModel
from src.models.inference_model import InferenceModel
from src.utils.helper_functions import compute_kl_divergence
from src.data.cip_dataset import CIPDataset, get_dataloader
from src.utils.utils import set_seed
from scipy.stats import spearmanr
# from scipy.stats import spearmanr
from scipy import stats
from src.utils.helper_functions import generate_perturbed_sequences
import os
import pickle

class VAEModel(pl.LightningModule):
    def __init__(self, config, dataset_collection):
        super(VAEModel, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.dataset_collection = dataset_collection
        self.generative_model = GenerativeModel(config)
        self.inference_model = InferenceModel(config)
        self.auxiliary_model = AuxiliaryModel(config, dataset_collection)
        
        self.lr = config['model']['lr']
        self.tau = config['exp']['tau']  # Planning Horizon
        self.a_dim = config['dataset']['treatment_size']  # 这个值为2，表示action的维度是 2，即代表有“两个并发的干预措施”
        self.predict_X = config['dataset']['predict_X']
        # self.init_AuxiliaryModel(config)
        self.automatic_optimization = False  # 手动控制pytorch-lightning 的优化步骤
        # 下面两个是充当临时容器，用于收集每一个 Batch（批次）训练或验证产生的中间结果（在这里特指 reg_loss），以便在当前 Epoch 结束时计算整体的平均值
        self.validation_step_outputs = []
        self.train_step_outputs = []
        # print(self.dataset_collection.train_scaling_params['output_stds'])
        try:
            self.std = self.dataset_collection.train_scaling_params[1]['cancer_volume']  # 将无量纲的归一化误差，转换回真实的肿瘤体积单位，以便在论文中报告真实的 Target Distance 或 RMSE
        except:
            self.std = 1
    
    def init_AuxiliaryModel(self, config):
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['exp']['patience'],
            mode='min')
        logger_board = TensorBoardLogger(
            save_dir=config['exp']['current_dir'], 
            name='', 
            version='')
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=None,
            monitor='val_loss',
            filename='checkpoint-{epoch:02d}-{val_loss:.3f}',
            save_top_k=3,
            mode='min')
        trainer = pl.Trainer(
            logger=logger_board,
            max_epochs=config['exp']['epochs'],
            enable_progress_bar=False,
            enable_model_summary=False, 
            devices=config['exp']['gpus'],
            callbacks=[early_stop_callback, checkpoint_callback])
        trainer.fit(self.auxiliary_model)

    def train_dataloader(self) -> DataLoader:
        return get_dataloader(CIPDataset(self.dataset_collection.train_f.data, self.config, train=True), batch_size=self.config['exp']['batch_size'], shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return get_dataloader(CIPDataset(self.dataset_collection.val_f.data, self.config), batch_size=self.config['exp']['batch_size'], shuffle=False)

    def forward(self, H_t, Y_targets, a_seq, X_targets=None):
        elbo, reg_loss, _ = self.calculate_elbo(H_t, Y_targets, a_seq, X_targets=X_targets, num_samples=self.config.exp.num_samples, optimize_a=False)
        return elbo, reg_loss

    def training_step(self, batch, batch_idx):
        self.train()
        
        H_t, targets = batch
        Y_targets = targets['outputs']
        X_targets = None if not self.predict_X else targets['vitals']
        a_seq = targets['current_treatments']

        opt = self.optimizers()
        opt.zero_grad()

        elbo, reg_loss = self(H_t, Y_targets, a_seq, X_targets=X_targets)
        self.manual_backward(elbo)
        opt.step()

        self.log('train_loss', elbo)
        self.train_step_outputs.append(reg_loss)
        return elbo

    def validation_step(self, batch, batch_idx):
        H_t, targets = batch
        Y_targets = targets['outputs']
        X_targets = None if not self.predict_X else targets['vitals']
        a_seq = targets['current_treatments']
        elbo, reg_loss = self(H_t, Y_targets, a_seq, X_targets=X_targets)
        self.log('val_loss', elbo)
        self.log('val_reg_loss', reg_loss)
        self.validation_step_outputs.append(reg_loss)
        return reg_loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x for x in self.validation_step_outputs]).mean()
        print(f"val_reg_loss loss: {avg_loss.item()}")
        self.validation_step_outputs.clear()
        return

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x for x in self.train_step_outputs]).mean()
        print(f"train_reg_loss loss: {avg_loss.item()}")
        self.train_step_outputs.clear()
        return

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def configure_optimizers(self):
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"Warning: {name} does not require gradients")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.config['exp']['weight_decay'])
        param_groups = optimizer.param_groups[0]['params']
        all_params = set(self.parameters())
        if len(param_groups) != len(list(all_params)):
            print("Warning: Not all parameters are in optimizer")
        print(f"Number of parameters: {len(list(all_params))}")
        
        return optimizer


    def optimize_interventions_(self, num_iterations=100, learning_rate=0.01):
        set_seed(self.config['exp']['seed'])
        dataloader = get_dataloader(CIPDataset(self.dataset_collection.val_f.data, self.config), batch_size=self.config['exp']['batch_size'], shuffle=False)
        # dataloader = get_dataloader(CIPDataset(self.dataset_collection.val_f.data, self.config), batch_size=1, shuffle=False)
        
        device = "cuda"
        # device = self.device

        self.inference_model.to(device)
        self.generative_model.to(device)
        self.auxiliary_model.to(device)

        self.inference_model.train()
        self.generative_model.train()
        self.auxiliary_model.train()

        results = []
        losses = []
        losses_2 = []
        ture_output_list = []
        output_after_actions_list = []
        ture_output_actions_list = []
        for batch in dataloader:
            # 如果未提供初始干预序列，则随机初始化
            initial_a_seq = None
            H_t, targets = batch
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)
            Y_targets = targets['outputs']
            batch_size = Y_targets.size(0)
            
            if initial_a_seq is None:
                # a_seq = torch.randn(batch_size, self.tau, self.a_dim, device=device, requires_grad=True)
                a_seq = torch.rand(batch_size, self.tau, self.a_dim, device=device, requires_grad=True)
            else:
                a_seq = initial_a_seq.clone().detach().requires_grad_(True)
            
            # 优化器（用于优化干预序列 a_seq）
            optimizer = torch.optim.Adam([a_seq], lr=learning_rate)
            print(f"-" * 50)
            for iteration in range(num_iterations):
                a_seq_sigmoid = torch.sigmoid(a_seq)
                optimizer.zero_grad()
                elbo, reg_loss = self.calculate_elbo(H_t, Y_targets, a_seq_sigmoid, num_samples=self.config.exp.num_samples, optimize_a=True)
                loss = elbo
                loss.backward()
                optimizer.step()
            optimized_a_seq = a_seq.detach()
            if num_iterations > 0:
                optimized_a_seq = torch.sigmoid(optimized_a_seq)

            output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, optimized_a_seq, self.dataset_collection.train_scaling_params)
            ture_output = targets['outputs'][:, -1, :].detach().cpu().numpy()

            true_actions = targets['current_treatments']
            ture_output_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, true_actions, self.dataset_collection.train_scaling_params)

            # print(f'shape of output_after_actions: {output_after_actions.shape}, shape of ture_output: {ture_output.shape}')
            loss = np.sqrt(((output_after_actions-ture_output) ** 2).mean())
            losses.append(loss)

            loss_2 = np.sqrt(((ture_output_actions-ture_output) ** 2).mean())
            losses_2.append(loss_2)

            ture_output_list.append(ture_output)
            output_after_actions_list.append(output_after_actions)
            ture_output_actions_list.append(ture_output_actions)

            results.append(optimized_a_seq)

        true_actions = targets['current_treatments']
        # print(f"true_actions: {true_actions}")
        ture_output_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, true_actions, self.dataset_collection.train_scaling_params)

         
        print(f"Mean loss v1: {sum(losses) / len(losses)}")
        print(f"Mean loss_2 v2: {sum(losses_2) / len(losses_2)}")
        ture_output_list = np.concatenate(ture_output_list, axis=0)
        output_after_actions_list = np.concatenate(output_after_actions_list, axis=0)
        ture_output_actions_list = np.concatenate(ture_output_actions_list, axis=0)
        loss = np.sqrt(((output_after_actions_list-ture_output_list) ** 2).mean())
        loss_2 = np.sqrt(((ture_output_actions_list-ture_output_list) ** 2).mean()) 
        print(f"Mean loss v2: {loss}")
        print(f"Mean loss_2 v2: {loss_2}")

        return torch.cat(results, dim=0), loss

    def optimize_interventions(self, num_iterations=100, learning_rate=0.01, batch_size=64):
        results = ['\n' + '-' * 50]
        tau = self.tau
        for i in range(1, 13):
        # for i in [2, 6, 9]:
            print(f'start predicting results for tau={i} ...')
            self.tau = i
            self.config.exp.tau = i
            print(self.tau, self.config.exp.tau)
            _, loss = self.optimize_interventions_onetime(num_iterations=num_iterations, learning_rate=learning_rate, batch_size=batch_size)
            results.append(f"Optimized interventions for tau={i}, {num_iterations} iterations, lr={learning_rate}: {loss}")
        return '\n'.join(results)

    def optimize_interventions_onetime(self, num_iterations=100, learning_rate=0.01, batch_size=64):
        set_seed(self.config['exp']['seed'])
        
        if not self.config.exp.test:
            data = self.dataset_collection.val_f.data
        else:
            data = self.dataset_collection.test_f.data
        # data = self.dataset_collection.val_f.data
        dataloader = get_dataloader(CIPDataset(data, self.config), batch_size=batch_size, shuffle=False)
        # dataloader = get_dataloader(CIPDataset(self.dataset_collection.test_f.data, self.config), batch_size=batch_size, shuffle=False)
        
        device = "cuda"

        self.inference_model.to(device)
        self.generative_model.to(device)
        self.auxiliary_model.to(device)

        self.inference_model.train()
        self.generative_model.train()
        self.auxiliary_model.train()

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
            print(f"-" * 50 + f"Batch {i}" + "-" * 50)
            H_t, targets = batch
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)
            Y_targets = targets['outputs']
            X_targets = None if not self.predict_X else targets['vitals']
            print(f"shape of Y_targets: {Y_targets.shape}")
            batch_size = Y_targets.size(0)
            
            # Initialize best tracking variables
            # best_a_seq = None
            best_elbo = float('inf')
            best_step = 0

            best_reg = float('inf')
            best_step_reg = 0
            
            a_seq = torch.rand(batch_size, self.tau, self.a_dim, device=device, requires_grad=True) # 生成一个均匀分布在 [0, 1) 之间的随机张量，形状为 (batch_size, self.tau, self.a_dim)
            # a_seq = torch.normal(mean=2, std=0.1, 
            #         size=(batch_size, self.tau, self.config.dataset.treatment_size),
            #         device=device, requires_grad=True)
            best_a_seq = torch.sigmoid(a_seq).clone().detach()
            best_a_seq_reg = a_seq.clone().detach()
            optimizer = torch.optim.Adam([a_seq], lr=learning_rate)  # 优化器的目标是优化干预序列 a_seq, 通过最小化ELBO损失来调整它的值

            print(f"-" * 50)
            for iteration in range(num_iterations):
                print('*' * 50)
                a_seq_sigmoid = torch.sigmoid(a_seq)
                optimizer.zero_grad()
                elbo, reg_loss, reg_loss_2 = self.calculate_elbo(H_t, Y_targets, a_seq_sigmoid, X_targets=X_targets,
                                                num_samples=self.config.exp.num_samples, 
                                                optimize_a=True)
                loss = elbo
                loss.backward()
                optimizer.step()
                # print(f"Iteration {iteration}: ELBO = {elbo.item()}, reg_loss = {reg_loss.item()}")
                
                # Check if current sequence is better
                if elbo.item() < best_elbo:
                    best_elbo = elbo.item()
                    best_a_seq = torch.sigmoid(a_seq.clone().detach())
                    elbos.append(best_elbo)
                    best_step = iteration

                if reg_loss.item() < best_reg:
                    best_reg = reg_loss.item()
                    best_step_reg = iteration
                    best_a_seq_reg = torch.sigmoid(a_seq.clone().detach())
                
                loss_outcome_after_actions, output_after_actions = self.get_loss_outcome_after_actions(H_t, Y_targets, a_seq_sigmoid.detach())
                ture_output = targets['outputs'][:, -1, :].detach().cpu().numpy()
                loss = np.sqrt(((output_after_actions-ture_output) ** 2).mean())
                print(f"Iteration {iteration}: elbo = {elbo.item()}, loss = {loss}")

                # print(f"Iteration {iteration}: elbo = {elbo.item()}, loss_after_actions = {loss_outcome_after_actions}, reg_loss: {reg_loss}")
                # print(f"targets['current_treatments']: {targets['current_treatments']}")
                # print(f"a_seq_sigmoid: {a_seq_sigmoid}")

            # Use the best sequence found
            print(f"Best step: {best_step}, best ELBO: {best_elbo}")

            if best_reg < 0:
                best_a_seq = best_a_seq_reg

            optimized_a_seq = best_a_seq
            
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
        print(f"losses: {losses}")
        print(f"Mean loss_2 v2: {sum(losses_2) / len(losses_2)}")
        
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

    def get_loss_outcome_after_actions(self, H_t, Y_targets, a_seq):
        output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, a_seq, self.dataset_collection.train_scaling_params)  # scaling_params是模拟生成结果的均值和方差
        ture_output = Y_targets[:, -1, :].detach().cpu().numpy()
        loss = np.sqrt(((output_after_actions-ture_output) ** 2).mean())
        loss = F.mse_loss(torch.tensor(output_after_actions, device=Y_targets.device), Y_targets[:, -1, :])
        return loss, output_after_actions

    def calculate_elbo(self, H_t, Y_targets, a_seq, X_targets=None, num_samples=10, optimize_a=False):
        H_rep = self.auxiliary_model.build_representations(H_t, only_last=True)
        
        Z_s_i, loss_hy, loss_deconfound = self.inference_model.ct_hidden_history(H_t)

        if self.config.dataset.treatment_mode == 'multilabel':
            predict_action_loss = self.generative_model.loss_predict_actions_bern(Z_s_i, a_seq)
        else:
            predict_action_loss = self.generative_model.loss_predict_actions_beta(Z_s_i, a_seq)
        
        # Reset states
        self.inference_model.reset_states()
        self.generative_model.reset_states()
        
        # Pre-compute action encodings
        a_seq_hiddens_1 = self.generative_model.build_action_encoding(a_seq)
        
        a_seq_hiddens = self.generative_model.build_reverse_action_encoding(a_seq)
        
        kl_losses = []
        reg_losses = []
        action_losses = []
        
        Y_last = Y_targets[:, -1, :]
        reg_loss_2 = None
        
        hidden_states = self.generative_model.build_hidden_states(Z_s_i, a_seq)
        reversed_hidden_states = self.generative_model.build_reverse_hidden_states(hidden_states)

        for s in range(self.tau):
            a_s = a_seq[:, s, :]
            a_s_hidden = a_seq_hiddens[:, s, :]
                
            Z_s_inf = torch.cat([Z_s_i, Z_s_i], dim=-1)
            
            if s < self.tau - 1:
                a_s_next = a_seq[:, s+1, :]
                if self.config.dataset.treatment_mode == 'multilabel':
                    action_loss = self.generative_model.bern_loss(Z_s_i, a_s_next)
                else:
                    action_loss = self.generative_model.beta_loss(Z_s_i, a_s_next)
                # action_loss = self.generative_model.mmd_loss(Z_s_i, a_s_next)
                action_losses.append(action_loss)
            
            # Inference network
            q_mu, q_logvar = self.inference_model(Z_s_inf, a_s_hidden, H_rep, Y_last)
            
            # Generative network
            p_mu, p_logvar = self.generative_model(Z_s_i, a_s_hidden)
            
            # Multiple sampling
            Z_samples = self.inference_model.reparameterize_multiple(q_mu, q_logvar, num_samples)
            # print(f"Z_samples shape: {Z_samples.shape}")
            
            # Compute losses
            kl_loss = compute_kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
            kl_losses.append(kl_loss)

            if s > 0 and s < self.tau - 1:
                Y_current = Y_targets[:, s, :].unsqueeze(0).expand(num_samples, -1, -1)
                a = a_seq_hiddens_1[:, -1, :].unsqueeze(0).expand(num_samples, -1, -1)
                reg_loss = self.generative_model.decoding_Y_loss_2(Z_samples, Y_current, a)
                if self.predict_X:
                    X_current = X_targets[:, s, :].unsqueeze(0).expand(num_samples, -1, -1)
                    reg_loss += self.generative_model.decoding_X_loss(Z_samples, X_current)
                reg_losses.append(reg_loss)

            if s == self.tau - 1:
                Y_current = Y_targets[:, s, :]
                a = a_seq_hiddens_1[:, -1, :]
                reg_losses.append(self.generative_model.decoding_Y_loss_2(Z_samples.mean(dim=0), Y_current, a))

                # Y_current = Y_targets[:, s, :].unsqueeze(0).expand(num_samples, -1, -1)
                # a = a_seq_hiddens_1[:, s, :].unsqueeze(0).expand(num_samples, -1, -1)
                # reg_losses.append(self.generative_model.decoding_Y_loss_2(Z_samples, Y_current, a))
                
                # reg_losses.append(self.generative_model.decoding_Y_loss(Z_samples.mean(dim=0), Y_current))
                
            if optimize_a:
                if s == self.tau - 1:
                    Y_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, a_seq.detach(), self.dataset_collection.train_scaling_params)
                    Y_preds = self.generative_model.decode(Z_samples)
                    Y_after_actions = torch.tensor(Y_after_actions, device=Y_preds.device)
                    # mean over each num_samples
                    Y_preds = Y_preds.mean(dim=0)
                    # print(f"Y_after_actions shape: {Y_after_actions.shape}")
                    # print(f"Y_preds shape: {Y_preds.shape}")
                    reg_loss_2 = F.mse_loss(Y_after_actions, Y_preds)

                    topk1 = self.print_topk_diff(Y_after_actions.flatten(), Y_preds.flatten())
                    topk2 = self.print_topk_diff(Y_after_actions.flatten(), Y_current.mean(dim=0).flatten())
                    topk3 = self.print_topk_diff(Y_preds.flatten(), Y_current.mean(dim=0).flatten())

                    reg_loss_3 = F.mse_loss(Y_current.mean(dim=0), Y_after_actions)
                    # print(f"reg_loss_2: {reg_loss_2.item()}, reg_loss_3: {reg_loss_3.item()}")
                    output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(H_t, a_seq.detach(), self.dataset_collection.train_scaling_params)
                    ture_output = Y_current.mean(dim=0).cpu().numpy()
                    loss_1 = np.sqrt(((output_after_actions-ture_output) ** 2).mean()) * self.std
                    loss_2 = np.sqrt(((Y_preds.detach().cpu().numpy()-ture_output) ** 2).mean()) * self.std
                    loss_3 = np.sqrt(((output_after_actions-Y_preds.detach().cpu().numpy()) ** 2).mean()) * self.std
                    
                    # print(f"loss_1: {loss_1}, loss_2: {loss_2}, loss_3: {loss_3}")

            # Update state
            Z_s_i = Z_samples.mean(dim=0)
            
        
        # Aggregate losses
        kl_loss = torch.stack(kl_losses).mean()
        if optimize_a:
            reg_loss = reg_losses[-1]
            # reg_loss = torch.stack(reg_losses).mean()
        else:
            reg_loss = torch.stack(reg_losses).mean()
            if self.config.exp.remove_aux:
                reg_loss = reg_losses[-1]
            reg_loss = reg_losses[-1]

        action_loss = torch.stack(action_losses).mean() if action_losses else 0
        
        # Compute final ELBO
        if optimize_a:
            elbo = reg_loss * self.config.exp.lambda_reg + \
                   kl_loss * self.config.exp.lambda_kl + \
                   action_loss * self.config.exp.lambda_step + \
                   predict_action_loss * self.config.exp.lambda_action

            # elbo = reg_loss + kl_loss
        else:
            elbo = reg_loss * self.config.exp.lambda_reg + \
                   kl_loss * self.config.exp.lambda_kl + \
                   action_loss * self.config.exp.lambda_step + \
                   predict_action_loss * self.config.exp.lambda_action + \
                   loss_hy * self.config.exp.lambda_hy + \
           loss_deconfound * 1.0
            # elbo = reg_loss + kl_loss
        
        return elbo, reg_loss, reg_loss_2

    def print_topk_diff(self, a, b, k=5):
        k = min(k, len(a))
        diff = torch.abs(a - b)
        topk = torch.topk(diff, k=k)
        # print(f"Top 5 differences: {topk.values}")
        # print(f"Indices: {topk.indices}")
        return topk

    def optimize_interventions_discrete(self):
        results = {}
        model = self.config.model.name.split('/')[0]
        self.eval()
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
        self.tau = tau

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
        if not self.config.exp.test:
            data = self.dataset_collection.val_f.data
        else:
            data = self.dataset_collection.test_f.data
        dataloader = get_dataloader(CIPDataset(data, self.config), 
                                batch_size=1, shuffle=False)
        device = "cuda"
        
        self.inference_model.to(device)
        self.generative_model.to(device)
        self.auxiliary_model.to(device)

        self.inference_model.eval()
        self.generative_model.eval()
        self.auxiliary_model.eval()
        
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
            X_targets = None if not self.predict_X else targets['vitals']
            true_actions = targets['current_treatments']
            
            
            # random_sequences = torch.rand(k-1, 1, self.tau, self.a_dim, device=device)
            # random_sequences = torch.sigmoid(random_sequences)
            # if self.config.dataset.treatment_mode == 'multilabel':
            #     random_sequences = (torch.rand(k-1, 1, self.tau, self.a_dim, device=device) > 0.5).float()
            # all_sequences = torch.cat([random_sequences, true_actions.unsqueeze(0)], dim=0)

            all_sequences = generate_perturbed_sequences(true_actions, k, self.tau, self.a_dim, device, treatment_mode=self.config.dataset.treatment_mode)
            
            # 计算ELBO和真实模拟loss
            elbos = []
            true_losses = []
            with torch.no_grad():
                for seq in all_sequences:
                    # 计算ELBO
                    elbo, _, _ = self.calculate_elbo(H_t, Y_targets, seq, X_targets=X_targets,
                                                num_samples=self.config.exp.num_samples,
                                                optimize_a=True)
                    elbos.append(elbo.item()) 
                    
                    # 计算真实模拟loss
                    output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(
                        H_t, seq, self.dataset_collection.train_scaling_params)
                    true_output = targets['outputs'][:, -1, :].detach().cpu().numpy()
                    true_loss = np.sqrt(((output_after_actions - true_output) ** 2).mean())
                    true_losses.append(true_loss)

            model_losses = np.array(elbos)
            true_losses = np.array(true_losses)

            corr_model_true, _ = spearmanr(model_losses, true_losses)
            
            case_info = {
                'individual_id': i,
                'model_losses': model_losses,
                'true_losses': true_losses,
                'correlations': {
                    'model_true': corr_model_true,
                },
                'true_sequence': true_actions.cpu().numpy(),
                'true_sequence_rank': np.sum(model_losses < model_losses[-1]) + 1
            }
            case_infos.append(case_info)
            
            elbos = np.array(elbos)
            true_losses = np.array(true_losses)

            
            correlation, p_value = spearmanr(elbos, true_losses)
            all_correlation.append(correlation)
            
            true_seq_rank = np.sum(elbos < elbos[-1]) + 1
            all_ranks.append(true_seq_rank)
            
            print(f"Individual {i} - True sequence rank: {true_seq_rank} out of {k}")
            print(f"ELBO for true sequence: {elbos[-1]}")
            print(f"True Loss for true sequence: {true_losses[-1]}")
            print(f"First 10 ELBOs: {elbos[:10]}")
            print(f"First 10 True Losses: {true_losses[:10]}")
            print(f"Rank correlation for this individual: {correlation:.3f} (p-value: {p_value:.3f})")
        
        # 计算平均结果
        all_correlation = [c for c in all_correlation if not np.isnan(c)]
        avg_rank = sum(all_ranks) / len(all_ranks)
        if len(all_correlation) > 0:
            avg_correlation = sum(all_correlation) / len(all_correlation)
        else:
            avg_correlation = 0
        print(f"Average rank of true sequences across all individuals: {avg_rank:.2f} out of {k}")
        print(f"Average rank correlation across all individuals: {avg_correlation:.3f}")
        
        return case_infos
        
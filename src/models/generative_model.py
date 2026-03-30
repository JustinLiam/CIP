import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss
import torch.distributions as dist
from src.models.dynamic_model import DynamicParamNetwork
from torch.distributions import Normal
from src.models.ct_history_encoder import CTHistoryEncoder, ProjectionHead

class GenerativeModel(nn.Module):
    def __init__(self, config):
        super(GenerativeModel, self).__init__()
        self.z_dim = config['model']['z_dim']
        self.hidden_dim = config['model']['generative']['hidden_dim']
        self.num_layers = config['model']['generative']['num_layers']
        self.treatment_size = config['dataset']['treatment_size']
        self.treatment_hidden_dim = config['model']['generative']['treatment_hidden_dim']
        self.output_dim = config['dataset']['output_size']
        self.input_size = config['dataset']['input_size']
        self.static_size = config['dataset']['static_size']
        self.autoregressive = config['dataset']['autoregressive']
        self.dropout = config['exp']['dropout']
        self.beta_bound = config['exp']['beta_bound']
        self.config = config
        self.predict_X = config['dataset']['predict_X']
        self.input_size = config['dataset']['input_size']

        self.ct_history_encoder = CTHistoryEncoder(x_dim=self.input_size + self.static_size, a_dim=self.treatment_size,
                                                   y_dim=self.output_dim,
                                                   d_model=64, num_heads=4, num_layers=self.num_layers,
                                                   dropout=self.dropout)
        self.projection_head = ProjectionHead(input_dim=64, hidden_dim=64, output_dim=self.z_dim)


        self.reverse_action_encoder = nn.LSTM(
            input_size=self.treatment_size,
            hidden_size=self.treatment_hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout
        )

        self.action_encoder = nn.LSTM(
            input_size=self.treatment_size,
            hidden_size=self.treatment_hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout
        )

        self.action_decoder = nn.Sequential()
        self.hidden_action_decoder = config['model']['generative']['hidden_action_decoder']
        output_size = self.treatment_size 
        if -1 not in self.hidden_action_decoder:
            for i in range(len(self.hidden_action_decoder)):
                if i == 0:
                    self.action_decoder.add_module('fc{}'.format(i), nn.Linear(self.treatment_hidden_dim + self.z_dim, self.hidden_action_decoder[i]))
                else:
                    self.action_decoder.add_module('elu{}'.format(i), nn.ELU())
                    self.action_decoder.add_module('fc{}'.format(i), nn.Linear(self.hidden_action_decoder[i-1], self.hidden_action_decoder[i]))
            self.action_decoder.add_module('elu{}'.format(len(self.hidden_action_decoder)), nn.ELU())
            self.action_decoder.add_module('fc{}'.format(len(self.hidden_action_decoder)), nn.Linear(self.hidden_action_decoder[-1], output_size))
        else:
            self.action_decoder.add_module('fc{}'.format(1), nn.Linear(self.treatment_hidden_dim + self.z_dim, output_size))

        self.action_decoder_beta = nn.Sequential()
        self.hidden_action_decoder = config['model']['generative']['hidden_action_decoder']
        output_size = self.treatment_size * 2
        if -1 not in self.hidden_action_decoder:
            for i in range(len(self.hidden_action_decoder)):
                if i == 0:
                    self.action_decoder_beta.add_module('fc{}'.format(i), nn.Linear(self.treatment_hidden_dim + self.z_dim, self.hidden_action_decoder[i]))
                else:
                    self.action_decoder_beta.add_module('elu{}'.format(i), nn.ELU())
                    self.action_decoder_beta.add_module('fc{}'.format(i), nn.Linear(self.hidden_action_decoder[i-1], self.hidden_action_decoder[i]))
            self.action_decoder_beta.add_module('elu{}'.format(len(self.hidden_action_decoder)), nn.ELU())
            self.action_decoder_beta.add_module('fc{}'.format(len(self.hidden_action_decoder)), nn.Linear(self.hidden_action_decoder[-1], output_size))
        else:
            self.action_decoder_beta.add_module('fc{}'.format(1), nn.Linear(self.treatment_hidden_dim + self.z_dim, output_size))

        self.action_decoder_step = nn.Sequential()
        self.hidden_action_decoder_step = config['model']['generative']['hidden_action_decoder_step']
        output_size = self.treatment_size * 2
        if -1 not in self.hidden_action_decoder_step:
            for i in range(len(self.hidden_action_decoder_step)):
                if i == 0:
                    self.action_decoder_step.add_module('fc{}'.format(i), nn.Linear(self.z_dim, self.hidden_action_decoder_step[i]))
                else:
                    self.action_decoder_step.add_module('elu{}'.format(i), nn.ELU())
                    self.action_decoder_step.add_module('fc{}'.format(i), nn.Linear(self.hidden_action_decoder_step[i-1], self.hidden_action_decoder_step[i]))
            self.action_decoder_step.add_module('elu{}'.format(len(self.hidden_action_decoder_step)), nn.ELU())
            self.action_decoder_step.add_module('fc{}'.format(len(self.hidden_action_decoder_step)), nn.Linear(self.hidden_action_decoder_step[-1], output_size))
        else:
            self.action_decoder_step.add_module('fc{}'.format(1), nn.Linear(self.z_dim, output_size))

        
        if self.static_size > 0:
            input_size = self.input_size + self.static_size + self.treatment_size
        else:
            input_size = self.input_size + self.treatment_size
        if self.autoregressive:
            input_size += self.output_dim
        if self.predict_X:
            input_size += self.input_size

        self.lstm_history = nn.LSTM(
            input_size=input_size,
            hidden_size=self.z_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )
        
        self.hidden_state = None
        self.cell_state = None

        self.lstm_input_dim = self.z_dim + self.treatment_hidden_dim
        # self.lstm_input_dim = self.z_dim + self.treatment_size
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )

        input_size = self.z_dim + self.treatment_size
        self.hidden_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.z_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )

        input_size = self.z_dim 
        self.hidden_lstm_reverse = nn.LSTM(
            input_size=input_size,
            hidden_size=self.z_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )

        self.hiddens_F_mu = config['model']['generative']['hiddens_F_mu']
        self.fc_mu = nn.Sequential()
        # input_size = self.hidden_dim + self.treatment_size
        input_size = self.hidden_dim + self.treatment_hidden_dim

        if -1 not in self.hiddens_F_mu:
            for i in range(len(self.hiddens_F_mu)):
                if i == 0:
                    self.fc_mu.add_module('fc{}'.format(i), nn.Linear(input_size, self.hiddens_F_mu[i]))
                else:
                    self.fc_mu.add_module('elu{}'.format(i), nn.ELU())
                    self.fc_mu.add_module('fc{}'.format(i), nn.Linear(self.hiddens_F_mu[i-1], self.hiddens_F_mu[i]))
            self.fc_mu.add_module('elu{}'.format(len(self.hiddens_F_mu)), nn.ELU())
            self.fc_mu.add_module('fc{}'.format(len(self.hiddens_F_mu)), nn.Linear(self.hiddens_F_mu[-1], self.z_dim))
        else:
            self.fc_mu.add_module('fc{}'.format(1), nn.Linear(input_size, self.z_dim))
        
        self.hiddens_F_logvar = config['model']['generative']['hiddens_F_logvar']
        self.fc_logvar = nn.Sequential()
        if -1 not in self.hiddens_F_logvar:
            for i in range(len(self.hiddens_F_logvar)):
                if i == 0:
                    self.fc_logvar.add_module('fc{}'.format(i), nn.Linear(input_size, self.hiddens_F_logvar[i]))
                else:
                    self.fc_logvar.add_module('elu{}'.format(i), nn.ELU())
                    self.fc_logvar.add_module('fc{}'.format(i), nn.Linear(self.hiddens_F_logvar[i-1], self.hiddens_F_logvar[i]))
            self.fc_logvar.add_module('elu{}'.format(len(self.hiddens_F_logvar)), nn.ELU())
            self.fc_logvar.add_module('fc{}'.format(len(self.hiddens_F_logvar)), nn.Linear(self.hiddens_F_logvar[-1], self.z_dim))
        else:
            self.fc_logvar.add_module('fc{}'.format(1), nn.Linear(input_size, self.z_dim))
        
        
        input_size = self.z_dim
        self.hiddens_decoder = config['model']['generative']['hiddens_decoder']
        self.decoder = nn.Sequential()
        if -1 not in self.hiddens_decoder:
            for i in range(len(self.hiddens_decoder)):
                if i == 0:
                    self.decoder.add_module('fc{}'.format(i), nn.Linear(input_size, self.hiddens_decoder[i]))
                else:
                    self.decoder.add_module('elu{}'.format(i), nn.ELU())
                    self.decoder.add_module('fc{}'.format(i), nn.Linear(self.hiddens_decoder[i-1], self.hiddens_decoder[i]))
            self.decoder.add_module('elu{}'.format(len(self.hiddens_decoder)), nn.ELU())
            self.decoder.add_module('fc{}'.format(len(self.hiddens_decoder)), nn.Linear(self.hiddens_decoder[-1], self.output_dim))
        else:
            self.decoder.add_module('fc{}'.format(1), nn.Linear(input_size, self.output_dim))

        input_size = self.z_dim
        self.hiddens_decoder = config['model']['generative']['hiddens_decoder']
        self.decoder_p = nn.Sequential()
        if -1 not in self.hiddens_decoder:
            for i in range(len(self.hiddens_decoder)):
                if i == 0:
                    self.decoder_p.add_module('fc{}'.format(i), nn.Linear(input_size, self.hiddens_decoder[i]))
                else:
                    self.decoder_p.add_module('elu{}'.format(i), nn.ELU())
                    self.decoder_p.add_module('fc{}'.format(i), nn.Linear(self.hiddens_decoder[i-1], self.hiddens_decoder[i]))
            self.decoder_p.add_module('elu{}'.format(len(self.hiddens_decoder)), nn.ELU())
            self.decoder_p.add_module('fc{}'.format(len(self.hiddens_decoder)), nn.Linear(self.hiddens_decoder[-1], self.output_dim * 2))
        else:
            self.decoder_p.add_module('fc{}'.format(1), nn.Linear(input_size, self.output_dim * 2))

        input_size = self.z_dim + self.treatment_hidden_dim
        self.hiddens_decoder = config['model']['generative']['hiddens_decoder']
        self.decoder_pa = nn.Sequential()
        if -1 not in self.hiddens_decoder:
            for i in range(len(self.hiddens_decoder)):
                if i == 0:
                    self.decoder_pa.add_module('fc{}'.format(i), nn.Linear(input_size, self.hiddens_decoder[i]))
                else:
                    self.decoder_pa.add_module('elu{}'.format(i), nn.ELU())
                    self.decoder_pa.add_module('fc{}'.format(i), nn.Linear(self.hiddens_decoder[i-1], self.hiddens_decoder[i]))
            self.decoder_pa.add_module('elu{}'.format(len(self.hiddens_decoder)), nn.ELU())
            self.decoder_pa.add_module('fc{}'.format(len(self.hiddens_decoder)), nn.Linear(self.hiddens_decoder[-1], self.output_dim * 2))
        else:
            self.decoder_pa.add_module('fc{}'.format(1), nn.Linear(input_size, self.output_dim * 2))

        if self.predict_X:
            input_size = self.z_dim
            self.hiddens_decoder = config['model']['generative']['hiddens_decoder']
            self.decoder_x = nn.Sequential()
            if -1 not in self.hiddens_decoder:
                for i in range(len(self.hiddens_decoder)):
                    if i == 0:
                        self.decoder_x.add_module('fc{}'.format(i), nn.Linear(input_size, self.hiddens_decoder[i]))
                    else:
                        self.decoder_x.add_module('elu{}'.format(i), nn.ELU())
                        self.decoder_x.add_module('fc{}'.format(i), nn.Linear(self.hiddens_decoder[i-1], self.hiddens_decoder[i]))
                self.decoder_x.add_module('elu{}'.format(len(self.hiddens_decoder)), nn.ELU())
                self.decoder_x.add_module('fc{}'.format(len(self.hiddens_decoder)), nn.Linear(self.hiddens_decoder[-1], self.input_size))
            else:
                self.decoder_x.add_module('fc{}'.format(1), nn.Linear(input_size, self.input_size))

            input_size = self.z_dim
            self.hiddens_decoder = config['model']['generative']['hiddens_decoder']
            self.decoder_p_x = nn.Sequential()
            if -1 not in self.hiddens_decoder:
                for i in range(len(self.hiddens_decoder)):
                    if i == 0:
                        self.decoder_p_x.add_module('fc{}'.format(i), nn.Linear(input_size, self.hiddens_decoder[i]))
                    else:
                        self.decoder_p_x.add_module('elu{}'.format(i), nn.ELU())
                        self.decoder_p_x.add_module('fc{}'.format(i), nn.Linear(self.hiddens_decoder[i-1], self.hiddens_decoder[i]))
                self.decoder_p_x.add_module('elu{}'.format(len(self.hiddens_decoder)), nn.ELU())
                self.decoder_p_x.add_module('fc{}'.format(len(self.hiddens_decoder)), nn.Linear(self.hiddens_decoder[-1], self.input_size * 2))
            else:
                self.decoder_p_x.add_module('fc{}'.format(1), nn.Linear(input_size, self.input_size * 2))

        self.epsilon = 1e-1

        input_dim = self.z_dim
        hidden_dim = self.hidden_dim
        output_dim = self.hidden_dim
        self.transition_network = DynamicParamNetwork(input_dim, hidden_dim, output_dim, num_rbf_centers=5)


    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        self.cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

    def build_H_t(self, H_t):
        if self.static_size > 0:
            if self.predict_X:
                x = H_t['vitals']
                x = torch.cat((x, H_t['static_features']), dim=-1)
            # when we don't predict x, we use static features as the current_covariates
            else:
                x = H_t['static_features']
        # if we use autoregressive, we need to use the previous output as the input
        if self.autoregressive:
            prev_outputs = H_t['prev_outputs']
            x = torch.cat((x, prev_outputs), dim=-1)
        
        previous_treatments = H_t['prev_treatments']
        x = torch.cat((x, previous_treatments), dim=-1) # (batch_size, seq_length, input_size)
        return x

    def init_hidden_history(self, H_t):
        x = self.build_H_t(H_t)
        Z_t, _ = self.lstm_history(x) # (batch_size, history_length, hidden_dim)
        # treatments = H_t['current_treatments']
        # y_hat = self.predict_y_history_net(torch.cat((Z_t, treatments), dim=-1))
        # loss = nn.MSELoss()(y_hat, H_t['outputs'])
        return Z_t[:, -1, :]

    def ct_hidden_history(self, H_t):

        # 1. 拆解字典，提取各自独立的特征序列
        covariates = H_t['vitals']
        if self.static_size > 0:
            # 静态特征一般和动态协变量拼接作为 x
            statics = H_t['static_features']
            covariates = torch.cat((covariates, statics), dim=-1)

        treatments = H_t['prev_treatments']  # A
        outputs = H_t['prev_outputs']  # Y

        # 2. 传入 Causal Transformer 提取时序表示
        # ct_rep 形状: (batch_size, history_length, d_model)
        ct_rep = self.ct_history_encoder(x=covariates, a=treatments, y=outputs)

        # 3. 通过投影头，映射到去交杂的 Latent 空间
        # Z_t 形状: (batch_size, history_length, z_dim)
        Z_t = self.projection_head(ct_rep)

        # 4. 辅助回归损失：保证提取出的表示依然具有预测病情的临床意义 (非常关键！)
        # 用当前的潜状态 Z_t 去预测真实的 outputs
        # y_hat = self.predict_y_history_net(torch.cat((Z_t, treatments), dim=-1))
        # loss_hy = nn.MSELoss()(y_hat, H_t['outputs'])

        # 5. 只取历史序列的最后一步作为当前时刻推演的起点 s_t
        Z_s_i = Z_t[:, -1, :]

        # ================= 新增的去交杂部分 =================
        # 4. 对比学习 Loss (参考 CoCo) 或 对抗 Loss (参考 CT) TODO
        # loss_deconfound = self.compute_causal_contrastive_loss(Z_t, treatments)

        # 返回给主循环
        return Z_s_i

    def build_reverse_action_encoding(self, a_seq):
        """
        a_seq: (batch_size, tau, treatment_size)
        """
        # reverse the action sequence
        reversed_a_seq = torch.flip(a_seq, [1])
        # reversed_a_seq = a_seq
        outs, (_, _) = self.reverse_action_encoder(reversed_a_seq) # (batch_size, tau, treatment_hidden_dim)
        # reverse the output
        outs = torch.flip(outs, [1])
        return outs

    def build_hidden_states_v3(self, Z, a_seq):
        hidden_states = []
        for i in range(a_seq.size(1)):
            out = self.hidden_lstm(torch.cat((Z.unsqueeze(1), a_seq[:, i, :].unsqueeze(1)), dim=-1))
            Z = out[0][:, -1, :]
            hidden_states.append(Z)

        return torch.stack(hidden_states, dim=1)
    
    def build_hidden_states_v2(self, Z, a_seq):
        Z_expanded = Z.unsqueeze(1).repeat(1, a_seq.size(1), 1)  # [batch_size, seq_len, Z_dim]
        
        lstm_input = torch.cat((Z_expanded, a_seq), dim=-1)  # [batch_size, seq_len, Z_dim + action_dim]
        
        out, _ = self.hidden_lstm(lstm_input)  # out: [batch_size, seq_len, hidden_dim]
        
        hidden_states = out  # [batch_size, seq_len, hidden_dim]
        return hidden_states

    def build_hidden_states(self, Z, a_seq):
        hidden_states = []
        h_t, c_t = None, None  
        seq_len = a_seq.size(1)

        for i in range(seq_len):
            input_t = torch.cat((Z.unsqueeze(1), a_seq[:, i, :].unsqueeze(1)), dim=-1)  # [batch_size, 1, Z_dim + action_dim]
            out, (h_t, c_t) = self.hidden_lstm(input_t, (h_t, c_t)) if h_t is not None else self.hidden_lstm(input_t)
            Z = out[:, -1, :]  
            hidden_states.append(Z)

        hidden_states = torch.stack(hidden_states, dim=1)  # [batch_size, seq_len, hidden_dim]
        return hidden_states

    def build_reverse_hidden_states(self, hidden_states):
        reversed_hidden_states = torch.flip(hidden_states, [1])
        outs, (_, _) = self.hidden_lstm_reverse(reversed_hidden_states)
        outs = torch.flip(outs, [1])
        return outs


    def build_action_encoding(self, a_seq):
        """
        a_seq: (batch_size, tau, treatment_size)
        """
        outs, (_, _) = self.action_encoder(a_seq.detach()) # (batch_size, tau, treatment_hidden_dim)
        # outs, (_, _) = self.action_encoder(a_seq)
        
        return outs

    def predict_actions(self, Z_init, a_seq):
        a_seq = a_seq[:, :-1, :]
        # append the initial zero hidden state
        a_seq = torch.cat((torch.zeros(a_seq.size(0), 1, a_seq.size(-1)).to(a_seq.device), a_seq), dim=1) # (batch_size, tau, treatment_size)
        a_seq, (_, _) = self.action_encoder(a_seq) # (batch_size, tau, treatment_size)
        Z_init_expanded = Z_init.unsqueeze(1).repeat(1, a_seq.size(1), 1)
        inputs = torch.cat((Z_init_expanded, a_seq), dim=-1) # (batch_size, tau, z_dim + treatment_size)
        action_hat = self.action_decoder(inputs) # (batch_size, tau, treatment_size)
        return action_hat

    def loss_predict_actions(self, Z_init, a_seq):
        action_hat = self.predict_actions(Z_init, a_seq)
        loss = nn.MSELoss()(action_hat, a_seq)
        return loss

    def loss_predict_actions_beta(self, Z_init, a_seq):
        a_seq_ori = a_seq.clone()
        a_seq = a_seq[:, :-1, :]
        a_seq = torch.cat((torch.zeros(a_seq.size(0), 1, a_seq.size(-1)).to(a_seq.device), a_seq), dim=1) # (batch_size, tau, treatment_size)
        a_seq, (_, _) = self.action_encoder(a_seq) # (batch_size, tau, treatment_size)
        
        Z_init_expanded = Z_init.unsqueeze(1).repeat(1, a_seq.size(1), 1)
        # print(f'a_seq: {a_seq.size()}, Z_init_expanded: {Z_init_expanded.size()}, Z_init: {Z_init.size()}')
        inputs = torch.cat((Z_init_expanded, a_seq), dim=-1) # (batch_size, tau, z_dim + treatment_size)
        out = self.action_decoder_beta(inputs) # (batch_size, tau, treatment_size)

        alpha, beta = out[:, :, :self.treatment_size], out[:, :, self.treatment_size:]
        alpha = F.softplus(alpha) + self.epsilon
        beta = F.softplus(beta) + self.epsilon
        
        dist_beta = dist.Beta(alpha, beta)
        entropy = dist_beta.entropy().mean()

        log_likelihood = dist_beta.log_prob(a_seq_ori.clamp(self.epsilon, 1 - self.epsilon)) 
        loss = self.loss_predict_actions(Z_init, a_seq_ori)

        # return -log_likelihood.sum(dim=(-1, -2)).mean() - entropy * self.config.model.generative.entropy_lambda + loss

        return -log_likelihood.sum(dim=(-1, -2)).mean() - entropy * self.config.model.generative.entropy_lambda + loss

    def predict_actions_binary(self, Z_init, a_seq):
        a_seq = a_seq[:, :-1, :]
        # append the initial zero hidden state
        a_seq = torch.cat((torch.zeros(a_seq.size(0), 1, a_seq.size(-1)).to(a_seq.device), a_seq), dim=1) # (batch_size, tau, treatment_size)
        a_seq, (_, _) = self.action_encoder(a_seq) # (batch_size, tau, treatment_size)
        Z_init_expanded = Z_init.unsqueeze(1).repeat(1, a_seq.size(1), 1)
        inputs = torch.cat((Z_init_expanded, a_seq), dim=-1) # (batch_size, tau, z_dim + treatment_size)
        
        logits = self.action_decoder(inputs) # (batch_size, tau, treatment_size)
        
        action_prob = torch.sigmoid(logits)
        action_prob = action_prob.clamp(self.epsilon, 1 - self.epsilon)  # 数值稳定性
        
        return action_prob

    def loss_predict_actions_binary(self, Z_init, a_seq):
        action_prob = self.predict_actions_binary(Z_init, a_seq)
    
        loss = nn.BCELoss()(action_prob, a_seq)
        return loss

    def loss_predict_actions_bern(self, Z_init, a_seq):
        a_seq_ori = a_seq
        a_seq = a_seq[:, :-1, :]
        a_seq = torch.cat((torch.zeros(a_seq.size(0), 1, a_seq.size(-1)).to(a_seq.device), a_seq), dim=1) # (batch_size, tau, treatment_size)
        a_seq, (_, _) = self.action_encoder(a_seq) # (batch_size, tau, treatment_size)
        a_seq = a_seq.clamp(self.epsilon, 1 - self.epsilon)
        Z_init_expanded = Z_init.unsqueeze(1).repeat(1, a_seq.size(1), 1)
        # print(f'a_seq: {a_seq.size()}, Z_init_expanded: {Z_init_expanded.size()}, Z_init: {Z_init.size()}')
        inputs = torch.cat((Z_init_expanded, a_seq), dim=-1) # (batch_size, tau, z_dim + treatment_size)
        out = self.action_decoder_beta(inputs) # (batch_size, tau, treatment_size)
        mu, logvar = out[:, :, :self.treatment_size], out[:, :, self.treatment_size:]

        probs = torch.sigmoid(mu)
        probs = probs.clamp(self.epsilon, 1 - self.epsilon)  # 数值稳定性
        # print(f"shape of beta {beta.shape}")
        dist_bern = torch.distributions.Bernoulli(probs=probs)
        entropy = dist_bern.entropy().mean()

        log_likelihood = dist_bern.log_prob(a_seq_ori) 
        loss = self.loss_predict_actions_binary(Z_init, a_seq_ori)

        return -log_likelihood.sum(dim=(-2,-1)).mean() - entropy * self.config.model.generative.entropy_lambda + loss


    def decode_action_step(self, Z_s):
        """
        Z_s: (batch_size, z_dim)
        """
        a_s = self.action_decoder_step(Z_s)
        # a_s = torch.sigmoid(a_s)
        # a_s = torch.clamp(a_s, min=0, max=1)
        mu, logvar = a_s[:, :self.treatment_size], a_s[:, self.treatment_size:]

        return mu, logvar

    def forward(self, Z_s_prev, a_s):
        """
        Z_s_prev: (batch_size, z_dim)
        a_s: (batch_size, treatment_dim)
        """
        batch_size = Z_s_prev.size(0)
        if self.hidden_state is None:
            self.init_hidden(batch_size)
        
        input = torch.cat([Z_s_prev, a_s], dim=-1).unsqueeze(1) 
        
        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
            input, (self.hidden_state, self.cell_state)
        )
        lstm_out = lstm_out.squeeze(1)  # 移除时间步维度
        lstm_out = torch.cat([lstm_out, a_s], dim=-1)

        # lstm_out = self.transition_network(Z_s_prev, a_s)
        
        p_mu = self.fc_mu(lstm_out)
        p_logvar = self.fc_logvar(lstm_out)
        # p_logvar = F.softplus(p_logvar)
        
        return p_mu, p_logvar
    
    def decode(self, Z_s):
        return self.decoder(Z_s)

    def decode_p(self, Z_s):
        out = self.decoder_p(Z_s)
        mu, p_logvar = out[..., :self.output_dim], out[..., self.output_dim:]
        # print(f"p_std : {p_logvar.mean()}, p_std : {p_logvar.shape}")
        
        return mu, p_logvar

    def decode_p_a(self, Z_s, a):
        # print(Z_s.shape, a.shape)
        input = torch.cat((Z_s, a), dim=-1)
        out = self.decoder_pa(input)
        mu, p_logvar = out[..., :self.output_dim], out[..., self.output_dim:]
        # print(f"p_std : {p_logvar.mean()}, p_std : {p_logvar.shape}")
        
        return mu, p_logvar

    def decoding_Y_loss(self, Z, y):
        mu, p_logvar = self.decode_p(Z)
        p_std = torch.exp(0.5 * p_logvar) + 1e-6
        normal_dist = Normal(mu, p_std)
        # print(f"mu shape:{mu.shape}")
        # print(f"y shape:{y.shape}")
        p = normal_dist.log_prob(y).sum(dim=-1).mean()
        # print(f"p_std : {p_std.mean()}")

        loss = F.mse_loss(mu, y)
        return (loss) * 0.1
        return -p

    def decoding_Y_loss_2(self, Z, y, a):
        mu, p_logvar = self.decode_p_a(Z, a)
        p_std = torch.exp(0.5 * p_logvar) + 1e-6
        p_std = torch.ones(p_std.shape, device=p_std.device) * 0.1
        normal_dist = Normal(mu, p_std)
        # print(f"mu shape:{mu.shape}")
        # print(f"y shape:{y.shape}")
        p = normal_dist.log_prob(y).sum(dim=-1).mean()
        # print(f"p_std : {p_std.mean()}")

        loss = F.mse_loss(mu, y)
        return loss
    
    def decode_x(self, Z_s):
        return self.decoder(Z_s)

    def decode_p_x(self, Z_s):
        out = self.decoder_p(Z_s)
        mu, p_logvar = out[..., :self.input_size], out[..., self.input_size:]
        # print(f"p_std : {p_logvar.mean()}, p_std : {p_logvar.shape}")
        
        return mu, p_logvar
    
    def decoding_X_loss(self, Z, x):
        mu, p_logvar = self.decode_p(Z)
        p_std = torch.exp(0.5 * p_logvar) + 1e-6
        normal_dist = Normal(mu, p_std)
        # print(f"mu shape:{mu.shape}")
        # print(f"y shape:{y.shape}")
        p = normal_dist.log_prob(x).sum(dim=-1).mean()
        # print(f"p_std : {p_std.mean()}")

        loss = F.mse_loss(mu, x)
        return loss
        return -p

        

    # def beta_loss(self, Z_s, a_s):
    #     # a_s (0,1)
    #     a_s = a_s.clamp(self.epsilon, 1 - self.epsilon)
    #     alpha, beta = self.decode_action_step(Z_s)
    #     alpha = alpha + 1.
    #     beta = beta + 1.
    #     # log_likelihood = (alpha - 1) * torch.log(a_s) + (beta - 1) * torch.log(1 - a_s)
    #     # log_likelihood -= torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    #     dist_beta = dist.Beta(alpha, beta)
    #     log_likelihood = dist_beta.log_prob(a_s)
    #     loss = -log_likelihood.sum(dim=-1).mean()
    #     return loss

    # def beta_loss(self, Z_s, a_s):
    #     a_s = a_s.clamp(self.epsilon, 1 - self.epsilon)
    #     alpha, beta = self.decode_action_step(Z_s)
        
    #     # 使用sigmoid将参数映射到合理范围
    #     max_param = 5
    #     min_param = 1
    #     alpha = min_param + (max_param - min_param) * torch.sigmoid(alpha)
    #     beta = min_param + (max_param - min_param) * torch.sigmoid(beta)
        
    #     dist_beta = dist.Beta(alpha, beta)
    #     log_likelihood = dist_beta.log_prob(a_s)
    #     loss = -log_likelihood.sum(dim=-1).mean()
    #     return loss

    # def beta_loss(self, Z_s, a_s):
    #     a_s = a_s.clamp(self.epsilon, 1 - self.epsilon)
    #     alpha, beta = self.decode_action_step(Z_s)
    #     alpha = F.softplus(alpha) + 1.
    #     beta = F.softplus(beta) + 1.
        
    #     # 添加正则化项，惩罚过大的参数值
    #     reg_strength = 0.1
    #     param_reg = reg_strength * (torch.mean(alpha**2) + torch.mean(beta**2))
        
    #     dist_beta = dist.Beta(alpha, beta)
    #     log_likelihood = dist_beta.log_prob(a_s)
    #     loss = -log_likelihood.sum(dim=-1).mean() + param_reg
    #     return loss

    def beta_loss(self, Z_s, a_s):
        a_s = a_s.clamp(self.epsilon, 1 - self.epsilon)
        alpha, beta = self.decode_action_step(Z_s)
        # alpha = F.softplus(mu) + self.epsilon  #  alpha > 0
        # beta = F.softplus(logvar) + self.epsilon    # beta > 0
        alpha = F.softplus(alpha) 
        beta = F.softplus(beta)
        
        dist_beta = dist.Beta(alpha, beta)
        log_likelihood = dist_beta.log_prob(a_s)
        
        entropy_reg = self.config.exp.entropy_reg
        entropy = dist_beta.entropy().mean()
        
        if self.beta_bound:
            loss_reg = log_likelihood.sum(dim=-1).mean().clamp(self.beta_bound) * 0.1
        else:
            loss_reg = log_likelihood.sum(dim=-1).mean() * 0.1
        loss = loss_reg - entropy_reg * entropy
        return loss

    def bern_loss(self, Z_s, a_s):
        logits, _ = self.decode_action_step(Z_s)
        probs = torch.sigmoid(logits)
        
        dist_bern = torch.distributions.Bernoulli(probs=probs)
        log_likelihood = dist_bern.log_prob(a_s)

        entropy_reg = 1
        entropy = dist_bern.entropy().mean()
        
        # 计算loss
        if self.beta_bound:
            loss_reg = log_likelihood.sum(dim=-1).mean().clamp(self.beta_bound) * 0.1
        else:
            loss_reg = log_likelihood.sum(dim=-1).mean() * 0.1
            
        loss = -loss_reg - entropy_reg * entropy
        
        return loss

    def bce_loss(self, Z_s, a_s):
        a_pred, _ = self.decode_action_step(Z_s)
        a_pred = torch.sigmoid(a_pred)
        bce_loss = -F.binary_cross_entropy(a_pred, a_s, reduction='mean')
        return bce_loss

    def mse_loss(self, Z_s, a_s):
        a_pred, _ = self.decode_action_step(Z_s)
        a_pred = torch.sigmoid(a_pred)
        mse_loss = F.mse_loss(a_pred, a_s, reduction='mean')
        return -mse_loss

    def wass_loss(self, Z_s, a_s, use_matrix=True):
        batch_size = Z_s.size(0)
        Za = torch.cat([Z_s, a_s], dim=1)
        
        idx = torch.randperm(batch_size)
        
        if use_matrix:
            indices = torch.zeros(batch_size, batch_size)
            indices[torch.arange(batch_size), idx] = 1
            indices = indices.to(a_s.device)
            a_shuffled = torch.mm(indices, a_s)
        else:
            a_shuffled = a_s[idx]
        
        Z_a_shuffled = torch.cat([Z_s, a_shuffled], dim=1)

        sinkhorn_loss = SamplesLoss(
            "sinkhorn",
            p=2,
            blur=0.05,
            scaling=0.9,
            backend="tensorized",
            debias=False
        )
        wass_distance = sinkhorn_loss(Za, Z_a_shuffled)
        
        return wass_distance

    def mmd_loss(self, Z_s, a_s):
        batch_size = Z_s.size(0)
        Za = torch.cat([Z_s, a_s], dim=1)
        
        idx = torch.randperm(batch_size)
        indices = torch.zeros(batch_size, batch_size)
        indices[torch.arange(batch_size), idx] = 1
        indices = indices.to(a_s.device)
        
        a_shuffled = torch.mm(indices, a_s)

        a_shuffled = a_s[idx]
        Z_a_shuffled = torch.cat([Z_s, a_shuffled], dim=1)
        
        def gaussian_kernel(x, y, sigma=1.0):
            dist = torch.cdist(x, y) ** 2
            return torch.exp(-dist / (2 * sigma ** 2))
        
        sigmas = [0.01, 0.1, 0.5, 1, 2]
        mmd = 0
        for sigma in sigmas:
            K_XX = gaussian_kernel(Za, Za, sigma)
            K_YY = gaussian_kernel(Z_a_shuffled, Z_a_shuffled, sigma)
            K_XY = gaussian_kernel(Za, Z_a_shuffled, sigma)
            
            mmd += K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        
        return mmd
        
    def reset_states(self):
        self.hidden_state = None
        self.cell_state = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
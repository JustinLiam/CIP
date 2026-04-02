import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.data.ct_transition_dataset import _covariate_stream_dim
from src.models.ct_deconfound import build_covariate_x
from src.models.dynamic_model import DynamicParamNetwork
from src.models.ct_history_encoder import CTHistoryEncoder, ProjectionHead


def _expand_static_time_dim(H_t: dict, seq_len: int) -> dict:
    """[B,d] static -> [B,T,d] so it can concat with prev_outputs / prev_treatments."""
    out = dict(H_t)
    if "static_features" in out:
        s = out["static_features"]
        if s.dim() == 2:
            out["static_features"] = s.unsqueeze(1).expand(-1, seq_len, -1)
    return out


class InferenceModel(nn.Module):
    def __init__(self, config):
        super(InferenceModel, self).__init__()
        self.config = config
        self.z_dim = config['model']['z_dim']
        self.hidden_dim = config['model']['inference']['hidden_dim']
        self.num_layers = config['model']['inference']['num_layers']
        self.do = config['model']['inference']['do']
        self.hiddens_F_mu = config['model']['inference']['hiddens_F_mu']
        self.hiddens_F_logvar = config['model']['inference']['hiddens_F_logvar']
        self.history_dim = config['model']['auxiliary']['hidden_dim']
        self.treatment_dim = config['dataset']['treatment_size']
        self.output_dim = config['dataset']['output_size']
        self.input_dim = self.history_dim + self.treatment_dim + self.output_dim + self.z_dim
        self.static_size = config['dataset']['static_size']
        self.autoregressive = config['dataset']['autoregressive']
        self.input_size = config['dataset']['input_size']
        self.treatment_size = config['dataset']['treatment_size']
        self.predict_X = config['dataset']['predict_X']
        self.treatment_hidden_dim = config['model']['generative']['treatment_hidden_dim']
        self.dropout = config['exp']['dropout']

        ds_dict = OmegaConf.to_container(config["dataset"], resolve=True)
        ct_x_dim = _covariate_stream_dim(ds_dict)

        self.ct_history_encoder = CTHistoryEncoder(
            x_dim=ct_x_dim,
            a_dim=self.treatment_dim,
            y_dim=self.output_dim,
            static_dim=self.static_size,
            d_model=64,
            num_heads=4,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.projection_head = ProjectionHead(input_dim=64, hidden_dim=64, output_dim=self.z_dim)
        

        if self.static_size > 0:
            input_size = self.input_size + self.static_size + self.treatment_size
        else:
            input_size = self.input_size + self.treatment_size
        if self.autoregressive:
            input_size += self.output_dim
        self.lstm_history = nn.LSTM(input_size, self.z_dim, self.num_layers, batch_first=True, dropout=self.dropout)
        
        # 存储隐状态
        self.hidden_state = None
        self.cell_state = None

        # input_size = self.z_dim + self.treatment_dim + self.output_dim
        input_size = self.z_dim * 2 + self.output_dim
        if self.do:
            input_size += self.treatment_hidden_dim
        print(f"input_size: {input_size}, self.do {self.do}")
        # input_size = self.z_dim + self.output_dim
        self.lstm = nn.LSTM(input_size, self.hidden_dim, self.num_layers, batch_first=True, dropout=self.dropout)
        
        # input_size = self.z_dim + self.treatment_size
        # input_size = self.z_dim + self.treatment_hidden_dim
        input_size = self.hidden_dim
        self.fc_mu = nn.Sequential()
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

        input_dim = self.z_dim + self.output_dim
        hidden_dim = self.hidden_dim
        output_dim = self.hidden_dim
        self.transition_network = DynamicParamNetwork(input_dim, hidden_dim, output_dim, num_rbf_centers=5)

        self.predict_y_history = config['model']['inference']['predict_y_history']
        # 如果 self.predict_y_history 的值是 16，那么 self.predict_y_history 实际上等价于 [16]，即只指定了一个隐藏层为 16 维。
        #
        # 下面的代码会自动构造如下网络结构：
        #   - 输入层: 维度为 (self.z_dim + self.treatment_size)
        #   - 隐藏层: 一个 Linear 层，输出 16 维，然后接一个 ELU 激活
        #   - 输出层: 一个 Linear 层，把 16 维变换为 output_dim 维（同时加一个 ELU 激活）
        #
        # 总体等价于:
        #   nn.Sequential(
        #       nn.Linear(self.z_dim + self.treatment_size, 16),
        #       nn.ELU(),
        #       nn.Linear(16, self.output_dim)
        #   )
        #
        # 这样设计的好处是灵活，可以配置多层或者单层隐藏层结构。这里指定 16 就是单隐藏层。

        self.predict_y_history_net = nn.Sequential()
        input_size = self.z_dim + self.treatment_size
        if -1 not in self.predict_y_history:
            for i in range(len(self.predict_y_history)):
                if i == 0:
                    self.predict_y_history_net.add_module('fc{}'.format(i), nn.Linear(input_size, self.predict_y_history[i]))
                else:
                    self.predict_y_history_net.add_module('elu{}'.format(i), nn.ELU())
                    self.predict_y_history_net.add_module('fc{}'.format(i), nn.Linear(self.predict_y_history[i-1], self.predict_y_history[i]))
            self.predict_y_history_net.add_module('elu{}'.format(len(self.predict_y_history)), nn.ELU())
            self.predict_y_history_net.add_module('fc{}'.format(len(self.predict_y_history)), nn.Linear(self.predict_y_history[-1], self.output_dim))
        else:
            self.predict_y_history_net.add_module('fc{}'.format(1), nn.Linear(input_size, self.output_dim))
    
    def init_hidden(self, batch_size):
        """初始化LSTM隐状态"""
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
        treatments = H_t['current_treatments']
        #  计算事实结果的辅助预测损失 (Factual Outcome Prediction Loss)，强制要求编码器提取出的潜状态 Z_t 必须包含足够的信息，能够准确预测出当前患者的真实医疗指标 Y_t
        y_hat = self.predict_y_history_net(torch.cat((Z_t, treatments), dim=-1))
        loss = nn.MSELoss()(y_hat, H_t['outputs'])
        return Z_t[:, -1, :], loss

    def ct_hidden_history(self, H_t):

        treatments = H_t['prev_treatments']  # A_{t-1}: 历史动作，供 history encoder 编码
        current_treatments = H_t['current_treatments']  # A_t: 与 outputs_t 对齐，供辅助监督
        outputs = H_t['prev_outputs']  # Y, 形状: [batch_size, seq_len, y_dim]
        seq_len = treatments.size(1)  # 获取当前历史序列的时间步长度

        # 1. 协变量 X 分支：与 train_ct / CTHistoryEncoder.x_dim（_covariate_stream_dim）一致
        H_aligned = _expand_static_time_dim(H_t, seq_len)
        x_input = build_covariate_x(H_aligned, self.config)

        # 2. 传入 Causal Transformer 提取时序表示
        # ct_rep 形状: (batch_size, history_length, d_model)
        ct_rep = self.ct_history_encoder(
            x=x_input,
            a=treatments,
            y=outputs,
            active_entries=H_t.get('active_entries', None),
            static_features=H_t.get('static_features', None),
        )

        # 3. 通过投影头，映射到去交杂的 Latent 空间
        # Z_t 形状: (batch_size, history_length, z_dim)
        Z_t = self.projection_head(ct_rep)

        # 4. 辅助回归损失：保证提取出的表示依然具有预测病情的临床意义 (非常关键！)
        # 用当前的潜状态 Z_t 去预测真实的 outputs
        y_hat = self.predict_y_history_net(torch.cat((Z_t, current_treatments), dim=-1))
        loss_hy = nn.MSELoss()(y_hat, H_t['outputs'])

        # 5. 只取历史序列的最后一步作为当前时刻推演的起点 s_t
        Z_s_i = Z_t[:, -1, :]

        # ================= 新增的去交杂部分 =================
        # 4. 对比学习 Loss (参考 CoCo) 或 对抗 Loss (参考 CT) TODO 后续在这里实现 CoCo 对比损失或对抗损失
        # loss_deconfound = self.compute_causal_contrastive_loss(Z_t, treatments)
        loss_deconfound = torch.tensor(0.0).to(Z_t.device)  # 临时占位符

        # 返回给主循环
        return Z_s_i, loss_hy, loss_deconfound
    
    def forward(self, Z_s_prev, a_s, H_t, Y_target):
        """
        Z_s_prev: (batch_size, z_dim)
        a_s: (batch_size, treatment_dim)
        H_t: (batch_size, history_dim)
        Y_target: (batch_size, output_dim)
        """
        # 如果隐状态未初始化，则初始化
        batch_size = Z_s_prev.size(0)
        if self.hidden_state is None:
            self.init_hidden(batch_size)
            
        # 前向传播
        # input = torch.cat([Z_s_prev, a_s, H_t, Y_target], dim=-1).unsqueeze(1)
        # print(f'Z_s_prev shape: {Z_s_prev.shape}, a_s shape: {a_s.shape}, Y_target shape: {Y_target.shape}')
        input = torch.cat([Z_s_prev, Y_target], dim=-1).unsqueeze(1)
        if self.do:
            input = torch.cat([Z_s_prev, Y_target, a_s], dim=-1).unsqueeze(1)

        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
            input, (self.hidden_state, self.cell_state)
        )
        lstm_out = lstm_out.squeeze(1)
        lstm_out = torch.cat([lstm_out], dim=-1)

        # input = torch.cat([Z_s_prev, Y_target], dim=-1).unsqueeze(1)
        # lstm_out = self.transition_network(input, a_s)
        
        q_mu = self.fc_mu(lstm_out)
        q_logvar = self.fc_logvar(lstm_out)
        return q_mu, q_logvar
    
    def reset_states(self):
        """重置LSTM隐状态"""
        self.hidden_state = None
        self.cell_state = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reparameterize_multiple(self, mu, logvar, num_samples=10):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(num_samples, *mu.shape).to(mu.device)
        return mu.unsqueeze(0) + eps * std.unsqueeze(0)
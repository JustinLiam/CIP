import torch
from torch import nn

class VariationalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer=1, dropout_rate=0.0):
        super().__init__()
        
        # 显式指定dtype
        self.dtype = torch.float32  # 或者使用 torch.float64
        
        self.lstm_layers = [nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)]
        if num_layer > 1:
            self.lstm_layers += [nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
                               for _ in range(num_layer - 1)]
        self.lstm_layers = nn.ModuleList(self.lstm_layers)
        
        # 确保所有LSTM层使用相同的数据类型
        for layer in self.lstm_layers:
            layer = layer.type(self.dtype)

        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def forward(self, x, init_states=None):
        # 确保输入数据类型一致
        x = x.type(self.dtype)
        
        for lstm_cell in self.lstm_layers:
            if init_states is None:
                hx = torch.zeros((x.shape[0], self.hidden_size), dtype=self.dtype, device=x.device)
                cx = torch.zeros((x.shape[0], self.hidden_size), dtype=self.dtype, device=x.device)
            else:
                # 确保init_states的数据类型一致
                init_states = init_states.type(self.dtype)
                hx, cx = init_states, init_states

            # 确保dropout mask的数据类型一致
            out_dropout = (torch.bernoulli(torch.full_like(hx, 1 - self.dropout_rate)) / (1 - self.dropout_rate)).type(self.dtype)
            h_dropout = (torch.bernoulli(torch.full_like(hx, 1 - self.dropout_rate)) / (1 - self.dropout_rate)).type(self.dtype)
            c_dropout = (torch.bernoulli(torch.full_like(cx, 1 - self.dropout_rate)) / (1 - self.dropout_rate)).type(self.dtype)

            output = []
            for t in range(x.shape[1]):
                # 确保所有输入到LSTM单元的张量类型一致
                hx, cx = hx.type(self.dtype), cx.type(self.dtype)
                current_input = x[:, t, :].type(self.dtype)
                
                hx, cx = lstm_cell(current_input, (hx, cx))
                
                if lstm_cell.training:
                    out = hx * out_dropout
                    hx, cx = hx * h_dropout, cx * c_dropout
                else:
                    out = hx
                output.append(out)

            x = torch.stack(output, dim=1)

        return x.type(self.dtype)

    def type(self, dtype):
        # 重写type方法以更新内部dtype
        super().type(dtype)
        self.dtype = dtype
        for layer in self.lstm_layers:
            layer.type(dtype)
        return self
import torch
import torch.nn as nn
import numpy as np
import math

class MultiquadricRBFNetwork(nn.Module):
    def __init__(self, centers, c):
        """
        This class constructs a set of multiquadric radial basis functions (RBFs).
        :param centers: list or torch.tensor, the centers of the RBFs
        :param c: float, the constant parameter of the multiquadric RBFs
        """
        super(MultiquadricRBFNetwork, self).__init__()
        self.centers = torch.tensor(centers, dtype=torch.float32)
        self.c = c
        self.num_of_basis = len(centers)
        
        if self.c <= 0:
            print('Parameter c should be positive!')
            raise ValueError
        
        if not isinstance(self.num_of_basis, int) or self.num_of_basis <= 0:
            print('Number of centers should be a positive integer')
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, [batch_size, features]
        :return: the value of each basis given x; [batch_size, self.num_of_basis * features]
        """
        features = x.shape[-1]
        out = torch.zeros(x.shape[0], self.num_of_basis, features, device=x.device)
        for i in range(features):
            for j, center in enumerate(self.centers):
                out[:, j, i] = torch.sqrt((x[:, i] - center) ** 2 + self.c ** 2)
        
        # Flatten the last two dimensions
        out = out.reshape(x.shape[0], self.num_of_basis * features)
        return out



class DynamicParamNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, features=2, num_rbf_centers=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_rbf_centers = num_rbf_centers
        
        centers = np.linspace(-1, 1, num_rbf_centers)
        c = 1
        self.dynamicNet = MultiquadricRBFNetwork(centers, c)
        self.w1 = nn.Parameter(torch.randn(input_dim, hidden_dim, num_rbf_centers * features))
        self.b1 = nn.Parameter(torch.randn(hidden_dim, num_rbf_centers * features))
        self.w2 = nn.Parameter(torch.randn(hidden_dim, output_dim, num_rbf_centers * features))
        
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def standardize(self, x):
        mean = x.mean(dim=1, keepdim=True) 
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + 1e-8)

    def forward(self, x, a):
        x = x.squeeze(1)
        basic = self.dynamicNet(a) # (batch_size, num_rbf_centers * features)
        basic = basic.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, num_rbf_centers * features)
        w1 = self.w1.unsqueeze(0) # (1, input_dim, hidden_dim, num_rbf_centers * features)
        w1 = w1 * basic # (batch_size, input_dim, hidden_dim, num_rbf_centers * features)
        w1 = w1.sum(-1)
        b1 = self.b1.unsqueeze(0) 
        b1 = b1 * basic.squeeze(1) # (batch_size, hidden_dim, num_rbf_centers * features)
        b1 = b1.sum(-1)
        # print(f'w1 shape: {w1.shape}, b1 shape: {b1.shape}, x shape: {x.shape}')
        x = torch.einsum('ab,abc->ac', x, w1) + b1 # (batch_size, hidden_dim)
        x = torch.relu(x)
        w2 = self.w2.unsqueeze(0)
        w2 = w2 * basic
        w2 = w2.sum(-1)
        out = torch.einsum('ab,abc->ac', x, w2) # (batch_size, output_dim)
        # standardize
        out = self.standardize(out)
        
        return out


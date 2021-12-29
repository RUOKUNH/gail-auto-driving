import torch
import pdb
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, net_dims, drop_rate=None):
        super(PolicyNetwork, self).__init__()
        self.net_dims = net_dims
        layers = []
        last_dim = state_dim
        for i in range(len(self.net_dims)):
            layers.append(torch.nn.Linear(last_dim, self.net_dims[i])),
            if drop_rate:
                layers.append(torch.nn.Dropout(drop_rate[i]))
            layers.append(torch.nn.LeakyReLU())
            last_dim = self.net_dims[i]
        layers.append(torch.nn.Linear(self.net_dims[-1], action_dim))
        self.layers = torch.nn.Sequential(*layers)


    def forward(self, states):
        mean, var = torch.chunk(self.layers(states), 2, dim=-1)
        mean /= 10
        var = torch.tanh(var) * 10
        var = F.softplus(var)
        cov_mat = torch.diag_embed(var)
        if mean.ndim > 1:
            mean[:, 0] = torch.tanh(mean[:, 0]) * 5
            mean[:, 1] = torch.tanh(mean[:, 1]) * 0.25
        else:
            mean[0] = torch.tanh(mean[0]) * 5
            mean[1] = torch.tanh(mean[0]) * 0.25
        dist = torch.distributions.MultivariateNormal(mean, cov_mat)

        return dist


class ValueNetwork(torch.nn.Module):
    def __init__(self, state_dim, net_dims, drop_rate=None):
        super(ValueNetwork, self).__init__()
        self.net_dims = net_dims
        layers = []
        last_dim = state_dim
        for i in range(len(self.net_dims)):
            layers.append(torch.nn.Linear(last_dim, self.net_dims[i]))
            if drop_rate:
                layers.append(torch.nn.Dropout(drop_rate[i]))
            layers.append(torch.nn.LeakyReLU())
            last_dim = self.net_dims[i]
        layers.append(torch.nn.Linear(self.net_dims[-1], 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, states):
        val = self.layers(states)
        return val


class Discriminator(torch.nn.Module):
    def __init__(self, state_dim, action_dim, net_dims, drop_rate=None):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        net_in_dim = state_dim + action_dim
        self.net_dims = net_dims
        layers = []
        last_dim = net_in_dim
        for i in range(len(self.net_dims)):
            layers.append(torch.nn.Linear(last_dim, self.net_dims[i]))
            if drop_rate:
                layers.append(torch.nn.Dropout(drop_rate[i]))
            layers.append(torch.nn.LeakyReLU())
            last_dim = self.net_dims[i]
        layers.append(torch.nn.Linear(self.net_dims[-1], 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        x = self.layers(x)
        return x

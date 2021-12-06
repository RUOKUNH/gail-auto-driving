import torch
import pdb


def init_weights_xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


init = 'xavier'
# init = None

class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, use_bn=False, init=init):
        super(PolicyNetwork, self).__init__()
        # self.net_dims = [64, 64, 128, 256]
        self.net_dims = [96, 96, 128, 128, 256]
        layers = [
            torch.nn.Linear(state_dim, self.net_dims[0])
        ]
        if use_bn:
            layers.append(torch.nn.BatchNorm1d(self.net_dims[0]))
        layers.append(torch.nn.Tanh())
        for i in range(len(self.net_dims)-1):
            layers.append(torch.nn.Linear(self.net_dims[i], self.net_dims[i+1]))
            if use_bn:
                layers.append(torch.nn.BatchNorm1d(self.net_dims[i+1]))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(self.net_dims[-1], action_dim))
        self.net = torch.nn.Sequential(*layers)
        # self.net = torch.nn.Sequential(
        #     torch.nn.Linear(state_dim, 64),
        #     # torch.nn.BatchNorm1d(96),
        #     # torch.nn.Tanh(),
        #     # torch.nn.Linear(32, 64),
        #     # torch.nn.BatchNorm1d(96),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(64, 64),
        #     # torch.nn.BatchNorm1d(128),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(64, 128),
        #     # torch.nn.BatchNorm1d(128),
        #     # torch.nn.Linear(state_dim, 64),
        #     # torch.nn.Tanh(),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(128, 256),
        #     # torch.nn.BatchNorm1d(256),
        #     # torch.nn.Linear(64, 128),
        #     # torch.nn.Tanh(),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(256, action_dim)
        #     # torch.nn.Linear(128, action_dim)
        # )

        self.state_dim = state_dim
        self.action_dim = action_dim

        if init == 'xavier':
            # pdb.set_trace()
            for module in self.modules():
                init_weights_xavier(module)

    def forward(self, states):
        _states = torch.FloatTensor(states)
        if _states.ndim == 1:
            _states = _states.view(1, -1)
        mean = self.net(_states)
        if states.ndim == 1:
            mean = mean[0]

        if mean.ndim > 1:
            mean[:, 0] = torch.tanh(mean[:, 0])
            mean[:, 1] = torch.tanh(mean[:, 1]) * 0.1
        else:
            mean[0] = torch.tanh(mean[0])
            mean[1] = torch.tanh(mean[0]) * 0.1

        cov_mtx = torch.FloatTensor([[5e-2, 0], [0, 25e-4]])

        distb = torch.distributions.MultivariateNormal(mean, cov_mtx)

        return distb


class ValueNetwork(torch.nn.Module):
    def __init__(self, state_dim, use_bn=False, init=init):
        super(ValueNetwork, self).__init__()
        # self.net_dims = [64, 64, 128]
        self.net_dims = [96, 96, 128, 128, 256]
        layers = [
            torch.nn.Linear(state_dim, self.net_dims[0])
        ]
        if use_bn:
            layers.append(torch.nn.BatchNorm1d(self.net_dims[0]))
        layers.append(torch.nn.Tanh())
        for i in range(len(self.net_dims)-1):
            layers.append(torch.nn.Linear(self.net_dims[i], self.net_dims[i + 1]))
            if use_bn:
                layers.append(torch.nn.BatchNorm1d(self.net_dims[i + 1]))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(self.net_dims[-1], 1))
        self.net = torch.nn.Sequential(*layers)
        # self.net = torch.nn.Sequential(
        #     torch.nn.Linear(state_dim, 64),
        #     # torch.nn.BatchNorm1d(96),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(64, 64),
        #     # torch.nn.BatchNorm1d(128),
        #     # torch.nn.Tanh(),
        #     # torch.nn.Linear(128, 128),
        #     # torch.nn.BatchNorm1d(128),
        #     # torch.nn.Linear(state_dim, 64),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(64, 128),
        #     # torch.nn.BatchNorm1d(256),
        #     # torch.nn.Linear(64, 128),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(128, 1)
        # )

        if init == 'xavier':
            for module in self.modules():
                init_weights_xavier(module)

    def forward(self, states):
        _states = torch.FloatTensor(states)
        if _states.ndim == 1:
            _states = _states.view(1, -1)
        val = self.net(_states)
        if states.ndim == 1:
            val = val[0]
        return val


class Discriminator(torch.nn.Module):
    def __init__(self, state_dim, action_dim, use_bn=False, init=init):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_in_dim = state_dim + action_dim
        # self.net_dims = [64, 64, 128, 128, 256]
        self.net_dims = [96, 96, 128, 128, 256]
        layers = [
            torch.nn.Linear(self.net_in_dim, self.net_dims[0])
        ]
        if use_bn:
            layers.append(torch.nn.BatchNorm1d(self.net_dims[0]))
        layers.append(torch.nn.Tanh())
        for i in range(len(self.net_dims)-1):
            layers.append(torch.nn.Linear(self.net_dims[i], self.net_dims[i + 1]))
            if use_bn:
                layers.append(torch.nn.BatchNorm1d(self.net_dims[i + 1]))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(self.net_dims[-1], 1))
        self.net = torch.nn.Sequential(*layers)

        # self.net = torch.nn.Sequential(
        #     torch.nn.Linear(self.net_in_dim, 64),
        #     # torch.nn.BatchNorm1d(96),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(64, 64),
        #     # torch.nn.BatchNorm1d(96),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(64, 128),
        #     # torch.nn.BatchNorm1d(128),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(128, 128),
        #     # torch.nn.BatchNorm1d(128),
        #     # torch.nn.Linear(state_dim, 64),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(128, 256),
        #     # torch.nn.BatchNorm1d(256),
        #     # torch.nn.Linear(64, 128),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(256, 1)
        # )

        if init == 'xavier':
            for module in self.modules():
                init_weights_xavier(module)

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        sa = torch.cat([states, actions], dim=-1)
        if sa.ndim == 1:
            sa = sa.view(1, -1)
            return self.net(sa)[0]
        else:
            return self.net(sa)

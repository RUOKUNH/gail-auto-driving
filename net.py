import torch
import pdb

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

#
# def init_weights(m):
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight)
#         m.bias.data.fill_(0.01)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 96),
            # torch.nn.ELU(),
            torch.nn.ELU(),
            torch.nn.Linear(96, 96),
            # torch.nn.ELU(),
            torch.nn.ELU(),
            torch.nn.Linear(96, 128),
            # torch.nn.ELU(),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            # torch.nn.Linear(state_dim, 64),
            # torch.nn.ELU(),
            torch.nn.ELU(),
            torch.nn.Linear(128, 256),
            # torch.nn.Linear(64, 128),
            # torch.nn.ELU(),
            torch.nn.ELU(),
            torch.nn.Linear(256, action_dim)
            # torch.nn.Linear(128, action_dim)
        )
        self.scale_net = torch.nn.Tanh()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cov = 1
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def forward(self, states):
        # pdb.set_trace()
        states = torch.FloatTensor(states)
        mean = self.net(states)

        if mean.ndim > 1:
            mean[:, 0] = torch.tanh(mean[:, 0])
            mean[:, 1] = torch.tanh(mean[:, 1]) * 0.1
        else:
            mean[0] = torch.tanh(mean[0])
            mean[1] = torch.tanh(mean[0]) * 0.1

        # cov_mtx = torch.FloatTensor([[5e-2, 0], [0, 25e-4]])
        std = torch.exp(self.log_std)
        cov_mtx = torch.eye(self.action_dim) * (std ** 2)

        distb = torch.distributions.MultivariateNormal(mean, cov_mtx)

        return distb


class ValueNetwork(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 96),
            torch.nn.ELU(),
            # torch.nn.ReLU(),
            torch.nn.Linear(96, 96),
            torch.nn.ELU(),
            # torch.nn.ReLU(),
            torch.nn.Linear(96, 128),
            torch.nn.ELU(),
            # torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            # torch.nn.Linear(state_dim, 64),
            torch.nn.ELU(),
            # torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            # torch.nn.Linear(64, 128),
            torch.nn.ELU(),
            # torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, states):
        states = torch.FloatTensor(states)
        return self.net(states)


class Discriminator(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net_in_dim = state_dim + action_dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.net_in_dim, 96),
            torch.nn.ELU(),
            # torch.nn.ReLU(),
            torch.nn.Linear(96, 96),
            torch.nn.ELU(),
            # torch.nn.ReLU(),
            torch.nn.Linear(96, 128),
            torch.nn.ELU(),
            # torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            # torch.nn.Linear(state_dim, 64),
            torch.nn.ELU(),
            # torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            # torch.nn.Linear(64, 128),
            torch.nn.ELU(),
            # torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        # pdb.set_trace()
        sa = torch.cat([states, actions], dim=-1)

        return self.net(sa)

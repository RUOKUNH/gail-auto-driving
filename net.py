import torch
import pdb


def init_weights_xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


init = 'xavier'
# init = None

# net 1
# p_dims = [64, 64, 128, 256]
# v_dims = [64, 64, 128]
# d_dims = [64, 64, 128, 128, 256]

# net 2
# p_dims = [96, 96, 128, 128, 256]
# v_dims = [96, 96, 128, 128, 256]
# d_dims = [96, 96, 128, 128, 256]

# net 3
# p_dims = [64, 64, 128, 256]
# v_dims = [96, 96, 128, 128, 256]
# d_dims = [96, 96, 128, 128, 256]

# net 4
# p_dims = [96, 96, 128, 128, 256]
# v_dims = [128] * 7
# d_dims = [128] * 7

# net 5
# p_dims = [128] * 5
# v_dims = [128] * 7
# d_dims = [128] * 7

# net 6
# p_dims = [128] * 4
# v_dims = [128] * 4
# d_dims = [128] * 4

# net 7
p_dims = [256, 128, 64, 32]
v_dims = [128] * 4
d_dims = [128] * 4

# net 8
# p_dims = [256, 128, 64, 32]
# v_dims = [256, 128, 64, 32]
# d_dims = [256, 128, 64, 32]


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.net_dims = p_dims
        self.layers = torch.nn.ModuleList()
        self.input = torch.nn.Linear(state_dim, self.net_dims[0])
        for i in range(len(self.net_dims) - 1):
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.net_dims[i], self.net_dims[i + 1]),
                    torch.nn.ELU())
            )

        self.output = torch.nn.Linear(self.net_dims[-1], action_dim)

        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, states):
        states = torch.FloatTensor(states)
        x = self.input(states)
        for layer in self.layers:
            # x = layer(x) + x
            x = layer(x)
        mean = self.output(x)

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
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.net_dims = v_dims
        self.layers = torch.nn.ModuleList()
        self.input = torch.nn.Linear(state_dim, self.net_dims[0])
        for i in range(len(self.net_dims)-1):
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.net_dims[i], self.net_dims[i + 1]),
                    torch.nn.ELU())
            )

        self.output = torch.nn.Linear(self.net_dims[-1], 1)

    def forward(self, states):
        states = torch.FloatTensor(states)
        x = self.input(states)
        for layer in self.layers:
            x = layer(x) + x
            # x = layer(x)
        x = self.output(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_in_dim = state_dim + action_dim
        self.net_dims = d_dims
        self.input = torch.nn.Linear(self.net_in_dim, self.net_dims[0])
        self.layers = torch.nn.ModuleList()
        for i in range(len(self.net_dims)-1):
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.net_dims[i], self.net_dims[i + 1]),
                    torch.nn.ELU())
            )
        self.output = torch.nn.Linear(self.net_dims[-1], 1)

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        x = self.input(x)
        for layer in self.layers:
            x = layer(x) + x
            # x = layer(x)
        x = self.output(x)
        return x

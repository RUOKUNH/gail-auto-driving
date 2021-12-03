import torch
import argparse
import os
from pathlib import Path
import sys
import numpy as np

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.Tanh(),
            # torch.nn.ReLU(),
            # torch.nn.Linear(32, 64),
            # torch.nn.Tanh(),
            # torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            # torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            # torch.nn.Linear(state_dim, 64),
            torch.nn.Tanh(),
            # torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            # torch.nn.Linear(64, 128),
            torch.nn.Tanh(),
            # torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim)
            # torch.nn.Linear(128, action_dim)
        )
        self.scale_net = torch.nn.Tanh()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cov = 1

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

        cov_mtx = torch.FloatTensor([[5e-2, 0], [0, 25e-4]])

        distb = torch.distributions.MultivariateNormal(mean, cov_mtx)

        return distb

pi = PolicyNetwork(20, 2)
model_path = os.path.dirname(os.path.abspath(__file__)) + '/bestmodelexp29.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))
pi.load_state_dict(model['action_net'])

sys.path.pop(-1)

def make_obs(obs):
    _obs = []
    ego_state = obs.ego_vehicle_state
    neighbors = obs.neighborhood_vehicle_states
    neighbor_idx = np.arange(len(neighbors))
    x, y = ego_state.position[:2]
    l = ego_state.bounding_box.length
    w = ego_state.bounding_box.width
    heading = ego_state.heading.real
    speed = ego_state.speed
    _obs += [heading, speed, x, y, l, w]
    neighbor_position = np.array([n.position[:2] for n in neighbors])
    if len(neighbor_position) > 0:
        dist = np.sqrt((neighbor_position[:, 0] - x)**2 + (neighbor_position[:, 1] - y)**2)
        nei_dist = [(neighbor_idx[i], dist[i]) for i in neighbor_idx]
        nei_dist.sort(key=lambda x: x[1])
    # closest 4 neighbors
    l = min(len(neighbor_position), 4)
    for i in range(l):
        n = neighbors[nei_dist[i][0]]
        n_x, n_y = n.position[:2]
        n_l = n.bounding_box.length
        n_w = n.bounding_box.width
        _obs += [n_x, n_y, n_l, n_w, nei_dist[i][1]]
    for i in range(l, 4):
        _obs += [0, 0, 0, 0, 0]
    return _obs

def make_obs_2(obs):
    heading, speed, x, y, l, w = obs[:6]
    _obs = [heading, speed, x, y]
    for i in range(4):
        n_x, n_y, n_l, n_w, d = obs[6+i*5:11+i*5]
        if d == 0:
            _obs += [0, 0, 0, 0]
        else:
            dx = n_x - x
            dy = n_y - y
            if dx == 0:
                r_heading = -np.pi/2
            else:
                r_heading = np.arctan(dy / dx)
                if r_heading > 0:
                    r_heading -= np.pi
            _obs += [dx, dy, d, r_heading]

    return _obs


def my_controller(observation, action_space, is_act_continuous=False):

    ob = make_obs(observation['obs'])
    ob = make_obs_2(ob)

    act = pi(ob).sample()

    act = [act.cpu().numpy()]

    return act

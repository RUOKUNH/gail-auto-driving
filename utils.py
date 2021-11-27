import numpy as np
import torch as torch


def get_flat_grads(f, net):
    flat_grads = torch.cat([
        grad.view(-1)
        for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
    ])

    return flat_grads

def get_flat_params(net):
    return torch.cat([param.view(-1) for param in net.parameters()])



def set_params(net, new_flat_params):
    start_idx = 0
    for param in net.parameters():
        end_idx = start_idx + np.prod(list(param.shape))
        param.data = torch.reshape(
            new_flat_params[start_idx:end_idx], param.shape
        )

        start_idx = end_idx


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


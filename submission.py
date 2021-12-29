import torch
import argparse
import torch.nn.functional as F
import os
from pathlib import Path
import sys
import numpy as np

class BC:
    def __init__(self, PolicyNet, device=None):
        self.policy = PolicyNet
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters())

    def action(self, state):
        # dist = self.policy(state)
        # return dist.sample()
        action = self.policy(state)
        if action.ndim == 2:
            action[:, 1] = 0
        else:
            action[1] = 0
        return action

    def dist(self, state):
        dist = self.policy(state)
        return dist

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def update(self, batch):
        state = batch['state'].float()
        action = batch['action'].detach().float()
        # _dist = self.dist(state)
        # log_probs = _dist.log_prob(action.reshape(_dist.sample().shape))
        # actor_loss = torch.mean(-log_probs)

        _action = self.action(state)
        actor_loss1 = F.mse_loss(_action[:, 0], action[:, 0])
        actor_loss2 = F.mse_loss(_action[:, 1], action[:, 1])
        # actor_loss2 = actor_loss1.detach()
        actor_loss = actor_loss2 + actor_loss1
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # loss = actor_loss.cpu().detach().numpy()
        return actor_loss1.cpu().detach().numpy(), actor_loss2.cpu().detach().numpy()

    def save_model(self, path, epoch):
        state = {'policy': self.policy.state_dict(),
                 'epoch': epoch}
        torch.save(state, path)

    def load_model(self, path):
        state = torch.load(path, map_location=torch.device('cpu'))
        self.policy.load_state_dict(state['policy'])
        return state['epoch']

class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.net_dims = [256, 128, 64, 32]
        layers = []
        last_dim = state_dim
        for i in range(len(self.net_dims)):
            layers.append(torch.nn.Linear(last_dim, self.net_dims[i]))
            layers.append(torch.nn.ReLU())
            last_dim = self.net_dims[i]
        layers.append(torch.nn.Linear(self.net_dims[-1], action_dim))
        self.layers = torch.nn.Sequential(*layers)

        # self.output = torch.nn.Linear(self.net_dims[-1], action_dim)

        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.var = torch.nn.Parameter(torch.ones(2), requires_grad=True)

    def forward(self, states):
        # x = torch.FloatTensor(states)
        # x = self.input(states)
        # for layer in self.layers:
        #     x = layer(x)
        mean = self.layers(states)
        # mean, var = torch.chunk(self.output(x), 2, dim=-1)
        # var = F.softplus(var)
        # if var.nim
        # var = torch.cat([torch.diag(var[i]).view(1, 2, 2) for i in range(len(var))])

        # mean = self.output(x)

        # mean = torch.tanh(mean) * 2
        # var = torch.tanh(self.var) * 0.1
        # var = torch.sigmoid(self.var) * 0.5
        # var = torch.diag(self.var)

        # if mean.ndim > 1:
        #     mean[:, 0] = torch.tanh(mean[:, 0])
        #     mean[:, 1] = torch.tanh(mean[:, 1]) * 0.1
        # else:
        #     mean[0] = torch.tanh(mean[0])
        #     mean[1] = torch.tanh(mean[0]) * 0.1
        #
        # var = torch.FloatTensor([[5e-2, 0], [0, 25e-4]])
        # dist = torch.distributions.MultivariateNormal(mean, var)

        # return dist
        return mean

    # def get_var(self):
    #     return torch.sigmoid(self.var) * 0.5

lane_heading = {}
lane_heading['gneE05a_0'] = [
    np.arctan((-1.75 + 1.98) / 20.75),
    np.arctan((-1.34 + 1.75) / (49.11 - 20.75)),
    np.arctan((-0.53 + 1.34) / (85.43 - 49.11)),
    np.arctan((1.40 + 0.53) / (132.38 - 85.43)),
    np.arctan((2.52 - 1.40) / (165.1 - 132.38)),
    np.arctan((2.57 - 2.52) / (180 - 165.1))
]
lane_heading['gneE05b_0'] = 0
lane_heading['gneE51_0'] = 0
lane_heading['gneE01_0'] = 0
lane_heading['gneE01_1'] = 0
lane_heading['gneE01_2'] = 0
lane_heading['gneE01_3'] = 0
lane_heading['gneE01_4'] = 0

lane_width = {
    'gneE05a_0': 4.25,
    'gneE05b_0': 7.4738,
    'gneE51_0': 6.21,
    'gneE01_0': 3.6576,
    'gneE01_1': 3.6576,
    'gneE01_2': 3.6576,
    'gneE01_3': 3.6576,
    'gneE01_4': 3.6576,
}



def clip(x, low, high):
    if x < low:
        return low
    if x > high:
        return high
    return x


def get_lane_info(lane_id, x):
    if lane_id == 'gneE05a_0':
        if 0 <= x < 20.75:
            heading = lane_heading[lane_id][0]
            y = -1.98 + (x-0) * np.tan(heading)
        elif 20.75 <= x < 49.11:
            heading = lane_heading[lane_id][1]
            y = -1.75 + (x-20.75) * np.tan(heading)
        elif 49.11 <= x < 85.43:
            heading = lane_heading[lane_id][2]
            y = -1.34 + (x-49.11) * np.tan(heading)
        elif 85.43 <= x < 132.38:
            heading = lane_heading[lane_id][3]
            y = -0.53 + (x-85.43) * np.tan(heading)
        elif 132.38 <= x < 165.1:
            heading = lane_heading[lane_id][4]
            y = 1.40 + (x-132.38) * np.tan(heading)
        else:
            heading = lane_heading[lane_id][5]
            y = 2.52 + (x-165.1) * np.tan(heading)
    else:
        heading = 0
        if lane_id == 'gneE05b_0':
            y = 2.87
        elif lane_id == 'gneE51_0':
            y = 3.50
        elif lane_id == 'gneE01_0':
            y = 8.41
        elif lane_id == 'gneE01_1':
            y = 12.10
        elif lane_id == 'gneE01_2':
            y = 15.79
        elif lane_id == 'gneE01_3':
            y = 19.48
        else:
            y = 23.18
    return heading, y


def conjugate_gradient(Av_func, b, max_iter=50, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b - Av_func(x)
    p = r
    rsold = r.norm() ** 2  # 2-norm

    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


# only available for gaussian distributions
# refer https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#
# As the cov mat is diag, so some operations like trace and inverse is achieved by sumation
def kl_divergence(dist, old_dist, action_dim):
    old_mean = old_dist.mean.detach()
    old_cov = old_dist.covariance_matrix.sum(-1).detach()
    mean = dist.mean.detach()
    cov = dist.covariance_matrix.sum(-1).detach()
    return 0.5 * ((old_cov / cov).sum(-1)
                  + (((old_mean - mean) ** 2) / cov).sum(-1)
                  - action_dim
                  + torch.log(cov).sum(-1)
                  - torch.log(old_cov).sum(-1)).mean()


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


# [heading, speed, x, y, l, w, nx,ny,nl,nw,nh,ns,....(4 neighbor)]
def expert_collector(obs):
    _obs = []
    ego_state = obs.ego_vehicle_state
    neighbors = obs.neighborhood_vehicle_states
    neighbor_idx = np.arange(len(neighbors))
    x, y = ego_state.position[:2]
    l = ego_state.bounding_box.length
    w = ego_state.bounding_box.width
    heading = ego_state.heading.real
    speed = ego_state.speed
    # _obs += [heading, speed, x, y, l, w]
    _obs = [ego_state]
    neighbor_position = np.array([n.position[:2] for n in neighbors])
    if len(neighbor_position) > 0:
        dist = np.sqrt((neighbor_position[:, 0] - x) ** 2 + (neighbor_position[:, 1] - y) ** 2)
        nei_dist = [(neighbor_idx[i], dist[i]) for i in neighbor_idx]
        nei_dist.sort(key=lambda x: x[1])
    # closest 4 neighbors
    l = min(len(neighbor_position), 4)
    for i in range(l):
        n = neighbors[nei_dist[i][0]]
        n_x, n_y = n.position[:2]
        n_l = n.bounding_box.length
        n_w = n.bounding_box.width
        n_heading = n.heading
        n_speed = n.speed
        _obs += [n_x, n_y, n_l, n_w, n_heading, n_speed]
    for i in range(l, 4):
        _obs += [0, 0, 0, 0, 0, 0]
    return _obs


def expert_collector2(obs):
    _obs = []
    ego_state = obs.ego_vehicle_state
    neighbors = obs.neighborhood_vehicle_states
    neighbor_idx = np.arange(len(neighbors))
    x, y = ego_state.position[:2]
    l = ego_state.bounding_box.length
    w = ego_state.bounding_box.width
    heading = ego_state.heading.real
    speed = ego_state.speed
    _obs = [ego_state]
    neighbor_position = np.array([n.position[:2] for n in neighbors])
    if len(neighbor_position) > 0:
        dist = np.sqrt((neighbor_position[:, 0] - x) ** 2 + (neighbor_position[:, 1] - y) ** 2)
        nei_dist = [(neighbor_idx[i], dist[i]) for i in neighbor_idx]
        nei_dist.sort(key=lambda x: x[1])
    # closest 4 neighbors
    l = min(len(neighbor_position), 4)
    for i in range(l):
        n = neighbors[nei_dist[i][0]]
        n_x, n_y = n.position[:2]
        n_l = n.bounding_box.length
        n_w = n.bounding_box.width
        n_heading = n.heading
        n_speed = n.speed
        _obs += [n_x, n_y, n_l, n_w, n_heading, n_speed]
    for i in range(l, 4):
        _obs += [0, 0, 0, 0, 0, 0]
    events = [0] * 4
    Event = obs.events
    if len(Event.collisions) > 0:
        events[0] = 1
    if Event.off_road:
        events[1] = 1
    if ego_state.speed < 0:
        events[2] = 1
    if np.pi/2 < normalize(ego_state.heading + np.pi/2, 0, np.pi*2, np.pi*2) < np.pi/2*3:
        events[3] = 1
    _obs += events
    return _obs


# 16 nearest neighbors
def expert_collector3(obs):
    _obs = []
    ego_state = obs.ego_vehicle_state
    neighbors = obs.neighborhood_vehicle_states
    neighbor_idx = np.arange(len(neighbors))
    x, y = ego_state.position[:2]
    l = ego_state.bounding_box.length
    w = ego_state.bounding_box.width
    heading = ego_state.heading.real
    speed = ego_state.speed
    _obs = [ego_state]
    neighbor_position = np.array([n.position[:2] for n in neighbors])
    if len(neighbor_position) > 0:
        dist = np.sqrt((neighbor_position[:, 0] - x) ** 2 + (neighbor_position[:, 1] - y) ** 2)
        nei_dist = [(neighbor_idx[i], dist[i]) for i in neighbor_idx]
        nei_dist.sort(key=lambda x: x[1])
    # closest 16 neighbors
    neighbor_num = 16
    l = min(len(neighbor_position), neighbor_num)
    for i in range(l):
        n = neighbors[nei_dist[i][0]]
        n_x, n_y = n.position[:2]
        n_l = n.bounding_box.length
        n_w = n.bounding_box.width
        n_heading = n.heading
        n_speed = n.speed
        _obs += [n_x, n_y, n_l, n_w, n_heading, n_speed]
    for i in range(l, neighbor_num):
        _obs += [0, 0, 0, 0, 0, 0]
    events = [0] * 4
    Event = obs.events
    if len(Event.collisions) > 0:
        events[0] = 1
    if Event.off_road:
        events[1] = 1
    if ego_state.speed < 0:
        events[2] = 1
    if np.pi/2 < normalize(ego_state.heading + np.pi/2, 0, np.pi*2, np.pi*2) < np.pi/2*3:
        events[3] = 1
    _obs += events
    return _obs


# heading normalize to [-pi, pi)
# record dist, relative speed-x, relative speed-y at four directions, line id (-1, 0, 1)
def feature2(obs):
    ego_state = obs[0]
    vehicles = []
    cars = 4
    for i in range(cars):
        vehicle_state = obs[6 * i + 1: 6 * i + 7]
        vehicles.append(vehicle_state)
    e_x, e_y = ego_state.position[:2]
    e_h = ego_state.heading.real
    e_s = ego_state.speed
    e_l = ego_state.bounding_box.length
    e_w = ego_state.bounding_box.width
    land_index = ego_state.lane_index
    # e_h, e_s, e_x, e_y, e_l, e_w = ego_state
    e_h += np.pi / 2
    e_h = normalize(e_h, -np.pi, np.pi, 2*np.pi)
    e_sx = e_s * np.cos(e_h)
    e_sy = e_s * np.sin(e_h)
    e_corners = get_corners(e_x, e_y, e_l, e_w, e_h)
    e_corners = np.array(e_corners)
    e_max_y = np.max(e_corners[:, 1])
    e_min_y = np.min(e_corners[:, 1])
    e_max_x = np.max(e_corners[:, 0])
    e_min_x = np.min(e_corners[:, 0])
    feature = np.array([
        e_x / 100, e_y / 10, e_h, e_s / 10, e_sx / 10, e_sy / 10, e_max_y / 10,
        e_min_y / 10, e_max_x / 100, e_min_x / 100])
    radius_sample = 8
    radius = np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample)
    e_dists = [get_cross_point_dist(e_x, e_y, r+e_h, e_corners, e_h) for r in radius]
    dists = np.ones(radius_sample) * 20
    r_speeds_x = np.zeros(radius_sample)
    r_speeds_y = np.zeros(radius_sample)
    for vehicle_state in vehicles:
        if not np.any(vehicle_state):
            continue
        # vehicle_state[4] += np.pi / 2
        # vehicle_state[4] = normalize(vehicle_state[4], -np.pi, np.pi, 2*np.pi)
        x, y, l, w, heading, speed = vehicle_state
        heading += np.pi / 2
        heading = normalize(heading, -np.pi, np.pi, 2*np.pi)
        corners = get_corners(x, y, l, w, heading)  # four corner of the boxing
        corners = np.array(corners)
        corner_angles = np.arctan((corners[:, 1] - e_y) / (corners[:, 0] - e_x + 1e-8))
        corner_angles[(corners[:, 0] - e_x) < 0] += np.pi
        corner_angles[corner_angles < 0] += 2*np.pi
        r_angles = corner_angles - e_h  # corner angles relative to ego_heading
        for i in range(len(r_angles)):
            r_angles[i] = normalize(r_angles[i], 0, 2*np.pi, 2*np.pi)
        min_angle, max_angle = np.min(r_angles), np.max(r_angles)
        for i in range(radius_sample):
            r = radius[i]
            if (max_angle-min_angle < np.pi) and (r<min_angle) or (r>max_angle):
                continue
            if (max_angle-min_angle > np.pi) and (min_angle < r <max_angle):
                continue
            real_r = e_h + r
            dist = get_cross_point_dist(e_x, e_y, real_r, corners, heading)
            dist -= e_dists[i]
            if dist < dists[i]:
                dists[i] = dist
                r_speeds_x[i] = speed * np.cos(heading) - e_s * np.cos(e_h)
                r_speeds_y[i] = speed * np.sin(heading) - e_s * np.sin(e_h)
    ego_land_index = np.zeros(5)
    ego_land_index[land_index] = 1
    dists /= 20
    r_speeds_x /= 10
    r_speeds_y /= 10
    feature = np.concatenate((feature, dists, r_speeds_x, r_speeds_y))
    return feature

def feature3(obs):
    ego_state = obs[0]
    vehicles = []
    for i in range(8):
        vehicle_state = obs[6 * i + 1: 6 * i + 7]
        vehicles.append(vehicle_state)
    events = obs[-4:]

    # ego info
    e_x, e_y = ego_state.position[:2]
    e_h = ego_state.heading.real
    e_s = ego_state.speed
    e_l = ego_state.bounding_box.length
    e_w = ego_state.bounding_box.width
    land_index = ego_state.lane_index
    # e_h, e_s, e_x, e_y, e_l, e_w = ego_state
    e_h += np.pi / 2
    e_h = normalize(e_h, -np.pi, np.pi, 2*np.pi)
    e_sx = e_s * np.cos(e_h)
    e_sy = e_s * np.sin(e_h)

    # neighbor info
    e_corners = get_corners(e_x, e_y, e_l, e_w, e_h)
    e_corners = np.array(e_corners)
    e_max_y = np.max(e_corners[:, 1])
    e_min_y = np.min(e_corners[:, 1])
    radius_sample = 8
    radius = np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample)
    e_dists = [get_cross_point_dist(e_x, e_y, r+e_h, e_corners, e_h) for r in radius]
    dists = np.ones(radius_sample) * 20
    r_speeds_l = np.zeros(radius_sample)
    r_speeds_r = np.zeros(radius_sample)
    for vehicle_state in vehicles:
        if not np.any(vehicle_state):
            continue
        # vehicle_state[4] += np.pi / 2
        # vehicle_state[4] = normalize(vehicle_state[4], -np.pi, np.pi, 2*np.pi)
        x, y, l, w, heading, speed = vehicle_state
        heading += np.pi / 2
        heading = normalize(heading, -np.pi, np.pi, 2*np.pi)
        corners = get_corners(x, y, l, w, heading)  # four corner of the boxing
        corners = np.array(corners)
        corner_angles = np.arctan((corners[:, 1] - e_y) / (corners[:, 0] - e_x + 1e-8))
        corner_angles[(corners[:, 0] - e_x) < 0] += np.pi
        corner_angles[corner_angles < 0] += 2*np.pi
        r_angles = corner_angles - e_h  # corner angles relative to ego_heading
        for i in range(len(r_angles)):
            r_angles[i] = normalize(r_angles[i], 0, 2*np.pi, 2*np.pi)
        min_angle, max_angle = np.min(r_angles), np.max(r_angles)
        for i in range(radius_sample):
            r = radius[i]
            if (max_angle-min_angle < np.pi) and (r<min_angle) or (r>max_angle):
                continue
            if (max_angle-min_angle > np.pi) and (min_angle < r <max_angle):
                continue
            real_r = e_h + r
            dist = get_cross_point_dist(e_x, e_y, real_r, corners, heading)
            dist -= e_dists[i]
            if dist < dists[i]:
                dists[i] = dist
                r_speeds_l[i] = speed * np.cos(heading - e_h) - e_s
                r_speeds_r[i] = speed * np.sin(heading - e_h)

    # lane info
    lane_id = ego_state.lane_id
    lane_h, lane_y = get_lane_info(lane_id, e_x)
    lane_offset = (e_y - lane_y) * np.cos(lane_h)
    lane_relative_h = e_h - lane_h
    e_s_lane = e_s * np.cos(lane_relative_h)
    e_s_lateral = e_s * np.sin(lane_relative_h)
    marker_dist_l = 25.02 - e_y
    if e_x < 180:
        _h, _y = get_lane_info('gneE05a_0', e_x)
        marker_dist_r = e_y - (_y - lane_width['gneE05a_0']/2/np.cos(_h))
    else:
        marker_dist_r = e_y - (3.49 - lane_width['gneE51_0']/2)
    ego_info = np.array([
        e_x / 100, e_y / 10, lane_relative_h, lane_offset / 2, e_l / 4, e_w / 2, e_s_lane / 10, e_s_lateral / 10,
        e_s / 10, marker_dist_l / 10, marker_dist_r / 10
    ])
    dists /= 20
    r_speeds_l /= 10
    r_speeds_r /= 10

    # all info
    feature = np.concatenate((ego_info, dists, r_speeds_l, r_speeds_r, events))
    return feature

def feature4(obs):
    ego_state = obs[0]
    vehicles = []
    for i in range(4):
        vehicle_state = obs[6 * i + 1: 6 * i + 7]
        vehicles.append(vehicle_state)

    # ego info
    e_x, e_y = ego_state.position[:2]
    e_h = ego_state.heading.real
    e_s = ego_state.speed
    e_l = ego_state.bounding_box.length
    e_w = ego_state.bounding_box.width
    e_h += np.pi / 2
    e_h = normalize(e_h, -np.pi, np.pi, 2*np.pi)

    # lane info
    lane_id = ego_state.lane_id
    lane_h, lane_y = get_lane_info(lane_id, e_x)
    lane_offset = (e_y - lane_y) * np.cos(lane_h)
    lane_relative_h = e_h - lane_h
    e_s_lane = e_s * np.cos(lane_relative_h)
    e_s_lateral = e_s * np.sin(lane_relative_h)
    marker_dist_l = 25.02 - e_y
    if e_x < 180:
        _h, _y = get_lane_info('gneE05a_0', e_x)
        marker_dist_r = e_y - (_y - lane_width['gneE05a_0'] / 2 / np.cos(_h))
    else:
        marker_dist_r = e_y - (3.49 - lane_width['gneE51_0'] / 2)

    # neighbor info
    e_corners = get_corners(e_x, e_y, e_l, e_w, e_h)
    e_corners = np.array(e_corners)
    radius_sample = 4
    radius = np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample) + lane_h
    e_dists = [get_cross_point_dist(e_x, e_y, r, e_corners, e_h) for r in radius]
    dists = np.ones(radius_sample) * 20
    r_speeds_x = np.zeros(radius_sample)
    r_speeds_y = np.zeros(radius_sample)
    for vehicle_state in vehicles:
        if not np.any(vehicle_state):
            continue
        # vehicle_state[4] += np.pi / 2
        # vehicle_state[4] = normalize(vehicle_state[4], -np.pi, np.pi, 2*np.pi)
        x, y, l, w, heading, speed = vehicle_state
        heading += np.pi / 2
        heading = normalize(heading, -np.pi, np.pi, 2*np.pi)
        corners = get_corners(x, y, l, w, heading)  # four corner of the boxing
        corners = np.array(corners)
        corner_angles = np.arctan((corners[:, 1] - e_y) / (corners[:, 0] - e_x + 1e-8))
        corner_angles[(corners[:, 0] - e_x) < 0] += np.pi
        corner_angles[corner_angles < 0] += 2*np.pi
        r_angles = corner_angles  # corner angles relative to lane
        for i in range(len(r_angles)):
            r_angles[i] = normalize(r_angles[i], 0, 2*np.pi, 2*np.pi)
        min_angle, max_angle = np.min(r_angles), np.max(r_angles)
        for i in range(radius_sample):
            r = radius[i]
            if (max_angle-min_angle < np.pi) and (r<min_angle) or (r>max_angle):
                continue
            if (max_angle-min_angle > np.pi) and (min_angle < r <max_angle):
                continue
            real_r = r
            dist = get_cross_point_dist(e_x, e_y, real_r, corners, heading)
            dist -= e_dists[i]
            if dist < dists[i]:
                dists[i] = dist
                r_speeds_x[i] = speed * np.cos(heading - lane_h) - e_s * np.cos(lane_relative_h)
                r_speeds_y[i] = speed * np.sin(heading - lane_h) - e_s * np.sin(lane_relative_h)

    ego_info = np.array([
        e_x / 100, e_y / 10, lane_relative_h, lane_offset / 2, e_l / 4, e_w / 2, e_s_lane / 10, e_s_lateral / 10,
        e_s / 10, marker_dist_l / 10, marker_dist_r / 10
    ])
    dists /= 20
    r_speeds_x /= 5
    r_speeds_y /= 5

    # all info
    feature = np.concatenate((ego_info, dists, r_speeds_x, r_speeds_y))
    return feature

def feature5(obs):
    ego_state = obs[0]
    vehicles = []
    cars = 4
    for i in range(cars):
        vehicle_state = obs[6 * i + 1: 6 * i + 7]
        vehicles.append(vehicle_state)
    e_x, e_y = ego_state.position[:2]
    e_h = ego_state.heading.real
    e_s = ego_state.speed
    e_l = ego_state.bounding_box.length
    e_w = ego_state.bounding_box.width
    land_index = ego_state.lane_index
    # e_h, e_s, e_x, e_y, e_l, e_w = ego_state
    e_h += np.pi / 2
    e_h = normalize(e_h, -np.pi, np.pi, 2*np.pi)
    e_sx = e_s * np.cos(e_h)
    e_sy = e_s * np.sin(e_h)
    e_corners = get_corners(e_x, e_y, e_l, e_w, e_h)
    e_corners = np.array(e_corners)
    e_max_y = np.max(e_corners[:, 1])
    e_min_y = np.min(e_corners[:, 1])
    e_max_x = np.max(e_corners[:, 0])
    e_min_x = np.min(e_corners[:, 0])
    feature = np.array([
        e_x / 100, e_y / 10, e_h, e_s / 10, e_sx / 10, e_sy / 10, e_max_y / 10,
        e_min_y / 10, e_max_x / 100, e_min_x / 100])
    radius_sample = 4
    radius = np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample)
    e_dists = [get_cross_point_dist(e_x, e_y, r+e_h, e_corners, e_h) for r in radius]
    dists = np.ones(radius_sample) * 20
    r_speeds_x = np.zeros(radius_sample)
    r_speeds_y = np.zeros(radius_sample)
    for vehicle_state in vehicles:
        if not np.any(vehicle_state):
            continue
        # vehicle_state[4] += np.pi / 2
        # vehicle_state[4] = normalize(vehicle_state[4], -np.pi, np.pi, 2*np.pi)
        x, y, l, w, heading, speed = vehicle_state
        heading += np.pi / 2
        heading = normalize(heading, -np.pi, np.pi, 2*np.pi)
        corners = get_corners(x, y, l, w, heading)  # four corner of the boxing
        corners = np.array(corners)
        corner_angles = np.arctan((corners[:, 1] - e_y) / (corners[:, 0] - e_x + 1e-8))
        corner_angles[(corners[:, 0] - e_x) < 0] += np.pi
        corner_angles[corner_angles < 0] += 2*np.pi
        r_angles = corner_angles - e_h  # corner angles relative to ego_heading
        for i in range(len(r_angles)):
            r_angles[i] = normalize(r_angles[i], 0, 2*np.pi, 2*np.pi)
        min_angle, max_angle = np.min(r_angles), np.max(r_angles)
        for i in range(radius_sample):
            r = radius[i]
            if (max_angle-min_angle < np.pi) and (r<min_angle) or (r>max_angle):
                continue
            if (max_angle-min_angle > np.pi) and (min_angle < r <max_angle):
                continue
            real_r = e_h + r
            dist = get_cross_point_dist(e_x, e_y, real_r, corners, heading)
            dist -= e_dists[i]
            if dist < dists[i]:
                dists[i] = dist
                r_speeds_x[i] = speed * np.cos(heading) - e_s * np.cos(e_h)
                r_speeds_y[i] = speed * np.sin(heading) - e_s * np.sin(e_h)
    ego_land_index = np.zeros(5)
    ego_land_index[land_index] = 1
    dists /= 20
    r_speeds_x /= 10
    r_speeds_y /= 10
    feature = np.concatenate((feature, dists, r_speeds_x, r_speeds_y))
    return feature

def feature6(obs):
    obs = feature3(obs)
    obs = obs[:-4]
    return obs

def feature7(obs):
    ego_state = obs[0]
    vehicles = []
    for i in range(4):
        vehicle_state = obs[6 * i + 1: 6 * i + 7]
        vehicles.append(vehicle_state)

    # ego info
    e_x, e_y = ego_state.position[:2]
    e_h = ego_state.heading.real
    e_s = ego_state.speed
    e_l = ego_state.bounding_box.length
    e_w = ego_state.bounding_box.width
    e_h += np.pi / 2
    e_h = normalize(e_h, -np.pi, np.pi, 2*np.pi)

    # lane info
    lane_id = ego_state.lane_id
    lane_h, lane_y = get_lane_info(lane_id, e_x)
    lane_offset = (e_y - lane_y) * np.cos(lane_h)
    lane_relative_h = e_h - lane_h
    e_s_lane = e_s * np.cos(lane_relative_h)
    e_s_lateral = e_s * np.sin(lane_relative_h)
    marker_dist_l = 25.02 - e_y
    if e_x < 180:
        _h, _y = get_lane_info('gneE05a_0', e_x)
        marker_dist_r = e_y - (_y - lane_width['gneE05a_0'] / 2 / np.cos(_h))
    else:
        marker_dist_r = e_y - (3.49 - lane_width['gneE51_0'] / 2)

    # neighbor info
    e_corners = get_corners(e_x, e_y, e_l, e_w, e_h)
    e_corners = np.array(e_corners)
    radius_sample = 8
    radius = np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample) + lane_h
    e_dists = [get_cross_point_dist(e_x, e_y, r, e_corners, e_h) for r in radius]
    dists = np.ones(radius_sample) * 20
    r_speeds_x = np.zeros(radius_sample)
    r_speeds_y = np.zeros(radius_sample)
    for vehicle_state in vehicles:
        if not np.any(vehicle_state):
            continue
        # vehicle_state[4] += np.pi / 2
        # vehicle_state[4] = normalize(vehicle_state[4], -np.pi, np.pi, 2*np.pi)
        x, y, l, w, heading, speed = vehicle_state
        heading += np.pi / 2
        heading = normalize(heading, -np.pi, np.pi, 2*np.pi)
        corners = get_corners(x, y, l, w, heading)  # four corner of the boxing
        corners = np.array(corners)
        corner_angles = np.arctan((corners[:, 1] - e_y) / (corners[:, 0] - e_x + 1e-8))
        corner_angles[(corners[:, 0] - e_x) < 0] += np.pi
        corner_angles[corner_angles < 0] += 2*np.pi
        r_angles = corner_angles  # corner angles relative to lane
        for i in range(len(r_angles)):
            r_angles[i] = normalize(r_angles[i], 0, 2*np.pi, 2*np.pi)
        min_angle, max_angle = np.min(r_angles), np.max(r_angles)
        for i in range(radius_sample):
            r = radius[i]
            if (max_angle-min_angle < np.pi) and (r<min_angle) or (r>max_angle):
                continue
            if (max_angle-min_angle > np.pi) and (min_angle < r <max_angle):
                continue
            real_r = r
            dist = get_cross_point_dist(e_x, e_y, real_r, corners, heading)
            dist -= e_dists[i]
            if dist < dists[i]:
                dists[i] = dist
                r_speeds_x[i] = speed * np.cos(heading - lane_h) - e_s * np.cos(lane_relative_h)
                r_speeds_y[i] = speed * np.sin(heading - lane_h) - e_s * np.sin(lane_relative_h)

    ego_info = np.array([
        e_x / 100, e_y / 10, lane_relative_h, lane_offset / 2, e_l / 4, e_w / 2, e_s_lane / 10, e_s_lateral / 10,
        e_s / 10, marker_dist_l / 10, marker_dist_r / 10
    ])
    dists /= 20
    r_speeds_x /= 5
    r_speeds_y /= 5

    # all info
    feature = np.concatenate((ego_info, dists, r_speeds_x, r_speeds_y))
    return feature

def feature9(obs):
    ego_state = obs[0]
    vehicles = []
    for i in range(8):
        vehicle_state = obs[6 * i + 1: 6 * i + 7]
        vehicles.append(vehicle_state)
    events = obs[-4:]

    # ego info
    e_x, e_y = ego_state.position[:2]
    e_h = ego_state.heading.real
    e_s = ego_state.speed
    e_l = ego_state.bounding_box.length
    e_w = ego_state.bounding_box.width
    land_index = ego_state.lane_index
    # e_h, e_s, e_x, e_y, e_l, e_w = ego_state
    e_h += np.pi / 2
    e_h = normalize(e_h, -np.pi, np.pi, 2*np.pi)
    e_sx = e_s * np.cos(e_h)
    e_sy = e_s * np.sin(e_h)

    # neighbor info
    e_corners = get_corners(e_x, e_y, e_l, e_w, e_h)
    e_corners = np.array(e_corners)
    e_max_y = np.max(e_corners[:, 1])
    e_min_y = np.min(e_corners[:, 1])
    radius_sample = 8
    radius = list(np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample))
    radius += [np.pi / 8, np.pi / 8 * 15]
    e_dists = [get_cross_point_dist(e_x, e_y, r+e_h, e_corners, e_h) for r in radius]
    dists = np.ones(10) * 20
    r_speeds_l = np.zeros(10)
    r_speeds_r = np.zeros(10)
    dx = np.ones(3) * 20
    dy = np.ones(3) * 20
    nei_l = np.zeros(3)
    nei_w = np.zeros(3)
    for vehicle_state in vehicles:
        if not np.any(vehicle_state):
            continue
        # vehicle_state[4] += np.pi / 2
        # vehicle_state[4] = normalize(vehicle_state[4], -np.pi, np.pi, 2*np.pi)
        x, y, l, w, heading, speed = vehicle_state
        heading += np.pi / 2
        heading = normalize(heading, -np.pi, np.pi, 2*np.pi)
        corners = get_corners(x, y, l, w, heading)  # four corner of the boxing
        corners = np.array(corners)
        corner_angles = np.arctan((corners[:, 1] - e_y) / (corners[:, 0] - e_x + 1e-8))
        corner_angles[(corners[:, 0] - e_x) < 0] += np.pi
        corner_angles[corner_angles < 0] += 2*np.pi
        r_angles = corner_angles - e_h  # corner angles relative to ego_heading
        for i in range(len(r_angles)):
            r_angles[i] = normalize(r_angles[i], 0, 2*np.pi, 2*np.pi)
        min_angle, max_angle = np.min(r_angles), np.max(r_angles)
        for i in range(len(radius)):
            r = radius[i]
            if (max_angle-min_angle < np.pi) and (r<min_angle) or (r>max_angle):
                continue
            if (max_angle-min_angle > np.pi) and (min_angle < r <max_angle):
                continue
            real_r = e_h + r
            dist = get_cross_point_dist(e_x, e_y, real_r, corners, heading)
            dist -= e_dists[i]
            if dist < dists[i]:
                dists[i] = dist
                r_speeds_l[i] = speed * np.cos(heading - e_h) - e_s
                r_speeds_r[i] = speed * np.sin(heading - e_h)
                if i == 8:
                    idx = 0
                elif i == 9:
                    idx = 2
                elif i == 0:
                    idx = 1
                else:
                    continue
                dx[idx] = x - e_x
                dy[idx] = y - e_y
                nei_l[idx] = l
                nei_w[idx] = w


    # lane info
    lane_id = ego_state.lane_id
    lane_h, lane_y = get_lane_info(lane_id, e_x)
    lane_offset = (e_y - lane_y) * np.cos(lane_h)
    lane_relative_h = e_h - lane_h
    e_s_lane = e_s * np.cos(lane_relative_h)
    e_s_lateral = e_s * np.sin(lane_relative_h)
    marker_dist_l = 25.02 - e_y
    if e_x < 180:
        _h, _y = get_lane_info('gneE05a_0', e_x)
        marker_dist_r = e_y - (_y - lane_width['gneE05a_0']/2/np.cos(_h))
    else:
        marker_dist_r = e_y - (3.49 - lane_width['gneE51_0']/2)
    ego_info = np.array([
        e_x / 100, e_y / 10, lane_relative_h, lane_offset / 2, e_l / 4, e_w / 2, e_s_lane / 10, e_s_lateral / 10,
        e_s / 10, marker_dist_l / 10, marker_dist_r / 10
    ])
    dists = dists[:radius_sample]
    r_speeds_l = r_speeds_l[:radius_sample]
    r_speeds_r = r_speeds_r[:radius_sample]
    dists /= 20
    r_speeds_l /= 10
    r_speeds_r /= 10
    dx /= 10
    dy /= 10
    nei_l /= 2
    nei_w /= 2

    # all info
    feature = np.concatenate((ego_info, dists, r_speeds_l, r_speeds_r, dx, dy, nei_l, nei_w))
    return feature

def feature10(obs):
    ego_state = obs[0]
    vehicles = []
    for i in range(16):
        vehicle_state = obs[6 * i + 1: 6 * i + 7]
        vehicles.append(vehicle_state)
    events = obs[-4:]

    # ego info
    e_x, e_y = ego_state.position[:2]
    e_h = ego_state.heading.real
    e_s = ego_state.speed
    e_l = ego_state.bounding_box.length
    e_w = ego_state.bounding_box.width
    e_h += np.pi / 2
    e_h = normalize(e_h, -np.pi, np.pi, 2*np.pi)

    # neighbor info
    e_corners = get_corners(e_x, e_y, e_l, e_w, e_h)
    e_corners = np.array(e_corners)
    e_max_y = np.max(e_corners[:, 1])
    e_min_y = np.min(e_corners[:, 1])
    radius_sample = 16
    radius = list(np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample))
    e_dists = [get_cross_point_dist(e_x, e_y, r+e_h, e_corners, e_h) for r in radius]
    max_dist = 100
    dists = np.ones(radius_sample) * max_dist
    r_speeds_l = np.zeros(radius_sample)
    r_speeds_r = np.zeros(radius_sample)
    dx = np.ones(5) * max_dist
    dy = np.ones(5) * max_dist
    nei_l = np.zeros(5)
    nei_w = np.zeros(5)
    for vehicle_state in vehicles:
        if not np.any(vehicle_state):
            continue
        # vehicle_state[4] += np.pi / 2
        # vehicle_state[4] = normalize(vehicle_state[4], -np.pi, np.pi, 2*np.pi)
        x, y, l, w, heading, speed = vehicle_state
        heading += np.pi / 2
        heading = normalize(heading, -np.pi, np.pi, 2*np.pi)
        corners = get_corners(x, y, l, w, heading)  # four corner of the boxing
        corners = np.array(corners)
        corner_angles = np.arctan((corners[:, 1] - e_y) / (corners[:, 0] - e_x + 1e-8))
        corner_angles[(corners[:, 0] - e_x) < 0] += np.pi
        corner_angles[corner_angles < 0] += 2*np.pi
        r_angles = corner_angles - e_h  # corner angles relative to ego_heading
        for i in range(len(r_angles)):
            r_angles[i] = normalize(r_angles[i], 0, 2*np.pi, 2*np.pi)
        min_angle, max_angle = np.min(r_angles), np.max(r_angles)
        for i in range(len(radius)):
            r = radius[i]
            if (max_angle-min_angle < np.pi) and (r<min_angle) or (r>max_angle):
                continue
            if (max_angle-min_angle > np.pi) and (min_angle < r <max_angle):
                continue
            real_r = e_h + r
            dist = get_cross_point_dist(e_x, e_y, real_r, corners, heading)
            if dist > max_dist:
                continue
            dist -= e_dists[i]
            if dist < dists[i]:
                dists[i] = dist
                r_speeds_l[i] = speed * np.cos(heading - e_h) - e_s
                r_speeds_r[i] = speed * np.sin(heading - e_h)
                if i == 2:
                    idx = 0
                elif i == 1:
                    idx = 1
                elif i == 0:
                    idx = 2
                elif i == 15:
                    idx = 3
                elif i == 14:
                    idx = 4
                else:
                    continue
                dx[idx] = x - e_x
                dy[idx] = y - e_y
                nei_l[idx] = l
                nei_w[idx] = w


    # lane info
    lane_id = ego_state.lane_id
    lane_h, lane_y = get_lane_info(lane_id, e_x)
    lane_offset = (e_y - lane_y) * np.cos(lane_h)
    lane_relative_h = e_h - lane_h
    e_s_lane = e_s * np.cos(lane_relative_h)
    e_s_lateral = e_s * np.sin(lane_relative_h)
    marker_dist_l = 25.02 - e_y
    if e_x < 180:
        _h, _y = get_lane_info('gneE05a_0', e_x)
        marker_dist_r = e_y - (_y - lane_width['gneE05a_0']/2/np.cos(_h))
    else:
        marker_dist_r = e_y - (3.49 - lane_width['gneE51_0']/2)
    ego_info = np.array([
        e_x / 100, e_y / 10, lane_relative_h, lane_offset / 2, e_l / 4, e_w / 2, e_s_lane / 10, e_s_lateral / 10,
        e_s / 10, marker_dist_l / 10, marker_dist_r / 10
    ])
    dists /= 20
    r_speeds_l /= 10
    r_speeds_r /= 10
    dx /= 10
    dy /= 10
    nei_l /= 2
    nei_w /= 2

    # all info
    feature = np.concatenate((ego_info, dists, r_speeds_l, r_speeds_r, dx, dy, nei_l, nei_w))
    return feature

def feature11(obs):
    ego_state = obs[0]
    vehicles = []
    for i in range(8):
        vehicle_state = obs[6 * i + 1: 6 * i + 7]
        vehicles.append(vehicle_state)

    # ego info
    e_x, e_y = ego_state.position[:2]
    e_h = ego_state.heading.real
    e_s = ego_state.speed
    e_l = ego_state.bounding_box.length
    e_w = ego_state.bounding_box.width
    e_h += np.pi / 2
    e_h = normalize(e_h, -np.pi, np.pi, 2*np.pi)

    # lane info
    lane_id = ego_state.lane_id
    lane_h, lane_y = get_lane_info(lane_id, e_x)
    lane_offset = (e_y - lane_y) * np.cos(lane_h)
    lane_relative_h = e_h - lane_h
    e_s_lane = e_s * np.cos(lane_relative_h)
    e_s_lateral = e_s * np.sin(lane_relative_h)
    marker_dist_l = 25.02 - e_y
    if e_x < 180:
        _h, _y = get_lane_info('gneE05a_0', e_x)
        marker_dist_r = e_y - (_y - lane_width['gneE05a_0'] / 2 / np.cos(_h))
    else:
        marker_dist_r = e_y - (3.49 - lane_width['gneE51_0'] / 2)

    # neighbor info
    e_corners = get_corners(e_x, e_y, e_l, e_w, e_h)
    e_corners = np.array(e_corners)
    radius_sample = 16
    radius = list(np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample))
    e_dists = [get_cross_point_dist(e_x, e_y, r+e_h, e_corners, e_h) for r in radius]
    dists = np.ones(radius_sample) * 100
    r_speeds_l = np.zeros(radius_sample)
    r_speeds_r = np.zeros(radius_sample)
    for vehicle_state in vehicles:
        if not np.any(vehicle_state):
            continue
        # vehicle_state[4] += np.pi / 2
        # vehicle_state[4] = normalize(vehicle_state[4], -np.pi, np.pi, 2*np.pi)
        x, y, l, w, heading, speed = vehicle_state
        heading += np.pi / 2
        heading = normalize(heading, -np.pi, np.pi, 2*np.pi)
        corners = get_corners(x, y, l, w, heading)  # four corner of the boxing
        corners = np.array(corners)
        corner_angles = np.arctan((corners[:, 1] - e_y) / (corners[:, 0] - e_x + 1e-8))
        corner_angles[(corners[:, 0] - e_x) < 0] += np.pi
        corner_angles[corner_angles < 0] += 2*np.pi
        r_angles = corner_angles - e_h  # corner angles relative to ego_heading
        for i in range(len(r_angles)):
            r_angles[i] = normalize(r_angles[i], 0, 2*np.pi, 2*np.pi)
        min_angle, max_angle = np.min(r_angles), np.max(r_angles)
        for i in range(len(radius)):
            r = radius[i]
            if (max_angle-min_angle < np.pi) and (r<min_angle) or (r>max_angle):
                continue
            if (max_angle-min_angle > np.pi) and (min_angle < r <max_angle):
                continue
            real_r = e_h + r
            dist = get_cross_point_dist(e_x, e_y, real_r, corners, heading)
            dist -= e_dists[i]
            if dist < dists[i]:
                dists[i] = dist
                r_speeds_l[i] = speed * np.cos(heading - lane_h) - e_s_lane
                r_speeds_r[i] = speed * np.sin(heading - lane_h) - e_s_lateral

    ego_info = np.array([
        e_x / 300, e_y / 20, lane_relative_h, lane_offset / 2, e_l / 4, e_w / 2, e_s_lane / 5, e_s_lateral,
        e_s / 5, marker_dist_l / 10, marker_dist_r / 10
    ])
    dists /= 100

    # all info
    feature = np.concatenate((ego_info, dists, r_speeds_l, r_speeds_r))
    return feature

def feature12(obs):
    ego_state = obs[0]
    vehicles = []
    for i in range(8):
        vehicle_state = obs[6 * i + 1: 6 * i + 7]
        vehicles.append(vehicle_state)

    # ego info
    e_x, e_y = ego_state.position[:2]
    e_h = ego_state.heading.real
    e_s = ego_state.speed
    e_l = ego_state.bounding_box.length
    e_w = ego_state.bounding_box.width
    e_h += np.pi / 2
    e_h = normalize(e_h, -np.pi, np.pi, 2*np.pi)

    # lane info
    lane_id = ego_state.lane_id
    lane_h, lane_y = get_lane_info(lane_id, e_x)
    lane_offset = (e_y - lane_y) * np.cos(lane_h)
    lane_relative_h = e_h - lane_h
    e_s_lane = e_s * np.cos(lane_relative_h)
    e_s_lateral = e_s * np.sin(lane_relative_h)
    marker_dist_l = 25.02 - e_y
    if e_x < 180:
        _h, _y = get_lane_info('gneE05a_0', e_x)
        marker_dist_r = e_y - (_y - lane_width['gneE05a_0'] / 2 / np.cos(_h))
    else:
        marker_dist_r = e_y - (3.49 - lane_width['gneE51_0'] / 2)

    # neighbor info
    e_corners = get_corners(e_x, e_y, e_l, e_w, e_h)
    e_corners = np.array(e_corners)
    radius_sample = 8
    radius = list(np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample))
    e_dists = [get_cross_point_dist(e_x, e_y, r+e_h, e_corners, e_h) for r in radius]
    dists = np.ones(radius_sample) * 100
    r_speeds_l = np.zeros(radius_sample)
    r_speeds_r = np.zeros(radius_sample)
    for vehicle_state in vehicles:
        if not np.any(vehicle_state):
            continue
        # vehicle_state[4] += np.pi / 2
        # vehicle_state[4] = normalize(vehicle_state[4], -np.pi, np.pi, 2*np.pi)
        x, y, l, w, heading, speed = vehicle_state
        heading += np.pi / 2
        heading = normalize(heading, -np.pi, np.pi, 2*np.pi)
        corners = get_corners(x, y, l, w, heading)  # four corner of the boxing
        corners = np.array(corners)
        corner_angles = np.arctan((corners[:, 1] - e_y) / (corners[:, 0] - e_x + 1e-8))
        corner_angles[(corners[:, 0] - e_x) < 0] += np.pi
        corner_angles[corner_angles < 0] += 2*np.pi
        r_angles = corner_angles - e_h  # corner angles relative to ego_heading
        for i in range(len(r_angles)):
            r_angles[i] = normalize(r_angles[i], 0, 2*np.pi, 2*np.pi)
        min_angle, max_angle = np.min(r_angles), np.max(r_angles)
        for i in range(len(radius)):
            r = radius[i]
            if (max_angle-min_angle < np.pi) and (r<min_angle) or (r>max_angle):
                continue
            if (max_angle-min_angle > np.pi) and (min_angle < r <max_angle):
                continue
            real_r = e_h + r
            dist = get_cross_point_dist(e_x, e_y, real_r, corners, heading)
            dist -= e_dists[i]
            if dist < dists[i]:
                dists[i] = dist
                r_speeds_l[i] = speed * np.cos(heading - lane_h) - e_s_lane
                r_speeds_r[i] = speed * np.sin(heading - lane_h) - e_s_lateral

    ego_info = np.array([
        e_y / 20, lane_relative_h, lane_offset / 2, e_l / 4, e_w / 2, e_s_lane / 5, e_s_lateral,
        e_s / 5, marker_dist_l / 10, marker_dist_r / 10
    ])
    dists /= 100

    # all info
    feature = np.concatenate((ego_info, dists, r_speeds_l, r_speeds_r))
    return feature

def feature13(obs):
    ego_state = obs[0]
    vehicles = []
    for i in range(8):
        vehicle_state = obs[6 * i + 1: 6 * i + 7]
        vehicles.append(vehicle_state)

    # ego info
    e_x, e_y = ego_state.position[:2]
    e_h = ego_state.heading.real
    e_s = ego_state.speed
    e_l = ego_state.bounding_box.length
    e_w = ego_state.bounding_box.width
    # e_h += np.pi / 2
    e_h = normalize(e_h, -np.pi, np.pi, 2*np.pi)

    # lane info
    lane_id = ego_state.lane_id
    lane_h, lane_y = get_lane_info(lane_id, e_x)
    lane_offset = (e_y - lane_y) * np.cos(lane_h)
    lane_relative_h = e_h - lane_h
    e_s_lane = e_s * np.cos(lane_relative_h)
    e_s_lateral = e_s * np.sin(lane_relative_h)
    marker_dist_l = 25.02 - e_y
    if e_x < 180:
        _h, _y = get_lane_info('gneE05a_0', e_x)
        marker_dist_r = e_y - (_y - lane_width['gneE05a_0'] / 2 / np.cos(_h))
    else:
        marker_dist_r = e_y - (3.49 - lane_width['gneE51_0'] / 2)

    # neighbor info
    e_corners = get_corners(e_x, e_y, e_l, e_w, e_h)
    e_corners = np.array(e_corners)
    radius_sample = 8
    radius = list(np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample))
    e_dists = [get_cross_point_dist(e_x, e_y, r+e_h, e_corners, e_h) for r in radius]
    dists = np.ones(radius_sample) * 100
    r_speeds_l = np.zeros(radius_sample)
    r_speeds_r = np.zeros(radius_sample)
    for vehicle_state in vehicles:
        if not np.any(vehicle_state):
            continue
        # vehicle_state[4] += np.pi / 2
        # vehicle_state[4] = normalize(vehicle_state[4], -np.pi, np.pi, 2*np.pi)
        x, y, l, w, heading, speed = vehicle_state
        # heading += np.pi / 2
        heading = normalize(heading, -np.pi, np.pi, 2*np.pi)
        corners = get_corners(x, y, l, w, heading)  # four corner of the boxing
        corners = np.array(corners)
        corner_angles = np.arctan((corners[:, 1] - e_y) / (corners[:, 0] - e_x + 1e-8))
        corner_angles[(corners[:, 0] - e_x) < 0] += np.pi
        corner_angles[corner_angles < 0] += 2*np.pi
        r_angles = corner_angles - e_h  # corner angles relative to ego_heading
        for i in range(len(r_angles)):
            r_angles[i] = normalize(r_angles[i], 0, 2*np.pi, 2*np.pi)
        min_angle, max_angle = np.min(r_angles), np.max(r_angles)
        for i in range(len(radius)):
            r = radius[i]
            if (max_angle-min_angle < np.pi) and (r<min_angle) or (r>max_angle):
                continue
            if (max_angle-min_angle > np.pi) and (min_angle < r <max_angle):
                continue
            real_r = e_h + r
            dist = get_cross_point_dist(e_x, e_y, real_r, corners, heading)
            dist -= e_dists[i]
            if dist < dists[i]:
                dists[i] = dist
                r_speeds_l[i] = speed * np.cos(heading - lane_h) - e_s_lane
                r_speeds_r[i] = speed * np.sin(heading - lane_h) - e_s_lateral

    ego_info = np.array([
        e_y / 20, lane_relative_h, lane_offset / 2, e_l / 4, e_w / 2, e_s_lane / 5, e_s_lateral,
        e_s / 5, marker_dist_l / 10, marker_dist_r / 10
    ])
    dists /= 100

    # all info
    feature = np.concatenate((ego_info, dists, r_speeds_l, r_speeds_r))
    return feature

def feature14(obs):
    ego_state = obs[0]
    vehicles = []
    for i in range(16):
        vehicle_state = obs[6 * i + 1: 6 * i + 7]
        vehicles.append(vehicle_state)

    # ego info
    e_x, e_y = ego_state.position[:2]
    e_h = ego_state.heading.real
    e_s = ego_state.speed
    e_l = ego_state.bounding_box.length
    e_w = ego_state.bounding_box.width
    e_h += np.pi / 2
    e_h = normalize(e_h, 0, 2*np.pi, 2*np.pi)
    e_sx = e_s * np.cos(e_h)
    e_sy = e_s * np.sin(e_h)

    # lane info
    lane_id = ego_state.lane_id
    lane_h, lane_y = get_lane_info(lane_id, e_x)
    lane_offset = (e_y - lane_y) * np.cos(lane_h)
    lane_relative_h = e_h - lane_h
    e_s_lane = e_s * np.cos(lane_relative_h)
    e_s_lateral = e_s * np.sin(lane_relative_h)
    marker_dist_l = 25.02 - e_y
    if e_x < 180:
        _h, _y = get_lane_info('gneE05a_0', e_x)
        marker_dist_r = e_y - (_y - lane_width['gneE05a_0'] / 2 / np.cos(_h))
    else:
        marker_dist_r = e_y - (3.49 - lane_width['gneE51_0'] / 2)

    # neighbor info
    e_corners = get_corners(e_x, e_y, e_l, e_w, e_h)
    e_corners = np.array(e_corners)
    radius_sample = 8
    radius = list(np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample))
    # dx, dy, dsx, dxy, h
    neibor_info = np.zeros((radius_sample, 5))
    closest_neibor = {i:None for i in range(radius_sample)}
    max_dists = 30
    for vehicle_state in vehicles:
        if not np.any(vehicle_state):
            continue
        # vehicle_state[4] += np.pi / 2
        # vehicle_state[4] = normalize(vehicle_state[4], -np.pi, np.pi, 2*np.pi)
        x, y, l, w, heading, speed = vehicle_state
        dist = np.sqrt((x-e_x)**2 + (y-e_y)**2)
        if dist > max_dists:
            continue
        r_h = np.arctan((y-e_y)/(x-e_x))
        if x-e_x<0:
            r_h += np.pi
        r_h = normalize(r_h, 0, 2*np.pi, 2*np.pi)
        for i in reversed(range(radius_sample)):
            if r_h >= radius[i]:
                closest_neibor[i] = vehicle_state
                break
    for i in range(radius_sample):
        if closest_neibor[i] is None:
            continue
        vehicle_state = closest_neibor[i]
        x, y, l, w, heading, speed = vehicle_state
        heading += np.pi / 2
        heading = normalize(heading, 0, 2*np.pi, 2 * np.pi)
        s_x = speed * np.cos(heading)
        s_y = speed * np.sin(heading)
        neibor_info[i][0] = x - e_x
        neibor_info[i][1] = y - e_y
        neibor_info[i][2] = s_x - e_sx
        neibor_info[i][3] = s_y - e_sy
        neibor_info[i][4] = np.sin(heading)

    ego_info = np.array([
        e_x, e_y, np.sin(lane_relative_h), lane_offset, e_l, e_w, e_s_lane, e_s_lateral,
        e_s, marker_dist_l, marker_dist_r
    ])

    # all info
    feature = np.concatenate((ego_info, neibor_info.reshape(-1)))
    return feature

def feature15(obs):
    ego_state = obs[0]
    vehicles = []
    for i in range(16):
        vehicle_state = obs[6 * i + 1: 6 * i + 7]
        vehicles.append(vehicle_state)

    # ego info
    e_x, e_y = ego_state.position[:2]
    e_h = ego_state.heading.real
    e_s = ego_state.speed
    e_l = ego_state.bounding_box.length
    e_w = ego_state.bounding_box.width
    e_h += np.pi / 2
    e_h = normalize(e_h, 0, 2*np.pi, 2*np.pi)
    e_sx = e_s * np.cos(e_h)
    e_sy = e_s * np.sin(e_h)

    # lane info
    lane_id = ego_state.lane_id
    lane_h, lane_y = get_lane_info(lane_id, e_x)
    lane_offset = (e_y - lane_y) * np.cos(lane_h)
    lane_relative_h = e_h - lane_h
    e_s_lane = e_s * np.cos(lane_relative_h)
    e_s_lateral = e_s * np.sin(lane_relative_h)
    marker_dist_l = 25.02 - e_y
    if e_x < 180:
        _h, _y = get_lane_info('gneE05a_0', e_x)
        marker_dist_r = e_y - (_y - lane_width['gneE05a_0'] / 2 / np.cos(_h))
    else:
        marker_dist_r = e_y - (3.49 - lane_width['gneE51_0'] / 2)

    # neighbor info
    radius_sample = 8
    radius = list(np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample))
    # dx, dy, dsx, dxy, h, l, w
    neibor_info = np.zeros((radius_sample, 7))
    closest_neibor = {i:None for i in range(radius_sample)}
    max_dists = 50
    closest_neibor_dist = np.ones(radius_sample) * max_dists
    for vehicle_state in vehicles:
        if not np.any(vehicle_state):
            continue
        # vehicle_state[4] += np.pi / 2
        # vehicle_state[4] = normalize(vehicle_state[4], -np.pi, np.pi, 2*np.pi)
        x, y, l, w, heading, speed = vehicle_state
        dist = np.sqrt((x-e_x)**2 + (y-e_y)**2)
        if dist >= max_dists:
            continue
        r_h = np.arctan((y-e_y)/(x-e_x))
        if x-e_x<0:
            r_h += np.pi
        r_h = normalize(r_h, 0, 2*np.pi, 2*np.pi)
        for i in reversed(range(radius_sample)):
            if r_h >= radius[i]:
                if dist < closest_neibor_dist[i]:
                    closest_neibor_dist[i] = dist
                    closest_neibor[i] = vehicle_state
                break
    for i in range(radius_sample):
        if closest_neibor[i] is None:
            continue
        vehicle_state = closest_neibor[i]
        x, y, l, w, heading, speed = vehicle_state
        heading += np.pi / 2
        heading = normalize(heading, 0, 2*np.pi, 2 * np.pi)
        s_x = speed * np.cos(heading)
        s_y = speed * np.sin(heading)
        neibor_info[i][0] = x - e_x
        neibor_info[i][1] = y - e_y
        neibor_info[i][2] = s_x - e_sx
        neibor_info[i][3] = s_y - e_sy
        neibor_info[i][4] = np.sin(heading)
        neibor_info[i][5] = l
        neibor_info[i][6] = w

    ego_info = np.array([
        e_x, e_y, np.sin(lane_relative_h), lane_offset, e_l, e_w, e_s_lane, e_s_lateral,
        e_s, marker_dist_l, marker_dist_r
    ])

    # all info
    feature = np.concatenate((ego_info, neibor_info.reshape(-1)))
    return feature

def feature17(obs):
    ego_state = obs[0]
    vehicles = []
    for i in range(16):
        vehicle_state = obs[6 * i + 1: 6 * i + 7]
        vehicles.append(vehicle_state)

    # ego info
    e_x, e_y = ego_state.position[:2]
    e_h = ego_state.heading.real
    e_s = ego_state.speed
    e_l = ego_state.bounding_box.length
    e_w = ego_state.bounding_box.width
    e_h += np.pi / 2
    e_h = normalize(e_h, 0, 2*np.pi, 2*np.pi)
    e_sx = e_s * np.cos(e_h)
    e_sy = e_s * np.sin(e_h)

    # lane info
    lane_id = ego_state.lane_id
    lane_h, lane_y = get_lane_info(lane_id, e_x)
    lane_offset = (e_y - lane_y) * np.cos(lane_h)
    lane_relative_h = e_h - lane_h
    e_s_lane = e_s * np.cos(lane_relative_h)
    e_s_lateral = e_s * np.sin(lane_relative_h)
    marker_dist_l = 25.02 - e_y
    if e_x < 180:
        _h, _y = get_lane_info('gneE05a_0', e_x)
        marker_dist_r = e_y - (_y - lane_width['gneE05a_0'] / 2 / np.cos(_h))
    else:
        marker_dist_r = e_y - (3.49 - lane_width['gneE51_0'] / 2)

    # neighbor info
    radius_sample = 6
    radius = list(np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample) + np.pi/6)
    # dx, dy, dsx, dxy, h, l, w
    neibor_info = np.zeros((radius_sample, 7))
    closest_neibor = {i:None for i in range(radius_sample)}
    max_dists = 50
    closest_neibor_dist = np.ones(radius_sample) * max_dists
    for vehicle_state in vehicles:
        if not np.any(vehicle_state):
            continue
        # vehicle_state[4] += np.pi / 2
        # vehicle_state[4] = normalize(vehicle_state[4], -np.pi, np.pi, 2*np.pi)
        x, y, l, w, heading, speed = vehicle_state
        dist = np.sqrt((x-e_x)**2 + (y-e_y)**2)
        if dist >= max_dists:
            continue
        r_h = np.arctan((y-e_y)/(x-e_x))
        if x-e_x<0:
            r_h += np.pi
        r_h = normalize(r_h, np.pi/6, 2*np.pi+np.pi/6, 2*np.pi)
        for i in reversed(range(radius_sample)):
            if r_h >= radius[i]:
                if dist < closest_neibor_dist[i]:
                    closest_neibor_dist[i] = dist
                    closest_neibor[i] = vehicle_state
                break
    for i in range(radius_sample):
        if closest_neibor[i] is None:
            continue
        vehicle_state = closest_neibor[i]
        x, y, l, w, heading, speed = vehicle_state
        heading += np.pi / 2
        heading = normalize(heading, 0, 2*np.pi, 2 * np.pi)
        s_x = speed * np.cos(heading)
        s_y = speed * np.sin(heading)
        neibor_info[i][0] = x - e_x
        neibor_info[i][1] = y - e_y
        neibor_info[i][2] = s_x - e_sx
        neibor_info[i][3] = s_y - e_sy
        neibor_info[i][4] = np.sin(heading)
        neibor_info[i][5] = l
        neibor_info[i][6] = w

    ego_info = np.array([
        e_x, e_y, np.sin(lane_relative_h), lane_offset, e_l, e_w, e_s_lane, e_s_lateral,
        e_s, marker_dist_l, marker_dist_r
    ])

    # all info
    feature = np.concatenate((ego_info, neibor_info.reshape(-1)))
    return feature


def feature11_descriptor1(obs):
    dists = obs[-48: -32]
    dists[dists == 1] = 0
    obs[-48: -32] = dists
    return obs


def feature11_descriptor2(obs):
    dists = obs[-48: -32]
    _dists = dists[dists != 1]
    if len(_dists) == 0:
        return obs
    else:
        mean_dist = np.mean(_dists)
    dists[dists == 1] = mean_dist
    obs[-48: -32] = dists
    return obs


def feature12_descriptor1(obs):
    dists = obs[-24: -16]
    _dists = dists[dists != 1]
    if len(_dists) == 0:
        return obs
    else:
        mean_dist = np.mean(_dists)
    dists[dists == 1] = mean_dist
    obs[-24: -16] = dists
    return obs


def feature12_descriptor2(obs):
    dists = obs[-24: -16]
    dists[dists == 1] = 0
    obs[-24: -16] = dists
    return obs

def filter(actions):
    action = np.array(actions)
    action[1:-1] = 0.5 * action[1:-1] + 0.25 * (action[:-2] + action[2:])
    return action.tolist()


def get_corners(x, y, l, w, heading):
    corner_pass = [(l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2)]
    corners = []
    for i in range(4):
        p = corner_pass[i]
        xc, yc = x, y
        xc += (p[0] * np.cos(heading) + p[1] * np.sin(heading))
        yc += (p[0] * np.sin(heading) - p[1] * np.cos(heading))
        corners.append((xc, yc))
    return corners


def normalize(angle, low, high, T):
    _angle = angle
    while _angle < low:
        _angle += T
    while _angle >= high:
        _angle -= T
    return _angle


def get_cross_point_dist(e_x, e_y, real_r, corners, n_h):
    _real_r = normalize(real_r, -np.pi/2, np.pi/2, np.pi)
    # pdb.set_trace()
    corner_h = [n_h+np.pi/2, n_h, n_h+np.pi/2, n_h]
    candidate_x = []
    candidate = []
    for i in range(4):
        if abs(_real_r - corner_h[i]) < 1e-4:
            continue
        if abs(_real_r - np.pi/2) < 1e-5 or abs(_real_r + np.pi/2) < 1e-5:
            if abs(corner_h[i] - np.pi / 2) < 1e-5 or abs(corner_h[i] + np.pi / 2) < 1e-5:
                continue
            x = e_x
        elif abs(corner_h[i] - np.pi/2) < 1e-5 or abs(corner_h[i] + np.pi/2) < 1e-5:
            x = corners[i][0]
        else:
            x = (corners[i][1]-e_y+e_x*np.tan(_real_r)-corners[i][0]*np.tan(corner_h[i])) \
            / (np.tan(_real_r) - np.tan(corner_h[i] + 1e-8))
        if abs(x) > 1e4:
            continue
        if np.abs(np.tan(_real_r)) > 1e4:
            y = corners[i][1] + (x-corners[i][0])*np.tan(corner_h[i])
        else:
            y = e_y + (x-e_x) * np.tan(_real_r)
        if abs(y) > 1e3:
            continue
        if (corners[i][0]-x)*(corners[(i+1)%4][0]-x)<=0 or (corners[i][1]-y)*(corners[(i+1)%4][1]-y)<=0:
            candidate_x.append(x)
            candidate.append((x, y))
    # pdb.set_trace()
    dist = [np.sqrt((p[0]-e_x)**2 + (p[1]-e_y)**2) for p in candidate]
    return np.min(dist)


base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))


pnet = PolicyNetwork(67, 2)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pth')
agent = BC(pnet)
agent.load_model(model_path)

sys.path.pop(-1)
normalize_weight = torch.tensor([ 1.5436e+02,  1.2777e+01,  3.6335e-03,  5.4334e-01,  4.9380e+00,
         2.0395e+00,  7.5032e+00,  1.8704e-02,  7.5063e+00,  1.2243e+01,
         1.3813e+01,  2.0715e+01,  5.0809e+00,  1.4941e+00, -1.5365e-02,
         1.6486e-03,  4.3419e+00,  1.7746e+00,  3.4777e+00,  6.7565e+00,
         2.3314e+00, -1.0470e-02,  5.9836e-04,  2.6194e+00,  1.1145e+00,
        -3.4800e+00,  6.7678e+00,  2.3359e+00, -8.4183e-03,  7.9320e-04,
         2.6172e+00,  1.1167e+00, -2.0791e+01,  5.1725e+00,  1.5713e+00,
        -4.0209e-03,  2.3666e-03,  4.2669e+00,  1.7622e+00, -2.0988e+01,
        -5.1395e+00, -5.7243e-01,  1.6389e-02,  7.6664e-03,  4.1940e+00,
         1.7885e+00, -3.5764e+00, -6.7323e+00, -6.3021e-01,  2.7680e-02,
         6.9276e-03,  2.6005e+00,  1.1194e+00,  3.6374e+00, -6.8230e+00,
        -6.3676e-01,  2.9537e-02,  7.0571e-03,  2.6441e+00,  1.1337e+00,
         2.0766e+01, -5.2590e+00, -7.1680e-01,  2.1353e-02,  6.9191e-03,
         4.1885e+00,  1.7602e+00])
normalize_std = torch.tensor([8.5796e+01, 6.5070e+00, 8.0803e-02, 7.0626e-01, 2.4470e+00, 2.7459e-01,
        3.9507e+00, 1.9977e-01, 3.9497e+00, 6.5070e+00, 6.6572e+00, 1.1139e+01,
        4.7625e+00, 4.3956e+00, 2.5322e-01, 7.3090e-02, 2.8874e+00, 7.3675e-01,
        4.6303e+00, 7.1874e+00, 4.8778e+00, 1.9390e-01, 4.3951e-02, 2.7289e+00,
        1.0303e+00, 4.6180e+00, 7.1808e+00, 4.8868e+00, 1.9233e-01, 4.3965e-02,
        2.7022e+00, 1.0293e+00, 1.1340e+01, 4.8205e+00, 4.4169e+00, 2.5563e-01,
        7.9083e-02, 2.7927e+00, 7.4498e-01, 1.0904e+01, 4.7788e+00, 3.7750e+00,
        2.8346e-01, 7.9700e-02, 2.5002e+00, 7.1213e-01, 4.6919e+00, 7.1411e+00,
        3.6834e+00, 2.1899e-01, 5.1494e-02, 2.7435e+00, 1.0330e+00, 4.7153e+00,
        7.1441e+00, 3.6766e+00, 2.1764e-01, 4.7552e-02, 2.7689e+00, 1.0327e+00,
        1.1331e+01, 4.9268e+00, 3.7598e+00, 2.8099e-01, 7.5412e-02, 2.6848e+00,
        7.4536e-01])


def my_controller(observation, action_space, is_act_continuous=False):
    state = feature15(expert_collector3(observation))
    state = torch.tensor(state).float()
    state = (state - normalize_weight) / normalize_std
    action = agent.action(state).detach()
    action /= 10

    return action.cpu().numpy().tolist()

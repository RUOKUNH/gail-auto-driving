import pdb

import numpy as np
import torch as torch


def clip(x, low, high):
    if x < low:
        return low
    if x > high:
        return high
    return x


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
    mean = dist.mean
    cov = dist.covariance_matrix.sum(-1)
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
    _obs += [heading, speed, x, y, l, w]
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


# heading normalize to [-pi, pi)
# record dist, relative speed-x, relative speed-y at eight directions
def feature_detection(obs):
    ego_state = obs[:6]
    vehicles = []
    for i in range(4):
        vehicle_state = obs[6 * (i + 1): 6 * (i + 2)]
        vehicles.append(vehicle_state)
    e_h, e_s, e_x, e_y, e_l, e_w = ego_state
    # ego_state[0] += np.pi / 2
    # ego_state[0] = normalize(ego_state[0], -np.pi, np.pi, 2*np.pi)
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
    feature = np.array([e_x, e_y, e_h, e_s, e_sx, e_sy, e_max_y, e_min_y, e_max_x, e_min_x])
    radius_sample = 8
    radius = np.arange(0, 2 * np.pi, 2 * np.pi / radius_sample)
    e_dists = [get_cross_point_dist(e_x, e_y, r+e_h, e_corners, e_h) for r in radius]
    # pdb.set_trace()
    dists = np.ones(radius_sample) * 30
    r_speeds_x = np.zeros(radius_sample)
    r_speeds_y = np.zeros(radius_sample)
    for vehicle_state in vehicles:
        # pdb.set_trace()
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
        # pdb.set_trace()
        min_angle, max_angle = np.min(r_angles), np.max(r_angles)
        for i in range(radius_sample):
            r = radius[i]
            if (max_angle-min_angle < np.pi) and (r<min_angle) or (r>max_angle):
                continue
            if (max_angle-min_angle > np.pi) and (min_angle < r <max_angle):
                continue
            # pdb.set_trace()
            real_r = e_h + r
            # pdb.set_trace()
            dist = get_cross_point_dist(e_x, e_y, real_r, corners, heading)
            dist -= e_dists[i]
            # pdb.set_trace()
            if dist < dists[i]:
                dists[i] = dist
                r_speeds_x[i] = speed * np.cos(heading) - e_s * np.cos(e_h)
                r_speeds_y[i] = speed * np.sin(heading) - e_s * np.sin(e_h)
    feature = np.concatenate((feature, dists, r_speeds_x, r_speeds_y))
    return feature


def get_corners(x, y, l, w, heading):
    corner_pass = [(l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2)]
    corners = []
    # pdb.set_trace()
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
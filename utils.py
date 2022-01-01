import pdb

import numpy as np
import torch as torch
from filter import kalman_filter
import matplotlib.pyplot as plt


# Lane informatino of the map
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
def kl_divergence(dist, old_dist, action_dim, require_grad=False):
    old_mean = old_dist.mean.detach()
    old_cov = old_dist.covariance_matrix.sum(-1).detach()
    if require_grad:
        mean = dist.mean
        cov = dist.covariance_matrix.sum(-1)
    else:
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


def get_l2_norm(net):
    param = get_flat_params(net)
    norm = torch.mean(param ** 2)
    return norm


def compute_discounted_rewards(rewards, gamma):
    rewards = np.array(rewards)
    discounted_rewards = np.zeros_like(rewards)
    discounted_rewards[-1] = rewards[-1]
    for i in reversed(range(len(rewards)-1)):
        discounted_rewards[i] = gamma * discounted_rewards[i+1] + rewards[i]
    return discounted_rewards.tolist()


def set_params(net, new_flat_params):
    start_idx = 0
    for param in net.parameters():
        end_idx = start_idx + np.prod(list(param.shape))
        param.data = torch.reshape(
            new_flat_params[start_idx:end_idx], param.shape
        )

        start_idx = end_idx


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


def feature15_descriptor(obs):
    return obs[2:]


def feature15_descriptor1(obs):
    return obs[1:]


def feature15_descriptor4(obs):
    e_x, e_y, sin_h, lane_offset, e_l, e_w, e_s_lane, e_s_lateral, \
    e_s, marker_dist_l, marker_dist_r = obs[:-56]
    _obs = [sin_h, lane_offset, marker_dist_r, marker_dist_l]
    front1 = obs[11:18]
    front2 = obs[-7:]
    lside1 = obs[18:25]
    lside2 = obs[25:32]
    back1 = obs[32:39]
    back2 = obs[39:46]
    rside1 = obs[46:53]
    rside2 = obs[53:60]
    frontinfo = get_info(front1, front2, 1, e_l, e_w)
    lsideinfo = get_info(lside1, lside2, 2, e_l, e_w)
    backinfo = get_info(back1, back2, 3, e_l, e_w)
    rsideinfo = get_info(rside1, rside2, 4, e_l, e_w)
    _obs = _obs + frontinfo + lsideinfo + backinfo + rsideinfo

    return _obs


def get_info(neighbor1, neighbor2, flag, el, ew):
    if neighbor1[0] == 0 and neighbor2[0] == 0:
        return [0, 0, 0, 0]
    if neighbor1[0] == 0:
        n = 2
    if neighbor2[0] == 0:
        n = 1
    d1 = neighbor1[0]**2 + neighbor1[1]**2
    d2 = neighbor2[0]**2 + neighbor2[1]**2
    if d1 < d2:
        n = 1
    else:
        n = 2
    if n == 2:
        l, w = neighbor2[-2:]
        dx = neighbor2[0]
        dy = neighbor2[1]
        dsx, dsy = neighbor2[2:4]
    if n == 1:
        l, w = neighbor1[-2:]
        dx = neighbor1[0]
        dy = neighbor1[1]
        dsx, dsy = neighbor1[2:4]
    if flag == 1:
        dx -= (l + el) / 2
    if flag == 2:
        dy -= (w + ew) / 2
    if flag == 3:
        dx += (l + el) / 2
    if flag == 4:
        dy += (w + ew) / 2
    return [dx, dy, dsx, dsy]


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


def KalmanFilter(z):
    n_iter = len(z)
    # 这里是假设A=1，H=1的情况

    # intial parameters

    sz = (n_iter,)  # size of array

    # Q = 1e-5 # process variance
    Q = 1e-6  # process variance
    # allocate space for arrays
    xhat = np.zeros(sz)  # a posteri estimate of x
    P = np.zeros(sz)  # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)  # a priori error estimate
    K = np.zeros(sz)  # gain or blending factor

    R = 0.1 ** 2  # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = z[0]
    P[0] = 1.0
    A = 1
    H = 1

    for k in range(1, n_iter):
        # time update
        xhatminus[k] = A * xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        Pminus[k] = A * P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        xhat[k] = xhatminus[k] + K[k] * (z[k] - H * xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        P[k] = (1 - K[k] * H) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
    return xhat


def get_x(x, y):
    virtual_x = np.zeros_like(x)
    dx = np.sqrt((y[1:]-y[:-1])**2 + (x[1:]-x[:-1])**2)
    for i in range(len(x)-1):
        virtual_x[i+1] = virtual_x[i]+dx[i]
    return virtual_x


def show_filter(v, a):
    vf, af, _ = kalman_filter(v, a, a, 0.1)
    plt.plot(range(len(v)), v)
    plt.plot(range(len(v)), vf)
    plt.savefig('test/vf.png')
    plt.close()
    plt.plot(range(len(v)), a)
    plt.plot(range(len(v)), af)
    plt.savefig('test/af.png')
    plt.close()
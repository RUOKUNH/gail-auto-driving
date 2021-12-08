import pdb
import numpy as np
import torch as torch


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


# 8 nearest neighbors
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
    # closest 8 neighbors
    neighbor_num = 4
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
    for i in range(4):
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

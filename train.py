import time
import os
import pdb
import random
import argparse
from BC import BC
from utils import *
import pickle as pkl
from net import *
import matplotlib.pyplot as plt
from filter import kalman_filter
# from multiagent_traffic_simulator import MATrafficSim

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device)


def get_gen_data(path):
    pkf = open(f'{path}/gen_data.pkl', 'rb')
    observation = []
    action = []
    while True:
        try:
            record = pkl.load(pkf)
            observation += record['observation']
            action += record['actions']
        except EOFError:
            pkf.close()
            break
    return observation, action


def train_bc(args):
    expert_path = './expert_data_feature16.pkl'
    path = args.exp
    state_dim = 65  # feature15
    # state_dim = 53
    descriptor = feature15_descriptor
    # feature = feature15
    # env = MATrafficSim(['../scenarios/ngsim'], 10)
    if not os.path.exists(f'{args.exp}'):
        os.mkdir(f'{args.exp}')
    expert_buffer = []
    pkf = open(expert_path, 'rb')
    while True:
        try:
            record = pkl.load(pkf)
            expert_obs = record['observation']
            if descriptor:
                expert_obs = [descriptor(ob) for ob in expert_obs]
            expert_act = record['actions']
            expert_buffer.append((expert_obs, expert_act))
        except EOFError:
            pkf.close()
            break
    if not os.path.exists(f'{path}/gen_data.pkl'):
        with open(f'{path}/gen_data.pkl', 'ab') as f:
            pass
    last_modified_time = time.ctime(os.stat(f'{path}/gen_data.pkl').st_mtime)
    loss_log1 = []
    loss_log2 = []
    smooth_loss_log1 = []
    smooth_loss_log2 = []
    pnet = PolicyNetwork(state_dim, 2*2, [256, 128, 64, 32]).to(device)
    agent = BC(pnet, device)
    epoch = 10000
    batch_size = 1024
    random.seed(1)
    random.shuffle(expert_buffer)
    expert_state = []
    expert_action = []
    for expert_obs, expert_act in expert_buffer:
        expert_state += expert_obs
        expert_action += expert_act
    expert_state = torch.tensor(expert_state).float().to(device)
    expert_action = torch.tensor(expert_action).float().to(device)
    normalize_weight = torch.mean(expert_state, dim=0)
    normalize_std = torch.std(expert_state, dim=0)
    expert_state = (expert_state - normalize_weight) / normalize_std
    train_states = expert_state
    train_actions = expert_action
    for _epoch in range(epoch):
        shuffle_idx = np.arange(len(train_states))
        random.shuffle(shuffle_idx)
        shuffle_idx = torch.from_numpy(shuffle_idx)
        train_states = train_states[shuffle_idx]
        train_actions = train_actions[shuffle_idx]
        print(f'Epoch{_epoch+1}')
        iter = 0
        agent.train()
        while (iter+1)*batch_size <= len(train_states):
            iter_idx = torch.arange(iter*batch_size, (iter+1)*batch_size)
            # iter_idx = shuffle_idx[iter_idx]
            states = train_states[iter_idx]
            actions = train_actions[iter_idx]
            batch = {'state': states, 'action': actions}
            loss1, loss2 = agent.update(batch)
            loss_log1.append(loss1)
            loss_log2.append(loss2)
            if (iter+1) % 10 == 0:
                smooth_loss_log1.append(np.mean(loss_log1[-10:]))
                smooth_loss_log2.append(np.mean(loss_log2[-10:]))
            if (iter+1) % 100 == 0:
                print(f"{args.exp} iter{iter+1}, "
                      "loss %.3f %.3f %.3f" % (np.mean(loss_log1[-100:]), np.mean(loss_log2[-100:]),
                    torch.mean(torch.cat([param.view(-1) for param in agent.policy.parameters()]) ** 2).cpu().detach().numpy()))
                plt.plot(np.arange(len(smooth_loss_log1)), smooth_loss_log1)
                plt.plot(np.arange(len(smooth_loss_log2)), smooth_loss_log2)
                plt.savefig(f'{args.exp}/smoothloss.png')
                plt.close()
            iter += 1
        agent.save_model(f'{args.exp}/model.pth', _epoch)


def train_gail(args):
    from gail_ppo import GAIL_PPO
    expert_path = './expert_data_feature16.pkl'
    action_dim = 2
    feature = feature15
    # state_dim = 62
    # descriptor = feature15_descriptor2
    state_dim = 65
    descriptor = feature15_descriptor
    agent = GAIL_PPO(state_dim, action_dim, args)
    agent.train(expert_path, feature, descriptor)


def evaluate(env, normalize_weight, normalize_std, agent, feature, descriptor=None):
    x_distance_log = []
    success_log = []
    state = env.reset()
    x_start = {}
    step = 0
    finish = 0
    while finish < 20:
        if len(state) == 0:
            break
        step += 1
        for agent_id in state.keys():
            if agent_id not in x_start.keys():
                x_start[agent_id] = state[agent_id].ego_vehicle_state.position[0]
        if descriptor:
            _state = [descriptor(feature(expert_collector3(state[agent_id]))) for agent_id in state.keys()]
        else:
            _state = [feature(expert_collector3(state[agent_id])) for agent_id in state.keys()]
        _state = torch.tensor(_state).float()
        _state = (_state - normalize_weight) / normalize_std
        _action = agent.action(_state)
        action = {}
        idx = -1
        for agent_id in state.keys():
            idx += 1
            action[agent_id] = _action[idx].detach().numpy()
        next_state, r, done, _ = env.step(action)
        for agent_id in done.keys():
            if done[agent_id]:
                finish += 1
                x_end = next_state[agent_id].ego_vehicle_state.position[0]
                x_distance_log.append(x_end - x_start[agent_id])
                if next_state[agent_id].events.reached_goal:
                    success_log.append(1)
                else:
                    success_log.append(0)
        if descriptor:
            _next_state = [descriptor(feature(expert_collector3(next_state[id]))) for id in state.keys()]
        else:
            _next_state = [feature(expert_collector3(next_state[id])) for id in state.keys()]
        _next_state = torch.tensor(_next_state).float()
        _next_state = (_next_state - normalize_weight) / normalize_std
        _done = [done[id] for id in state.keys()]

        state = {}
        for agent_id in next_state.keys():
            if done[agent_id]:
                continue
            state[agent_id] = next_state[agent_id]

    print(f"mdist {np.mean(x_distance_log)} srate {np.mean(success_log)} {len(x_distance_log)} {len(success_log)}")
    return x_distance_log, success_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',
                        help='experiment name',
                        type=str)
    parser.add_argument('--con',
                        help='whether continue with the experiment',
                        default=False,
                        type=bool)
    parser.add_argument('--optim',
                        default='ppo',
                        type=str)
    args = parser.parse_args()

    # main(args)

    # train_bc(args)

    train_gail(args)


import os
import json
from traffic_simulator import TrafficSim
from gail_trpo import GAIL
import torch
import argparse
from gail_ppo import GAIL_PPO


def main(args):
    expert_path = './expert_data5.pkl'
    with open('config.json') as f:
        config = json.load(f)

    state_dim = 34  # feature2
    # state_dim = 39    # feature3
    action_dim = 2

    if args.optim == 'trpo':
        model = GAIL(state_dim, action_dim, config, args)
    if args.optim == 'ppo':
        model = GAIL_PPO(state_dim, action_dim, config, args)

    pi, v, d = model.train(expert_path)
    #
    # state = {'action_net': pi.state_dict(),
    #          'value_net': v.state_dict(),
    #          'disc_net': d.state_dict()}
    # torch.save(state, 'model.pth')


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
    main(args)

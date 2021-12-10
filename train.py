import os
import json
from traffic_simulator import TrafficSim
from gail_trpo import GAIL
import torch
import argparse
from gail_ppo import GAIL_PPO
from utils import *


def main(args):
    ###### train settings ######
    # expert_path = './expert_data_feature2.pkl'
    # state_dim = 34  # feature2
    # feature = feature2

    # expert_path = './expert_data_feature3.pkl'
    # state_dim = 39    # feature3
    # feature = feature3

    # expert_path = './expert_data_feature4.pkl'
    # state_dim = 23  # feature4
    # feature = feature4

    # expert_path = './expert_data_feature5.pkl'
    # state_dim = 22  # feature5
    # feature = feature5

    expert_path = './expert_data_feature6.pkl'
    state_dim = 35  # feature6
    feature = feature6

    # expert_path = './expert_data_feature7.pkl'
    # state_dim = 35  # feature7
    # feature = feature7

    # kld_limit = False
    kld_limit = True

    synchronize = 5
    epoch = 5
    d_iters = 50
    v_iters = 25
    beta = 0.995

    action_dim = 2
    ############################

    with open('config.json') as f:
        config = json.load(f)

    if args.optim == 'trpo':
        model = GAIL(state_dim, action_dim, config, args)
    else:
        model = GAIL_PPO(state_dim, action_dim, config, args, synchronize=synchronize)

    model.train(expert_path, feature,
                kld_limit=kld_limit,
                epoch=epoch,
                d_iters=d_iters,
                v_iters=v_iters,
                beta=beta)


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

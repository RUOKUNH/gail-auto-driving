import os
import json
from traffic_simulator import TrafficSim
from gail import GAIL
import torch
import argparse


def main(args):
    expert_path = './expert_simple_2.pkl'
    with open('config.json') as f:
        config = json.load(f)

    # env = TrafficSim(["./scenarios/ngsim"])
    env = TrafficSim(["./ngsim"])
    print('env created')

    model = GAIL(20, 2, config, args)

    pi, v, d = model.train(env, expert_path)
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
    args = parser.parse_args()
    main(args)

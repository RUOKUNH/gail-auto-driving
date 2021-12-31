import os
import pdb
import sys
import argparse
import pickle
import sqlite3
import numpy as np
import pickle as pkl
from gail_ppo import GAIL_PPO
from BC import BC
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from utils import *
from net import PolicyNetwork

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR.parent))

from eval_sampler import ParallelPathSampler
from fixed_traffic_simulator import FixedMATrafficSim
from math_utils import DiscreteFrechet

sys.setrecursionlimit(25000)


def benchmark(vehicle_id_groups_list, agent_num, ngsim_path, agent, feature, descriptor, args):
    sampler = ParallelPathSampler(
        [make_env(vehicle_id_groups, agent_num, ngsim_path) for vehicle_id_groups in vehicle_id_groups_list],
        agent,
        feature,
        descriptor,
        args
    )
    rollout_trajs = sampler.collect_samples()
    compute_metrics(rollout_trajs, ngsim_path, args)


def compute_metrics(rollout_trajs, ngsim_path, args):
    def _compute_frechet_distance(sample_traj, expert_traj):
        # pdb.set_trace()
        frechet_solver = DiscreteFrechet(
            dist_func=lambda p, q: np.linalg.norm(p - q)
        )
        frechet_distance = frechet_solver.distance(
            np.stack([traj[:2] for traj in expert_traj], axis=0),
            np.stack([traj for traj in sample_traj["observations"]], axis=0),
        )
        return frechet_distance

    def _compute_distance_travelled(sample_traj):
        # NOTE(zbzhu): we only care about the distance in horizontal direction
        distance_travelled = abs(
            sample_traj["observations"][-1][0]
            - sample_traj["observations"][0][0]
        )
        return distance_travelled

    def _judge_success(sample_traj):
        if sample_traj["infos"][-1]["reached_goal"]:
            return 1.0
        else:
            return 0.0

    def _judge_wrong_way(sample_traj):
        wrong_way = []
        for info in sample_traj["infos"]:
            if info["wrong_way"]:
                wrong_way.append(1.0)
            else:
                wrong_way.append(0.0)
        return wrong_way

    def _judge_collision(sample_traj):
        if sample_traj["infos"][-1]["cossision"]:
            return 1.0
        else:
            return 0.0

    metrics = {}
    demo_path = Path(ngsim_path) / "i80_0400-0415.shf"
    dbconxn = sqlite3.connect(demo_path)
    cur = dbconxn.cursor()
    print(">>> Evaluating...")
    for vehicle_id in tqdm(rollout_trajs.keys()):
        if vehicle_id not in metrics.keys():
            metrics[vehicle_id] = defaultdict(list)
        query = """SELECT position_x, position_y, heading_rad, speed
                    FROM Trajectory
                    WHERE vehicle_id = ?
                    """
        cur.execute(query, [vehicle_id])
        expert_traj = cur.fetchall()
        for sample_traj in rollout_trajs[vehicle_id]:
            metrics[vehicle_id]["Frechet Distance"].append(_compute_frechet_distance(sample_traj, expert_traj))
            metrics[vehicle_id]["Distance Travelled"].append(_compute_distance_travelled(sample_traj))
            metrics[vehicle_id]["Success"].append(_judge_success(sample_traj))
            metrics[vehicle_id]["Wrong Way"] += _judge_wrong_way(sample_traj)
            metrics[vehicle_id]["Collision"].append(_judge_collision(sample_traj))

    cur.close()

    with open(f"evaluation/{args.model}_{args.agent_num}.pkl", "wb") as f:
        pkl.dump(metrics, f)

    metric_tot = defaultdict(list)
    for vehicle_id in metrics.keys():
        metric_tot["Frechet Distance"] += metrics[vehicle_id]["Frechet Distance"]
        metric_tot["Distance Travelled"] += metrics[vehicle_id]["Distance Travelled"]
        metric_tot["Success"] += metrics[vehicle_id]["Success"]
        metric_tot["Wrong Way"] += metrics[vehicle_id]["Wrong Way"]
        metric_tot["Collision"] += metrics[vehicle_id]["Collision"]


    print(f"{args.model} {args.agent_num}")
    print("Average Frechet Distance: {}".format(np.mean(metric_tot["Frechet Distance"])))
    print("Average Distance Travelled: {}".format(np.mean(metric_tot["Distance Travelled"])))
    print("Success Rate: {}".format(np.mean(metric_tot["Success"])))
    print("Collision Rate: {}".format(np.mean(metric_tot["Collision"])))
    print("Wrong Way Rate: {}".format(np.mean(metric_tot["Wrong Way"])))
    print("Max Distance Travelled: {}".format(np.max(metric_tot["Distance Travelled"])))


def make_env(vehicle_id_groups, agent_number, ngsim_path):
    def _init():
        return FixedMATrafficSim(
            scenarios=[ngsim_path],
            agent_number=agent_number,
            vehicle_id_groups=vehicle_id_groups,
        ), len(vehicle_id_groups)
    return _init


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        # default='../scenarios/ngsim',
        type=str,
        # help="Path to your ngsim scenario folder"
    )
    parser.add_argument(
        "--agent_num",
        default=10,
        type=int,
        help="Num of vehicles controlled during sampling"
    )
    args = parser.parse_args()
    ngsim_path = '../scenarios/ngsim'
    env_num = 1
    agent_num = args.agent_num

    action_dim = 2
    feature = feature15
    if args.model == 'multi':
        descriptor = feature15_descriptor
        state_dim = 65
        exp_path = 'save_model'
        agent = GAIL_PPO(state_dim, action_dim, None, policy_net=[512, 256, 128, 64])
        agent.load_model(f"{exp_path}/{args.model}.pth")
    elif args.model == 'single':
        descriptor = feature15_descriptor4
        state_dim = 20
        exp_path = 'save_model'
        agent = GAIL_PPO(state_dim, action_dim, None, policy_net=[512, 256, 128, 64], alim=False)
        agent.load_model(f"{exp_path}/{args.model}.pth")
    elif args.model == 'bc':
        exp_path = 'bc-feature16'
        state_dim = 66
        descriptor = feature15_descriptor1
        pnet = PolicyNetwork(state_dim, 4, [256, 128, 64, 64, 32])
        agent = BC(pnet)
        agent.load_model(f"{exp_path}/model.pth")
    else:
        raise KeyError("Incorrect Model Type")

    with open(os.path.join(BASE_DIR, "test_ids.pkl"), "rb") as f:
        eval_vehicle_ids = pickle.load(f)

    eval_vehicle_ids = eval_vehicle_ids[238:338]

    eval_vehicle_groups = []
    for idx in range(len(eval_vehicle_ids) - agent_num+1):
        eval_vehicle_groups.append(eval_vehicle_ids[idx: idx + agent_num])

    vehicle_id_groups_list = np.array_split(
        eval_vehicle_groups,
        env_num,
    )

    benchmark(vehicle_id_groups_list, agent_num, ngsim_path, agent, feature, descriptor, args)

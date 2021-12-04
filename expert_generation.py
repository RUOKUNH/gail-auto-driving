import copy
import pdb
import os

import numpy as np
import pickle
import argparse
import time

from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType

from utils import *


def acceleration_count(obs, obs_next, acc_dict, ang_v_dict, avg_dis_dict):
    acc_dict = {}
    for car in obs.keys():
        car_state = obs[car].ego_vehicle_state
        angular_velocity = car_state.yaw_rate
        ang_v_dict.append(angular_velocity)
        dis_cal = car_state.speed * 0.1
        if car in avg_dis_dict:
            avg_dis_dict[car] += dis_cal
        else:
            avg_dis_dict[car] = dis_cal
        if car not in obs_next.keys():
            continue
        car_next_state = obs_next[car].ego_vehicle_state
        acc_cal = (car_next_state.speed - car_state.speed) / 0.1
        acc_dict.append(acc_cal)


def cal_action(obs, obs_next, dt=0.1):
    act = {}
    for car in obs.keys():
        if car not in obs_next.keys():
            continue
        car_state = obs[car].ego_vehicle_state
        car_next_state = obs_next[car].ego_vehicle_state
        acceleration = (car_next_state.speed - car_state.speed) / dt
        angular_velocity = car_state.yaw_rate
        act[car] = np.array([acceleration, angular_velocity])
    return act


def main(scenario):
    # dataset_name = 'expert_simple.pkl' # [tan(heading), speed, x, y, l, w, ....]
    # dataset_name = 'expert_simple_2.pkl' # [heading, speed, x, y, l, w, ....]
    # dataset_name = 'expert_simple_3.pkl'  # [heading, speed, x, y, l, w, ....(10 neighbor)]
    # dataset_name = 'expert_data.pkl'  # [heading, speed, x, y, l, w, nx,ny,nl,nw,nh,ns,....(4 neighbor)]
    dataset_name = 'expert_data3.pkl'
    """Collect expert observations.

    Each input scenario is associated with some trajectory files. These trajectories
    will be replayed on SMARTS and observations of each vehicle will be collected and
    stored in a dict.

    Args:
        scenarios: A string of the path to scenarios to be processed.

    Returns:
        A dict in the form of {"observation": [...], "next_observation": [...], "done": [...]}.
    """

    agent_spec = AgentSpec(
        interface=AgentInterface(
            max_episode_steps=None,
            waypoints=False,
            neighborhood_vehicles=True,
            ogm=False,
            rgb=False,
            lidar=False,
            action=ActionSpaceType.Imitation,
        )
    )

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
    )
    scenarios_iterator = Scenario.scenario_variations(
        [scenario],
        list([]),
    )

    smarts.reset(next(scenarios_iterator))

    # expert_obs = []
    # expert_acts = []
    # expert_obs_next = []
    # expert_terminals = []
    cars_obs = {}
    cars_act = {}
    # cars_ego_states = {}
    # cars_neighbors = {}
    # cars_obs_next = {}
    # cars_terminals = {}

    vehicles_to_end = set()
    prev_vehicles = set()
    prev_obs = None
    t = time.time()
    ts = 0
    if os.path.exists(dataset_name):
        os.remove(dataset_name)
    while True:
        smarts.step({})

        current_vehicles = smarts.vehicle_index.social_vehicle_ids()
        done_vehicles = prev_vehicles - current_vehicles
        vehicles_to_end = vehicles_to_end | done_vehicles
        prev_vehicles = current_vehicles

        if len(current_vehicles) == 0:
            break

        smarts.attach_sensors_to_vehicles(
            agent_spec, smarts.vehicle_index.social_vehicle_ids()
        )
        obs, _, _, dones = smarts.observe_from(
            smarts.vehicle_index.social_vehicle_ids()
        )
        # pdb.set_trace()

        if done_vehicles:
            ts = time.time() - t
        for v in done_vehicles:
            car = f"Agent-{v}"
            # cars_terminals[car][-1] = True
            print(f"{car} Ended, survive {len(current_vehicles)}, finish {len(vehicles_to_end)}"
                  f", time {ts}")
            observation = cars_obs[car][:-1]
            # ego_state = cars_ego_states[car][:-1]
            # neighbors = cars_neighbors[car][:-1]
            # next_observation = cars_obs[car][1:]
            acts = np.array(cars_act[car])
            # terminals = np.array(cars_terminals[car][:-1])
            with open(dataset_name, "ab") as f:
                pickle.dump(
                    {
                        "car": car,
                        "observation": observation,
                        # "ego_state": ego_state,
                        # "neighbors": neighbors,
                        "actions": acts,
                    },
                    f,
                )
            cars_obs.pop(car)
            # cars_terminals.pop(car)
            cars_act.pop(car)
            # cars_ego_states.pop(car)
            # cars_neighbors.pop(car)

        # handle actions
        if prev_obs is not None:
            act = cal_action(prev_obs, obs)
            for car in act.keys():
                if cars_act.__contains__(car):
                    cars_act[car].append(act[car])
                else:
                    cars_act[car] = [act[car]]
        prev_obs = copy.copy(obs)

        # handle observations
        cars = obs.keys()
        for car in cars:
            if cars_obs.__contains__(car):
                cars_obs[car].append(expert_collector(obs[car]))
            else:
                cars_obs[car] = [expert_collector(obs[car])]

    smarts.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="../scenarios/ngsim",
    )
    args = parser.parse_args()
    main(scenario=args.scenario)

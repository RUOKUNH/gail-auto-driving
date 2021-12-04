import pdb
import numpy as np
import random
from dataclasses import replace

from smarts.core.smarts import SMARTS
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.scenario import Scenario
from smarts.core.traffic_history_provider import TrafficHistoryProvider
from envision.client import Client as EnvisionClient


def get_action_adapter():
    def action_adapter(model_action):
        assert len(model_action) == 2
        return (model_action[0], model_action[1])

    return action_adapter


class TrafficSim:
    def __init__(self, scenarios, envision=True, collectors=1):
        self.collectors = collectors
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self._init_scenario()
        self.obs_stacked_size = 1
        self.agent_spec = AgentSpec(
            interface=AgentInterface(
                max_episode_steps=None,
                waypoints=False,
                neighborhood_vehicles=True,
                ogm=False,
                rgb=False,
                lidar=False,
                action=ActionSpaceType.Imitation,
            ),
            action_adapter=get_action_adapter(),
        )
        self.smarts = []
        self.vehicle_id = [0] * self.collectors
        for i in range(collectors):
            if envision:
                self.smarts.append(SMARTS(
                    agent_interfaces={},
                    traffic_sim=None,
                    envision=EnvisionClient(
                        sim_name='single_agent',
                        ),
                ))
            else:
                self.smarts.append(SMARTS(
                    agent_interfaces={},
                    traffic_sim=None,
                ))
            print(f'env {i} created')

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action, smart_id=0):

        observations, rewards, dones, _ = self.smarts[smart_id].step(
            {self.vehicle_id[smart_id]: self.agent_spec.action_adapter(action)}
        )

        return (
            observations[self.vehicle_id[smart_id]],
            rewards[self.vehicle_id[smart_id]],
            dones[self.vehicle_id[smart_id]],
            {},
        )

    def reset(self, smart_id=0):
        vehicle_itr = random.randint(0, len(self.vehicle_ids[smart_id])-1)
        # if self.vehicle_itr >= len(self.vehicle_ids[smart_id]):
        #     self.vehicle_itr = 0

        self.vehicle_id[smart_id] = self.vehicle_ids[smart_id][vehicle_itr]
        vehicle_mission = self.vehicle_missions[smart_id][self.vehicle_id[smart_id]]

        traffic_history_provider = self.smarts[smart_id].get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider
        traffic_history_provider.start_time = vehicle_mission.start_time

        modified_mission = replace(vehicle_mission, start_time=0.0)
        self.scenarios[smart_id].set_ego_missions({self.vehicle_id[smart_id]: modified_mission})
        self.smarts[smart_id].switch_ego_agents({self.vehicle_id[smart_id]: self.agent_spec.interface})

        observations = self.smarts[smart_id].reset(self.scenarios[smart_id])
        # self.vehicle_itr += 1
        # print(self.vehicle_id[smart_id])
        return observations[self.vehicle_id[smart_id]]

    def _init_scenario(self):
        # pdb.set_trace()
        random.seed()
        self.vehicle_itr = 0
        self.scenarios = []
        self.vehicle_missions = []
        self.vehicle_ids = []
        for i in range(self.collectors):
            self.scenarios.append(next(self.scenarios_iterator))
            self.vehicle_missions.append(self.scenarios[i].discover_missions_of_traffic_histories())
            self.vehicle_ids.append(list(self.vehicle_missions[i].keys()))
            random.shuffle(self.vehicle_ids[i])
            print(f'scenario {i} created')

    def destroy(self):
        for i in range(self.collectors):
            if self.smarts[i] is not None:
                self.smarts[i].destroy()

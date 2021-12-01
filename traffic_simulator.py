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
        self.collectors = collectors
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

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action, smart_id=0):

        observations, rewards, dones, _ = self.smarts[smart_id].step(
            {self.vehicle_id: self.agent_spec.action_adapter(action)}
        )

        return (
            observations[self.vehicle_id],
            rewards[self.vehicle_id],
            dones[self.vehicle_id],
            {},
        )

    def reset(self, smart_id=0):
        self.vehicle_itr = random.randint(0, len(self.vehicle_ids)-1)
        # if self.vehicle_itr >= len(self.vehicle_ids):
        #     self.vehicle_itr = 0

        self.vehicle_id = self.vehicle_ids[self.vehicle_itr]
        vehicle_mission = self.vehicle_missions[self.vehicle_id]

        traffic_history_provider = self.smarts[smart_id].get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider
        traffic_history_provider.start_time = vehicle_mission.start_time

        modified_mission = replace(vehicle_mission, start_time=0.0)
        self.scenario.set_ego_missions({self.vehicle_id: modified_mission})
        self.smarts[smart_id].switch_ego_agents({self.vehicle_id: self.agent_spec.interface})

        observations = self.smarts[smart_id].reset(self.scenario)
        # self.vehicle_itr += 1
        return observations[self.vehicle_id]

    def _init_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.vehicle_ids = list(self.vehicle_missions.keys())
        np.random.shuffle(self.vehicle_ids)
        self.vehicle_itr = 0

    def destroy(self):
        for i in range(self.collectors):
            if self.smarts[i] is not None:
                self.smarts[i].destroy()

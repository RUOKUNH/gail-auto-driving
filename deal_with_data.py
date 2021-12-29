import pdb
import pickle as pkl
import os

import matplotlib.pyplot as plt

import numpy as np

from utils import *
from filter import kalman_filter


# [ego_state, nx,ny,nl,nw,nh,ns,....(4 neighbor), events]
dataset_name = 'expert_data_feature18.pkl'
pkf = open('expert_data_full.pkl', 'rb')
if os.path.exists(dataset_name):
    os.remove(dataset_name)
i = 0
while True:
    try:
        record = pkl.load(pkf)
        car = record['car']
        observation = record['observation']
        observation = [expert_collector3(obs) for obs in observation]
        h = [ob[0].heading.real for ob in observation]
        observation = [feature18(obs) for obs in observation]
        acts = list(record['actions'])
        observation = np.array(observation)
        acts = np.array(acts)
        x = observation[:, 0].copy()
        y = observation[:, 1].copy()
        x = get_x(x, y)
        v = observation[:, 8].copy()
        a = acts[:, 0].copy()
        if np.abs(a[0]) > 5:
            a[0] = 0
        w = acts[:, 1].copy()
        if np.abs(w[0]) > 2:
            w[0] = 0
        hf, wf, _ = kalman_filter(h, w, w, 0.1)
        xf, vf, af = kalman_filter(x, v, a, 0.1)
        # vf, af, _ = kalman_filter(v, a, a, 0.1)
        acts[:, 0] = np.array(af)
        acts[:, 1] = np.array(wf)
        if np.max(np.abs(acts)) > 10:
            pdb.set_trace()
        acts = acts.tolist()
        observation = observation.tolist()

        with open(dataset_name, "ab") as f:
            pkl.dump(
                {
                    "car": car,
                    "observation": observation,
                    "actions": acts,
                },
                f,
            )
        i += 1
        print(f'collect {i}')
    except EOFError:
        pkf.close()
        break
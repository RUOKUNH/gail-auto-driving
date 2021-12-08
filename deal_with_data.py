import pickle as pkl
import os
from utils import *


# [ego_state, nx,ny,nl,nw,nh,ns,....(4 neighbor), events]
dataset_name = 'expert_data_feature5.pkl'
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
        observation = [feature5(obs) for obs in observation]
        acts = list(record['actions'])
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
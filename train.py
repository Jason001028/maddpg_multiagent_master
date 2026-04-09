import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import random
import torch
import numpy as np
import pandas as pd
import torch.multiprocessing as mp

from arguments import Args as args
from core.runner import Runner


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def load_obstacle_states():
    L1 = pd.read_excel("origin_obstacle_states_mid.xlsx", engine="openpyxl", sheet_name="Sheet1")
    states = []
    for i in range(16):
        for j in range(16):
            y = L1.iloc[j, i] if j < L1.shape[0] else 0
            if y != 0:
                states.append([i, y])
    return states


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default=None, choices=['qmix', 'vdn', 'legacy_maddpg', 'iql', 'ra_maddpg', 'ef_maddpg'])
    parser.add_argument('--epochs', type=int, default=None)
    cli = parser.parse_args()

    if cli.algo is not None:
        args.algo_name = cli.algo
        args.train_params['env_name'] = cli.algo + '_grid_world_' + "seed" + str(args.seed) + '_' + args.train_params['date']

    if cli.epochs is not None:
        # evalue_interval steps per epoch
        args.train_params['learner_step'] = cli.epochs * args.train_params['evalue_interval']

    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['MKL_NUM_THREADS'] = "1"
    torch.set_num_threads(1)
    mp.set_sharing_strategy('file_system')

    setup_seed(args.seed)

    origin_obstacle_states = load_obstacle_states()

    runner = Runner(args, args.env_params, args.train_params, origin_obstacle_states)
    runner.run()

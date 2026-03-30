import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())

from core.registry import get_algorithm
from Env.env import Gridworld
from Env.reward_wrapper import RewardWrapper
from arguments import Args

import torch
import time
import traceback
import numpy as np 
from copy import deepcopy

env_params = Args.env_params
train_params = Args.train_params
max_timesteps = env_params.max_timesteps
store_interval = train_params.store_interval
n_agents = env_params.n_agents

@torch.no_grad()
def actor_worker(
    data_queue,
    actor_queue,
    actor_index,
    origin_obstacle_states
):
    try:
        from core.logger import Logger
        logger = Logger(logger=f"actor_{actor_index}")
        logger.info(f"Actor {actor_index} started.")
        # init env
        env = RewardWrapper(Gridworld(obstacles=origin_obstacle_states, agent_configs=Args.role_configs))
        store_item = ['obs', 'next_obs', 'acts', 'reward', 'dones', 'role_features']
        policy = get_algorithm(Args.algo_name, Args, env_params, device='cpu')
        init_flag = False
        rolltime_count = 0
        # sampling ..
        while True:
            # update model params periodly
            if not actor_queue.empty():
                data = actor_queue.get()
                policy.sync_actor(data)
                init_flag = True
            # first time initialization
            elif not init_flag:
                time.sleep(5)
                continue
            mb_store_dict = {item: [] for item in store_item}
            rolltime_count += 1
            for rollouts_times in range(store_interval):
                ep_store_dict = {item: [] for item in store_item}
                #在这里实际重置环境
                obs = env.reset() # reset the environment
                # start to collect samples
                count_agentself_total = [0, 0, 0]
                for t in range(max_timesteps):
                    ##探索工作量统计列表
                    actions = policy.act(obs, explore=True)
                    _, _, reward, next_obs, done, info = env.step(t, actions)
                    escape_rate = info[0].get('escape_rate', 0)
                    count_agentself_total = list(np.add(count_agentself_total, info[0]['step_cover_delta']))
                    save_fig = info[0] if t == max_timesteps - 1 else None
                    save_fig_path = f'results_png/demo_{rolltime_count}_{rollouts_times}.png' if save_fig else None
                    #此处包含实时绘制参数
                    env.render(escape_rate, reward, done, save_fig_path)
                    # role_features: list of dicts → (n_agents, 2) float array
                    rf = info[0].get('role_features', [{'task_rate': 0.0, 'viewrange': 0.0}] * n_agents)
                    rf_arr = np.array([[d['task_rate'], d['viewrange']] for d in rf], dtype=np.float32)
                    store_data = {
                        'obs': obs,
                        'next_obs': next_obs if t != max_timesteps - 1 else obs,
                        'acts': actions,
                        'reward': reward,
                        'dones': np.array(done, dtype=np.float32).reshape(n_agents, 1),
                        'role_features': rf_arr,
                    }
                    # append rollouts
                    for key, val in store_data.items():
                        ep_store_dict[key].append(val.copy())
                    obs = next_obs
                for key in store_item:
                    mb_store_dict[key].append(deepcopy(ep_store_dict[key]))
            # convert them into arrays and send as dict
                episode_dict = {key: np.array(mb_store_dict[key]) for key in store_item}
                data_queue.put(episode_dict, block=True)
            # real_size = self.buffer.check_real_cur_size()
            logger.info(f'actor {actor_index} send data, current data_queue size is {store_interval * data_queue.qsize()}')
    except KeyboardInterrupt:
        logger.critical(f"interrupt")
    except Exception as e:
        logger.error(f"Exception in worker process {actor_index}")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
        # test actor
    from core import actor

    # #! /usr/bin/env python
    import random
    import torch
    import time
    import torch.multiprocessing as mp
    import numpy as np 

    from arguments import Args as args
    from core.logger import Logger
    from core.actor import actor_worker

    import os

    # set logging level 
    logger = Logger(logger="dual_arm_multiprocess")
    train_params = args.train_params
    env_params = args.env_params
    actor_num = train_params.actor_num
    model_path = os.path.join(train_params.save_dir, train_params.env_name)
    if not os.path.exists(train_params.save_dir):
        os.mkdir(train_params.save_dir)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # starting multiprocess
    ctx = mp.get_context("spawn") # using shared cuda tensor should use 'spawn'
    # queue to transport data
    data_queue = ctx.Queue()
    actor_queues = [ctx.Queue() for _ in range(1)]
    actor_processes = []
    for i in range(1):
        actor = ctx.Process(
            target = actor_worker,
            args = (
                data_queue,
                actor_queues[i],
                i,
                logger,
            )
        )
        logger.info(f"Starting actor:{i} process...")
        actor.start()
        actor_processes.append(actor)
        time.sleep(1)

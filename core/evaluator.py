#! /usr/bin/env python
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())

import time
from core.registry import get_algorithm
from Env.env import Gridworld
from Env.reward_wrapper import RewardWrapper
from arguments import Args


class Evaluator:
    """
    基于论文公式的多项式评估系统。
    权重 ω1~ω6 对应: success, reward, time, energy, collision, distance
    """
    def __init__(self, env, algo, max_timesteps,
                 w1=1.0, w2=1.0, w3=1.0, w4=1.0, w5=1.0, w6=1.0):
        self.env = env
        self.algo = algo
        self.max_timesteps = max_timesteps
        self.weights = (w1, w2, w3, w4, w5, w6)

    def evaluate_model(self, n_episodes: int) -> dict:
        w1, w2, w3, w4, w5, w6 = self.weights

        n_success = 0
        sum_reward = 0.0
        sum_time_success = 0.0
        sum_energy = 0.0
        sum_collision = 0.0
        sum_distance_success = 0.0
        sum_coverage = 0.0

        for _ in range(n_episodes):
            obs = self.env.reset()
            ep_reward = 0.0
            ep_energy = 0.0
            ep_collision = 0
            ep_distance = 0.0
            ep_steps = 0
            ep_success = False

            for t in range(self.max_timesteps):
                actions = self.algo.act(obs, explore=False)
                step_result = self.env.step(t, actions)
                if len(step_result) == 6:
                    _, _, reward, obs, dones, info = step_result
                else:
                    obs, reward, dones, info = step_result

                info0 = info[0]
                ep_reward += sum(reward)
                ep_energy += info0.get('energy_cost', 0.0)
                ep_collision += info0.get('collisions', 0)
                ep_distance += info0.get('distance_delta', 0.0)
                ep_steps += 1

                ep_coverage = info0.get('coverage_rate', 0.0)
                if dones[0]:
                    ep_success = info0.get('is_success', False)
                    break

            sum_reward += ep_reward
            sum_energy += ep_energy
            sum_collision += ep_collision
            sum_coverage += ep_coverage

            if ep_success:
                n_success += 1
                sum_time_success += ep_steps
                sum_distance_success += ep_distance

        E_success = n_success / n_episodes
        E_reward = sum_reward / n_episodes
        E_coverage = sum_coverage / n_episodes
        # 防除零：成功数为0时用最大惩罚值
        E_time = sum_time_success / n_success if n_success > 0 else float(self.max_timesteps)
        E_energy = sum_energy / n_episodes
        E_collision = sum_collision / n_episodes
        E_distance = sum_distance_success / n_success if n_success > 0 else float(self.max_timesteps)

        fitness = (w1 * E_success + w2 * E_reward
                   - w3 * E_time - w4 * E_energy
                   - w5 * E_collision - w6 * E_distance)

        return {
            'success_rate': E_success,
            'mean_reward': E_reward,
            'mean_coverage': E_coverage,
            'mean_time': E_time,
            'mean_energy': E_energy,
            'mean_collision': E_collision,
            'mean_distance': E_distance,
            'fitness': fitness,
        }


def evaluate_worker(
        __train_params,
        env_params,
        plot_path,
        evalue_time,
        evalue_queue,
        origin_obstacle_states,
    ):
    from core.logger import Logger
    logger = Logger(logger="evaluator")
    env = RewardWrapper(Gridworld(obstacles=origin_obstacle_states, agent_configs=Args.role_configs))
    algo = get_algorithm(Args.algo_name, Args, env_params, device='cpu')

    evaluator = Evaluator(env, algo, max_timesteps=env_params.max_timesteps)

    while True:
        if not evalue_queue.empty():
            data = evalue_queue.get()
            evaluate_step = data['step']
            algo.sync_actor(data)
            for a in algo.model.actors:
                a.eval()

            metrics = evaluator.evaluate_model(n_episodes=evalue_time)
            metrics['step'] = evaluate_step
            metrics['actor_loss'] = data.get('actor_loss', 0.0)
            metrics['critic_loss'] = data.get('critic_loss', 0.0)

            logger.info(
                f"epoch={evaluate_step // Args.train_params.evalue_interval} | eval step={evaluate_step} | "
                f"success={metrics['success_rate']:.3f} | "
                f"coverage={metrics['mean_coverage']:.3f} | "
                f"reward={metrics['mean_reward']:.2f} | "
                f"time={metrics['mean_time']:.1f} | "
                f"energy={metrics['mean_energy']:.2f} | "
                f"collision={metrics['mean_collision']:.2f} | "
                f"distance={metrics['mean_distance']:.2f} | "
                f"fitness={metrics['fitness']:.3f}"
            )

            from core.logger import log_eval_metrics
            log_eval_metrics(plot_path, metrics)
        else:
            time.sleep(30)

import math
import gym
import numpy as np


class MARLRewardWrapper(gym.Wrapper):
    """
    奖励整形挂载点。env.step() 仅做物理转移，本 Wrapper 接管全部奖励计算。

    标准输出元组：(obs, actions, rewards, next_obs, dones, infos)
      - infos[i]['role_features']：角色特征向量 E_i，直接喂给 Actor 的 f_role 编码器

    扩展接口（后续阶段注入）：
      - _compute_marginal_contribution(agent_id, state, action)
      - _compute_kl_regularization(agent_id)
    """

    def step(self, t, actions, obs=None):
        next_obs, _, dones, infos = self.env.step(t, actions)
        rewards = self._compute_rewards(actions, infos)
        rewards = self._compute_marginal_contribution_hook(rewards, infos)
        rewards = self._compute_kl_regularization_hook(rewards, infos)
        return obs, actions, rewards, next_obs, dones, infos

    def _compute_rewards(self, actions, infos):
        env = self.env
        count_oneclear_total = infos[0]['step_cover_delta']
        valid_actions_list   = infos[0]['valid_actions']

        rewards = [-0.005] * env.agent_num
        cita    = [1.0]    * env.agent_num

        for i, action in enumerate(actions if not hasattr(actions, 'tolist') else actions.tolist()):
            if env.agent_cover_count[i] >= env.smog_initial_count * env.agent_task_rate[i]:
                rewards[i] -= 1
                cita[i] = 0.1
            rewards[i] += cita[i] * count_oneclear_total[i]

            valid_actions   = valid_actions_list[i]
            discrete_action = int(np.argmax(action))
            if valid_actions[discrete_action] == 0:
                if env.agent_cover_count[i] < env.agent_task_rate[i] * 200:
                    rewards[i] -= 15
            else:
                next_state  = env.current_state[i]
                shorted_dis = min(env.get_distance(next_state, env.current_state[j])
                                  for j in range(env.agent_num))
                avoid_rate  = 10
                if i == 0 or i == 2:
                    if shorted_dis <= env.safe_dis:
                        if shorted_dis == 5:
                            rewards[i] -= 0.05 * avoid_rate
                        elif shorted_dis == 4:
                            rewards[i] -= 0.45 * avoid_rate
                        elif shorted_dis == 3:
                            rewards[i] -= 0.65 * avoid_rate
                        elif shorted_dis == 2:
                            rewards[i] -= 0.85 * avoid_rate
                        elif shorted_dis <= 1:
                            rewards[i] -= 1.0  * avoid_rate
                        else:
                            rewards[i] += 0.15

                if i == 1:
                    d0 = env.get_distance(next_state, env.current_state[0])
                    d2 = env.get_distance(next_state, env.current_state[2])
                    target_dis = d0 if env.postman_target == 0 else d2
                    rewards[i] += 3.0 / (target_dis + 1)
                    if target_dis <= 2:
                        rewards[i] += 5.0
                    if d0 > 8 and d2 > 8:
                        rewards[i] -= 2.0

        total_r = ((sum(count_oneclear_total) - count_oneclear_total[1]) * 0.65
                   - env.smog_realtime_count * 0.1
                   + (env.smog_initial_count - env.smog_realtime_count) * 0.3)

        escape_rate = infos[0]['escape_rate']
        if 0.25 < escape_rate < 4:
            total_r += 10 * math.atan(1 / (escape_rate - 1))
            if 0.5 < escape_rate < 2:
                total_r += 10 * math.atan(1 / (escape_rate - 1))
            if escape_rate == 1:
                total_r += 20
        else:
            if escape_rate > 1:
                total_r -= 5 * math.atan(escape_rate - 1)
            elif 0 < escape_rate < 1:
                total_r -= 5 * math.atan(1 / escape_rate)
            elif escape_rate == 0:
                total_r -= 6

        if count_oneclear_total.count(0) == len(count_oneclear_total) and env.smog_realtime_count > 20:
            total_r -= 5

        return [r + total_r for r in rewards]

    # ------------------------------------------------------------------ #
    # 扩展接口：后续阶段注入
    # ------------------------------------------------------------------ #
    def _compute_marginal_contribution(self, _agent_id, _state, _action):
        """预留：局部边际贡献奖励项。"""
        return 0.0

    def _compute_kl_regularization(self, _agent_id):
        """预留：KL 散度多样性正则化惩罚项。"""
        return 0.0

    def _compute_marginal_contribution_hook(self, rewards, _infos):
        return rewards

    def _compute_kl_regularization_hook(self, rewards, _infos):
        return rewards


# 向后兼容别名，供旧导入 `from Env.reward_wrapper import RewardWrapper` 使用
RewardWrapper = MARLRewardWrapper

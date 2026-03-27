import math
import gymnasium as gym


class RewardWrapper(gym.Wrapper):
    """
    将奖励计算从 Gridworld 中解耦。
    外部挂载方式：env = RewardWrapper(Gridworld(...))
    扩展接口：
      - _marginal_contribution_hook(rewards, infos)  -> rewards
      - _kl_diversity_hook(rewards, infos)           -> rewards
    """

    def step(self, t, actions):
        next_obs, _, dones, infos = self.env.step(t, actions)
        rewards = self._compute_rewards(actions, infos)
        rewards = self._marginal_contribution_hook(rewards, infos)
        rewards = self._kl_diversity_hook(rewards, infos)
        return next_obs, rewards, dones, infos

    def _compute_rewards(self, actions, infos):
        env = self.env
        # 从 infos 取本步所需数据
        count_oneclear_total = infos[0]['step_cover_delta']
        valid_actions_list   = infos[0]['valid_actions']
        prev_states          = infos[0]['prev_states']

        rewards = [-0.005] * env.agent_num
        cita    = [1.0]    * env.agent_num

        for i, action in enumerate(actions if not hasattr(actions, 'tolist') else actions.tolist()):
            # 任务量上限惩罚
            if env.agent_cover_count[i] >= env.smog_initial_count * env.agent_task_rate[i]:
                rewards[i] -= 1
                cita[i] = 0.1
            # 迷雾清除奖励
            rewards[i] += cita[i] * count_oneclear_total[i]

            valid_actions = valid_actions_list[i]
            # 无效动作惩罚
            if valid_actions[action] == 0:
                if env.agent_cover_count[i] < env.agent_task_rate[i] * 200:
                    rewards[i] -= 15
            else:
                next_state  = env.current_state[i]   # 物理推进后的位置
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

        # 集体奖励
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

        rewards = [r + total_r for r in rewards]
        return rewards

    # ------------------------------------------------------------------ #
    # 扩展接口：后续注入局部边际贡献与 KL 散度多样性惩罚
    # ------------------------------------------------------------------ #
    def _marginal_contribution_hook(self, rewards, infos):
        """预留：局部边际贡献奖励注入点"""
        return rewards

    def _kl_diversity_hook(self, rewards, infos):
        """预留：KL 散度多样性惩罚注入点"""
        return rewards

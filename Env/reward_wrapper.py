import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import gym


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
        rewards = self._compute_rewards(infos)
        rewards = self._compute_marginal_contribution_hook(rewards, infos)
        rewards = self._compute_kl_regularization_hook(rewards, infos)
        return obs, actions, rewards, next_obs, dones, infos

    def _compute_rewards(self, infos):
        env = self.env
        count_oneclear_total = infos[0]['step_cover_delta']

        # --- Layer 1: Global shared rewards ---
        global_r = -0.1  # 连坐时间惩罚
        global_r += sum(count_oneclear_total) * 0.5  # 全局增量奖励
        if env.smog_realtime_count <= 0:
            global_r += 50.0  # 终局巨额奖励

        # --- Layer 2 & 3: Per-agent rewards ---
        rewards = [global_r] * env.agent_num
        quota = [env.smog_initial_count * env.agent_task_rate[i] for i in range(env.agent_num)]

        for i in range(env.agent_num):
            delta = count_oneclear_total[i]
            # 收益递减配额：配额内高奖励，超出低正奖励
            in_quota  = max(0, min(delta, quota[i] - env.agent_cover_count[i] + delta))
            over_quota = max(0, delta - in_quota)
            rewards[i] += 1.0 * in_quota + 0.2 * over_quota

            next_state = env.current_state[i]

            # 碰撞约束（仅 0、2 号智能体）
            if i == 0 or i == 2:
                shorted_dis = min(env.get_distance(next_state, env.current_state[j])
                                  for j in range(env.agent_num) if j != i)
                if shorted_dis <= env.safe_dis:
                    penalty = min(1.0, (env.safe_dis - shorted_dis + 1) * 0.2)
                    rewards[i] -= penalty

            # Postman 角色约束（1 号智能体）
            if i == 1:
                d0 = env.get_distance(next_state, env.current_state[0])
                d2 = env.get_distance(next_state, env.current_state[2])
                target_dis = d0 if env.postman_target == 0 else d2
                rewards[i] += 0.5 / (target_dis + 1)

        return rewards

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

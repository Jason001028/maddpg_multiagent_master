"""
RA-MADDPG: Role-Aware Multi-Agent Deep Deterministic Policy Gradient

改进点（相对于 DiscreteMADDPG）：
    - Critic 替换为 RoleAwareCritic，引入 FiLM 角色调制机制
    - 静态角色特征向量 role_feats 从 Args.role_configs 构建，存为网络 buffer
    - Critic 的每次前向传播均注入 role_feats，Actor 结构不变

使用方式：
    在 arguments.py 中设置 algo_name = 'ra_maddpg'
    确保 Args.role_configs 已正确配置
"""

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy

from core.base_algo import BaseMARLAlgorithm
from core.model import RAMADDPGNet


def _role_configs_to_array(role_configs: list) -> np.ndarray:
    """将 role_configs 列表转换为 (n_agents, role_dim) 的 numpy 数组。

    Args:
        role_configs: [{'task_rate': 0.34, 'viewrange': 2}, ...]
                      来自 Args.role_configs，顺序对应 agent 0, 1, 2

    Returns:
        role_feats_np: (n_agents, role_dim) float32 数组
                       键按字典序排列以保证顺序一致性
    """
    # 使用排序键名确保特征顺序在每次运行中一致
    keys = sorted(role_configs[0].keys())   # e.g. ['task_rate', 'viewrange']
    arr = np.array(
        [[cfg[k] for k in keys] for cfg in role_configs],
        dtype=np.float32
    )  # (n_agents, role_dim)
    # 对每个特征维度做 min-max 归一化，消除量纲差异（task_rate~0.66 vs viewrange~3）
    col_min = arr.min(axis=0)
    col_max = arr.max(axis=0)
    denom = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
    return (arr - col_min) / denom


class RAMADDPG(BaseMARLAlgorithm):
    """Role-Aware MADDPG 算法。

    与 DiscreteMADDPG 的差异：
        1. 网络使用 RAMADDPGNet（含 RoleAwareCritic）
        2. Critic 的 forward 传入 self.model.role_feats（buffer，无需手动迁移设备）
        3. Actor 结构与更新逻辑保持不变（Gumbel-Softmax 离散动作）
    """

    def __init__(self, args, env_params, device='cpu'):
        self.env_params   = env_params
        self.train_params = args.train_params
        self.device       = device
        self.n_agents     = env_params.n_agents
        self.gamma        = self.train_params.gamma
        self.polyak       = self.train_params.polyak
        self.update_tar_interval = self.train_params.update_tar_interval

        # ── 构建静态角色特征矩阵 ──────────────────────────────────────────────
        # role_feats_np: (n_agents, role_dim)，来自 Args.role_configs
        # e.g. [[0.34, 2], [0.0, 0], [0.66, 3]]  → task_rate 和 viewrange 按字典序排列
        role_feats_np = _role_configs_to_array(args.role_configs)

        # ── 初始化网络（role_feats 注册为 buffer，自动随 .to(device) 迁移）────
        self.model = RAMADDPGNet(
            env_params,
            role_feats_np,
            role_embed_dim=64,   # FiLM 中角色嵌入维度，可通过 train_params 扩展
            device=device,
        )

        # ── 优化器 ────────────────────────────────────────────────────────────
        self.actor_optimizers = [
            Adam(self.model.actors[i].parameters(), lr=self.train_params.lr_actor)
            for i in range(self.n_agents)
        ]
        # 所有 Critic 共用一个优化器（与 DiscreteMADDPG 一致）
        self.critic_optimizer = Adam(
            self.model.critics.parameters(), lr=self.train_params.lr_critic
        )

    @torch.no_grad()
    def act(self, obs, explore=True, current_eps=0.05, available_actions=None):
        """Actor 推断（与 DiscreteMADDPG 完全相同，结构不变）。"""
        obs_t = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)
        if explore and np.random.rand() < current_eps:
            if available_actions is not None:
                actions = np.array([
                    np.random.choice(np.where(available_actions[i] == 1)[0])
                    for i in range(self.n_agents)
                ])
            else:
                actions = np.random.randint(
                    0, self.env_params.dim_action, size=self.n_agents
                )
            one_hot = np.zeros(
                (self.n_agents, self.env_params.dim_action), dtype=np.float32
            )
            one_hot[np.arange(self.n_agents), actions] = 1.0
            return one_hot
        else:
            logits = torch.stack(
                [self.model.actors[i](obs_t[i]) for i in range(self.n_agents)]
            ).cpu().numpy()
            if available_actions is not None:
                logits[available_actions == 0] = -np.inf
            actions = np.argmax(logits, axis=-1)
            one_hot = np.zeros_like(logits)
            one_hot[np.arange(self.n_agents), actions] = 1.0
            return one_hot

    def update(self, transitions, logger, step=0, **kwargs):
        """核心更新步骤。

        关键改动：每次调用 critic.forward 时传入 self.model.role_feats。
        role_feats 是 (n_agents, role_dim) 的 buffer，expand 为 (B, n_agents*role_dim)
        在 RoleAwareCritic.forward 内部完成，此处无需预处理。
        """
        def to_t(key):
            arr = [
                v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for v in transitions[key]
            ]
            return torch.tensor(np.array(arr), dtype=torch.float32).to(self.device)

        r_key   = 'reward' if 'reward' in transitions else 'r'
        obs      = to_t('obs').reshape(-1, self.n_agents, self.env_params.dim_observation)
        obs_next = to_t('next_obs').reshape(-1, self.n_agents, self.env_params.dim_observation)
        acts_i   = to_t('acts').reshape(-1, self.n_agents, self.env_params.dim_action)
        reward   = to_t(r_key).reshape(-1, self.n_agents)
        dones    = (
            to_t('dones') if 'dones' in transitions
            else torch.zeros_like(reward)
        ).reshape(-1, self.n_agents)

        B = obs.shape[0]
        # 全局拼接向量，供 Critic 使用
        global_obs      = obs.reshape(B, -1)        # (B, dim_obs * n_agents)
        global_obs_next = obs_next.reshape(B, -1)   # (B, dim_obs * n_agents)
        global_acts     = acts_i.reshape(B, -1)     # (B, dim_act * n_agents)

        # role_feats: (n_agents, role_dim)，从 buffer 取，已在正确 device 上
        # RoleAwareCritic.forward 内部通过 expand 完成 batch 维度广播（零拷贝）
        role_feats = self.model.role_feats  # (n_agents, role_dim)

        # ── Critic 更新 ────────────────────────────────────────────────────────
        with torch.no_grad():
            # Target Actor 推断下一步动作（argmax → one-hot，离散动作）
            next_logits = torch.stack(
                [self.model.actors_target[i](obs_next[:, i, :])
                 for i in range(self.n_agents)],
                dim=1,
            )  # (B, n_agents, dim_act)
            next_actions_idx = next_logits.argmax(dim=-1)   # (B, n_agents)
            next_one_hot = torch.zeros_like(next_logits)
            next_one_hot.scatter_(-1, next_actions_idx.unsqueeze(-1), 1.0)
            global_acts_next = next_one_hot.reshape(B, -1)  # (B, dim_act * n_agents)

        critic_loss_total = 0.0
        self.critic_optimizer.zero_grad()

        for i in range(self.n_agents):
            with torch.no_grad():
                # Target Critic 计算 Q'(s', a')，注入 role_feats
                q_next = self.model.critics_target[i](
                    global_obs_next, global_acts_next, role_feats
                )  # (B, 1)
                r_i      = reward[:, i:i+1]              # (B, 1)
                d_i      = dones[:, i:i+1]               # (B, 1)
                target_q = r_i + self.gamma * q_next * (1 - d_i)  # (B, 1)

            # 当前 Critic 计算 Q(s, a)，注入 role_feats
            real_q = self.model.critics[i](global_obs, global_acts, role_feats)  # (B, 1)
            loss_i = F.mse_loss(real_q, target_q)
            critic_loss_total = critic_loss_total + loss_i

        (critic_loss_total / self.n_agents).backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.critics.parameters(), max_norm=1.0
        ).item()
        self.critic_optimizer.step()

        # ── Actor 更新（Gumbel-Softmax，冻结 Critic）──────────────────────────
        for p in self.model.critics.parameters():
            p.requires_grad_(False)

        actor_loss_total = 0.0
        actor_grad_norms = []
        entropies = []
        for i in range(self.n_agents):
            logits_i       = self.model.actors[i](obs[:, i, :])
            # 策略熵：从原始 logits 的 softmax 分布计算，不参与梯度图
            with torch.no_grad():
                probs_i    = F.softmax(logits_i.detach(), dim=-1)
                entropy_i  = -(probs_i * torch.log(probs_i + 1e-8)).sum(dim=-1).mean().item()
            entropies.append(entropy_i)

            # Gumbel-Softmax：可微分的 one-hot 近似，使 argmax 可反向传播
            soft_one_hot_i = F.gumbel_softmax(logits_i, tau=1.0, hard=True)

            # 构建 global_acts：agent i 替换为 soft_one_hot，其余 agent 动作固定
            acts_list = []
            for j in range(self.n_agents):
                if j == i:
                    acts_list.append(soft_one_hot_i)
                else:
                    acts_list.append(acts_i[:, j, :].detach())
            all_acts = torch.cat(acts_list, dim=-1)  # (B, dim_act * n_agents)

            # Critic 评估当前 Actor 策略质量，注入 role_feats
            q_i    = self.model.critics[i](global_obs, all_acts, role_feats)  # (B, 1)
            loss_i = -q_i.mean()

            self.actor_optimizers[i].zero_grad()
            loss_i.backward()
            actor_grad_norm_i = torch.nn.utils.clip_grad_norm_(
                self.model.actors[i].parameters(), max_norm=1.0
            ).item()
            actor_grad_norms.append(actor_grad_norm_i)
            self.actor_optimizers[i].step()
            actor_loss_total += loss_i.detach()

        for p in self.model.critics.parameters():
            p.requires_grad_(True)

        if step % self.update_tar_interval == 0:
            self._soft_update_targets()

        extra = {
            'grad_norm_critic': critic_grad_norm,
            'grad_norm_actor':  sum(actor_grad_norms) / len(actor_grad_norms),
            'entropy':          entropies,  # [explorer, postman, surveyor]
        }
        return (
            (actor_loss_total / self.n_agents).item(),
            (critic_loss_total / self.n_agents).item(),
            extra,
        )

    def _soft_update_targets(self):
        def _soft(target_list, source_list):
            for target, source in zip(target_list, source_list):
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.copy_(
                        (1 - self.polyak) * sp.data + self.polyak * tp.data
                    )
        _soft(self.model.actors_target,  self.model.actors)
        _soft(self.model.critics_target, self.model.critics)

    def save(self, path):
        torch.save({
            'actors':     [a.state_dict() for a in self.model.actors],
            'critics':    [c.state_dict() for c in self.model.critics],
            'role_feats': self.model.role_feats.cpu().numpy(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for i, sd in enumerate(ckpt['actors']):
            self.model.actors[i].load_state_dict(sd)
        for i, sd in enumerate(ckpt['critics']):
            self.model.critics[i].load_state_dict(sd)

    def get_actor_state_dict(self):
        return {
            'actor_dict': [deepcopy(a).cpu().state_dict() for a in self.model.actors]
        }

    def sync_actor(self, state_dict):
        for i, sd in enumerate(state_dict['actor_dict']):
            self.model.actors[i].load_state_dict(sd)

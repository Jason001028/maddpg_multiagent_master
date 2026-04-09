"""
EF-MADDPG: Early Fusion MADDPG（消融实验基线）

改进点（相对于 DiscreteMADDPG）：
    - Actor 输入层：concat(obs_i, role_i)，输入维度扩展为 dim_obs + role_dim
    - Critic 输入层：concat(global_obs, all_acts, all_roles)，输入维度扩展为
      (dim_obs + dim_act + role_dim) * n_agents
    - 无任何额外特征提取/注意力/归一化，MLP 主干与 Vanilla MADDPG 完全一致
    - 目的：验证"低维静态角色特征直接拼接高维动态观测会导致梯度稀释"的消融假设

使用方式：
    在 arguments.py 中设置 algo_name = 'ef_maddpg'
    确保 Args.role_configs 已正确配置
"""

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from copy import deepcopy

from core.base_algo import BaseMARLAlgorithm
from core.model import EFMADDPGNet
from core.ra_maddpg import _role_configs_to_array


class EFMADDPG(BaseMARLAlgorithm):
    """Early Fusion MADDPG 算法（消融实验）。

    与 RAMADDPG 的差异：
        1. 网络使用 EFMADDPGNet（朴素 concat，无 FiLM）
        2. Actor forward 需额外传入 role_i（单个 agent 的角色特征行）
        3. Critic forward 需额外传入 all_roles（全体 agent 角色特征展平）
    """

    def __init__(self, args, env_params, device='cpu'):
        self.env_params   = env_params
        self.train_params = args.train_params
        self.device       = device
        self.n_agents     = env_params.n_agents
        self.gamma        = self.train_params.gamma
        self.polyak       = self.train_params.polyak
        self.update_tar_interval = self.train_params.update_tar_interval

        role_feats_np = _role_configs_to_array(args.role_configs)

        self.model = EFMADDPGNet(env_params, role_feats_np, device=device)

        self.actor_optimizers = [
            Adam(self.model.actors[i].parameters(), lr=self.train_params.lr_actor)
            for i in range(self.n_agents)
        ]
        self.critic_optimizer = Adam(
            self.model.critics.parameters(), lr=self.train_params.lr_critic
        )

        decay = self.train_params.decay_steps
        self.actor_schedulers = [
            LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=decay)
            for opt in self.actor_optimizers
        ]
        self.critic_scheduler = LinearLR(
            self.critic_optimizer, start_factor=1.0, end_factor=0.1, total_iters=decay
        )

    @torch.no_grad()
    def act(self, obs, explore=True, current_eps=0.05, available_actions=None):
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
            # role_feats: (n_agents, role_dim)，取每行对应 agent i 的角色特征
            role_feats = self.model.role_feats  # (n_agents, role_dim)
            logits = torch.stack([
                self.model.actors[i](obs_t[i].unsqueeze(0), role_feats[i].unsqueeze(0)).squeeze(0)
                for i in range(self.n_agents)
            ]).cpu().numpy()  # (n_agents, dim_act)
            if available_actions is not None:
                logits[available_actions == 0] = -np.inf
            actions = np.argmax(logits, axis=-1)
            one_hot = np.zeros_like(logits)
            one_hot[np.arange(self.n_agents), actions] = 1.0
            return one_hot

    def update(self, transitions, logger, step=0, **kwargs):
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
        global_obs      = obs.reshape(B, -1)       # (B, dim_obs * n_agents)
        global_obs_next = obs_next.reshape(B, -1)
        global_acts     = acts_i.reshape(B, -1)    # (B, dim_act * n_agents)

        role_feats = self.model.role_feats          # (n_agents, role_dim)
        # all_roles: (B, role_dim * n_agents)，全 batch 共享，零拷贝广播
        all_roles = role_feats.view(1, -1).expand(B, -1)  # (B, n_agents * role_dim)

        # ── Critic 更新 ────────────────────────────────────────────────────────
        with torch.no_grad():
            # Target Actor 推断：每个 agent 需传入对应的 role_i
            next_logits_list = []
            for i in range(self.n_agents):
                role_i = role_feats[i].unsqueeze(0).expand(B, -1)  # (B, role_dim)
                next_logits_list.append(
                    self.model.actors_target[i](obs_next[:, i, :], role_i)
                )
            next_logits = torch.stack(next_logits_list, dim=1)  # (B, n_agents, dim_act)
            next_actions_idx = next_logits.argmax(dim=-1)
            next_one_hot = torch.zeros_like(next_logits)
            next_one_hot.scatter_(-1, next_actions_idx.unsqueeze(-1), 1.0)
            global_acts_next = next_one_hot.reshape(B, -1)  # (B, dim_act * n_agents)

        critic_loss_total = 0.0
        self.critic_optimizer.zero_grad()

        for i in range(self.n_agents):
            with torch.no_grad():
                q_next   = self.model.critics_target[i](global_obs_next, global_acts_next, all_roles)
                r_i      = reward[:, i:i+1]
                d_i      = dones[:, i:i+1]
                target_q = r_i + self.gamma * q_next * (1 - d_i)

            real_q = self.model.critics[i](global_obs, global_acts, all_roles)
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
            role_i = role_feats[i].unsqueeze(0).expand(B, -1)  # (B, role_dim)
            logits_i = self.model.actors[i](obs[:, i, :], role_i)

            with torch.no_grad():
                probs_i   = F.softmax(logits_i.detach(), dim=-1)
                entropy_i = -(probs_i * torch.log(probs_i + 1e-8)).sum(dim=-1).mean().item()
            entropies.append(entropy_i)

            soft_one_hot_i = F.gumbel_softmax(logits_i, tau=1.0, hard=True)

            acts_list = []
            for j in range(self.n_agents):
                acts_list.append(soft_one_hot_i if j == i else acts_i[:, j, :].detach())
            all_acts = torch.cat(acts_list, dim=-1)  # (B, dim_act * n_agents)

            q_i    = self.model.critics[i](global_obs, all_acts, all_roles)
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

        for sched in self.actor_schedulers:
            sched.step()
        self.critic_scheduler.step()

        extra = {
            'grad_norm_critic': critic_grad_norm,
            'grad_norm_actor':  sum(actor_grad_norms) / len(actor_grad_norms),
            'entropy':          entropies,
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

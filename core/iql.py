import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy

from core.base_algo import BaseMARLAlgorithm
from core.model import IQLNet


class IQL(BaseMARLAlgorithm):
    """Independent Q-Learning / Independent Actor-Critic.
    Each agent trains entirely on its own (obs_i, act_i, r_i) — no cooperation.
    """

    def __init__(self, args, env_params, device='cpu'):
        self.env_params   = env_params
        self.train_params = args.train_params
        self.device       = device
        self.n_agents     = env_params.n_agents
        self.gamma        = self.train_params.gamma
        self.polyak       = self.train_params.polyak
        self.update_tar_interval = self.train_params.update_tar_interval

        self.model = IQLNet(env_params, device)

        self.actor_optimizers = [
            Adam(self.model.actors[i].parameters(), lr=self.train_params.lr_actor)
            for i in range(self.n_agents)
        ]
        self.critic_optimizers = [
            Adam(self.model.critics[i].parameters(), lr=self.train_params.lr_critic)
            for i in range(self.n_agents)
        ]

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
                actions = np.random.randint(0, self.env_params.dim_action, size=self.n_agents)
        else:
            logits = torch.stack(
                [self.model.actors[i](obs_t[i]) for i in range(self.n_agents)]
            ).cpu().numpy()
            if available_actions is not None:
                logits[available_actions == 0] = -np.inf
            actions = np.argmax(logits, axis=-1)

        one_hot = np.zeros((self.n_agents, self.env_params.dim_action), dtype=np.float32)
        one_hot[np.arange(self.n_agents), actions] = 1.0
        return one_hot

    def update(self, transitions, logger, step=0, **kwargs):
        def to_t(key):
            arr = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                   for v in transitions[key]]
            return torch.tensor(np.array(arr), dtype=torch.float32).to(self.device)

        r_key    = 'reward' if 'reward' in transitions else 'r'
        obs      = to_t('obs').reshape(-1, self.n_agents, self.env_params.dim_observation)
        obs_next = to_t('next_obs').reshape(-1, self.n_agents, self.env_params.dim_observation)
        acts_i   = to_t('acts').reshape(-1, self.n_agents, self.env_params.dim_action)
        reward   = to_t(r_key).reshape(-1, self.n_agents)
        dones    = (to_t('dones') if 'dones' in transitions
                    else torch.zeros_like(reward)).reshape(-1, self.n_agents)

        # ---- Critic 更新（各智能体独立）------------------------------------
        total_critic_loss = 0.0
        with torch.no_grad():
            def _argmax_onehot(logits):
                idx = logits.argmax(dim=-1, keepdim=True)
                return torch.zeros_like(logits).scatter_(-1, idx, 1.0)

        for i in range(self.n_agents):
            with torch.no_grad():
                next_act_i = _argmax_onehot(self.model.actors_target[i](obs_next[:, i, :]))
                q_next = self.model.critics_target[i](obs_next[:, i, :], next_act_i)  # (B,1)
                # 使用共享奖励（全局均值）作为 TD target，也可改为 reward[:,i:i+1]
                r_i    = reward[:, i:i+1]
                d_i    = dones[:, i:i+1]
                target = r_i + self.gamma * q_next * (1 - d_i)

            q_cur = self.model.critics[i](obs[:, i, :], acts_i[:, i, :])
            critic_loss = (target - q_cur).pow(2).mean()

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.critics[i].parameters(), max_norm=1.0)
            self.critic_optimizers[i].step()
            total_critic_loss += critic_loss.item()

        # ---- Actor 更新（各智能体独立，Gumbel-Softmax 保持可导）-----------
        total_actor_loss = 0.0
        for i in range(self.n_agents):
            for p in self.model.critics[i].parameters():
                p.requires_grad_(False)

            act_pred = F.gumbel_softmax(self.model.actors[i](obs[:, i, :]), tau=1.0, hard=True)
            actor_loss = -self.model.critics[i](obs[:, i, :], act_pred).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.actors[i].parameters(), max_norm=1.0)
            self.actor_optimizers[i].step()

            for p in self.model.critics[i].parameters():
                p.requires_grad_(True)

            total_actor_loss += actor_loss.item()

        if step % self.update_tar_interval == 0:
            self._soft_update_targets()

        return total_actor_loss / self.n_agents, total_critic_loss / self.n_agents

    def _soft_update_targets(self):
        def _soft(target_list, source_list):
            for target, source in zip(target_list, source_list):
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.copy_((1 - self.polyak) * sp.data + self.polyak * tp.data)
        _soft(self.model.actors_target,  self.model.actors)
        _soft(self.model.critics_target, self.model.critics)

    def save(self, path):
        torch.save({
            'actors':  [a.state_dict() for a in self.model.actors],
            'critics': [c.state_dict() for c in self.model.critics],
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for i, sd in enumerate(ckpt['actors']):
            self.model.actors[i].load_state_dict(sd)
        for i, sd in enumerate(ckpt['critics']):
            self.model.critics[i].load_state_dict(sd)

    def get_actor_state_dict(self):
        return {'actor_dict': [deepcopy(a).cpu().state_dict() for a in self.model.actors]}

    def sync_actor(self, state_dict):
        for i, sd in enumerate(state_dict['actor_dict']):
            self.model.actors[i].load_state_dict(sd)

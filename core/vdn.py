import numpy as np
import torch
from torch.optim import Adam
from copy import deepcopy

from core.base_algo import BaseMARLAlgorithm
from core.model import VDNNet


class ContinuousVDN(BaseMARLAlgorithm):

    def __init__(self, args, env_params, device='cpu'):
        self.env_params   = env_params
        self.train_params = args.train_params
        self.device       = device
        self.n_agents     = env_params.n_agents
        self.gamma        = self.train_params.gamma
        self.polyak       = self.train_params.polyak
        self.update_tar_interval = self.train_params.update_tar_interval
        self.noise_eps    = self.train_params.noise_eps
        self.action_max   = env_params.action_max

        self.model = VDNNet(env_params, device)

        self.actor_optimizers = [
            Adam(self.model.actors[i].parameters(), lr=self.train_params.lr_actor)
            for i in range(self.n_agents)
        ]
        critic_params = list(self.model.critics.parameters())
        self.critic_optimizer = Adam(critic_params, lr=self.train_params.lr_critic)

    @torch.no_grad()
    def act(self, obs, explore=True):
        obs_t = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)
        actions = torch.stack(
            [self.model.actors[i](obs_t[i]) for i in range(self.n_agents)]
        ).cpu().numpy()  # (n_agents, dim_act)
        if explore:
            actions += self.noise_eps * self.action_max * np.random.randn(*actions.shape)
            actions = np.clip(actions, 0, 1)
        return actions

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

        r_global    = reward.sum(dim=1, keepdim=True)           # (batch, 1)
        done_global = dones.max(dim=1, keepdim=True).values     # (batch, 1)

        # ---- Critic 更新 ------------------------------------------------
        with torch.no_grad():
            tq = []
            for i in range(self.n_agents):
                a_next = self.model.actors_target[i](obs_next[:, i, :])
                tq.append(self.model.critics_target[i](obs_next[:, i, :], a_next))
            target_q_tot = torch.stack(tq, dim=1).sum(dim=1)   # (batch, 1)
            target_q     = r_global + self.gamma * target_q_tot * (1 - done_global)

        q_locals = torch.stack(
            [self.model.critics[i](obs[:, i, :], acts_i[:, i, :]) for i in range(self.n_agents)],
            dim=1
        )  # (batch, n_agents, 1)
        q_tot = self.model.mixer(q_locals)  # (batch, 1)

        critic_loss = (target_q - q_tot).pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.critics.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ---- Actor 更新（冻结 Critic 参数）-------------------------------
        # VDN 线性求和：∇_θi Q_tot ≡ ∇_θi Q_i，直接用 -Q_i 作 loss
        for p in self.model.critics.parameters():
            p.requires_grad_(False)

        actor_loss_total = 0.0
        for i in range(self.n_agents):
            loss_i = -self.model.critics[i](
                obs[:, i, :], self.model.actors[i](obs[:, i, :])
            ).mean()
            self.actor_optimizers[i].zero_grad()
            loss_i.backward()
            torch.nn.utils.clip_grad_norm_(self.model.actors[i].parameters(), max_norm=1.0)
            self.actor_optimizers[i].step()
            actor_loss_total += loss_i.detach()

        for p in self.model.critics.parameters():
            p.requires_grad_(True)

        if step % self.update_tar_interval == 0:
            self._soft_update_targets()

        return (actor_loss_total / self.n_agents).item(), critic_loss.item()

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

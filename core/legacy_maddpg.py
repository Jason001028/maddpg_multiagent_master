import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())

import random
import numpy as np
import torch
from torch.optim import Adam
from copy import deepcopy

from core.base_algo import BaseMARLAlgorithm
from core.model import Net


class LegacyMADDPG(BaseMARLAlgorithm):

    def __init__(self, args, env_params, device='cpu'):
        self.env_params = env_params
        self.train_params = args.train_params
        self.device = device

        self.n_agents = env_params.n_agents
        self.noise_eps = self.train_params.noise_eps
        self.action_max = env_params.action_max
        self.gamma = self.train_params.gamma
        self.polyak = self.train_params.polyak
        self.update_tar_interval = self.train_params.update_tar_interval

        self.model = Net(env_params, device)
        self.actor_optimizer = Adam(self.model.actor.parameters(), lr=self.train_params.lr_actor)
        self.critic_optimizer = Adam(self.model.critic.parameters(), lr=self.train_params.lr_critic)

    @torch.no_grad()
    def act(self, obs, explore=True):
        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)
        action = self.model.actor(obs_tensor).cpu().numpy()
        if action.ndim == 1:
            action = action[np.newaxis, :]
        if explore:
            if random.random() < 0.1:
                action = np.random.uniform(-1, 1, action.shape)
            action += self.noise_eps * self.action_max * np.random.randn(*action.shape)
            action = np.clip(action, -self.action_max, self.action_max)
        return np.clip(action, 0, 1)

    @staticmethod
    def _to_numpy(x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def update(self, transitions, logger, step=0):
        def to_tensor(key):
            arr = [self._to_numpy(v) for v in transitions[key]]
            return torch.tensor(np.array(arr), dtype=torch.float32).to(self.device)

        r_key = 'reward' if 'reward' in transitions else 'r'

        obs      = to_tensor('obs')
        obs_next = to_tensor('next_obs')
        acts     = to_tensor('acts')
        r        = to_tensor(r_key)
        batch = obs.shape[0]
        n = self.n_agents

        with torch.no_grad():
            obs      = obs.reshape(batch, n, -1)
            obs_next = obs_next.reshape(batch, n, -1)
            acts_i   = acts.reshape(batch, n, -1)
            r_i      = r.reshape(batch, n, -1)

            acts_next = self.model.actors_target(obs_next).reshape(batch, -1).unsqueeze(1).repeat(1, n, 1)
            q_next    = self.model.critics_target(
                obs_next.reshape(batch, -1).unsqueeze(1).repeat(1, n, 1), acts_next)
            target_q  = r_i + self.gamma * q_next

        real_q = self.model.critic(
            obs.reshape(batch, -1).unsqueeze(1).repeat(1, n, 1),
            acts_i.reshape(batch, -1).unsqueeze(1).repeat(1, n, 1))
        critic_loss = (target_q - real_q).pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        acts_real = self.model.actor(obs)
        actor_loss = -self.model.critic(
            obs.reshape(batch, -1).unsqueeze(1).repeat(1, n, 1),
            acts_real.reshape(batch, -1).unsqueeze(1).repeat(1, n, 1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if step % self.update_tar_interval == 0:
            self._soft_update_targets()

        return actor_loss.item(), critic_loss.item()

    def _soft_update_targets(self):
        def _soft(target, source):
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_((1 - self.polyak) * sp.data + self.polyak * tp.data)
        _soft(self.model.actors_target, self.model.actor)
        _soft(self.model.critics_target, self.model.critic)

    def save(self, path):
        torch.save([self.model.actor.state_dict(), self.model.critic], path)

    def load(self, path):
        act_sd, cr = torch.load(path, map_location=self.device)
        self.model.actor.load_state_dict(act_sd)
        self.model.critic = cr

    def get_actor_state_dict(self):
        return {'actor_dict': deepcopy(self.model.actor).cpu().state_dict()}

    def sync_actor(self, state_dict):
        self.model.actor.load_state_dict(state_dict['actor_dict'])

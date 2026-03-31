import numpy as np
import torch
from torch.optim import Adam
from copy import deepcopy

from core.base_algo import BaseMARLAlgorithm
from core.model import QMIXNet


class ContinuousQMIX(BaseMARLAlgorithm):

    def __init__(self, args, env_params, device='cpu'):
        self.env_params   = env_params
        self.train_params = args.train_params
        self.device       = device
        self.n_agents     = env_params.n_agents
        self.gamma        = self.train_params.gamma
        self.polyak       = self.train_params.polyak
        self.update_tar_interval = self.train_params.update_tar_interval
        self.action_max   = env_params.action_max

        embed_dim = getattr(args.train_params, 'mixer_embed_dim', 32)
        self.model = QMIXNet(env_params, embed_dim, device)

        self.actor_optimizers = [
            Adam(self.model.actors[i].parameters(), lr=self.train_params.lr_actor)
            for i in range(self.n_agents)
        ]
        critic_params = list(self.model.critics.parameters()) + list(self.model.mixer.parameters())
        self.critic_optimizer = Adam(critic_params, lr=self.train_params.lr_critic)

    @torch.no_grad()
    def act(self, obs, explore=True, current_eps=0.05, available_actions=None):
        obs_t = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)
        if explore and np.random.rand() < current_eps:
            # random exploration, respecting action mask if provided
            if available_actions is not None:
                actions = np.array([
                    np.random.choice(np.where(available_actions[i] == 1)[0])
                    for i in range(self.n_agents)
                ])
            else:
                actions = np.random.randint(0, self.env_params.dim_action, size=self.n_agents)
            # one-hot encode
            one_hot = np.zeros((self.n_agents, self.env_params.dim_action), dtype=np.float32)
            one_hot[np.arange(self.n_agents), actions] = 1.0
            return one_hot
        else:
            # greedy: actor outputs logits/Q-values, argmax
            logits = torch.stack(
                [self.model.actors[i](obs_t[i]) for i in range(self.n_agents)]
            ).cpu().numpy()  # (n_agents, dim_action)
            if available_actions is not None:
                logits[available_actions == 0] = -np.inf
            actions = np.argmax(logits, axis=-1)
            one_hot = np.zeros_like(logits)
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

        B = obs.size(0)
        r_global    = reward.sum(dim=1, keepdim=True)           # (B,1)
        done_global = dones.max(dim=1, keepdim=True).values     # (B,1)

        global_state      = obs.view(B, -1)       # (B, N*obs_dim)
        global_state_next = obs_next.view(B, -1)  # (B, N*obs_dim)

        # ---- Critic + Mixer 更新 -----------------------------------------
        with torch.no_grad():
            tq = [self.model.critics_target[i](obs_next[:, i, :],
                      self.model.actors_target[i](obs_next[:, i, :]))
                  for i in range(self.n_agents)]
            q_locals_next = torch.cat(tq, dim=-1)                       # (B,N)
            q_tot_next    = self.model.mixer_target(q_locals_next, global_state_next)  # (B,1)
            target_q      = r_global + self.gamma * q_tot_next * (1 - done_global)

        q_locals = torch.cat(
            [self.model.critics[i](obs[:, i, :], acts_i[:, i, :]) for i in range(self.n_agents)],
            dim=-1
        )  # (B,N)
        q_tot = self.model.mixer(q_locals, global_state)  # (B,1)

        critic_loss = (target_q - q_tot).pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- Actor 更新（FACMAC：梯度流经 Mixer）--------------------------
        for p in list(self.model.critics.parameters()) + list(self.model.mixer.parameters()):
            p.requires_grad_(False)

        acts_pred   = [self.model.actors[i](obs[:, i, :]) for i in range(self.n_agents)]
        q_locals_a  = torch.cat(
            [self.model.critics[i](obs[:, i, :], acts_pred[i]) for i in range(self.n_agents)],
            dim=-1
        )  # (B,N)
        q_tot_actor = self.model.mixer(q_locals_a, global_state)  # (B,1)

        actor_loss = -q_tot_actor.mean()
        for opt in self.actor_optimizers:
            opt.zero_grad()
        actor_loss.backward()
        for opt in self.actor_optimizers:
            opt.step()

        for p in list(self.model.critics.parameters()) + list(self.model.mixer.parameters()):
            p.requires_grad_(True)

        if step % self.update_tar_interval == 0:
            self._soft_update_targets()

        return actor_loss.item(), critic_loss.item()

    def _soft_update_targets(self):
        def _soft(target_list, source_list):
            for target, source in zip(target_list, source_list):
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.copy_((1 - self.polyak) * sp.data + self.polyak * tp.data)
        _soft(self.model.actors_target,  self.model.actors)
        _soft(self.model.critics_target, self.model.critics)
        for tp, sp in zip(self.model.mixer_target.parameters(), self.model.mixer.parameters()):
            tp.data.copy_((1 - self.polyak) * sp.data + self.polyak * tp.data)

    def save(self, path):
        torch.save({
            'actors':  [a.state_dict() for a in self.model.actors],
            'critics': [c.state_dict() for c in self.model.critics],
            'mixer':   self.model.mixer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for i, sd in enumerate(ckpt['actors']):
            self.model.actors[i].load_state_dict(sd)
        for i, sd in enumerate(ckpt['critics']):
            self.model.critics[i].load_state_dict(sd)
        self.model.mixer.load_state_dict(ckpt['mixer'])

    def get_actor_state_dict(self):
        return {'actor_dict': [deepcopy(a).cpu().state_dict() for a in self.model.actors]}

    def sync_actor(self, state_dict):
        for i, sd in enumerate(state_dict['actor_dict']):
            self.model.actors[i].load_state_dict(sd)

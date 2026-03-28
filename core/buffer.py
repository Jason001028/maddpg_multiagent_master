from abc import ABC, abstractmethod
import torch
import numpy as np

# role_features per agent: {'task_rate': float, 'viewrange': float} → dim 2
_ROLE_FEAT_DIM = 2


class BaseBuffer(ABC):
    @abstractmethod
    def push(self, data): ...

    @abstractmethod
    def sample(self, batch_size): ...

    @abstractmethod
    def clear(self): ...

    @abstractmethod
    def __len__(self): ...


class ReplayBuffer(BaseBuffer):
    def __init__(self, env_params, train_params, logger=None):
        self.T = env_params.max_timesteps
        self.device = train_params.device
        self.size = int(train_params.buffer_size // self.T)
        n = env_params.n_agents
        self.specs = dict(
            obs=dict(size=(self.T, n, env_params.dim_observation), dtype=torch.float32),
            acts=dict(size=(self.T, n * env_params.dim_action), dtype=torch.float32),
            next_obs=dict(size=(self.T, n, env_params.dim_observation), dtype=torch.float32),
            reward=dict(size=(self.T, n, 1), dtype=torch.float32),
            dones=dict(size=(self.T, n, 1), dtype=torch.float32),
            role_features=dict(size=(self.T, n, _ROLE_FEAT_DIM), dtype=torch.float32),
        )
        self.buffers = {key: [] for key in self.specs}
        for _ in range(self.size):
            for key in self.specs:
                self.buffers[key].append(torch.zeros(**self.specs[key]).to(self.device))
        self.current_size = 0
        self.demo_length = 0
        if logger:
            logger.info(f'ReplayBuffer created: T={self.T}, size={self.size}')

    def push(self, episode_batch):
        """
        episode_batch: dict with keys obs, acts, next_obs, reward, dones, role_features
                       each value is np.array of shape (batch_episodes, T, ...)
        """
        batch_size = episode_batch['obs'].shape[0]
        idxs = self._get_storage_idx(inc=batch_size)
        for key in self.specs:
            arr = episode_batch[key]
            for i, idx in enumerate(idxs):
                self.buffers[key][idx] = torch.tensor(
                    arr[i], dtype=self.specs[key]['dtype']
                ).to(self.device)

    def sample(self, batch_size):
        valid = {key: self.buffers[key][:self.current_size] for key in self.buffers}
        ep_idxs = np.random.randint(0, self.current_size, batch_size)
        t_idxs = np.random.randint(self.T, size=batch_size)
        transitions = {
            key: torch.stack([valid[key][ep][t].to('cpu') for ep, t in zip(ep_idxs, t_idxs)])
            for key in valid
        }
        transitions = {k: v.reshape(batch_size, *v.shape[1:]) for k, v in transitions.items()}
        transitions['reward'] = transitions['reward'].detach().clone().to(self.specs['reward']['dtype'])
        return {k: v.to(self.device) for k, v in transitions.items()}

    def clear(self):
        self.current_size = 0
        self.demo_length = 0

    def __len__(self):
        return self.current_size

    def _get_storage_idx(self, inc=1):
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx = np.concatenate([
                np.arange(self.current_size, self.size),
                np.random.randint(self.demo_length, self.current_size, overflow)
            ])
        else:
            idx = np.random.randint(self.demo_length, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        return idx


class RolloutBuffer(BaseBuffer):
    """On-policy rollout buffer for PPO/A2C-style algorithms."""

    def __init__(self, capacity, obs_shape, act_shape, n_agents, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, n_agents, *obs_shape), dtype=np.float32)
        self.acts = np.zeros((capacity, n_agents, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity, n_agents), dtype=np.float32)
        self.log_probs = np.zeros((capacity, n_agents), dtype=np.float32)
        self.returns = None
        self.advantages = None
        self.ptr = 0
        self.full = False

    def push(self, obs, acts, rewards, dones, values, log_probs):
        idx = self.ptr % self.capacity
        self.obs[idx] = obs
        self.acts[idx] = acts
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        self.values[idx] = values
        self.log_probs[idx] = log_probs
        self.ptr += 1
        self.full = self.ptr >= self.capacity

    def sample(self, batch_size=None):
        assert self.returns is not None, "Call compute_returns_and_advantage() first."
        n = self.capacity if self.full else self.ptr
        idx = np.random.randint(0, n, batch_size) if batch_size else np.arange(n)
        return {
            'obs': torch.tensor(self.obs[idx]).to(self.device),
            'acts': torch.tensor(self.acts[idx]).to(self.device),
            'log_probs': torch.tensor(self.log_probs[idx]).to(self.device),
            'returns': torch.tensor(self.returns[idx]).to(self.device),
            'advantages': torch.tensor(self.advantages[idx]).to(self.device),
        }

    def compute_returns_and_advantage(self, last_values, gamma=0.99, gae_lambda=0.95):
        """GAE advantage estimation — placeholder for future implementation."""
        raise NotImplementedError

    def clear(self):
        self.ptr = 0
        self.full = False
        self.returns = None
        self.advantages = None

    def __len__(self):
        return self.capacity if self.full else self.ptr

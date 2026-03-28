from abc import ABC, abstractmethod


class BaseMARLAlgorithm(ABC):

    @abstractmethod
    def __init__(self, args, env_params, device='cpu'):
        pass

    @abstractmethod
    def act(self, obs, explore=True):
        """Forward pass + action sampling. Returns numpy array."""
        pass

    @abstractmethod
    def update(self, transitions, logger):
        """Gradient computation + soft update. Returns (actor_loss, critic_loss)."""
        pass

    @abstractmethod
    def save(self, path):
        """Serialize model weights to disk."""
        pass

    @abstractmethod
    def load(self, path):
        """Deserialize model weights from disk."""
        pass

    @abstractmethod
    def get_actor_state_dict(self):
        """Return {'actor_dict': cpu_state_dict} for IPC."""
        pass

    @abstractmethod
    def sync_actor(self, state_dict):
        """Load actor weights from IPC dict."""
        pass

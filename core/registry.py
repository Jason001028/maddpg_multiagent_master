from core.legacy_maddpg import DiscreteMADDPG as LegacyMADDPG
from core.vdn import ContinuousVDN
from core.qmix import ContinuousQMIX
from core.iql import IQL
from core.ra_maddpg import RAMADDPG
from core.ef_maddpg import EFMADDPG

_REGISTRY = {
    'legacy_maddpg': LegacyMADDPG,
    'vdn': ContinuousVDN,
    'qmix': ContinuousQMIX,
    'iql': IQL,
    'ra_maddpg': RAMADDPG,
    'ef_maddpg': EFMADDPG,
}


def get_algorithm(algo_name, args, env_params, device='cpu'):
    if algo_name not in _REGISTRY:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[algo_name](args, env_params, device=device)

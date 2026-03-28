from core.legacy_maddpg import LegacyMADDPG

_REGISTRY = {
    'legacy_maddpg': LegacyMADDPG,
}


def get_algorithm(algo_name, args, env_params, device='cpu'):
    if algo_name not in _REGISTRY:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[algo_name](args, env_params, device=device)

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

agent_num = 3
class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params.action_max
        self.FC1 = nn.Linear(36 * agent_num + 15, 256)   # 48+6+6+3
        self.FC2 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)
        self.RELU = nn.ReLU()

    def forward(self, obs, acts):  # 前向传播  acts 6   hand 6
        combined = th.cat([obs, acts], dim=-1)  # 将各个agent的观察和动作联合到一起
        result = self.RELU(self.FC1(combined))  # relu为激活函数 if输入大于0，直接返回作为输入值；else 是0或更小，返回值0。
        result = self.RELU(self.FC2(result))
        q_value = self.q_out(result)
        return q_value

class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params.action_max
        self.FC1 = nn.Linear(36, 256)  # 24+3
        self.FC2 = nn.Linear(256, 256)   # FC为full_connected ，即初始化一个全连接网络
        self.action_out = nn.Linear(256, env_params.dim_action)
        self.RELU = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self, obs_and_g):
        result = self.RELU(self.FC1(obs_and_g))
        result = self.RELU(self.FC2(result))
        actions = self.action_out(result)  # raw logits
        return actions

class Net():
    def __init__(self, env_params, device = 'cpu'):
        self.device = device
        self.env_params = env_params
        self.n_agents = env_params.n_agents
        # self.disc = Discriminator(args.env_params)  # if imitation learning
        self.actor = actor(env_params).to(device)
        self.critic = critic(env_params).to(device)
        self.actors_target = deepcopy(self.actor)
        self.critics_target = deepcopy(self.critic)
        # load the weights into the target networks 可以将预训练的参数权重加载到新的模型之中
        self.actors_target.load_state_dict(self.actor.state_dict())
        self.critics_target.load_state_dict(self.critic.state_dict())

    def update(self, model):
        self.actor.load_state_dict(model.actor.state_dict())


# ── VDN 所需网络组件 ──────────────────────────────────────────────────────────

class LocalCritic(nn.Module):
    """每个智能体独立的局部 Critic：输入 (obs_i, act_i)，输出 Q_i"""
    def __init__(self, dim_obs, dim_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_obs + dim_act, 256), nn.ReLU(),
            nn.Linear(256, 256),               nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs_i, act_i):
        return self.net(th.cat([obs_i, act_i], dim=-1))  # (batch, 1)


class VDNMixer(nn.Module):
    """Q_tot = sum(Q_i)，无可学习参数"""
    def forward(self, q_locals):
        return q_locals.sum(dim=1)  # (batch, n_agents, 1) -> (batch, 1)


class ActorNetwork(nn.Module):
    def __init__(self, dim_obs, dim_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_obs, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
            nn.Linear(256, dim_act)
        )

    def forward(self, obs_i):
        return self.net(obs_i)  # raw logits


class DiscreteActorNetwork(nn.Module):
    def __init__(self, dim_obs, dim_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_obs, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
            nn.Linear(256, dim_act)
        )

    def forward(self, obs_i):
        return self.net(obs_i)  # raw logits，由调用方决定如何使用


class VDNNet(nn.Module):
    """n_agents 个独立 ActorNetwork + LocalCritic，支持异构智能体"""
    def __init__(self, env_params, device='cpu'):
        super().__init__()
        self.n_agents = env_params.n_agents
        dim_obs = env_params.dim_observation
        dim_act = env_params.dim_action

        self.actors  = nn.ModuleList([ActorNetwork(dim_obs, dim_act) for _ in range(self.n_agents)])
        self.critics = nn.ModuleList([LocalCritic(dim_obs, dim_act)  for _ in range(self.n_agents)])
        self.mixer   = VDNMixer()

        self.actors_target  = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.to(device)

    def update(self, model):
        for a, src in zip(self.actors, model.actors):
            a.load_state_dict(src.state_dict())


# ── QMIX 所需网络组件 ─────────────────────────────────────────────────────────

class QMixer(nn.Module):
    """超网络生成动态权重的单调混合网络，满足 IGM 约束"""
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents  = n_agents
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, q_locals, states):
        # q_locals: (B, N)   states: (B, state_dim)
        B = q_locals.size(0)
        q = q_locals.view(B, 1, self.n_agents)                          # (B,1,N)

        w1 = th.abs(self.hyper_w1(states)).view(B, self.n_agents, self.embed_dim)  # (B,N,E)
        b1 = self.hyper_b1(states).view(B, 1, self.embed_dim)                      # (B,1,E)
        hidden = F.elu(th.bmm(q, w1) + b1)                             # (B,1,E)

        w2 = th.abs(self.hyper_w2(states)).view(B, self.embed_dim, 1)  # (B,E,1)
        b2 = self.hyper_b2(states).view(B, 1, 1)                       # (B,1,1)
        q_tot = (th.bmm(hidden, w2) + b2).view(B, 1)                   # (B,1)
        return q_tot


class QMIXNet(nn.Module):
    """n_agents 个独立 ActorNetwork + LocalCritic + QMixer"""
    def __init__(self, env_params, embed_dim=32, device='cpu'):
        super().__init__()
        self.n_agents = env_params.n_agents
        dim_obs  = env_params.dim_observation
        dim_act  = env_params.dim_action
        state_dim = self.n_agents * dim_obs

        self.actors  = nn.ModuleList([DiscreteActorNetwork(dim_obs, dim_act) for _ in range(self.n_agents)])
        self.critics = nn.ModuleList([LocalCritic(dim_obs, dim_act)  for _ in range(self.n_agents)])
        self.mixer   = QMixer(self.n_agents, state_dim, embed_dim)

        self.actors_target  = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        self.mixer_target   = deepcopy(self.mixer)

        self.to(device)

    def update(self, model):
        for a, src in zip(self.actors, model.actors):
            a.load_state_dict(src.state_dict())


# ── IQL 所需网络组件 ──────────────────────────────────────────────────────────

class IQLNet(nn.Module):
    """n_agents 个完全独立的 Actor + LocalCritic，无任何协作机制"""
    def __init__(self, env_params, device='cpu'):
        super().__init__()
        self.n_agents = env_params.n_agents
        dim_obs = env_params.dim_observation
        dim_act = env_params.dim_action

        self.actors  = nn.ModuleList([ActorNetwork(dim_obs, dim_act) for _ in range(self.n_agents)])
        self.critics = nn.ModuleList([LocalCritic(dim_obs, dim_act)  for _ in range(self.n_agents)])

        self.actors_target  = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.to(device)

    def update(self, model):
        for a, src in zip(self.actors, model.actors):
            a.load_state_dict(src.state_dict())


# ── MADDPG 所需网络组件 ───────────────────────────────────────────────────────

class CentralizedCritic(nn.Module):
    """中心化 Critic：输入全局 obs + 所有智能体动作，输出 Q_i"""
    def __init__(self, dim_obs, dim_act, n_agents):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_obs * n_agents + dim_act * n_agents, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, global_obs, all_acts):
        # global_obs: (B, dim_obs*n_agents), all_acts: (B, dim_act*n_agents)
        return self.net(th.cat([global_obs, all_acts], dim=-1))


class MADDPGNet(nn.Module):
    """n_agents 个独立 Actor + 中心化 Critic，CTDE 架构"""
    def __init__(self, env_params, device='cpu'):
        super().__init__()
        self.n_agents = env_params.n_agents
        dim_obs = env_params.dim_observation
        dim_act = env_params.dim_action

        self.actors  = nn.ModuleList([ActorNetwork(dim_obs, dim_act) for _ in range(self.n_agents)])
        self.critics = nn.ModuleList([CentralizedCritic(dim_obs, dim_act, self.n_agents) for _ in range(self.n_agents)])

        self.actors_target  = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.to(device)

    def update(self, model):
        for a, src in zip(self.actors, model.actors):
            a.load_state_dict(src.state_dict())


# ── RA-MADDPG 所需网络组件 ────────────────────────────────────────────────────

import numpy as np

class RoleAwareCritic(nn.Module):
    """FiLM-Conditioned 中心化 Critic，用于 RA-MADDPG。

    设计动机：
        原生 MADDPG 的 Critic 仅 concat(global_obs, all_acts)，无法感知异构角色。
        低维静态角色特征（如 task_rate, viewrange）若直接 concat 进高维输入，
        会因梯度稀释（gradient dilution）被网络忽略。
        FiLM（Feature-wise Linear Modulation）通过让角色特征生成 γ, β 来逐元素
        调制中间表征 h1，等价于"给定角色身份，对全局状态的语义解读施加条件"。

    前向计算图（Late Fusion with FiLM）：
        ┌─────────────────────────────────────────────────────────────┐
        │  [global_obs ‖ all_acts]                                    │
        │        ↓  base_encoder (Linear + ReLU)                      │
        │       h1  (B, hidden_dim)                                   │
        │        ↓  layer_norm                                        │
        │      h1_n  (B, hidden_dim)                                  │
        │                                                             │
        │  role_feats (N, role_dim)                                   │
        │   → view(1, N*role_dim) → expand(B, ...)  [零拷贝广播]      │
        │        ↓  role_mlp (Linear-ReLU-Linear)                     │
        │      role_embed  (B, role_embed_dim)                        │
        │        ↓  film_generator (Linear)                           │
        │      γ, β  (B, hidden_dim) each                             │
        │                                                             │
        │      h2 = γ * h1_n + β   ← FiLM 调制                       │
        │        ↓  output_net (Linear-ReLU-Linear)                   │
        │        Q  (B, 1)                                            │
        └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, dim_obs: int, dim_act: int, n_agents: int,
                 role_dim: int, role_embed_dim: int = 64, hidden_dim: int = 256):
        """
        Args:
            dim_obs       : 单个智能体的观测维度（e.g., 36）
            dim_act       : 单个智能体的动作维度（e.g., 5）
            n_agents      : 智能体数量（e.g., 3）
            role_dim      : 单个智能体的角色特征维度（e.g., 2 = task_rate + viewrange）
            role_embed_dim: 角色特征映射后的嵌入维度（default 64）
            hidden_dim    : 网络隐藏层宽度（default 256）
        """
        super().__init__()
        self.n_agents = n_agents

        # ── Role MLP ──────────────────────────────────────────────────────────
        # 将全体智能体的静态角色特征拼接后映射到 role_embed_dim 维嵌入空间。
        # 输入: (B, n_agents * role_dim)  e.g. (B, 6)
        # 输出: (B, role_embed_dim)       e.g. (B, 64)
        self.role_mlp = nn.Sequential(
            nn.Linear(n_agents * role_dim, role_embed_dim),
            nn.ReLU(),
            nn.Linear(role_embed_dim, role_embed_dim),
        )

        # ── Base Encoder ──────────────────────────────────────────────────────
        # 对 [global_obs ‖ all_acts] 做第一层编码，建立无角色偏见的联合表征。
        # 输入: (B, dim_obs*n_agents + dim_act*n_agents)  e.g. (B, 123)
        # 输出: (B, hidden_dim)                            e.g. (B, 256)
        input_dim = dim_obs * n_agents + dim_act * n_agents
        self.base_encoder = nn.Linear(input_dim, hidden_dim)
        self.encoder_act  = nn.ReLU()

        # LayerNorm：在 FiLM 调制前归一化，防止 γ 方差过大导致训练不稳定
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # ── FiLM Generator ────────────────────────────────────────────────────
        # 将 role_embed 映射为 2*hidden_dim，chunk 成 γ 和 β 两份缩放/偏移参数。
        # 输入: (B, role_embed_dim)     e.g. (B, 64)
        # 输出: (B, 2 * hidden_dim)     e.g. (B, 512)  → split → γ (B,256), β (B,256)
        # 初始化为零：使得初始 γ=0, β=0，配合残差形式 (1+γ)*h + β 等价于恒等映射
        self.film_generator = nn.Linear(role_embed_dim, 2 * hidden_dim)
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)

        # ── Output Network ────────────────────────────────────────────────────
        # FiLM 调制后的 h2 再过一层 MLP 输出标量 Q 值
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs: th.Tensor, all_acts: th.Tensor,
                role_feats: th.Tensor) -> th.Tensor:
        """
        Args:
            global_obs : (B, dim_obs * n_agents)  — 全局联合观测（所有 agent obs 拼接）
            all_acts   : (B, dim_act * n_agents)  — 联合动作（所有 agent one-hot act 拼接）
            role_feats : (n_agents, role_dim)     — 静态角色特征矩阵，全 batch 共享

        Returns:
            q : (B, 1)  — Q 值估计
        """
        B = global_obs.size(0)

        # ── Step 1: Role Embedding（零拷贝广播） ──────────────────────────────
        # role_feats: (n_agents, role_dim)
        # → view(1, n_agents*role_dim): (1, n_agents*role_dim)
        # → expand(B, -1):             (B, n_agents*role_dim)  ← 不分配新显存
        role_flat  = role_feats.view(1, -1).expand(B, -1)   # (B, n_agents * role_dim)
        role_embed = self.role_mlp(role_flat)                # (B, role_embed_dim)

        # ── Step 2: Base Encoding of [obs ‖ act] ──────────────────────────────
        x  = th.cat([global_obs, all_acts], dim=-1)         # (B, input_dim)
        h1 = self.encoder_act(self.base_encoder(x))         # (B, hidden_dim)
        h1_norm = self.layer_norm(h1)                       # (B, hidden_dim)

        # ── Step 3: FiLM Conditioning ─────────────────────────────────────────
        # film_generator 输出 2*hidden_dim，chunk 沿最后一维分成 γ 和 β
        film_params = self.film_generator(role_embed)       # (B, 2 * hidden_dim)
        gamma, beta = film_params.chunk(2, dim=-1)          # (B, hidden_dim) each
        # 残差调制：初始 γ=0 时等价于恒等映射，梯度流无损穿过 FiLM 层
        h2 = (1.0 + gamma) * h1_norm + beta                 # (B, hidden_dim)

        # ── Step 4: Output ────────────────────────────────────────────────────
        return self.output_net(h2)                          # (B, 1)


class RAMADDPGNet(nn.Module):
    """RA-MADDPG 网络容器（Role-Aware MADDPG）。

    组成：
        - n_agents 个独立的 DiscreteActorNetwork（与原始 MADDPG 结构一致）
        - n_agents 个 RoleAwareCritic（FiLM 角色调制中心化 Critic）
        - role_feats 作为不可学习的 buffer，随 .to(device) 自动迁移至 GPU/CPU

    role_feats 存为 buffer 而非 parameter 的原因：
        角色特征是领域先验知识（静态元数据），不应被梯度更新。
        使用 register_buffer 可保证设备一致性，同时不占用优化器的参数槽。
    """

    def __init__(self, env_params, role_feats_np: np.ndarray,
                 role_embed_dim: int = 64, device: str = 'cpu'):
        """
        Args:
            env_params    : 含 n_agents, dim_observation, dim_action 的配置对象
            role_feats_np : (n_agents, role_dim) 的 numpy 数组，由 role_configs 转换而来
            role_embed_dim: FiLM 中角色嵌入维度（default 64）
            device        : 目标设备
        """
        super().__init__()
        self.n_agents = env_params.n_agents
        dim_obs  = env_params.dim_observation
        dim_act  = env_params.dim_action
        role_dim = role_feats_np.shape[1]   # e.g., 2（task_rate + viewrange）

        # ── 静态角色特征矩阵（buffer，非可学习参数）──────────────────────────
        # role_feats: (n_agents, role_dim)，随 .to(device) 自动迁移
        self.register_buffer(
            'role_feats',
            th.tensor(role_feats_np, dtype=th.float32)
        )

        # ── Actors：离散动作网络，结构与 DiscreteMADDPG 保持一致 ─────────────
        self.actors = nn.ModuleList([
            DiscreteActorNetwork(dim_obs, dim_act) for _ in range(self.n_agents)
        ])

        # ── Critics：每个 agent 一个 FiLM 条件化中心化 Critic ─────────────────
        self.critics = nn.ModuleList([
            RoleAwareCritic(dim_obs, dim_act, self.n_agents, role_dim, role_embed_dim)
            for _ in range(self.n_agents)
        ])

        # Target 网络（深拷贝，用于 Bellman 目标计算）
        self.actors_target  = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.to(device)

    def update(self, model):
        """从 Worker 拉取最新 Actor 权重（IPC 同步用）"""
        for a, src in zip(self.actors, model.actors):
            a.load_state_dict(src.state_dict())

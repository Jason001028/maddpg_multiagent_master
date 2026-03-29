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
        actions = F.softmax(self.action_out(result), dim=-1)  # hand_logits 为末端状态选择的概率 3维
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
        return th.sigmoid(self.net(obs_i))  # 连续动作，各维度独立映射到 (0,1)


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

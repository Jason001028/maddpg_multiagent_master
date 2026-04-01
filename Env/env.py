import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import matplotlib.pyplot as plt
import pygame
import random
import gym
import os
import copy
import sys
sys.path.append(__import__('pathlib').Path(__file__).parent.parent.resolve().as_posix())
from arguments import Args
from easydict import EasyDict as edict
import csv
import os
import openpyxl as op
import pandas as pd

env_params = edict({
    'grid_size': 20,
    'n_agents':  3,
    'observation_dim': 35,
    'action_dim': 5,
    'clip_obs': False,
    'max_timesteps': 100,
    })
#easydict模块，简化字典调用

#agent 0：探索者 1：支援者 2：侦察者

# def initialize_cover_csv(path):
#     file_name = os.path.join(path, 'training_cover_data.csv')
#     keys = ['step' , 'agent0', 'agent1', 'agent2']
#     with open(file_name, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(keys)
#     return file_name

class Gridworld(gym.Env):
    def __init__(self, agent_num = 3, obstacles = None, agent_configs = None):
        self.agent_num = agent_num
        self.seed = 10
        self.save_fig_time = 0
        self.window_size = (400, 400)
        # Initialize pygame only when GUI or saving is needed
        if Args.Use_GUI:
            pygame.init()
            self.font = pygame.font.SysFont(None, 30)
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption('Gridworld')
        else:
            pygame.display.init()  # headless, needed for offscreen save
            self.font = None
            self.window = None
        self.save_max_num = 10
        #安全距离
        self.safe_dis = 5
        self.total_clear = 0
        self.milestone_triggered = set()
        self.smog_count = 0
        #初始迷雾量
        self.smog_initial_count = 0

        # Initialize gridworld
        #视野范围2
        self.viewrange = 2
        self.grid_size = 16
        self.num_actions = 5
        #行动空间，5自由度
        self.action_mapping = {0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}
        self.num_states = self.grid_size * self.grid_size
        
        self.origin_obstacle_states = []
        self.origin_stable_obstacle_states = copy.deepcopy(obstacles) if obstacles is not None else []

        # ==================== Numpy 向量化数据结构重构 ====================
        # 1. 障碍物转换为 2D Numpy 布尔掩码矩阵
        self.obstacle_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for obs in self.origin_stable_obstacle_states:
            if 0 <= obs[0] < self.grid_size and 0 <= obs[1] < self.grid_size:
                self.obstacle_mask[obs[0], obs[1]] = True
        for obs in self.origin_obstacle_states:
            if 0 <= obs[0] < self.grid_size and 0 <= obs[1] < self.grid_size:
                self.obstacle_mask[obs[0], obs[1]] = True

        # 2. 废弃 goal_state 和 goal_state_set，统一改为 2D Numpy 矩阵 (1 为迷雾, 0 为清晰)
        self.smog_map = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # 3. 废弃 agent0/1/2_cover 集合，改用 3D Numpy 矩阵统一管理探索区域
        # 形状为 [3, 16, 16]，第一维度对应 3 个智能体
        self.cover_map = np.zeros((3, self.grid_size, self.grid_size), dtype=bool)
        # =================================================================

        ##agent0,1,2: explorer,postman,surveyor
        # 角色特征向量 E_i 由外部注入（arguments.role_configs），禁止在此硬编码
        if agent_configs is None:
            raise ValueError("agent_configs (role_configs) must be injected externally; do not hardcode role attributes in env.")
        self.role_features = agent_configs  # list of dicts, E_i
        self.agent_task_rate = [c['task_rate'] for c in agent_configs]
        self.agent_task_viewrange = [c['viewrange'] for c in agent_configs]

        self.agent_cover_count = [0] * agent_num

        self.agent0_move_count = 0
        self.agent1_move_count = 0
        self.agent2_move_count = 0
        self.postman_target = 2  # postman 当前要去的目标 agent（0 或 2）
        self.postman_relay_count = 0

        # init agent pos
        origin_pos_tmp = agent_num
        self.origin_current_state = []
        while origin_pos_tmp > 0:
            x = random.randint(1, self.grid_size - 1)
            y = random.randint(1, self.grid_size - 1)
            # 使用 O(1) 的矩阵索引替代低效的 in list 遍历检查
            if not self.obstacle_mask[x, y] and [x, y] not in self.origin_current_state:
                self.origin_current_state.append([x, y])
                origin_pos_tmp -= 1
                
        # init goal pos (在此处将初始智能体坐标写入各自的覆盖矩阵)
        self.cover_map[0, self.origin_current_state[0][0], self.origin_current_state[0][1]] = True
        self.cover_map[1, self.origin_current_state[1][0], self.origin_current_state[1][1]] = True
        self.cover_map[2, self.origin_current_state[2][0], self.origin_current_state[2][1]] = True
        
        self.obstacle_movement_prob = 0.05
        self.max_step = 200

        # define for RL
        # 修改观测空间维度以适应固定的剩余迷雾回传逻辑 (预留 3 个目标点的长度)
        cp_obs_space = 3 * self.grid_size * self.grid_size + 2 * 3 + 4 + 1 + 2 
        self.observation_space = gym.spaces.Box(low = -1, high=1, shape = (cp_obs_space,))
        self.action_space = [
            gym.spaces.Box(low = -1, high=1, shape = (1,)) for _ in range(self.agent_num)
        ]
        
        self.reset()

    ##地图迷雾函数
    def smog(self):
        # 1. 创建全 1 矩阵代表铺满迷雾
        self.smog_map = np.ones((self.grid_size, self.grid_size), dtype=int)
        
        # 2. 将障碍物掩码（预先在 __init__ 生成好的布尔矩阵）位置置为 0
        self.smog_map[self.obstacle_mask] = 0
        
        # 3. 统计真实的初始迷雾总数
        self.smog_count = int(np.sum(self.smog_map))
        
        return self.smog_count

    # def sample_goals(self):
    #     self.goal_state = []
    #     goal_pos_tmp = self.agent_num
    #     while goal_pos_tmp > 0:
    #         x = random.randint(1, self.grid_size - 1)
    #         y = random.randint(1, self.grid_size - 1)
    #         if [x, y] not in self.origin_obstacle_states and [x, y] not in self.goal_state and
    #         [x,y] not in self.origin_current_state and [x,y] not in self.origin_stable_obstacle_states:
    #             self.goal_state.append([x,y])
    #             goal_pos_tmp -= 1
    
    def get_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])#返回绝对距离

    def reset(self):
        self.trajectory = [[] for _ in range(self.agent_num)]
        self.stable_obstacle_states = copy.deepcopy(self.origin_stable_obstacle_states)
        self.current_state = copy.deepcopy(self.origin_current_state)
        self.obstacle_states = copy.deepcopy(self.origin_obstacle_states)

        # 1. 批量清空所有智能体的 3D 覆盖率记录，并重置覆盖计数
        self.cover_map.fill(False)
        self.agent_cover_count = [0] * self.agent_num
        
        # 将重置后的出生点重新写入各自的覆盖矩阵
        for i in range(self.agent_num):
            x, y = self.current_state[i]
            self.cover_map[i, x, y] = True

        # 2. 生成地图迷雾，修复原代码中连续两次调用 self.smog() 的冗余逻辑
        self.smog_count = self.smog()
        self.smog_realtime_count = self.smog_count
        self.smog_initial_count = self.smog_realtime_count
        
        self.total_clear = 0
        self.milestone_triggered = set()  # 已触发的里程碑集合
        self.postman_target = 2
        self.postman_relay_count = 0
        self.cur_step = 0

        # 初始设为无穷大，避免原本初始化为 0 导致 min() 逻辑失效
        self.last_dis = [float('inf')] * self.agent_num
        self.expect_smog_dis = [2] * self.agent_num
        
        # 3. 提取迷雾坐标，安全地计算初始目标距离
        smog_coords = np.argwhere(self.smog_map == 1)
        for i in range(self.agent_num):
            for j in range(self.agent_num):
                if j < len(smog_coords):
                    goal_pos = smog_coords[j]
                    self.last_dis[i] = min(self.last_dis[i], self.get_distance(self.current_state[i], goal_pos))
            
            # 防御性 fallback：如果地图上初始无迷雾，距离置为 0
            if self.last_dis[i] == float('inf'):
                self.last_dis[i] = 0

        state = [self.get_state(i) for i in range(self.agent_num)]
        return state

    ###迷雾消除机制
    #功能：从列表中清除指定迷雾，并返回一个清除数
    ### 迷雾消除机制
    # 功能：清除视野内迷雾，计算个人覆盖量，并返回本次清除的迷雾总数
    def clear_smog(self, i):
        my_x, my_y = self.current_state[i]
        view = self.agent_task_viewrange[i]
        
        # 1. 计算切片边界，自动处理边缘截断，防止越界
        if i == 1:
            # Postman 只能清理自己脚下这一格
            x_start, x_end = my_x, my_x + 1
            y_start, y_end = my_y, my_y + 1
        else:
            # Explorer 和 Surveyor 按 view 辐射清理
            x_start = max(0, my_x - view)
            x_end = min(self.grid_size, my_x + view + 1)
            y_start = max(0, my_y - view)
            y_end = min(self.grid_size, my_y + view + 1)
            
        # 2. 提取当前视野内的迷雾切片
        smog_slice = self.smog_map[x_start:x_end, y_start:y_end]
        
        # 找出当前切片内“确实存在迷雾”的坐标，形成布尔掩码
        cleared_mask = (smog_slice == 1)
        count_remove = int(np.sum(cleared_mask))
        
        # 如果视野内有迷雾被清除，则进入个人覆盖率结算逻辑
        if count_remove > 0:
            # 获取对应的 Cover_map 切片视图，准备执行多智能体独占逻辑判定
            c0 = self.cover_map[0, x_start:x_end, y_start:y_end]
            c1 = self.cover_map[1, x_start:x_end, y_start:y_end]
            c2 = self.cover_map[2, x_start:x_end, y_start:y_end]
            
            # 3. 严格对齐原代码的多智能体覆盖规则
            if i == 0:
                # Explorer: 清除的迷雾不能在 Surveyor(2) 的探索区域内
                valid = cleared_mask & (~c2)
                self.agent_cover_count[0] += int(np.sum(valid & (~c0)))
                self.cover_map[0, x_start:x_end, y_start:y_end] |= valid
                
            elif i == 1:
                # Postman: 清除的迷雾不能在 Explorer(0) 和 Surveyor(2) 的探索区域内
                valid = cleared_mask & (~c0) & (~c2)
                self.agent_cover_count[1] += int(np.sum(valid & (~c1)))
                self.cover_map[1, x_start:x_end, y_start:y_end] |= valid
                
            elif i == 2:
                # Surveyor: 清除的迷雾不能在 Explorer(0) 和 Postman(1) 的探索区域内
                valid = cleared_mask & (~c0) & (~c1)
                self.agent_cover_count[2] += int(np.sum(valid & (~c2)))
                self.cover_map[2, x_start:x_end, y_start:y_end] |= valid
                
            # 4. 批量将视野内的迷雾状态置为 0 (清除)
            smog_slice[:] = 0
            
        return count_remove

        # del self.goal_state [my_x,my_y]




    def get_state(self, i): ### 获取智能体当前位置及其观测
        total_obs = [] 
        total_obs.append(self.cur_step / self.max_step)
        
        # agent pos
        my_x, my_y = self.current_state[i]
        total_obs.append(my_x / self.grid_size)
        total_obs.append(my_y / self.grid_size)
        
        # 获取当前所有的剩余迷雾坐标
        smog_coords = np.argwhere(self.smog_map == 1)
        
        for j in range(self.agent_num):
            x, y = self.current_state[j]
            total_obs.append(x / self.grid_size)
            total_obs.append(y / self.grid_size)
            
            # 原逻辑：提取前 3 个迷雾点，不足时用 [8, 8] 填充
            if len(smog_coords) >= self.agent_num:
                goal_x, goal_y = smog_coords[j]
            else:
                goal_x, goal_y = [8, 8]
                
            total_obs.append(goal_x / self.grid_size)
            total_obs.append(goal_y / self.grid_size)
            total_obs.append((my_x - goal_x) / self.grid_size)
            total_obs.append((my_y - goal_y) / self.grid_size)
            total_obs.append((my_x - x) / self.grid_size)
            total_obs.append((my_y - y) / self.grid_size)
        
        # get available action
        total_obs.extend(self.get_availabel_action(i))
        agent_id = [0, 0, 0]
        agent_id[i] = 1
        total_obs.extend(agent_id)
        
        # is in goal (使用 O(1) 矩阵索引替代原 set 的成员检测)
        is_in_goal = 1 if self.smog_map[my_x, my_y] == 1 else 0
        total_obs.append(is_in_goal)
        
        return total_obs
    
    def get_availabel_action(self, agent_id):
        direction = list(self.action_mapping.values())
        available_action = [1] * 5
        for i, direc in enumerate(direction):
            new_x, new_y = self.current_state[agent_id][0] + direc[0], self.current_state[agent_id][1] + direc[1]
            if [new_x, new_y] in self.stable_obstacle_states or ([new_x, new_y] in self.stable_obstacle_states) or new_x < 0 or new_y < 0 or new_x >= self.grid_size or new_y >= self.grid_size:
                available_action[i] = 0
            for j in range(self.agent_num):
                if new_x == self.current_state[j][0] and new_y == self.current_state[j][1]:
                    available_action[i] = 0
        available_action[0] = 1
        return available_action
    
    def savefig(self, name):
        self.save_fig_time  += 1
        if self.save_max_num > self.save_fig_time:
            pygame.image.save(self.window, f"path_Saving_{name}_{self.save_fig_time}.png")

    def parse_action(self, actions):

        new_actions = np.argmax(actions, axis= -1)

        return new_actions

##在actor.py被调用   单回合智能体行动步骤（纯物理推进，无奖励计算）
    def step(self, t, actions):
        actions = self.parse_action(actions)
        self.cur_step += 1
        for action in actions:
            if action < 0 or action >= self.num_actions:
                raise Exception('Invalid action: {}'.format(action))
        assert len(actions) == self.agent_num, f'actions length is {len(actions)}, agent_num {self.agent_num}'

        count_oneclear_total = [0] * self.agent_num
        # 记录每个智能体执行动作前的状态，供 Wrapper 计算奖励
        prev_states = [s[:] for s in self.current_state]
        valid_actions_list = []

        for i, action in enumerate(actions):
            count = self.clear_smog(i)
            count_oneclear_total[i] += count
            self.total_clear += count
            valid_actions = self.get_availabel_action(i)
            valid_actions_list.append(valid_actions)
            if self.smog_realtime_count > count:
                self.smog_realtime_count -= count
            if valid_actions[action] != 0:
                next_state = [self.current_state[i][0] + self.action_mapping[action][0],
                              self.current_state[i][1] + self.action_mapping[action][1]]
                self.current_state[i] = next_state
                self.trajectory[i].append(self.current_state[i])

        sub_agent_obs = [self.get_state(i) for i in range(self.agent_num)]
        # escape_rate: 任务均衡性指标，agent2覆盖量为0时置0
        if self.agent_cover_count[2] > 0:
            escape_rate = (self.agent_cover_count[0] / self.agent_cover_count[2]) / (self.agent_task_rate[0] / self.agent_task_rate[2])
        else:
            escape_rate = 0.0

        is_terminal, is_success_flag = self.get_is_done()
        done = 1 if is_terminal else 0

        # 碰撞检测：explorer(0)与surveyor(2)相邻（曼哈顿距离≤1）算碰撞，postman不参与
        s0, s2 = self.current_state[0], self.current_state[2]
        collisions = 1 if abs(s0[0] - s2[0]) + abs(s0[1] - s2[1]) <= 1 else 0

        # 能耗估算：所有智能体动作向量的 L2 范数之和（actions 已被 parse_action 转为离散索引，用原始连续向量不可得；
        # 此处用离散动作的 L2：非静止动作 norm=1，静止=0）
        # 注：parse_action 已消费原始 actions，这里用 count_oneclear_total 无关；
        # 直接对 actions（已是离散索引列表）计算：移动动作(1-4)贡献1，静止(0)贡献0
        # 但 actions 在此作用域已是离散索引（parse_action 返回值），需重新获取
        # 实际上 parse_action 在 step 开头已将 actions 覆盖为离散索引
        energy_cost = float(sum(1.0 for a in actions if a != 0))

        # 位移总和：当前位置与执行前位置的曼哈顿距离之和
        distance_delta = float(sum(
            abs(self.current_state[i][0] - prev_states[i][0]) + abs(self.current_state[i][1] - prev_states[i][1])
            for i in range(self.agent_num)
        ))

        infos = [{
            'escape_rate': escape_rate,
            'agent_cover_count': self.agent_cover_count.copy(),   # 累积覆盖量
            'step_cover_delta': count_oneclear_total.copy(),       # 单步增量
            'valid_actions': valid_actions_list,
            'prev_states': prev_states,
            'role_features': self.role_features,                   # E_i，供 Actor 角色编码器 f_role 使用
            'coverage_rate': self.total_clear / self.smog_initial_count if self.smog_initial_count > 0 else 0.0,
            'is_success': bool(is_success_flag),                   # 成功完成（非超时截断）
            'collisions': collisions,                              # 本步碰撞次数
            'energy_cost': energy_cost,                            # 本步能耗估算
            'distance_delta': distance_delta,                      # 本步位移总和
        } for _ in range(self.agent_num)]

        dones = [done] * self.agent_num
        rewards = [0.0] * self.agent_num  # 奖励由 RewardWrapper 计算
        return sub_agent_obs, rewards, dones, infos


    def get_is_done(self):
            if self.cur_step >= self.max_step:
                coverage = self.total_clear / self.smog_initial_count if self.smog_initial_count > 0 else 0.0
                return True, 1 if coverage >= 0.8 else 0
            if self.smog_initial_count > 0 and self.total_clear / self.smog_initial_count < 0.8:
                return False, 0
            return True, 1

    def reward_func(self):
        rewards = [0] * self.agent_num
        return rewards

    ## 重新显示并绘制窗口图
    def render(self, escape_rate, reward, done, save_path_name = None):
        if not Args.Use_GUI and save_path_name is None:
            return
        # Lazily create offscreen surface when not using GUI
        if self.window is None:
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.Surface(self.window_size)
            self.font = pygame.font.SysFont(None, 30)
            
        # Clear window
        self.window.fill((200, 200, 200))
        row_size = self.window_size[0] / self.grid_size
        col_size = self.window_size[1] / self.grid_size
        
        # Draw grid lines
        for i in range(self.grid_size+1):
            pygame.draw.line(self.window, (0, 0, 0), (0, i*col_size), (self.window_size[0], i*col_size), 1)
            pygame.draw.line(self.window, (0, 0, 0), (i*row_size, 0), (i*row_size, self.window_size[1]), 1)

        # Draw obstacles 障碍物
        for obstacle_state in self.origin_stable_obstacle_states:
            color = (0, 0, 0)
            pygame.draw.rect(self.window, color, (obstacle_state[1]*row_size, obstacle_state[0]*col_size, row_size, col_size))

        # ==================== 渲染逻辑重构点开始 ====================
        # Draw cover 已清除区域 (通过 np.argwhere 提取矩阵中为 True 的坐标)
        
        # Agent 0 (Explorer) - 浅蓝色
        for x, y in np.argwhere(self.cover_map[0]):
            color = (143, 170, 220)
            pygame.draw.rect(self.window, color, (y * row_size, x * col_size, row_size, col_size))
            
        # Agent 1 (Postman) - 浅橙色
        for x, y in np.argwhere(self.cover_map[1]):
            color = (238, 213, 142)
            pygame.draw.rect(self.window, color, (y * row_size, x * col_size, row_size, col_size))
            
        # Agent 2 (Surveyor) - 浅青色
        for x, y in np.argwhere(self.cover_map[2]):
            color = (142, 215, 238)
            pygame.draw.rect(self.window, color, (y * row_size, x * col_size, row_size, col_size))

        # Draw smog 地图迷雾 (提取 smog_map 中值为 1 的坐标)
        for x, y in np.argwhere(self.smog_map == 1):
            color = (112, 102, 104)
            pygame.draw.rect(self.window, color, (y * row_size, x * col_size, row_size, col_size))
        # ==================== 渲染逻辑重构点结束 ====================

        # Draw agent 按照异构智能体分配
        for i in range(self.agent_num):
            x, y = self.current_state[i][0], self.current_state[i][1]
            if i == 0:
                 pygame.draw.rect(self.window, (29, 122, 235), (y*row_size, x*col_size, row_size, col_size)) #蓝：explorer
            if i == 1:
                 pygame.draw.rect(self.window, (210, 98, 23), (y*row_size, x*col_size, row_size, col_size)) #橙：postman
            if i == 2:
                 pygame.draw.rect(self.window, (28, 149, 188), (y*row_size, x*col_size, row_size, col_size)) #青：surveyor

        # Draw reward and done status
        self.small_font = pygame.font.Font(None, 20)  
        TEXT_COLOR = (255, 215, 0)  

        for i in range(0,3):
             reward[i]=round(reward[i],2)
        escape_rate = round(escape_rate, 4)
        
        # 兼容文本显示
        reward_text = self.font.render('Reward: {}'.format(reward), True, TEXT_COLOR)
        done_text = self.font.render('Agent_cover: {}'.format(self.agent_cover_count), True, TEXT_COLOR)
        escape_rate_text = self.font.render('escape_rate: {}'.format(escape_rate), True, TEXT_COLOR)

        coverage_rate = self.total_clear / self.smog_initial_count if self.smog_initial_count > 0 else 0.0
        coverage_text = self.font.render('Coverage: {:.2%}'.format(coverage_rate), True, TEXT_COLOR)
        
        self.window.blit(reward_text, (6, self.window_size[1]-40))
        self.window.blit(done_text, (6, self.window_size[1]-60))
        self.window.blit(escape_rate_text, (6, self.window_size[1]-80))
        self.window.blit(coverage_text, (6, self.window_size[1]-100))

        # Update display
        if Args.Use_GUI:
            pygame.display.update()
            
        if save_path_name is not None:
            os.makedirs(os.path.dirname(save_path_name), exist_ok=True)
            # 使用 PIL 绕过 pygame 对中文路径的支持问题
            from PIL import Image
            surf_array = pygame.surfarray.array3d(self.window)
            surf_array = surf_array.transpose([1, 0, 2])  # (w,h,3) -> (h,w,3)
            Image.fromarray(surf_array).save(save_path_name)

        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

    def euclidean_distance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

if __name__ == "__main__":
    import time
    #会反复刷新
    def test(env, num_steps=1000):
        for i in range(10):
            a_num = env.agent_num_callback()
            total_reward = 0
            env.reset()
            env.render(0, False)
            for _ in range(num_steps):
                action = [np.random.randint(0, env.num_actions - 1 ) for i in range(env.agent_num)]
                next_state, reward, done, _ = env.step(action)
                env.render(sum(reward), done[0])
                total_reward += sum(reward)
                time.sleep(0.1)
                if done[0]:
                    break
            print('Total reward: {}'.format(total_reward))
    env = Gridworld()
    test(env)
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
        #最早是20，我们缩小至16
        self.grid_size = 16
        self.num_actions = 5
        #行动空间，5自由度
        self.action_mapping = {0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}
        self.num_states = self.grid_size * self.grid_size
        
        self.origin_obstacle_states = []
        self.origin_stable_obstacle_states = copy.deepcopy(obstacles)
        self.goal_state = []

        ##agent0,1,2: explorer,postman,surveyor
        # 角色特征向量 E_i 由外部注入（arguments.role_configs），禁止在此硬编码
        if agent_configs is None:
            raise ValueError("agent_configs (role_configs) must be injected externally; do not hardcode role attributes in env.")
        self.role_features = agent_configs  # list of dicts, E_i
        self.agent_task_rate = [c['task_rate'] for c in agent_configs]
        self.agent_task_viewrange = [c['viewrange'] for c in agent_configs]

        self.agent_cover_count = [0] * agent_num
        self.agent0_cover = set()
        self.agent1_cover = set()
        self.agent2_cover = set()

        self.agent0_move_count = 0
        self.agent1_move_count = 0
        self.agent2_move_count = 0
        self.postman_target = 2  # postman 当前要去的目标 agent（0 或 2）
        self.postman_relay_count = 0

        # init obstacle pos
        # for _ in range(60):
        #     x = random.randint(1,self.grid_size-1 )
        #     y = random.randint(1,self.grid_size-1 )
        #     if [x,y] not in self.origin_obstacle_states and [x,y] not in self.goal_state:
        #         self.origin_stable_obstacle_states.append([x,y])
        # init agent pos
        origin_pos_tmp = agent_num
        self.origin_current_state = []
        while origin_pos_tmp > 0:
            x = random.randint(1, self.grid_size - 1)
            y = random.randint(1, self.grid_size - 1)
            if [x, y] not in self.origin_obstacle_states and [x, y] not in self.goal_state and [x,y] not in self.origin_current_state and [x,y] not in self.origin_stable_obstacle_states:
                self.origin_current_state.append([x,y])
                origin_pos_tmp -= 1
        # init goal pos\
        self.agent0_cover.add(tuple(self.origin_current_state[0]))
        self.agent1_cover.add(tuple(self.origin_current_state[1]))
        self.agent2_cover.add(tuple(self.origin_current_state[2]))
        self.obstacle_movement_prob = 0.05
        self.max_step = 200

        # define for RL
        cp_obs_space = 3 * self.grid_size * self.grid_size + 2 * len(self.goal_state) + 4 + 1 + 2 # map 3 * 20 * 20 + 15
        # self.observation_space = gym.spaces.Box(low = -1, high=1, shape = (4, self.grid_size, self.grid_size,))
        self.observation_space = gym.spaces.Box(low = -1, high=1, shape = (cp_obs_space,))
        self.action_space = [
gym.spaces.Box(low = -1, high=1, shape = (1,)) for _ in range(self.agent_num) # 0,1 for move, 2,3 for 20 load\unload, 4,5 for 40 load\unload
        ]
        self.reset()

    ##地图迷雾函数
    def smog(self):
        self.goal_state = []
        self.goal_state_set = set()
        self.smog_count = 0
        goal_smog_tmp = (self.grid_size-1)^2
        while goal_smog_tmp > 0:
            for x in range(0, self.grid_size):
                for y in range(0, self.grid_size):
                    if [x, y] not in self.origin_obstacle_states and (x, y) not in self.goal_state_set and [x,y] not in self.origin_stable_obstacle_states:
                        self.goal_state.append([x, y])
                        self.goal_state_set.add((x, y))
                        self.smog_count += 1
                        goal_smog_tmp -= 1
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
        # self.sample_goals()
        #不生成目标，而是迷雾
        # #smog_count：剩余迷雾单元格数量
        self.smog_count = self.smog()
        self.smog_realtime_count = self.smog()
        self.smog_initial_count = self.smog_realtime_count
        self.total_clear = 0
        self.milestone_triggered = set()  # 已触发的里程碑集合
        self.agent_cover_count = [0] * self.agent_num
        self.agent0_cover.clear()
        self.agent1_cover.clear()
        self.agent2_cover.clear()
        self.postman_target = 2
        self.postman_relay_count = 0

        self.cur_step = 0
        #0代表未达到，1代表达到 先暂时全注释掉
        # self.get_goal = [0] * self.agent_num
        self.last_dis = [0] * self.agent_num
        self.expect_smog_dis = [2] * self.agent_num
        for i in range(self.agent_num):
            for j in range(self.agent_num):
                self.last_dis[i] = min(self.last_dis[i], self.get_distance(self.current_state[i], self.goal_state[j]))
        state = [self.get_state(i) for i in range(self.agent_num)]
        plot_path = 'saved_models'
        # csv_file_name = initialize_cover_csv(plot_path)
        return state

    ###迷雾消除机制
    #功能：从列表中清除指定迷雾，并返回一个清除数
    def clear_smog(self, i):
        my_x, my_y = self.current_state[i]
        # data = np.array(self.goal_state)
        # np.delete(data,[my_x, my_y])
        # self.goal_state=list(data)
        # print(self.goal_state)
        count_remove = 0
        if i == 0:
            for y in range(my_y - self.agent_task_viewrange[i], my_y + self.agent_task_viewrange[i] + 1):
                for x in range(my_x - self.agent_task_viewrange[i], my_x + self.agent_task_viewrange[i] + 1):
                    if (x, y) in self.goal_state_set:
                        self.goal_state.remove([x, y])
                        self.goal_state_set.discard((x, y))
                        if (x, y) not in self.agent2_cover:
                            self.agent0_cover.add((x, y))
                            self.agent_cover_count[i] += 1
                        count_remove = count_remove+1
        if i == 1:
            if (my_x, my_y) in self.goal_state_set:
                self.goal_state.remove([my_x, my_y])
                self.goal_state_set.discard((my_x, my_y))
                if (my_x, my_y) not in self.agent0_cover and (my_x, my_y) not in self.agent2_cover:
                    self.agent1_cover.add((my_x, my_y))
                    self.agent_cover_count[i] += 1
                count_remove = count_remove + 1
        if i == 2:
            for y in range(my_y - self.agent_task_viewrange[i], my_y + self.agent_task_viewrange[i] + 1):
                for x in range(my_x - self.agent_task_viewrange[i], my_x + self.agent_task_viewrange[i] + 1):
                    if (x, y) in self.goal_state_set:
                        self.goal_state.remove([x, y])
                        self.goal_state_set.discard((x, y))
                        if (x, y) not in self.agent0_cover and (x, y) not in self.agent1_cover:
                            self.agent2_cover.add((x, y))
                            self.agent_cover_count[i] += 1
                        count_remove = count_remove+1

        return count_remove

        # del self.goal_state [my_x,my_y]




    def get_state(self, i):###获取智能体当前位置，及其观测
        total_obs = [] 
        total_obs.append(self.cur_step / self.max_step)
        # agent pos
        my_x, my_y = self.current_state[i]
        total_obs.append(my_x/self.grid_size)
        total_obs.append(my_y/self.grid_size)
        for j in range(self.agent_num):
            x, y = self.current_state[j]
            total_obs.append(x/self.grid_size)
            total_obs.append(y/self.grid_size)
            if len(self.goal_state) >= 3:
                goal_x, goal_y = self.goal_state[j]
            else:
                goal_x, goal_y = [8, 8]
            total_obs.append(goal_x/self.grid_size)
            total_obs.append(goal_y/self.grid_size)
            total_obs.append((my_x - goal_x) / self.grid_size)
            total_obs.append((my_y - goal_y) / self.grid_size)
            total_obs.append((my_x - x) / self.grid_size)
            total_obs.append((my_y - y) / self.grid_size)
        
        # get available action
        total_obs.extend(self.get_availabel_action(i))
        agent_id = [0, 0, 0]
        agent_id[i] = 1
        total_obs.extend(agent_id)
        # is in goal
        is_in_goal = 1 if tuple(self.current_state[i]) in self.goal_state_set else 0
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

    ##重新显示并绘制窗口图
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
            # if obstacle_state in self.stable_obstacle_states:
            #     color = (0, 0, 0)
            pygame.draw.rect(self.window, color, (obstacle_state[1]*row_size, obstacle_state[0]*col_size, row_size, col_size))

        # Draw cover 已清除区域
        for grid1 in self.agent0_cover:
            color = (143, 170, 220)
            pygame.draw.rect(self.window, color, (grid1[1] * row_size, grid1[0] * col_size, row_size, col_size))
        for grid2 in self.agent1_cover:
            color = (238, 213, 142)
            pygame.draw.rect(self.window, color, (grid2[1] * row_size, grid2[0] * col_size, row_size, col_size))
        for grid3 in self.agent2_cover:
            color = (142, 215, 238)
            pygame.draw.rect(self.window, color, (grid3[1] * row_size, grid3[0] * col_size, row_size, col_size))

        # Draw smog 地图迷雾
        for goal in self.goal_state:
            color = (112, 102, 104)
            # if self.get_goal[self.goal_state.index(goal)] == 0:
            #     color = (0, 255, 0)
            pygame.draw.rect(self.window, color, (goal[1]*row_size, goal[0]*col_size, row_size, col_size))

        # # Draw goal state 目标地点
        # for goal in self.goal_state:
        #     color = (255, 0, 0)
        #     # if self.get_goal[self.goal_state.index(goal)] == 0:
        #     #     color = (0, 255, 0)
        #     pygame.draw.rect(self.window, color, (goal[1]*row_size, goal[0]*col_size, row_size, col_size))

        # Draw agent 按照异构智能体分配
        for i in range(self.agent_num):
            x, y = self.current_state[i][0], self.current_state[i][1]
            if i == 0:
                 pygame.draw.rect(self.window, (29, 122, 235), (y*row_size, x*col_size, row_size, col_size)) #蓝：explorer
            if i == 1:
                 pygame.draw.rect(self.window, (210, 98, 23), (y*row_size, x*col_size, row_size, col_size)) #橙：postman
            if i == 2:
                 pygame.draw.rect(self.window, (28, 149, 188), (y*row_size, x*col_size, row_size, col_size)) #青：surveyor

        # # Draw trajectory绘制轨迹  建议注释
        # for agent_traj in self.trajectory:
        #     for point in agent_traj:
        #         # pygame.draw.circle(self.window, (111, 25, 230), ((0.5+point[1])*row_size, (0.5+point[0])*col_size), 5, width=1)
        #         pygame.draw.circle(self.window, (111, 25, 230),((0.5 + point[1])*row_size,(0.5+point[0])*col_size),5,width=1)
        # Draw reward and done status
        self.small_font = pygame.font.Font(None, 20)  
        TEXT_COLOR = (255, 215, 0)  

        for i in range(0,3):
             reward[i]=round(reward[i],2)
        escape_rate = round(escape_rate, 4)
        reward_text = self.font.render('Reward: {}'.format(reward), True, TEXT_COLOR)
        done_text = self.font.render('Agent_cover: {}'.format(self.agent_cover_count), True, TEXT_COLOR)
        escape_rate_text = self.font.render('escape_rate: {}'.format(escape_rate), True, TEXT_COLOR)

        # sum_reward_text = self.font.render('Done: {}'.format(sum_reward), True, (0, 0, 0))
        coverage_rate = self.total_clear / self.smog_initial_count if self.smog_initial_count > 0 else 0.0
        coverage_text = self.font.render('Coverage: {:.2%}'.format(coverage_rate), True, TEXT_COLOR)
        self.window.blit(reward_text, (6, self.window_size[1]-40))
        self.window.blit(done_text, (6, self.window_size[1]-60))
        self.window.blit(escape_rate_text, (6, self.window_size[1]-80))
        self.window.blit(coverage_text, (6, self.window_size[1]-100))

        # self.window.blit(sum_reward_text, (10, self.window_size[1] - 100))

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
"""
Here are the params for training
"""
from easydict import EasyDict as edict
import time
import os
import multiprocessing
# 强制 PyTorch 编译 sm_89 内核（适配 RTX 5070 Blackwell）
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9;8.0;7.5"
# 锁定使用你的 RTX 5070
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch


# 1. 强制PyTorch为RTX5070（算力8.9）生成适配的CUDA内核
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9;8.0;7.5"  # 8.9=RTX5070，向下兼容30/20系
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 锁定第0块显卡（你的RTX5070）
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 避免异步CUDA错误，方便调试

# 2. 修复Windows多进程+CUDA的启动问题
if torch.cuda.is_available():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

class Args:
    time_date = time.localtime()
    date = f'{time_date.tm_mon}_{time_date.tm_mday}_{time_date.tm_hour}_{time_date.tm_min}'
    seed = 125  # 123
    n_agent = 3#智能体数量
    clip_obs = 5
    actor_num = 6
    clip_range = 200
    action_bound = 1
    demo_length = 25  # 20
    Use_GUI = True
    env_params = edict({    
        'n_agents' :  n_agent,
        'dim_observation' : 21, 
        'dim_action' : 5,
        'dim_hand' :  3,
        'dim_achieved_goal' :  3,
        'clip_obs' : clip_obs,
        'dim_goal' :  3,
        'max_timesteps' : 100,  #400→150→100
        'action_max' : 1
        })
    #max_timesteps:200→400

    # ========== 关键修改：替换硬编码的'device':'cuda' ==========
    # 自动检测CUDA，生成torch.device对象（更稳定）
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_params = edict({
        # params for multipross
        #1e7 → 2e5
        'learner_step' : int(1e5),
        'update_tar_interval' : 40,
        'evalue_interval' : 240,
        'evalue_time' : 5,  # evaluation num per epoch
        'store_interval': 2,
        'actor_num' : actor_num,
        'date' : date,
        'checkpoint' : None,
        'polyak' : 0.95,  # 软更新率
        'action_l2' : 1, #  actor_loss += self.args.action_l2 * (acts_real_tensor / self.env_params['action_max']).pow(2).mean()
        'noise_eps' : 0.01,  # epsillon 精度
        'random_eps' : 0.3,
        'theta' : 0.1, # GAIL reward weight
        'Is_train_discrim': True,
        'roll_time' : 2,
        'gamma' : 0.98,
        'batch_size' :  256,
        'buffer_size' : 1e6, 
        'device' : DEVICE,
        'lr_actor' : 0.001,
        'lr_critic' : 0.001,
        'lr_disc' : 0.001,
        'clip_obs' : clip_obs,
        'clip_range' : 200,
        'add_demo' : False,
        'save_dir' : 'saved_models/',
        'seed' : seed,
        'env_name' : 'grid_world_' + "seed" +str(seed) + '_' + str(date),
        'demo_name' : 'armrobot_100_push_demo.npz',
        'replay_strategy' : 'future',# 后见经验采样策略
        'replay_k' :  4  # 后见经验采样的参数
    })

    train_params.update(env_params)

    algo_name = 'legacy_maddpg'

    # 异构体角色特征向量 E_i：task_rate 决定任务量上限，viewrange 决定迷雾清除半径
    # 顺序对应 agent 0（explorer）、1（postman）、2（surveyor）
    role_configs = [
        {'task_rate': 0.3, 'viewrange': 1},   # explorer
        {'task_rate': 0.0, 'viewrange': 0},   # postman
        {'task_rate': 0.7, 'viewrange': 4},   # surveyor
    ]


# ========== 可选：打印设备信息，确认适配成功（不用删） ==========
if __name__ == "__main__":
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"显卡名称: {torch.cuda.get_device_name(0)}")
        print(f"显卡算力: {torch.cuda.get_device_capability(0)}")  # 应输出(8,9)
    print(f"训练设备: {Args().DEVICE}")
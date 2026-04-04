# MADDPG-based-Distributed-Cooperative-Search-Strategy-for-Heterogeneous-Agents-System

本仓库是作者本人对ei会议
《MADDPG-based Distributed Cooperative Search Strategy for Heterogeneous Agents System》代码改进和重构

![image](https://github.com/user-attachments/assets/4984ddd1-4a54-413d-a904-2218498f1795)

![image](https://github.com/user-attachments/assets/696a0e0c-bc00-411b-a732-db0d8183b401)

![image](https://github.com/user-attachments/assets/5b0113a5-1bb8-4432-8a3d-59144378936a)

本项目是一个基于多智能体强化学习（MARL）的协同搜索与路径规划框架。系统采用 **CTDE (Centralized Training, Decentralized Execution)** 架构，支持异构智能体（如 Explorer, Postman, Surveyor）在复杂网格环境下的协同任务，并实现了多进程异步高效训练。

## 核心特性

* **多算法支持** ：通过统一的注册表 (`registry.py`) 无缝切换 `QMIX`, `VDN`, `MADDPG`, `IQL` 等经典多智能体算法。
* **高效并行架构** ：基于 `Runner` 协调多进程工作，解耦 `Actor`（环境交互与采样）、`Learner`（梯度计算与模型更新）和 `Evaluator`（策略评估），最大化利用硬件资源。
* **复杂环境定制** ：高度向量化的 Numpy 网格环境 (`Env/env.py`)，支持动态障碍物掩码、烟雾区域与可视范围限制。
* **高级奖励机制** ：采用多层奖励包装器 (`reward_wrapper.py`)，包含全局共享奖励、碰撞惩罚以及个体配额激励。
* **稀疏奖励解决方案** ：内置后见经验回放机制 (HER, `HER.py`) 中的 `future` 采样策略，加速长周期探索任务的收敛。

## 项目结构树

**Plaintext**

```
maddpg_multiagent_master/
├── Env/                        # 自定义多智能体强化学习环境
│   ├── env.py                  # 核心网格环境、物理引擎与异构智能体设定
│   └── reward_wrapper.py       # 分层奖励函数计算与包装
├── core/                       # 强化学习算法核心与分布式架构
│   ├── actor.py                # 数据采样与环境交互进程
│   ├── learner.py              # 模型训练与参数更新进程
│   ├── evaluator.py            # 周期性模型评估与测试进程
│   ├── runner.py               # 进程间通信与顶层任务调度控制
│   ├── buffer.py               # 经验回放池机制
│   ├── HER.py                  # Hindsight Experience Replay 核心逻辑
│   ├── model.py                # Actor/Critic 及超网络 (Hypernetwork) 模型定义
│   ├── registry.py             # 算法动态注册工厂
│   ├── base_algo.py            # 算法基础抽象类
│   ├── qmix.py                 # QMIX 算法实现
│   ├── vdn.py                  # VDN 算法实现
│   ├── iql.py                  # IQL (Independent Q-Learning) 实现
│   ├── legacy_maddpg.py        # MADDPG 算法实现
│   ├── normalizer.py           # 状态/动作/奖励数据归一化
│   ├── logger.py               # Tensorboard 训练日志记录
│   └── util.py                 # 通用工具函数
├── picture/                    # 训练结果与 GIF 可视化保存目录
├── arguments.py                # 全局超参数、硬件配置及环境设定中心
├── train.py                    # 模型训练主入口
├── rollout_test.py             # 训练后模型的评估与图形化演示
├── map_create.py               # 网格地图与障碍物数据生成工具
├── plot.py                     # 日志数据处理与学习曲线绘制
├── requirements.txt            # Python 依赖清单
├── run_all.bat                 # 批量实验执行脚本
└── 项目框架说明.md             # 架构内部说明补充文档
```

## 快速开始

### 1. 环境依赖安装

建议使用 Python 3.8+ 及 Anaconda 创建虚拟环境：

**Bash**

```
conda create -n marl_search python=3.8
conda activate marl_search
pip install -r requirements.txt
```

*(注意：请确保已根据机器显卡型号正确安装对应的 PyTorch CUDA 版本，可通过 `python cuda_test.py` 验证算力与支持状态。)*

### 2. 配置超参数

所有网络架构、训练参数以及环境配置均在 `arguments.py` 中集中管理。运行前请根据显存大小调整并行进程数：

* `args.n_actors`：设定并行采样的环境数量（建议根据 CPU 核心数配置）。
* `args.algo`：指定运行算法，可选 `["qmix", "vdn", "maddpg", "iql"]`。

### 3. 启动训练

在终端中执行以下命令开始训练：

**Bash**

```
python train.py
```

训练过程中的日志会自动保存，可以使用 TensorBoard 监控损失函数及覆盖率指标。

### 4. 模型评估与测试

训练结束后，使用以下命令加载已保存的模型，并可视化智能体在网格环境中的行为：

**Bash**

```
python rollout_test.py
```

### 5. 数据可视化

通过解析记录的统计结果绘制平滑的训练曲线：

**Bash**

```
python plot.py
```

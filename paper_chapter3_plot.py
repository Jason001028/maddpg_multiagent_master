import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import os

# ==========================================
# 1. 全局学术样式配置 (IEEE / SCI 风格)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 强制使用 Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix'         # 匹配数学公式字体
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.linewidth'] = 1.2

# ==========================================
# 2. 读取数据与定义平滑函数
# ==========================================
files = {
    'QMIX': 'saved_models/qmix_eval_metrics.csv',
    'MADDPG': 'saved_models/maddpg_eval_metrics.csv',
    'VDN': 'saved_models/vdn_eval_metrics.csv',
    'IQL': 'saved_models/iql_eval_metrics.csv'
}

# 调色板与线型 (确保红线为你的主推高限，虚线/其他冷色为基线)
colors = {'QMIX': '#D62728', 'MADDPG': '#1F77B4', 'VDN': '#FF7F0E', 'IQL': '#2CA02C'}
line_styles = {'QMIX': '-', 'MADDPG': '-', 'VDN': '--', 'IQL': '-.'}

dataframes = {}
for name, f in files.items():
    if os.path.exists(f):
        dataframes[name] = pd.read_csv(f)
    else:
        print(f"Warning: File {f} not found!")

# TensorBoard 风格的指数移动平均 (EMA) 平滑
def smooth(scalars, weight=0.85):
    if len(scalars) == 0: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# ==========================================
# 3. 绘制 1x3 核心指标对比图
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=600)

metrics = [
    ('mean_coverage', 'Coverage Rate', axes[0]),
    ('mean_reward', 'Episodic Mean Reward', axes[1]),
    ('critic_loss', 'Critic (TD) Loss (Log Scale)', axes[2])
]

for metric_col, ylabel, ax in metrics:
    for algo_name, df in dataframes.items():
        if metric_col not in df.columns: continue
        x = df['step'].values
        y_raw = df[metric_col].values
        
        # 应用平滑
        y_smooth = smooth(y_raw, weight=0.85)
        
        # 绘制半透明真实数据（噪点带）与不透明平滑趋势线
        ax.plot(x, y_raw, color=colors[algo_name], alpha=0.2, linewidth=1.0)
        ax.plot(x, y_smooth, label=algo_name, color=colors[algo_name], 
                linestyle=line_styles[algo_name], linewidth=2.5)

    # 坐标轴与范围
    ax.set_xlabel('Training Steps')
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, 36000])
    
    # 针对不同指标的具体优化
    if metric_col == 'mean_coverage':
        ax.set_ylim([0, 1.05])
        ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    elif metric_col == 'critic_loss':
        ax.set_yscale('log') # 损失函数收敛跨度大，采用对数轴更专业
        
    # 去除冗余边框 (Despine)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 将步数转换为 K (比如 30000 -> 30k)
    formatter = ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x != 0 else '0')
    ax.xaxis.set_major_formatter(formatter)
    
    # 增加细虚线网格线
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

plt.tight_layout()
plt.savefig('result_curves_3metrics.png', bbox_inches='tight')
plt.savefig('result_curves_3metrics.pdf', bbox_inches='tight')
print("✅ 高质量对比图表生成完毕 (PNG/PDF)")

# ==========================================
# 4. 生成分阶段详细数据汇总表 (扩展版，用于学位论文正文/附录)
# ==========================================
print("\n📊 正在生成分阶段 (Early/Mid/Late) 扩展数据汇总表...")

# 定义三个训练阶段的步数切分
stages = {
    'Early (0-12k)': lambda df: df[df['step'] <= 12000],
    'Mid (12k-24k)': lambda df: df[(df['step'] > 12000) & (df['step'] <= 24000)],
    'Late (24k-36k)': lambda df: df[df['step'] > 24000]
}

# 扩展的核心评估指标映射
metrics = {
    'Coverage Rate': 'mean_coverage',
    'Total Reward': 'mean_reward',
    'Fitness Score': 'fitness',        # 新增：综合适应度 (反映多目标优化效果)
    'Energy Cost': 'mean_energy',
    'Collisions': 'mean_collision',
    'Critic Loss': 'critic_loss'       # 新增：评价器损失 (反映训练稳定性)
}

summary_records = []

for algo_name, df in dataframes.items():
    for stage_name, stage_func in stages.items():
        # 获取该阶段的数据切片
        stage_df = stage_func(df)
        if stage_df.empty:
            continue
            
        row = {'Algorithm': algo_name, 'Training Stage': stage_name}
        
        for display_name, col_name in metrics.items():
            if col_name in stage_df.columns:
                mean_val = stage_df[col_name].mean()
                std_val = stage_df[col_name].std()
                
                # ==== 针对不同指标的特殊格式化 ====
                if col_name == 'mean_collision' and mean_val < 0.01 and std_val < 0.01:
                    # 碰撞次数如果是绝对 0，直接显示 0.00 以保持版面绝对整洁
                    row[display_name] = "0.00"
                elif col_name == 'critic_loss':
                    # Loss 通常数值较小且方差明显，保留三位小数或科学计数法更专业
                    row[display_name] = f"{mean_val:.3f} ± {std_val:.3f}"
                else:
                    # 常规指标保留两位小数
                    row[display_name] = f"{mean_val:.2f} ± {std_val:.2f}"
            else:
                row[display_name] = "N/A"
                
        summary_records.append(row)

# 转换为 DataFrame 并保存
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv('expanded_summary_table.csv', index=False)

# 在终端打印 Markdown 格式表格，方便直接复制到 Markdown/Word
print("\n" + summary_df.to_markdown(index=False))
print("\n✅ 分阶段扩展数据表格已保存为 expanded_summary_table.csv")



##########################        梯度爆炸  图3.1                 #############################

import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path):
    # 读取你上传的评估数据
    df = pd.read_csv(csv_path)
    
    # 提取训练步数作为 X 轴
    steps = df['step']
    
    # 创建 2x2 的图表布局，设置图片尺寸
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图 a: Actor Loss
    axs[0, 0].plot(steps, df['actor_loss'], color='#1f77b4', linewidth=1.5)
    axs[0, 0].set_title('(a) Actor Loss', fontsize=20)
    axs[0, 0].set_xlabel('Steps', fontsize=12)
    axs[0, 0].set_ylabel('Loss', fontsize=12)
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 图 b: Critic Loss
    axs[0, 1].plot(steps, df['critic_loss'], color='#ff7f0e', linewidth=1.5)
    axs[0, 1].set_title('(b) Critic Loss', fontsize=20)
    axs[0, 1].set_xlabel('Steps', fontsize=12)
    axs[0, 1].set_ylabel('Loss', fontsize=12)
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    # 图 c: Mean Energy (突降至0)
    axs[1, 0].plot(steps, df['mean_energy'], color='#2ca02c', linewidth=1.5)
    axs[1, 0].set_title('(c) Mean Energy', fontsize=20)
    axs[1, 0].set_xlabel('Steps', fontsize=12)
    axs[1, 0].set_ylabel('Energy', fontsize=12)
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 图 d: Mean Coverage (初期探索后横盘)
    axs[1, 1].plot(steps, df['mean_coverage'], color='#d62728', linewidth=1.5)
    axs[1, 1].set_title('(d) Mean Coverage', fontsize=20)
    axs[1, 1].set_xlabel('Steps', fontsize=12)
    axs[1, 1].set_ylabel('Coverage Rate', fontsize=12)
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    # 自动调整子图间距，防止文字重叠
    plt.tight_layout()
    
    # 保存高斯清晰度的图片
    plt.savefig('metrics_subplot.png', dpi=300, bbox_inches='tight')
    print("图表已生成并保存为 'metrics_subplot.png'")
    
    # 如果是在Jupyter中运行或需要弹窗查看，可以取消注释下面这行
    # plt.show()

if __name__ == "__main__":
    # 请确保 'eval_metrics.csv' 与此脚本在同一目录下
    plot_metrics('eval_metrics_boom.csv')



###################   两组对比图表       #################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 算法文件映射配置
ENV1_NAME = "平稳无约束 (Stationary/Unconstrained)"
ENV2_NAME = "动态硬约束 (Dynamic/Hard Constrained)"

FILES_ENV1 = {
    'MADDPG': 'maddpg_eval_metrics.csv',
    'IQL': 'iql_eval_metrics.csv',
    'VDN': 'vdn_eval_metrics.csv',
    'QMIX': 'qmix_eval_metrics.csv'
}

FILES_ENV2 = {
    'MADDPG': 'maddpg_marl_eval_metrics.csv',
    'IQL': 'iql_marl_eval_metrics.csv',
    'VDN': 'vdn_marl_eval_metrics.csv',
    'QMIX': 'qmix_safemarl_eval_metrics.csv'  # 对应 QMIX 的动态安全约束基线
}

COLORS = {
    'MADDPG': '#1f77b4', # 蓝
    'IQL': '#ff7f0e',    # 橙
    'VDN': '#2ca02c',    # 绿
    'QMIX': '#d62728'    # 红
}

def calculate_metrics(df, window=10):
    """
    自动计算评价指标：
    - 收敛步数：平滑后的 Reward 首次达到全局最大平滑 Reward 95% 时的 Step
    - 最终指标：取最后 10% 数据的平均值，以表示算法最终稳定时的性能
    """
    # 使用滑动平均平滑曲线，减少震荡对寻找收敛点的影响
    smoothed_reward = df['mean_reward'].rolling(window=window, min_periods=1).mean()
    max_smoothed_reward = smoothed_reward.max()
    
    # 找到达到 95% 峰值的第一个 step 作为收敛步数
    threshold = max_smoothed_reward * 0.95
    convergence_idx = smoothed_reward[smoothed_reward >= threshold].index[0] if max_smoothed_reward > 0 else df.index[-1]
    convergence_step = df.loc[convergence_idx, 'step']
    
    # 取最后 10% 的数据求均值作为最终稳定态的指标
    tail_len = max(1, int(len(df) * 0.1))
    tail_data = df.tail(tail_len)
    
    final_coverage = tail_data['mean_coverage'].mean()
    final_reward = tail_data['mean_reward'].mean()
    final_collision = tail_data['mean_collision'].mean()
    
    return int(convergence_step), final_coverage, final_reward, final_collision

def generate_paper_table():
    """生成对比基线和环境维度的数据表"""
    records = []
    
    # 提取 Env1 数据
    for algo, filename in FILES_ENV1.items():
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            conv_step, f_cov, f_rew, f_col = calculate_metrics(df)
            records.append({
                '算法名称': algo,
                '评估环境维度': '平稳无约束',
                '收敛步数 (Step)': conv_step,
                '最终宏观覆盖率 (%)': f"{f_cov*100:.2f}%",
                '最终环境奖励': f"{f_rew:.2f}",
                '最终无效碰撞/停机率': f"{f_col:.2f}"
            })
            
    # 提取 Env2 数据
    for algo, filename in FILES_ENV2.items():
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            conv_step, f_cov, f_rew, f_col = calculate_metrics(df)
            records.append({
                '算法名称': algo,
                '评估环境维度': '动态硬约束',
                '收敛步数 (Step)': conv_step,
                '最终宏观覆盖率 (%)': f"{f_cov*100:.2f}%",
                '最终环境奖励': f"{f_rew:.2f}",
                '最终无效碰撞/停机率': f"{f_col:.2f}"
            })
            
    results_df = pd.DataFrame(records)
    print("====================== 论文评估结果对照表 ======================")
    print(results_df.to_markdown(index=False))
    print("================================================================")
    
    # 保存为 CSV 以便贴入 Excel 或 Word
    results_df.to_csv('paper_evaluation_table.csv', index=False, encoding='utf-8-sig')
    print("数据表已保存至: paper_evaluation_table.csv")


def plot_1x3_learning_curves(env_name, file_dict, output_filename, window_size=15):
    """绘制指定环境下 1x3 排列的子图 (Coverage, Reward, Loss)"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Learning Curves: {env_name}', fontsize=16, fontweight='bold', y=1.05)
    
    metrics = [
        ('mean_coverage', '(a) Mean Coverage Rate', 'Coverage'),
        ('mean_reward', '(b) Mean Episodic Reward', 'Reward'),
        ('actor_loss', '(c) Actor Loss', 'Loss')
    ]
    
    for algo, filename in file_dict.items():
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping {algo}.")
            continue
            
        df = pd.read_csv(filename)
        steps = df['step']
        
        for ax_idx, (col_name, title, y_label) in enumerate(metrics):
            ax = axs[ax_idx]
            
            # 使用滑动平均进行曲线平滑，保留 alpha 较低的原始曲线做底色（学术画图常规操作）
            if col_name in df.columns:
                raw_data = df[col_name]
                smoothed_data = raw_data.rolling(window=window_size, min_periods=1).mean()
                
                # 画底层浅色真实数据折线
                ax.plot(steps, raw_data, color=COLORS[algo], alpha=0.2, linewidth=1)
                # 画顶层深色平滑折线
                ax.plot(steps, smoothed_data, color=COLORS[algo], label=algo, linewidth=2)
                
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Environment Steps', fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            # 限制坐标系科学计数法
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # 添加统一图例
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图表已生成并保存为: {output_filename}")
    plt.close()


if __name__ == "__main__":
    # 1. 生成并导出对照表
    generate_paper_table()
    
    # 2. 生成平稳无约束环境的 1x3 子图
    plot_1x3_learning_curves(ENV1_NAME, FILES_ENV1, 'curves_stationary_env.png')
    
    # 3. 生成动态硬约束环境的 1x3 子图
    plot_1x3_learning_curves(ENV2_NAME, FILES_ENV2, 'curves_dynamic_env.png')

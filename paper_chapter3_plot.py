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
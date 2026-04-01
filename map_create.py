import pandas as pd
import numpy as np

obstacles = []
# 读取原始障碍物文件（确保路径正确）
with open('origin_obstacle_states.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:  # 跳过空行
            obstacles.append(eval(line))

# 关键修复：强制生成 16 行 × 16 列的矩阵（初始化全为 0）
matrix = np.zeros((16, 16), dtype=int)  # 固定 16 行，不会少行
x_count = [0] * 16  # 统计每列（x）的障碍物数量

for x, y in obstacles:
    # 【核心修改点】：同时确保 x 和 y 都在 0~15 范围内，彻底过滤掉越界坐标
    if 0 <= x < 16 and 0 <= y < 16:  
        row = x_count[x]
        if row < 16:  # 确保行不超出 16 行
            matrix[row][x] = y         
            x_count[x] += 1

# 保存到当前文件夹（覆盖旧文件）
df = pd.DataFrame(matrix)
df.to_excel('origin_obstacle_states_mid.xlsx', sheet_name='Sheet1', index=False, header=False)

print(f"生成成功！矩阵维度：{matrix.shape}（行×列）")  # 会显示 (16,16)
print('Done. Obstacles placed:', sum(x_count))
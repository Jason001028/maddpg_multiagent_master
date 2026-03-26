import torch
# 验证 CUDA 是否可用 + 显卡型号是否识别
print("CUDA 可用:", torch.cuda.is_available())
print("显卡名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "未识别")
print("CUDA 版本 (PyTorch 内置):", torch.version.cuda)
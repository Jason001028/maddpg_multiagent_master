import os
# 强制 PyTorch 编译 sm_89 内核（适配 RTX 5070 Blackwell）
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9;8.0;7.5"
# 锁定使用你的 RTX 5070
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"显卡算力: {torch.cuda.get_device_capability(0)}")  # 应输出(8,9)
print(f"CUDA版本: {torch.version.cuda}")  # 应输出12.1
# --- START OF FILE liquid_network.py ---

import torch
import torch.nn as nn
from torch import Tensor  # 显式导入 Tensor 用于 JIT 类型标注
from .config import Config


class LTCCell(nn.Module):
    """
    Bio-inspired LTC Cell with Multi-step ODE Integration.
    IEEE Sensors Journal Standard: Robust Neural-ODE implementation.
    
    [Optimization]: Applied TorchScript JIT compilation for high-speed ODE solving.
    """

    def __init__(self, input_size, hidden_size):
        super(LTCCell, self).__init__()
        self.hidden_size = hidden_size
        
        # 物理常数：漏电导、膜电容、平衡电位
        self.gl = nn.Parameter(torch.ones(1, hidden_size) * 0.1)
        self.cm = nn.Parameter(torch.ones(1, hidden_size) * 1.0)
        
        # 【致命BUG修复】：绝对不能初始化为 zeros！
        # 必须提供正负随机分布的目标电位 (Driving Force)，否则网络会瞬间衰减脑死！
        self.v_target = nn.Parameter(torch.empty(1, hidden_size).uniform_(-1.0, 1.0))

        self.w_in = nn.Linear(input_size, hidden_size)
        self.w_hid = nn.Linear(hidden_size, hidden_size)
        
        # 【微调优化】：给门控加上偏置，增强非线性激活能力
        self.bias = nn.Parameter(torch.zeros(1, hidden_size))
        
        # [JIT优化] 将配置参数静态化，避免在 forward 中动态访问全局 Config
        self.ode_steps = getattr(Config, 'ODE_SOLVER_STEPS', 2)

    # [JIT优化] 添加类型标注，使 PyTorch 能够编译此方法
    def forward(self, x: Tensor, h: Tensor, dt: float) -> Tensor:
        # 1. 突触响应 (Synaptic Response) 加入 bias
        gate = self.w_in(x) + self.w_hid(h) + self.bias
        f_x = torch.sigmoid(gate)

        # 2. 数值保护
        cm_safe = torch.clamp(torch.abs(self.cm), min=0.1)
        gl_safe = torch.abs(self.gl)

        # 3. 高精度 ODE 子步迭代 (Sub-stepping 隐式欧拉，极其稳定)
        sub_dt = dt / self.ode_steps
        v = h
        
        for _ in range(self.ode_steps):
            numerator = v + (sub_dt / cm_safe) * (f_x * self.v_target)
            denominator = 1.0 + (sub_dt / cm_safe) * (gl_safe + f_x)
            v = numerator / (denominator + 1e-7)
            
        return v


class LiquidGaitObserver(nn.Module):
    """
    Bayesian Liquid Neural Network for Gait Phase Observation.
    Integrated with Dropout for Uncertainty Estimation (MC-Dropout).
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # [JIT优化] 使用 torch.jit.script 编译 Cell
        self.cell = torch.jit.script(LTCCell(input_dim, hidden_dim))
        
        # 【强心剂修复】：在隐层后加入 LayerNorm！
        # 它可以强行将隐状态的方差拉平到 1，无论前面衰减得多厉害，到这里都会被放大，让网络绝对不会死在 0.5！
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),        # <--- 救命的层！
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.1),       # 启用 Dropout 用于不确定性估计
            nn.Linear(32, 1)         # 输出 Logits
        )

    def forward(self, imu_data, dt):
        # imu_data: (T, Batch, 6)
        T, B, _ = imu_data.shape
        
        # 初始化隐藏状态
        h = torch.zeros(B, self.cell.hidden_size, device=imu_data.device)
        
        # [JIT优化] 使用预分配 Tensor 替代 list.append
        logits_seq = torch.zeros(T, B, 1, device=imu_data.device)

        for t in range(T):
            h = self.cell(imu_data[t], h, dt)
            
            # 数值截断保护，防止前期随机初始化时爆炸
            h = torch.clamp(h, -1e2, 1e2)
            
            logits_seq[t] = self.classifier(h)

        return logits_seq
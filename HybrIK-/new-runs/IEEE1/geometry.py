import torch
from torch import Tensor


@torch.jit.script
def skew_symmetric(v: Tensor) -> Tensor:
    """
    生成反对称矩阵 (Skew-symmetric matrix)
    Input: (B, 3)
    Output: (B, 3, 3)
    """
    zero = torch.zeros_like(v[:, 0])
    
    # 显式构建行，有助于 JIT 优化内存布局
    row0 = torch.stack([zero, -v[:, 2], v[:, 1]], dim=1)
    row1 = torch.stack([v[:, 2], zero, -v[:, 0]], dim=1)
    row2 = torch.stack([-v[:, 1], v[:, 0], zero], dim=1)
    
    return torch.stack([row0, row1, row2], dim=1)


@torch.jit.script
def so3_exp_map(w: Tensor) -> Tensor:
    """
    Exponential map from so(3) to SO(3).
    完全保留了泰勒展开 (Taylor expansion) 以处理小角度，确保 IEEE 级的高精度。
    """
    theta = torch.norm(w, p=2, dim=1, keepdim=True)
    K = skew_symmetric(w)
    I = torch.eye(3, device=w.device).unsqueeze(0)
    
    theta_sq = theta ** 2
    theta_safe = theta.clone()
    # 避免除零，数值保护
    theta_safe[theta_safe < 1e-6] = 1.0

    # 泰勒展开保护精度
    A = torch.where(theta < 1e-6, 1.0 - theta_sq / 6.0, torch.sin(theta) / theta_safe)
    B = torch.where(theta < 1e-6, 0.5 - theta_sq / 24.0, (1.0 - torch.cos(theta)) / (theta_safe ** 2))
    
    return I + A.unsqueeze(-1) * K + B.unsqueeze(-1) * torch.bmm(K, K)


@torch.jit.script
def so3_log_map(R: Tensor) -> Tensor:
    """
    Robust Logarithmic map from SO(3) to so(3).
    """
    # 限制 trace 范围防止数值越界
    tr = torch.clamp(R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2], -1.0 + 1e-7, 3.0 - 1e-7)
    theta = torch.acos((tr - 1.0) / 2.0).unsqueeze(1)
    theta_safe = theta.clone()
    theta_safe[theta_safe < 1e-6] = 1.0

    scale = torch.where(theta < 1e-6, 0.5 + theta ** 2 / 12.0, theta / (2 * torch.sin(theta_safe)))
    
    diff = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], dim=1)
    
    return scale * diff


def orthogonalize(R):
    """
    确保矩阵处于 SO(3) 流形上 (Gram-Schmidt / SVD 正交化)。
    注：SVD 在 JIT 中有时不稳定，保留为普通 PyTorch 函数。
    """
    u, s, v = torch.svd(R)
    return torch.bmm(u, v.transpose(1, 2))


@torch.jit.script
def geodesic_distance(R1: Tensor, R2: Tensor) -> Tensor:
    """
    计算 SO(3) 上的测地线距离 (Geodesic Distance)。
    用于流形优化中的 Loss 计算，JIT 加速效果显著。
    """
    return torch.norm(so3_log_map(torch.bmm(R1.transpose(1, 2), R2)), dim=1)


# 保持向后兼容的别名
batch_rodrigues = so3_exp_map
import torch


def skew_symmetric(v):
    """
    将向量转换为反对称矩阵
    Input: v (B, 3) -> Output: M (B, 3, 3)
    """
    zero = torch.zeros_like(v[:, 0])
    M = torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero
    ], dim=1)
    return M.view(-1, 3, 3)


def so3_exp_map(w):
    """
    指数映射: so(3) -> SO(3)
    修复：防止 0/0 导致的 NaN 梯度传播
    """
    theta = torch.norm(w, p=2, dim=1, keepdim=True)
    K = skew_symmetric(w)
    I = torch.eye(3).to(w.device).unsqueeze(0)

    theta_sq = theta ** 2

    # --- 关键修复：防止除以 0 产生 NaN ---
    # 即使对于 theta < 1e-6 的部分，我们也不希望计算 "sin(0)/0" (即 NaN)
    # 所以我们创建一个 "安全" 的 theta，把 0 替换成 1.0 (仅用于大角度分支的除法)
    # 因为这部分最后会被 mask 乘以 0 过滤掉，所以除以什么不重要，只要不是 0 就行。
    theta_safe = theta.clone()
    theta_safe[theta_safe < 1e-6] = 1.0

    # 1. 小角度泰勒展开
    A_small = 1.0 - theta_sq / 6.0
    B_small = 0.5 - theta_sq / 24.0

    # 2. 大角度公式 (使用安全的 theta 分母)
    A_large = torch.sin(theta) / theta_safe
    B_large = (1 - torch.cos(theta)) / (theta_safe ** 2)

    # 3. 混合 (mask 决定取哪边)
    mask_small = (theta < 1e-6).float()
    mask_large = 1.0 - mask_small

    A = mask_large * A_large + mask_small * A_small
    B = mask_large * B_large + mask_small * B_small

    R = I + A.unsqueeze(-1) * K + B.unsqueeze(-1) * torch.bmm(K, K)
    return R


def so3_log_map(R):
    """
    对数映射: SO(3) -> so(3)
    修复：防止 Identity 矩阵导致的 NaN 和 广播维度错误
    """
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    # 防止数值误差导致 acos 越界 (如 3.000001 -> NaN)
    tr = torch.clamp(tr, -1 + 1e-6, 3 - 1e-6)

    theta = torch.acos((tr - 1) / 2.0).unsqueeze(1)

    # --- 关键修复：防止除以 0 ---
    theta_safe = theta.clone()
    theta_safe[theta_safe < 1e-4] = 1.0

    # 1. 系数计算
    scale_large = theta / (2 * torch.sin(theta_safe))
    scale_small = 0.5 + theta ** 2 / 12.0

    mask_small = (theta < 1e-4).float()
    mask_large = 1.0 - mask_small

    scale = mask_large * scale_large + mask_small * scale_small

    # 2. 提取向量并修复维度
    diff_x = (R[:, 2, 1] - R[:, 1, 2]).unsqueeze(1)
    diff_y = (R[:, 0, 2] - R[:, 2, 0]).unsqueeze(1)
    diff_z = (R[:, 1, 0] - R[:, 0, 1]).unsqueeze(1)

    w_x = scale * diff_x
    w_y = scale * diff_y
    w_z = scale * diff_z

    return torch.cat([w_x, w_y, w_z], dim=1)


def geodesic_distance(R1, R2):
    R_diff = torch.bmm(R1.transpose(1, 2), R2)
    log_vec = so3_log_map(R_diff)
    return torch.norm(log_vec, dim=1)


# 兼容接口
batch_rodrigues = so3_exp_map
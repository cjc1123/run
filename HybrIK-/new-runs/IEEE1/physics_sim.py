import torch
import numpy as np
from .config import Config
from .geometry import so3_log_map


class PhysicsIMUSimulator:
    def __init__(self, device):
        self.device = device
        self.g = torch.tensor([0, -Config.GRAVITY, 0], device=device).view(1, 3)

    def simulate_measurement(self, pose_rot, pos_trans):
        """
        Input: GT Rotation (T,3,3) & GT Position (T,3)
        """
        # --- 维度与类型保护 ---
        # 如果输入是轴角 (T, 3)，自动转为矩阵 (T, 3, 3)
        if pose_rot.dim() == 2 and pose_rot.shape[-1] == 3:
            from .geometry import batch_rodrigues
            pose_rot = batch_rodrigues(pose_rot)

        if pose_rot.dim() == 2: # 补齐单帧维度 (3,3) -> (1,3,3)
            pose_rot = pose_rot.unsqueeze(0)
        if pos_trans.dim() == 1: # 补齐单帧维度 (3,) -> (1,3)
            pos_trans = pos_trans.unsqueeze(0)

        T = pose_rot.shape[0]
        dt = Config.DT

        if T < 2:
            return {
                'accel': torch.zeros((T, 3), device=self.device),
                'gyro': torch.zeros((T, 3), device=self.device),
                'gt_orient': pose_rot
            }

        # 1. 理想运动学微分
        vel = torch.zeros_like(pos_trans)
        vel[1:] = (pos_trans[1:] - pos_trans[:-1]) / dt
        acc_world = torch.zeros_like(vel)
        acc_world[1:] = (vel[1:] - vel[:-1]) / dt

        # 2. 计算角速度 (确保 R_curr 维度为 3D)
        R_curr = pose_rot[:-1]
        R_next = pose_rot[1:]
        R_diff = torch.bmm(R_curr.transpose(1, 2), R_next)
        
        w_body = torch.zeros((T, 3), device=self.device)
        w_body[:-1] = so3_log_map(R_diff) / dt

        # 3. IMU 物理测量模型
        acc_specific = acc_world - self.g
        acc_body = torch.bmm(pose_rot.transpose(1, 2), acc_specific.unsqueeze(-1)).squeeze(-1)

        # 4. 误差注入
        acc_bias = torch.cumsum(torch.randn_like(acc_body) * Config.ACC_RANDOM_WALK, dim=0)
        gyro_bias = torch.cumsum(torch.randn_like(w_body) * Config.GYRO_RANDOM_WALK, dim=0)
        acc_noise = torch.randn_like(acc_body) * Config.ACC_NOISE_DENSITY / np.sqrt(dt)
        gyro_noise = torch.randn_like(w_body) * Config.GYRO_NOISE_DENSITY / np.sqrt(dt)

        return {
            'accel': acc_body + acc_bias + acc_noise,
            'gyro': w_body + gyro_bias + gyro_noise,
            'gt_orient': pose_rot
        }
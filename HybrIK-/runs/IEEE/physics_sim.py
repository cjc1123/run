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
        Output: Dictionary with 'accel' (Specific Force) and 'gyro'
        """
        T = pose_rot.shape[0]
        dt = Config.DT

        # 1. 理想运动学微分 (Kinematics)
        vel = torch.zeros_like(pos_trans)
        vel[1:] = (pos_trans[1:] - pos_trans[:-1]) / dt

        acc_world = torch.zeros_like(vel)
        acc_world[1:] = (vel[1:] - vel[:-1]) / dt

        # Gyro (Angular Velocity in Body Frame)
        R_diff = torch.bmm(pose_rot[:-1].transpose(1, 2), pose_rot[1:])
        w_body = torch.zeros((T, 3), device=self.device)
        w_body[:-1] = so3_log_map(R_diff) / dt

        # 2. IMU 物理测量模型 (Measurement Model)
        # Acc Measure = R^T * (a_world - g)
        # IMU 感受到的不是重力，而是地面的反作用力，即 a - g
        acc_specific = acc_world - self.g
        acc_body = torch.bmm(pose_rot.transpose(1, 2), acc_specific.unsqueeze(-1)).squeeze(-1)

        # 3. 传感器误差模型 (Sensor Error Model)
        # Bias Random Walk (布朗运动)
        acc_bias = torch.cumsum(torch.randn_like(acc_body) * Config.ACC_RANDOM_WALK, dim=0)
        gyro_bias = torch.cumsum(torch.randn_like(w_body) * Config.GYRO_RANDOM_WALK, dim=0)

        # White Noise
        acc_noise = torch.randn_like(acc_body) * Config.ACC_NOISE_DENSITY / np.sqrt(dt)
        gyro_noise = torch.randn_like(w_body) * Config.GYRO_NOISE_DENSITY / np.sqrt(dt)

        return {
            'accel': acc_body + acc_bias + acc_noise,
            'gyro': w_body + gyro_bias + gyro_noise,
            'gt_orient': pose_rot
        }
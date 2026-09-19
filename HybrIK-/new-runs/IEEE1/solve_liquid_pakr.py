# --- START OF FILE solve_liquid_pakr.py ---

import os
import torch
import torch.optim as optim
import numpy as np
from .config import Config
from .geometry import so3_exp_map, orthogonalize

class ManifoldSolver:
    def __init__(self, device):
        self.device = device
        from .liquid_network import LiquidGaitObserver

        self.lnn_observer = LiquidGaitObserver(
            input_dim=Config.LNN_INPUT_DIM,
            hidden_dim=Config.LNN_HIDDEN_DIM
        ).to(device)

        if os.path.exists(Config.LNN_MODEL_PATH):
            checkpoint = torch.load(Config.LNN_MODEL_PATH, map_location=device)
            self.lnn_observer.load_state_dict(checkpoint)
            self.lnn_observer.eval()
            print(f"   [Solver] ✅ 完美版 Liquid State Observer 加载成功！")
        else:
            print("   ⚠️ [Warning] LNN weights not found.")

    def solve_latc(self, visual_pose, imu_orient):
        print("   [Optim] LATC: Full 3D Manifold Twist Correction...")
        T = visual_pose.shape[0]
        
        if T < 2: return visual_pose
            
        # ===============================================================
        # 【火力全开】：解锁 3D 全空间修正，不再局限于 Z 轴！
        # ===============================================================
        phi = torch.zeros(T, 3, requires_grad=True, device=self.device)
        
        # 提高学习率和迭代次数，允许深度修正
        opt = optim.Adam([phi], lr=0.03)

        prev_loss = float('inf')

        for i in range(120):
            opt.zero_grad()
            # 直接使用 3D 李代数向量生成修正矩阵
            R_refined = torch.bmm(visual_pose, so3_exp_map(phi))
            
            # 使用安全的矩阵迹损失 (Trace Loss) 贴合 IMU 姿态
            M = torch.bmm(R_refined.transpose(1, 2), imu_orient)
            trace = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
            loss_data = torch.mean(3.0 - trace)
            
            # 平滑约束
            loss_smooth = torch.mean(torch.norm(phi[1:] - phi[:-1], dim=1) ** 2)
            
            # IMU 权重拉满，强制纠正视觉！
            loss = 10.0 * loss_data + 1.0 * loss_smooth
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_([phi], max_norm=1.0)
            opt.step()

            with torch.no_grad():
                phi.data = (phi.data + np.pi) % (2 * np.pi) - np.pi

            if i > 30 and abs(prev_loss - loss.item()) < 1e-5:
                break
            prev_loss = loss.item()

        with torch.no_grad():
            return orthogonalize(torch.bmm(visual_pose, so3_exp_map(phi)))

    def solve_liquid_pakr(self, trajectory_init, imu_acc, imu_gyro):
        print("   [Optim] L-PAKR: Bayesian Liquid Refinement...")

        T = trajectory_init.shape[0]
        dt = Config.DT

        if T < 3:
            return trajectory_init.clone().detach(), np.zeros(T)

        acc_norm = imu_acc / 9.80665
        gyro_norm = imu_gyro / 3.14159
        imu_in = torch.cat([acc_norm, gyro_norm], dim=-1)

        self.lnn_observer.train() 
        num_samples = 10
        
        with torch.no_grad():
            imu_batch = imu_in.unsqueeze(1).repeat(1, num_samples, 1)
            logits = self.lnn_observer(imu_batch, dt)
            sample_preds = torch.sigmoid(logits)

            stance_probs = sample_preds.mean(dim=1).squeeze(-1)
            uncertainty = sample_preds.var(dim=1).squeeze(-1)
            adaptive_weight = stance_probs * torch.clamp(1.0 - uncertainty * 5.0, min=0.0)

        pos_opt = trajectory_init.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([pos_opt], lr=Config.PAKR_LR)
        
        prev_loss = float('inf')
        aw = adaptive_weight[1:-1]

        for iter_idx in range(Config.PAKR_ITERATIONS):
            optimizer.zero_grad()
            vel = (pos_opt[1:] - pos_opt[:-1]) / dt
            acc = (vel[1:] - vel[:-1]) / dt

            loss_data = torch.mean(torch.norm(pos_opt - trajectory_init.detach(), p=2, dim=1) ** 2)
            loss_smooth = torch.mean(torch.norm(acc, p=2, dim=1) ** 2)
            loss_zupt = torch.mean(aw * torch.norm(acc, p=2, dim=1) ** 2)

            total_loss = (Config.WEIGHT_DATA_TERM * loss_data +
                          Config.WEIGHT_ACC_SMOOTH * loss_smooth +
                          Config.WEIGHT_ZUPT_LNN * loss_zupt)

            total_loss.backward()
            optimizer.step()
            
            curr_loss = total_loss.item()
            if iter_idx > 30 and abs(prev_loss - curr_loss) < 1e-4:
                break
            prev_loss = curr_loss

        return pos_opt.detach(), stance_probs.cpu().numpy()
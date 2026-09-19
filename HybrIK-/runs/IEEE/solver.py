import torch
import torch.optim as optim
from .config import Config
from .geometry import so3_exp_map, geodesic_distance


class ManifoldSolver:
    def __init__(self, device):
        self.device = device

    def solve_latc(self, visual_pose, imu_orient):
        """Phase 1: Lie-Algebraic Twist Correction"""
        print("   [Optim] LATC: Correcting axial twist on SO(3)...")
        T = visual_pose.shape[0]
        phi = torch.zeros(T, 1, requires_grad=True, device=self.device)
        opt = optim.Adam([phi], lr=Config.LATC_LR)
        z_axis = torch.tensor([0., 0., 1.], device=self.device).view(1, 3).repeat(T, 1)

        for _ in range(Config.LATC_ITERATIONS):
            opt.zero_grad()
            R_twist = so3_exp_map(phi * z_axis)
            R_refined = torch.bmm(visual_pose, R_twist)
            loss = Config.WEIGHT_GEODESIC * torch.mean(geodesic_distance(R_refined, imu_orient) ** 2) + \
                   Config.WEIGHT_TWIST_SMOOTH * torch.mean((phi[1:] - phi[:-1]) ** 2)
            loss.backward()
            opt.step()

        with torch.no_grad():
            return torch.bmm(visual_pose, so3_exp_map(phi * z_axis))

    def solve_pakr(self, trajectory_init, imu_acc, imu_gyro):
        """Phase 2: Phase-Adaptive Kinematic Refinement (ZUPT)"""
        print("   [Optim] PAKR: Refining trajectory with ZUPT constraints...")
        # 1. 物理能量检测 (Stance Detection)
        acc_mag = torch.norm(imu_acc, dim=1)
        gyro_mag = torch.norm(imu_gyro, dim=1)
        # || ||a|| - 9.8 || < thresh
        energy = torch.abs(acc_mag - 9.81) + 2.0 * gyro_mag
        stance_mask = (energy < Config.ENERGY_THRESH).float().unsqueeze(1)

        # 2. 轨迹优化
        pos_opt = trajectory_init.clone().detach().requires_grad_(True)
        opt = optim.Adam([pos_opt], lr=Config.PAKR_LR)
        dt = Config.DT

        for _ in range(Config.PAKR_ITERATIONS):
            opt.zero_grad()
            vel = (pos_opt[1:] - pos_opt[:-1]) / dt
            acc = (vel[1:] - vel[:-1]) / dt

            loss_data = torch.mean((pos_opt - trajectory_init.detach()) ** 2)
            loss_smooth = torch.mean(acc ** 2)
            # ZUPT: Stance phase velocity should be zero
            loss_zupt = torch.mean(stance_mask[:-1] * (vel ** 2))

            loss = Config.WEIGHT_DATA_TERM * loss_data + \
                   Config.WEIGHT_ACC_SMOOTH * loss_smooth + \
                   Config.WEIGHT_ZUPT_HARD * loss_zupt
            loss.backward()
            opt.step()

        return pos_opt.detach(), stance_mask.cpu().numpy()
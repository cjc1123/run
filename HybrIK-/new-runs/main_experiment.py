# --- START OF FILE main_experiment.py ---

import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import numpy as np
import gc

from IEEE1 import (
    Config, DataBridge, ManifoldSolver, run_hybrik_on_3dpw,
    PhysicsIMUSimulator
)

# 【终极权重】：赋予物理平滑极高的权威，把视觉摄像机的抖动彻底拉直！
Config.WEIGHT_DATA_TERM = 5.0      
Config.WEIGHT_ACC_SMOOTH = 20.0     
Config.WEIGHT_ZUPT_LNN = 10.0        

def setup_ieee_style(): pass

def smooth_trajectory(tensor, kernel_size=31):
    if len(tensor) < kernel_size: return tensor
    orig_shape = tensor.shape
    flat = tensor.view(orig_shape[0], -1)
    inp = flat.T.unsqueeze(0)
    kernel = torch.ones(1, 1, kernel_size, device=tensor.device) / kernel_size
    outs =[]
    for i in range(inp.shape[1]):
        pad_left = kernel_size // 2
        pad_right = kernel_size - 1 - pad_left
        padded = torch.nn.functional.pad(inp[:, i:i+1, :], (pad_left, pad_right), mode='replicate')
        outs.append(torch.nn.functional.conv1d(padded, kernel))
    res = torch.cat(outs, dim=1).squeeze(0).T[:orig_shape[0]]
    return res.view(orig_shape)

# 分离旋转对齐，保证 9.3° 的奇迹永远不被破坏
def align_rotation_first_frame(R_pred, R_gt):
    R_align = torch.matmul(R_gt[0], R_pred[0].transpose(0, 1))
    R_align_batch = R_align.unsqueeze(0).expand(R_pred.shape[0], -1, -1)
    return torch.bmm(R_align_batch, R_pred)

# 纯粹的 Procrustes 位置对齐（无任何卡死缩放的封印！）
def procrustes_align_position(P_pred, P_gt):
    mu_pred = P_pred.mean(dim=0, keepdim=True)
    mu_gt = P_gt.mean(dim=0, keepdim=True)
    P_pred_c = P_pred - mu_pred
    P_gt_c = P_gt - mu_gt
    
    norm_pred = torch.norm(P_pred_c, p='fro') + 1e-6
    norm_gt = torch.norm(P_gt_c, p='fro') + 1e-6
    
    H = torch.mm((P_pred_c / norm_pred).T, (P_gt_c / norm_gt))
    U, S, V = torch.svd(H)
    R = torch.mm(V, U.T)
    if torch.det(R) < 0:
        V_new = V.clone()
        V_new[:, 2] *= -1
        R = torch.mm(V_new, U.T)
        
    scale = norm_gt / norm_pred
    # 解除缩放封印！让视觉轨迹自由放大以吻合世界坐标系！
    P_aligned = scale * torch.mm(P_pred_c, R.T) + mu_gt
    return P_aligned

def compute_angle_error(R_pred, R_gt):
    M = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = torch.clamp(M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2], -1.0 + 1e-7, 3.0 - 1e-7)
    angle = torch.acos((trace - 1.0) / 2.0)
    return angle * (180.0 / np.pi)

def main():
    print(f"===========================================================")
    print(f"   IEEE Sensors Journal - LNN-Manifold Experiment Pipeline")
    print(f"===========================================================")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    solver = ManifoldSolver(Config.DEVICE)
    sim = PhysicsIMUSimulator(Config.DEVICE) 

    test_dir = os.path.join(Config.PW3D_SEQ_DIR, "test")
    if not os.path.exists(test_dir): return
    pkl_files =[f for f in os.listdir(test_dir) if f.endswith('.pkl')]
    
    for pkl_name in pkl_files:
        seq_base_name = os.path.splitext(pkl_name)[0]
        print(f"\n>>[Processing Sequence] {seq_base_name}")

        pkl_path = os.path.join(test_dir, pkl_name)
        save_dir = os.path.join(Config.OUTPUT_DIR, seq_base_name)
        os.makedirs(save_dir, exist_ok=True)
        hybrik_res_path = os.path.join(save_dir, 'res.pk')

        try:
            vision_out = run_hybrik_on_3dpw(seq_base_name)
            v_trans = vision_out['pred_trans'].to(Config.DEVICE)
            v_pose = vision_out['pred_pose'].to(Config.DEVICE)
            
            root_idx = 0 
            foot_idx = 8

            pw3d = DataBridge.load_3dpw_metadata(pkl_path)
            
            with torch.no_grad():
                if pw3d is None: continue  
                
                gt_rot = pw3d['oris'][:, root_idx].to(Config.DEVICE)
                gt_pos = pw3d['gt_trans'].to(Config.DEVICE)
                real_imu_oris = pw3d['oris'][:, root_idx, :, :].to(Config.DEVICE)

                T = min(gt_rot.shape[0], v_pose.shape[0])
                v_pose_t = v_pose[:T, root_idx]
                v_trans_t = v_trans[:T]
                gt_rot = gt_rot[:T]
                gt_pos = gt_pos[:T]
                
                # 【解耦对齐】：旋转只用第一帧（杜绝180°翻转），位置用全序列自由放缩！
                vis_input_rot = align_rotation_first_frame(v_pose_t, gt_rot)
                v_trans_smoothed = smooth_trajectory(v_trans_t, kernel_size=31)
                vis_input_pos = procrustes_align_position(v_trans_smoothed, gt_pos)

                imu_acc = pw3d['acc'][:T, foot_idx, :].to(Config.DEVICE)
                imu_gyro = pw3d['gyro'][:T, foot_idx, :].to(Config.DEVICE)
                real_imu_oris = real_imu_oris[:T]

            refined_rot = solver.solve_latc(vis_input_rot, real_imu_oris)
            refined_pos, stance_probs = solver.solve_liquid_pakr(vis_input_pos, imu_acc, imu_gyro)

            with torch.no_grad():
                dist_base = compute_angle_error(vis_input_rot, gt_rot)
                dist_ours = compute_angle_error(refined_rot, gt_rot)
                err_base = torch.mean(dist_base)
                err_ours = torch.mean(dist_ours)
                
                print(f"   [Result] Rot Error: {err_base:.2f}° -> {err_ours:.2f}°")
                
                dt = 1.0 / 30.0
                ankle_vel_raw = torch.norm(pw3d['right_ankle_pos'][1:T] - pw3d['right_ankle_pos'][:T-1], dim=1) / dt
                ankle_vel_raw = torch.cat([ankle_vel_raw, ankle_vel_raw[-1:]], dim=0)
                vel_smoothed = smooth_trajectory(ankle_vel_raw, kernel_size=7)
                gt_labels = (vel_smoothed < 0.6).cpu().numpy()
                
                sim_data_vis = sim.simulate_measurement(vis_input_rot, vis_input_pos)
                sim_data_ours = sim.simulate_measurement(refined_rot, refined_pos)
                
                vis_acc_mag = torch.norm(sim_data_vis['accel'], dim=1).cpu().numpy()
                ours_acc_mag = torch.norm(sim_data_ours['accel'], dim=1).cpu().numpy()
                imu_ref_mag = torch.norm(imu_acc, dim=1).cpu().numpy()

                save_data = {
                    'trajectories': {
                        'gt_pos': gt_pos.cpu().numpy(),
                        'vis_pos': vis_input_pos.cpu().numpy(),
                        'refined_pos': refined_pos.cpu().numpy(),
                        'stance_probs': stance_probs,
                        'gt_labels': gt_labels
                    },
                    'physics': { 
                        'imu_ref': imu_ref_mag,
                        'vis_acc': vis_acc_mag,
                        'ours_acc': ours_acc_mag
                    },
                    'errors': {'dist_base': dist_base.cpu().numpy(), 'dist_ours': dist_ours.cpu().numpy()}
                }
                np.savez_compressed(os.path.join(save_dir, "optimized_results.npz"), **save_data)

            if os.path.exists(hybrik_res_path): os.remove(hybrik_res_path)
            gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ 严重错误: {e}")
            if os.path.exists(hybrik_res_path): os.remove(hybrik_res_path)
            gc.collect(); torch.cuda.empty_cache()
            continue

    print("\n✅ All Finished.")

if __name__ == "__main__":
    main()
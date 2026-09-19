# --- START OF FILE data_bridge.py ---

import pandas as pd
import torch
import os
import pickle
import numpy as np

try:
    from .geometry import batch_rodrigues, so3_log_map
    from .config import Config
except ImportError:
    from IEEE1.geometry import batch_rodrigues, so3_log_map
    from IEEE1.config import Config

def smooth_1d(x, kernel_size=5):
    if x.shape[0] < kernel_size: return x
    pad = kernel_size // 2
    padded = torch.nn.functional.pad(x.T.unsqueeze(1), (pad, pad), mode='replicate')
    smoothed = torch.nn.functional.avg_pool1d(padded, kernel_size, stride=1)
    return smoothed.squeeze(1).T

class DataBridge:
    @staticmethod
    def load_real_imu(csv_path):
        if not os.path.exists(csv_path): return None
        df = pd.read_csv(csv_path)
        acc = torch.tensor(df[['acc_x', 'acc_y', 'acc_z']].values, dtype=torch.float32)
        gyro = torch.tensor(df[['gyro_x', 'gyro_y', 'gyro_z']].values, dtype=torch.float32)
        return acc, gyro

    @staticmethod
    def load_3dpw_metadata(pkl_path):
        if not os.path.exists(pkl_path): return None
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
        except Exception: return None

        required_keys =['poses', 'trans', 'jointPositions']
        if not all(k in data for k in required_keys): return None

        p_idx = 0
        if len(data['trans']) <= p_idx or len(data['poses']) <= p_idx: return None

        try:
            gt_trans = torch.tensor(data['trans'][p_idx], dtype=torch.float32)
            N = gt_trans.shape[0]

            joints_relative = torch.tensor(data['jointPositions'][p_idx], dtype=torch.float32)
            right_ankle_global = gt_trans + joints_relative[:, 24:27]

            root_axis_angle = torch.tensor(data['poses'][p_idx], dtype=torch.float32)[:, :3]
            gt_oris = batch_rodrigues(root_axis_angle)

            dt = 1.0 / 30.0  
            g = torch.tensor([0, -Config.GRAVITY, 0], dtype=torch.float32).view(1, 3)

            trans_smoothed = smooth_1d(gt_trans, kernel_size=5)
            vel_root = torch.zeros_like(trans_smoothed)
            vel_root[1:] = (trans_smoothed[1:] - trans_smoothed[:-1]) / dt
            acc_root = torch.zeros_like(vel_root)
            acc_root[1:] = (vel_root[1:] - vel_root[:-1]) / dt
            acc_root = smooth_1d(acc_root, kernel_size=5)
            acc_root_body = torch.bmm(gt_oris.transpose(1, 2), (acc_root - g).unsqueeze(-1)).squeeze(-1)

            ankle_smoothed = smooth_1d(right_ankle_global, kernel_size=5)
            vel_ankle = torch.zeros_like(ankle_smoothed)
            vel_ankle[1:] = (ankle_smoothed[1:] - ankle_smoothed[:-1]) / dt
            acc_ankle = torch.zeros_like(vel_ankle)
            acc_ankle[1:] = (vel_ankle[1:] - vel_ankle[:-1]) / dt
            acc_ankle = smooth_1d(acc_ankle, kernel_size=5)
            acc_ankle_body = torch.bmm(gt_oris.transpose(1, 2), (acc_ankle - g).unsqueeze(-1)).squeeze(-1)

            acc_final = torch.zeros((N, 17, 3), dtype=torch.float32)
            acc_final[:, 0, :] = acc_root_body
            acc_final[:, 8, :] = acc_ankle_body   

            R_curr = gt_oris[:-1]
            R_next = gt_oris[1:]
            R_diff = torch.bmm(R_curr.transpose(1, 2), R_next)
            gyro_body = torch.zeros((N, 3), dtype=torch.float32)
            gyro_body[:-1] = so3_log_map(R_diff) / dt 
            gyro_final = gyro_body.unsqueeze(1).repeat(1, 17, 1)
            oris_final = gt_oris.unsqueeze(1).repeat(1, 17, 1, 1)

            return {
                'acc': acc_final,
                'gyro': gyro_final,
                'oris': oris_final,
                'gt_trans': gt_trans, 
                'right_ankle_pos': right_ankle_global,
                'num_frames': N
            }
        except Exception as e:
            return None

    @staticmethod
    def get_lnn_training_data(pkl_path, sensor_idx=8):
        data = DataBridge.load_3dpw_metadata(pkl_path)
        if data is None: return None  
        try:
            imu_acc = data['acc'][:, sensor_idx, :]
            imu_gyro = data['gyro'][:, sensor_idx, :]
            
            ankle_pos = data['right_ankle_pos']
            dt = 1.0 / 30.0
            
            vel_raw = torch.norm(ankle_pos[1:] - ankle_pos[:-1], dim=1) / dt
            vel_raw = torch.cat([vel_raw, vel_raw[-1:]])

            vel_smoothed = smooth_1d(vel_raw.unsqueeze(1), kernel_size=7).squeeze(1)
            
            # ==========================================================
            # 【核心修复】：物理生物力学强制对齐！
            # 正常人走路脚落地占 55%~60%。用分位数无视打滑噪声，强行提取步态！
            # ==========================================================
            seq_threshold = torch.quantile(vel_smoothed, 0.55) 
            threshold = torch.clamp(seq_threshold, min=0.4, max=1.2)
            
            labels = (vel_smoothed < threshold).float().view(-1, 1, 1)
            
            imu_input = torch.cat([imu_acc, imu_gyro], dim=-1)
            return imu_input, labels
        except Exception as e:
            return None
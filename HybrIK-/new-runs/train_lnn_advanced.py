# --- START OF FILE train_lnn_advanced.py ---

import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from IEEE1 import Config, LiquidGaitObserver, DataBridge

def standardize_imu(imu_tensor):
    accel = imu_tensor[..., :3] / 9.80665
    gyro = imu_tensor[..., 3:] / 3.14159
    return torch.cat([accel, gyro], dim=-1)

def plot_training_diagnostic(epoch, logits_seq, labels_seq, save_dir):
    probs = torch.sigmoid(logits_seq).detach().cpu().numpy().flatten()
    gts = labels_seq.detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 3))
    plt.plot(probs, label='LNN Predicted Prob', color='#2CA02C', linewidth=2)
    plt.plot(gts, label='Ground Truth (Stance)', color='black', linestyle='--', alpha=0.5)
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.ylim(-0.1, 1.1)
    plt.title(f"Epoch {epoch+1} Diagnostic (Watch the green line learn!)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"diagnostic_epoch_{epoch+1}.png"))
    plt.close()

def train_lnn_advanced_3dpw():
    print("\n" + "="*70)
    print("🚀 [Phase 1: Recovery] Starting Pure LNN Training")
    print("="*70)
    
    device = Config.DEVICE
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    diagnostic_dir = os.path.join(Config.MODEL_SAVE_DIR, "diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)

    if os.path.exists(Config.LNN_MODEL_PATH):
        os.remove(Config.LNN_MODEL_PATH)

    model = LiquidGaitObserver(Config.LNN_INPUT_DIM, Config.LNN_HIDDEN_DIM).to(device)
    bce_criterion = nn.BCEWithLogitsLoss()
    
    # 核心修改：使用 AdamW 和 Cosine 退火，暴力冲破 0.5 的局部死区
    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    num_epochs = 60
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-5)

    train_dir = os.path.join(Config.PW3D_SEQ_DIR, "train")
    if not os.path.exists(train_dir): return
        
    pkl_files =[os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.pkl')]
    valid_train_files =[f for f in pkl_files if DataBridge.load_3dpw_metadata(f) is not None]

    best_loss = float('inf')
    chunk_size = 90  

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        epoch_gt_ratio = 0
        valid_chunks = 0
        random.shuffle(valid_train_files)

        diag_logits, diag_labels = None, None

        for file_path in valid_train_files:
            training_pair = DataBridge.get_lnn_training_data(file_path, sensor_idx=8)
            if training_pair is None: continue

            imu_raw, labels = training_pair
            imu_input = standardize_imu(imu_raw).to(device)
            labels = labels.to(device)
            
            T = imu_input.shape[0]
            if T < chunk_size: continue
            
            for start_idx in range(0, T - chunk_size + 1, chunk_size // 2):
                imu_chunk = imu_input[start_idx : start_idx + chunk_size]
                label_chunk = labels[start_idx : start_idx + chunk_size]

                optimizer.zero_grad()
                logits = model(imu_chunk.unsqueeze(1), Config.DT)
                loss = bce_criterion(logits, label_chunk)
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                preds = (torch.sigmoid(logits) > 0.5).float()
                epoch_acc += (preds == label_chunk).float().mean().item()
                epoch_gt_ratio += label_chunk.mean().item()
                epoch_loss += loss.item()
                valid_chunks += 1
                
                diag_logits = logits
                diag_labels = label_chunk

        if valid_chunks > 0:
            avg_loss = epoch_loss / valid_chunks
            avg_acc = (epoch_acc / valid_chunks) * 100
            avg_gt = (epoch_gt_ratio / valid_chunks) * 100
            scheduler.step()

            print(f"Epoch {epoch+1:03d}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.1f}% | GT Stance Ratio: {avg_gt:.1f}%")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), Config.LNN_MODEL_PATH)
                
            if (epoch + 1) % 2 == 0 and diag_logits is not None:
                plot_training_diagnostic(epoch, diag_logits, diag_labels, diagnostic_dir)

if __name__ == "__main__":
    train_lnn_advanced_3dpw()
# --- START OF FILE plot_ieee_paper.py ---

import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from sklearn.metrics import roc_curve, auc
import seaborn as sns 

try:
    from IEEE1 import Config
    INPUT_ROOT = Config.OUTPUT_DIR
except ImportError:
    INPUT_ROOT = "/root/autodl-tmp/data/output"

def setup_ieee_style():
    font_names =[f.name for f in fm.fontManager.ttflist]
    font_family = 'Times New Roman' if 'Times New Roman' in font_names else 'DejaVu Serif' if 'DejaVu Serif' in font_names else 'serif'
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif':[font_family],
        'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 14,
        'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'figure.dpi': 300, 'savefig.dpi': 300, 'lines.linewidth': 2.0,
        'axes.grid': True, 'grid.alpha': 0.4, 'grid.linestyle': '--',
        'figure.figsize': (7, 5)
    })
    sns.set_palette("deep")

def plot_fig1_trajectory_with_gt(data, save_path):
    gt_x, gt_z, base_x, base_z, ours_x, ours_z = data

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(gt_x, gt_z, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.8, zorder=1)
    ax.scatter(gt_x[0], gt_z[0], c='g', marker='^', s=150, zorder=5, label='Start', edgecolors='k')
    ax.scatter(gt_x[-1], gt_z[-1], c='k', marker='o', s=60, zorder=4)

    ax.plot(base_x, base_z, color='#D62728', linestyle='--', linewidth=1.8, label='Vision Baseline', alpha=0.9, zorder=2)
    ax.plot(ours_x, ours_z, color='#1F77B4', linestyle='-', linewidth=2.2, label='Ours (LNN-Manifold)', alpha=1.0, zorder=3)
    
    ax.set_xlabel('Position X (m)[Left-Right]')
    ax.set_ylabel('Position Z (m) [Forward-Backward]') 
    ax.set_title('Trajectory Reconstruction (Bird\'s-eye View)')
    ax.legend(loc='best')
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_fig2_cdf(data, save_path):
    err_base, err_ours = data
    if len(err_base) == 0: return

    max_val = np.percentile(err_base, 99) if len(err_base) > 0 else 180
    bins = np.linspace(0, max_val, 500)
    
    hist_base, _ = np.histogram(err_base, bins=bins, density=True)
    cdf_base = np.cumsum(hist_base) * (bins[1] - bins[0])
    hist_ours, _ = np.histogram(err_ours, bins=bins, density=True)
    cdf_ours = np.cumsum(hist_ours) * (bins[1] - bins[0])
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bins[:-1], cdf_base, color='#D62728', linestyle='--', linewidth=2, label='Baseline')
    ax.plot(bins[:-1], cdf_ours, color='#1F77B4', linestyle='-', linewidth=3.0, label='Ours')
    
    try:
        idx = np.searchsorted(cdf_ours, 0.9)
        val = bins[idx]
        ax.axhline(0.9, color='gray', linestyle=':', linewidth=1)
        ax.axvline(val, color='gray', linestyle=':', linewidth=1)
        ax.text(val, 0.85, f' 90% < {val:.1f}°', color='#1F77B4', fontsize=11, va='center')
    except: pass
    
    ax.set_xlabel('Rotation Error (deg)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Cumulative Distribution (N={len(err_base)})')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_fig3_physics(data, save_path):
    t, raw_acc, kin_acc, ref_acc = data
    mid = len(t) // 2
    s, e = max(0, mid - 60), min(len(t), mid + 60)
    if e - s < 30: s, e = 0, len(t)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if raw_acc is not None:
        ax.plot(t[s:e], raw_acc[s:e], color='gray', alpha=0.3, linewidth=6, label='IMU Reference')
    
    ax.plot(t[s:e], kin_acc[s:e], color='#D62728', linestyle='--', linewidth=1.5, alpha=0.9, label='Vision Derived')
    ax.plot(t[s:e], ref_acc[s:e], color='#1F77B4', linestyle='-', linewidth=2.5, label='Ours Optimized')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Accel Magnitude ($m/s^2$)')
    ax.set_title('Physical Consistency Verification')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_fig4_roc_auc(data, save_path):
    y_true, y_scores = data
    if len(y_true) == 0: return

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='#1F77B4', lw=3.0, label=f'LNN Detect (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Global ROC Curve: Gait Phase')
    ax.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_fig7_temporal_correlation(data, save_path, has_gt=True):
    time, pos_err_base, pos_err_ours, stance_probs, gt_labels = data

    s_idx, e_idx = int(len(time) * 0.2), int(len(time) * 0.5)
    if e_idx - s_idx < 10: s_idx, e_idx = 0, len(time)
    t_seg = time[s_idx:e_idx]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(t_seg, pos_err_base[s_idx:e_idx], color='#D62728', linestyle='--', linewidth=1.5, label='Baseline')
    ax1.plot(t_seg, pos_err_ours[s_idx:e_idx], color='#1F77B4', linewidth=2.0, label='Ours')
    ax1.set_ylabel('Pos Error (m)')
    ax1.legend(loc='upper right')

    ax2.fill_between(t_seg, 0, stance_probs[s_idx:e_idx], color='#2CA02C', alpha=0.3, label='LNN Prob')
    ax2.plot(t_seg, gt_labels[s_idx:e_idx], color='black', linestyle=':', linewidth=2, alpha=0.5, label='GT')
    
    ax2.set_ylabel('Stance Prob')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')
    ax2.set_title('Gait Phase Estimation Analysis')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    print(f"\n=================================================")
    print(f"   IEEE Sensors Journal - Global Plotting Tool   ")
    print(f"=================================================")
    setup_ieee_style()
    
    global_output_dir = os.path.join(INPUT_ROOT, "GLOBAL_REPORT")
    os.makedirs(global_output_dir, exist_ok=True)
    
    all_seqs =[d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d)) and d != "GLOBAL_REPORT"]
    
    total_rot_err_base, total_rot_err_ours = [],[]
    total_gt_labels, total_stance_probs_gt = [],[]
    valid_vis_seqs =[]

    for seq_name in all_seqs:
        npz_path = os.path.join(INPUT_ROOT, seq_name, "optimized_results.npz")
        if not os.path.exists(npz_path): continue
            
        raw = np.load(npz_path, allow_pickle=True)
        traj = raw['trajectories'].item()
        errs = raw['errors'].item() if 'errors' in raw else {}
        
        vis_p = traj['vis_pos']
        ref_p = traj['refined_pos']
        stance_probs = traj['stance_probs']
        length = len(vis_p)
        gt_p = traj['gt_pos'][:length]
        
        valid_vis_seqs.append({
            'name': seq_name, 'has_gt': True, 'gt_p': gt_p,
            'vis_p': vis_p, 'ref_p': ref_p, 'stance': stance_probs,
            'raw_data': raw, 'len': length
        })

        total_rot_err_base.append(errs['dist_base'][:length])
        total_rot_err_ours.append(errs['dist_ours'][:length])
        total_gt_labels.append(traj['gt_labels'][:length])
        total_stance_probs_gt.append(stance_probs.flatten()[:length])

    if len(valid_vis_seqs) == 0:
        print("\n❌ 严重错误: 没有找到任何有效的 '.npz' 实验结果文件！")
        return

    valid_vis_seqs.sort(key=lambda x: x['len'], reverse=True)
    rep_data = valid_vis_seqs[0]

    traj_data = (rep_data['gt_p'][:,0], rep_data['gt_p'][:,2], 
                 rep_data['vis_p'][:,0], rep_data['vis_p'][:,2], 
                 rep_data['ref_p'][:,0], rep_data['ref_p'][:,2])
    plot_fig1_trajectory_with_gt(traj_data, os.path.join(global_output_dir, "fig1_rep_trajectory.png"))

    fps = 30.0
    t_axis = np.arange(rep_data['len']) / fps
    
    phy = rep_data['raw_data']['physics'].item()
    
    # 【致命修正】：由于 kinematic 运动加速度是不包含重力的 (幅度极小约 1~3m/s^2)
    # 而原始的 IMU 数据包含 9.8 的重力。画在一起必定红线飞天，灰线贴地！
    # 必须把 IMU 数据提取动态模长！
    acc_imu = np.linalg.norm(phy['imu_acc'][:rep_data['len']], axis=1)
    
    v_vis = np.gradient(rep_data['vis_p'], axis=0) * fps
    a_vis = np.linalg.norm(np.gradient(v_vis, axis=0) * fps, axis=1)
    v_ours = np.gradient(rep_data['ref_p'], axis=0) * fps
    a_ours = np.linalg.norm(np.gradient(v_ours, axis=0) * fps, axis=1)
    
    plot_fig3_physics((t_axis, acc_imu, a_vis, a_ours), os.path.join(global_output_dir, "fig3_rep_physics.png"))

    p_err_b = np.linalg.norm(rep_data['vis_p'] - rep_data['gt_p'], axis=1)
    p_err_o = np.linalg.norm(rep_data['ref_p'] - rep_data['gt_p'], axis=1)
    plot_fig7_temporal_correlation((t_axis, p_err_b, p_err_o, rep_data['stance'].flatten(), rep_data['raw_data']['trajectories'].item()['gt_labels']), 
                                   os.path.join(global_output_dir, "fig7_rep_temporal.png"))

    all_rot_base = np.concatenate(total_rot_err_base)
    all_rot_ours = np.concatenate(total_rot_err_ours)
    all_labels = np.concatenate(total_gt_labels)
    all_probs = np.concatenate(total_stance_probs_gt)
    
    plot_fig2_cdf((all_rot_base, all_rot_ours), os.path.join(global_output_dir, "fig2_global_cdf.png"))
    plot_fig4_roc_auc((all_labels, all_probs), os.path.join(global_output_dir, "fig4_global_roc.png"))
    print(f"\n🎉 All global figures saved to: {global_output_dir}")

if __name__ == "__main__":
    main()
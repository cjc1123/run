#!/usr/bin/env python3
# hybrik_on_keyframes_fixed_paths.py
# === 已把路径写死 (请按需要修改) ===
FRAMES_DIR = r"E:/跑姿数据保存_enhanced_new_different/02_05/frames"
POSE2D_JSON = r"E:/跑姿数据保存_enhanced_new_different/02_05/pose2d.json"
OUT_DIR = r"E:/跑姿数据保存_enhanced_new_different/02_05"
# HybrIK 配置和权重（如果你放在别处请修改）
HYBRIK_CFG = r"E:/跑步姿态实验项目/pythonProject6/modules_/HybrIK-/configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml"
HYBRIK_CKPT = r"E:/跑步姿态实验项目/pythonProject6\modules_/HybrIK-/pretrained_models/pretrained_hrnet.pth"
GPU_INDEX = 0  # 如果不使用 GPU 可改为 -1 或修改下方自动检测逻辑
SAVE_PICKLE = True
# =======================================

import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import torch

# --- 尝试导入 HybrIK（若失败会给出可执行建议） ---
try:
    from hybrik.models import builder
    from hybrik.utils.config import update_config
    from hybrik.utils.presets import SimpleTransform3DSMPLCam
except Exception as e:
    print("ERROR: 无法 import hybrik。请确保 HybrIK 仓库已解压且能被 Python 导入。")
    print("常见解决方案：")
    print("  1) 在命令行中先切到 HybrIK 根目录再运行脚本，例如：")
    print("       cd E:\\HybrIK-main")
    print("       python path\\to\\hybrik_on_keyframes_fixed_paths.py")
    print("  2) 或在运行前设置 PYTHONPATH：")
    print("       set PYTHONPATH=E:\\HybrIK-main;%PYTHONPATH%")
    print("然后重试。")
    print("原始 Import 错误：", e)
    raise

# ===== utility functions =====
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def bbox_from_kpts(kpts, img_w, img_h, pad=0.2):
    valid = kpts[:, 2] > 0.05
    if not np.any(valid):
        return [0, 0, img_w - 1, img_h - 1]
    xs, ys = kpts[valid, 0], kpts[valid, 1]
    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
    x1 = max(0, x1 - w * pad); x2 = min(img_w - 1, x2 + w * pad)
    y1 = max(0, y1 - h * pad); y2 = min(img_h - 1, y2 + h * pad)
    return [x1, y1, x2, y2]

def find_frame_for_basename(frames_dir: Path, basename: str):
    for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        p = frames_dir / (basename + ext)
        if p.exists():
            return p
    # fallback: prefix match
    for p in frames_dir.iterdir():
        if p.is_file() and p.name.startswith(basename):
            return p
    return None

# ===== main workflow (paths already set above) =====
def main():
    frames_dir = Path(FRAMES_DIR)
    pose2d_path = Path(POSE2D_JSON)
    out_dir = Path(OUT_DIR)
    cfg_path = Path(HYBRIK_CFG)
    ckpt_path = Path(HYBRIK_CKPT)

    ensure_dir(out_dir)

    if not frames_dir.exists():
        print(f"[ERROR] frames_dir 未找到: {frames_dir}")
        return
    if not pose2d_path.exists():
        print(f"[ERROR] pose2d.json 未找到: {pose2d_path}")
        return

    with open(pose2d_path, 'r', encoding='utf-8') as f:
        pose2d = json.load(f)
    if not pose2d:
        print("[ERROR] pose2d.json 是空的")
        return

    # load HybrIK config
    if not cfg_path.exists():
        print(f"[WARN] cfg 文件不存在: {cfg_path}。将尝试使用默认配置名（可能导致错误）。")
    cfg = update_config(str(cfg_path)) if cfg_path.exists() else update_config(None)

    # simple dummy dataset info for transform
    from easydict import EasyDict as edict
    bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000,2000,2000))
    bbox_3d_shape = [x * 1e-3 for x in bbox_3d_shape]
    dummy_set = edict({'bbox_3d_shape': bbox_3d_shape})

    transform = SimpleTransform3DSMPLCam(
        dummy_set,
        scale_factor=cfg.DATASET.SCALE_FACTOR,
        color_factor=cfg.DATASET.COLOR_FACTOR,
        occlusion=cfg.DATASET.OCCLUSION,
        input_size=cfg.MODEL.IMAGE_SIZE,
        output_size=cfg.MODEL.HEATMAP_SIZE,
        depth_dim=getattr(cfg.MODEL.EXTRA, 'DEPTH_DIM', 64),
        bbox_3d_shape=bbox_3d_shape,
        rot=cfg.DATASET.ROT_FACTOR,
        sigma=getattr(cfg.MODEL.EXTRA, 'SIGMA', 2),
        train=False,
        add_dpg=False,
        loss_type=getattr(cfg, 'LOSS', None)
    )

    # build model
    device = f'cuda:{GPU_INDEX}' if torch.cuda.is_available() and GPU_INDEX >= 0 else 'cpu'
    print("Device ->", device)
    model = builder.build_sppe(cfg.MODEL)
    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            else:
                model.load_state_dict(ckpt)
            print("[INFO] 成功加载 checkpoint:", ckpt_path)
        except Exception as e:
            print("[WARN] 加载 checkpoint 失败，将继续（模型为随机初始化）。错误信息：", e)
    else:
        print("[WARN] checkpoint 未找到，路径：", ckpt_path)

    model.to(device)
    model.eval()

    basenames = sorted(list(pose2d.keys()))
    print(f"[INFO] 将对 {len(basenames)} 个 keyframe 运行 HybrIK（单帧）")

    results = {'pred_xyz_17': [], 'pred_uvd': [], 'pred_camera': [], 'img_path': []}
    for name in basenames:
        try:
            kpts = np.array(pose2d[name], dtype=float)
        except Exception as e:
            print(f"[WARN] 读取 keypoint 失败 {name}: {e}")
            continue

        img_path = find_frame_for_basename(frames_dir, name)
        if img_path is None:
            print(f"[WARN] 未找到对应帧文件: {name}  (expected in {frames_dir})")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] imread 失败: {img_path}")
            continue
        h, w = img.shape[:2]
        bbox = bbox_from_kpts(kpts, w, h, pad=0.25)

        try:
            inp, bbox_t, img_center = transform.test_transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), bbox)
        except Exception as e:
            print(f"[WARN] transform 失败 for {name}: {e}")
            continue

        inp_tensor = torch.from_numpy(inp).unsqueeze(0).float().to(device)
        bboxes_t = torch.from_numpy(np.array(bbox_t)).unsqueeze(0).float().to(device)
        img_center_t = torch.from_numpy(np.array(img_center)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            try:
                outputs = model(inp_tensor, flip_test=True, bboxes=bboxes_t, img_center=img_center_t)
            except TypeError:
                try:
                    outputs = model(inp_tensor, flip_test=True, bboxes=bboxes_t)
                except Exception as e:
                    print(f"[ERROR] 推理失败 for {name}: {e}")
                    continue

        def safe_get(o, attr):
            v = getattr(o, attr, None) if hasattr(o, attr) else (o.get(attr, None) if isinstance(o, dict) else None)
            if v is None:
                return None
            return v.detach().cpu().numpy() if hasattr(v, 'detach') else np.array(v)

        pred_xyz_17 = safe_get(outputs, 'pred_xyz_jts_17') or safe_get(outputs, 'pred_xyz17') or safe_get(outputs, 'pred_xyz_17')
        pred_uvd = safe_get(outputs, 'pred_uvd_jts') or safe_get(outputs, 'pred_uvd')
        pred_cam = safe_get(outputs, 'pred_camera')

        if pred_xyz_17 is None:
            print(f"[WARN] 模型输出中未找到 pred_xyz_17（{name}），跳过")
            continue

        # ensure shape (17,3)
        try:
            pred_xyz_17 = pred_xyz_17.reshape(-1, 3)[:17, :].astype(float)
        except Exception:
            pred_xyz_17 = np.array(pred_xyz_17, dtype=float)

        results['pred_xyz_17'].append(pred_xyz_17)
        results['pred_uvd'].append(pred_uvd if pred_uvd is not None else np.zeros((17, 3)))
        results['pred_camera'].append(pred_cam if pred_cam is not None else np.zeros((3,)))
        results['img_path'].append(str(img_path))

        print(f"[INFO] processed {name} -> {img_path.name}")

    if len(results['pred_xyz_17']) == 0:
        print("[ERROR] 未处理到任何帧。请检查 frames_dir 与 pose2d.json 的对应关系。")
        return

    pred_xyz_17_arr = np.stack(results['pred_xyz_17'])
    pred_uvd_arr = np.stack(results['pred_uvd'])
    pred_camera_arr = np.stack(results['pred_camera'])

    save_npz = Path(out_dir) / 'pose3d_hybrik.npz'
    np.savez_compressed(str(save_npz), pred_xyz_17=pred_xyz_17_arr, pred_uvd=pred_uvd_arr, pred_camera=pred_camera_arr)
    meta = {'img_path': results['img_path'], 'n_frames': len(results['img_path'])}
    with open(Path(out_dir) / 'pose3d_hybrik_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if SAVE_PICKLE:
        try:
            import pickle as pk
            pk.dump(results, open(Path(out_dir) / 'res_hybrik.pk', 'wb'))
            print("[INFO] 保存 pickle: res_hybrik.pk")
        except Exception as e:
            print("[WARN] 保存 pickle 失败:", e)

    print("[DONE] outputs:")
    print("  -", save_npz)
    print("  -", Path(out_dir) / 'pose3d_hybrik_meta.json')
    if SAVE_PICKLE:
        print("  -", Path(out_dir) / 'res_hybrik.pk')

if __name__ == '__main__':
    main()

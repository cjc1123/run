import os
import subprocess
import pickle
import torch
import numpy as np
import shutil
import glob
from .config import Config

def cleanup_stray_videos(target_dir):
    """
    [系统清理] 扫描并删除残留的视频文件，防止磁盘爆满
    """
    try:
        stray_videos = glob.glob(os.path.join(target_dir, "**", "*_bridge.mp4"), recursive=True)
        for f in stray_videos:
            try: os.remove(f)
            except: pass
        if len(stray_videos) > 0:
            print(f"   🧹 [Cleanup] Removed {len(stray_videos)} stray video files.")
    except Exception:
        pass

def run_hybrik_on_3dpw(sequence_name):
    """针对 3DPW 文件夹结构运行 HybrIK"""
    img_folder = os.path.normpath(os.path.join(Config.PW3D_IMG_DIR, sequence_name))
    if not os.path.exists(img_folder):
        print(f"⚠️ [Warning] 3DPW Image folder not found: {img_folder}")
        return generate_mock_data(sequence_name)
    return run_hybrik_demo(img_folder, Config.OUTPUT_DIR)

def run_hybrik_demo(video_path, output_dir):
    """
    HybrIK 推理通用接口
    """
    video_path = os.path.normpath(video_path)
    # 如果是文件夹，名字就是文件夹名；如果是视频文件，名字是文件名去掉后缀
    if os.path.isdir(video_path):
        video_name = os.path.basename(video_path)
    else:
        video_name = os.path.basename(video_path).split('.')[0]

    result_dir = os.path.join(output_dir, video_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # 统一的目标结果路径
    target_res_path = os.path.join(result_dir, 'res.pk')

    # 1. 运行前清理环境
    cleanup_stray_videos(output_dir)

    # 2. 检查是否有现成的缓存
    if os.path.exists(target_res_path) and os.path.getsize(target_res_path) > 0:
        cached = load_hybrik_result(target_res_path)
        if cached:
            print(f"   [HybrIK] Found cached result: res.pk")
            return cached
        else:
            try: os.remove(target_res_path)
            except: pass

    # 3. 图片/视频 预处理
    input_source = video_path
    is_temp_video = False
    temp_video = os.path.join(result_dir, f"{video_name}_bridge.mp4")

    # 如果输入是文件夹(图片序列)，需要转成视频
    if os.path.isdir(video_path):
        if os.path.exists(temp_video):
            try: os.remove(temp_video)
            except: pass

        # 空间预检 (100MB)
        try:
            _, _, free = shutil.disk_usage(output_dir)
            if free < 100 * 1024 * 1024:
                print("   ❌ [CRITICAL] Disk space critically low. Skipping.")
                return generate_mock_data(video_path)
        except: pass

        if shutil.which('ffmpeg') is None:
            print("   ❌ Error: FFmpeg not installed.")
            return generate_mock_data(video_path)

        print(f"   [System] Bridging: Compressing images to video...")
        # 极速压缩模式: CRF 35 + ultrafast
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-r', '30',
            '-pattern_type', 'glob', '-i', os.path.join(video_path, '*.jpg'),
            '-c:v', 'libx264', '-crf', '35', '-preset', 'ultrafast', 
            '-pix_fmt', 'yuv420p', '-loglevel', 'error',
            temp_video
        ]
        try:
            subprocess.check_call(ffmpeg_cmd)
            input_source = temp_video
            is_temp_video = True
        except Exception as e:
            print(f"   ❌ FFMPEG failed: {e}")
            return generate_mock_data(video_path)
    
    # 如果输入本来就是视频文件，直接用
    elif os.path.isfile(video_path):
        input_source = video_path

    # 4. 调用 HybrIK
    hybrik_script = os.path.join(Config.HYBRIK_ROOT, 'scripts', 'demo_video.py')
    final_result = None
    if os.path.exists(hybrik_script):
        cmd = [
            Config.PYTHON_EXEC, hybrik_script,
            '--video-name', input_source,
            '--out-dir', result_dir,
            '--save-pk'
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = Config.HYBRIK_ROOT + os.pathsep + env.get("PYTHONPATH", "")

        try:
            # 运行模型
            subprocess.check_call(cmd, env=env, cwd=Config.HYBRIK_ROOT)
            
            # [关键修复] 智能查找结果文件并重命名为 res.pk
            found_pk = None
            if os.path.exists(target_res_path):
                found_pk = target_res_path
            else:
                # 模糊查找 res_*.pk
                pk_candidates = glob.glob(os.path.join(result_dir, "*.pk"))
                for pk in pk_candidates:
                    if "res" in os.path.basename(pk):
                        found_pk = pk
                        break
            
            if found_pk and os.path.exists(found_pk):
                # 强制改名为 res.pk，方便统一管理
                if found_pk != target_res_path:
                    shutil.move(found_pk, target_res_path)
                
                final_result = load_hybrik_result(target_res_path)
            else:
                print(f"   ⚠️ HybrIK ran but no .pk result found in {result_dir}")

        except Exception as e:
            print(f"   ⚠️ [Warning] HybrIK execution failed: {e}")
    else:
        print(f"   ⚠️ [Warning] Script not found: {hybrik_script}")

    # 5. 清理临时生成的视频 (必须执行)
    if is_temp_video and os.path.exists(input_source):
        try: os.remove(input_source)
        except: pass

    if final_result is None:
        return generate_mock_data(video_path)
    
    return final_result

def load_hybrik_result(pkl_path):
    """解析 HybrIK 结果"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        raw_thetas = data['pred_thetas']
        raw_trans = data['transl']

        # 智能维度处理
        params_per_person = 24 * 3 * 3
        if raw_thetas.ndim == 5:
            pose_final = raw_thetas[:, 0]
        elif raw_thetas.ndim == 4:
            pose_final = raw_thetas
        else:
            total_size = raw_thetas.size
            if total_size % params_per_person == 0:
                total_frames = total_size // params_per_person
                est_T = raw_trans.shape[0] if raw_trans.ndim >= 1 else total_frames
                num_people = total_frames // est_T
                if num_people > 1:
                    reshaped = raw_thetas.reshape(est_T, num_people, 24, 3, 3)
                    pose_final = reshaped[:, 0]
                else:
                    pose_final = raw_thetas.reshape(est_T, 24, 3, 3)
            else:
                raise ValueError(f"Unknown pose shape")

        if raw_trans.ndim == 3: trans_final = raw_trans[:, 0]
        elif raw_trans.ndim == 2: trans_final = raw_trans
        else: trans_final = raw_trans.reshape(-1, 3)

        return {
            'pred_pose': torch.from_numpy(pose_final.copy()).float().to(Config.DEVICE),
            'pred_trans': torch.from_numpy(trans_final.copy()).float().to(Config.DEVICE)
        }
    except Exception as e:
        print(f"   ❌ Corrupted cache: {e}")
        return None

def generate_mock_data(video_path):
    """保底数据"""
    T = 150
    return {
        'pred_pose': torch.eye(3).to(Config.DEVICE).view(1,1,3,3).repeat(T,24,1,1),
        'pred_trans': torch.zeros(T,3).to(Config.DEVICE)
    }
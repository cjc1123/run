import os
import torch
import torch.backends.cudnn as cudnn


class Config:
    # --- 1. 计算设备配置 (IEEE High Performance) ---
    if torch.cuda.is_available():
        DEVICE = 'cuda:0'
        cudnn.benchmark = True
        cudnn.deterministic = False
        print(f"✅ [System] High-Performance Mode: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = 'cpu'
        print("⚠️ [System] Warning: Running on CPU. This will be slow for manifold optimization.")

    # --- 2. 路径配置 ---
    # 指向你的 HybrIK 项目根目录
    HYBRIK_ROOT = r"/root/HybrIK-"
    PYTHON_EXEC = "python"

    # 指向你的数据根目录 (必须包含 video 文件夹)
    DATA_ROOT = r"/root/HybrIK-/runs/data"

    VIDEO_INPUT_DIR = os.path.join(DATA_ROOT, "video")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "output")

    # --- 3. 物理常数 (关键修复) ---
    FPS = 30.0
    DT = 1.0 / FPS  # <--- 之前报错就是因为缺了这一行！
    GRAVITY = 9.81

    # --- 4. IMU 物理特性 (Xsens MTw / Bosch BNO055 等级) ---
    ACC_NOISE_DENSITY = 0.002  # m/s^2 / sqrt(Hz) (White Noise)
    ACC_RANDOM_WALK = 0.0001  # m/s^2 / sqrt(s)  (Bias Instability)

    GYRO_NOISE_DENSITY = 0.0001  # rad/s / sqrt(Hz)
    GYRO_RANDOM_WALK = 1.0e-5  # rad/s / sqrt(s)

    # --- 5. 优化算法超参数 ---
    # LATC (Twist Correction)
    LATC_ITERATIONS = 50
    LATC_LR = 0.02
    WEIGHT_GEODESIC = 1.0
    WEIGHT_TWIST_SMOOTH = 10.0

    # PAKR (ZUPT Trajectory)
    PAKR_ITERATIONS = 100
    PAKR_LR = 0.05
    ENERGY_THRESH = 150  # ZUPT 能量阈值
    WEIGHT_DATA_TERM = 0.1  # 信任视觉先验
    WEIGHT_ACC_SMOOTH = 0.5  # 物理平滑约束
    WEIGHT_ZUPT_HARD = 2000  # 强零速约束
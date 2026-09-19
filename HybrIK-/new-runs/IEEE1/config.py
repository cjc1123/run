import os
import torch
import torch.backends.cudnn as cudnn


class Config:
    # --- 1. 计算设备配置 (IEEE High Performance Mode) ---
    if torch.cuda.is_available():
        DEVICE = 'cuda:0'
        # 针对固定输入的卷积和矩阵运算开启加速
        cudnn.benchmark = True
        cudnn.deterministic = False
        print(f"✅ [System] High-Performance GPU Mode: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = 'cpu'
        print("⚠️ [System] Warning: Running on CPU. This will significantly impact LNN-ODE solving speed.")

    # --- 2. 路径与模型管理 ---
    HYBRIK_ROOT = r"/root/HybrIK-"
    PYTHON_EXEC = "python"
    DATA_ROOT = "/root/autodl-tmp/data"

    # --- [新增] 3DPW 数据集路径配置 ---
    PW3D_ROOT = os.path.join(DATA_ROOT, "3DPW_Dataset")
    PW3D_SEQ_DIR = os.path.join(PW3D_ROOT, "sequenceFiles")
    PW3D_IMG_DIR = os.path.join(PW3D_ROOT, "imageFiles")

    # [新增] 3DPW IMU 对应关系 (SMPL 17个传感器索引)
    IMU_IDX_ROOT = 0  # 躯干/质心
    IMU_IDX_ANKLE_R = 0  # 右脚踝 (用于步态识别训练)

    # 结果输出与权重存储
    VIDEO_INPUT_DIR = os.path.join(DATA_ROOT, "video")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "output")
    MODEL_SAVE_DIR = os.path.join(DATA_ROOT, "models")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 液态神经网络预训练权重路径
    LNN_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "lnn_gait_observer.pth")

    # --- 3. 环境与物理动力学常数 ---
    FPS = 30.0
    DT = 1.0 / FPS
    GRAVITY = 9.80665  # 采用标准重力加速度 (Standard Gravity), 投稿 IEEE 必备的精确性

    # --- 4. IMU 物理特性 (Xsens MTw / Bosch BNO055 工业级规格) ---
    # 这些参数用于 PhysicsIMUSimulator 生成高质量合成数据
    ACC_NOISE_DENSITY = 0.002  # m/s^2 / sqrt(Hz) (高频高斯噪声)
    ACC_RANDOM_WALK = 0.0001  # m/s^2 / sqrt(s)  (零偏不稳定性/随机游走)
    GYRO_NOISE_DENSITY = 0.0001  # rad/s / sqrt(Hz)
    GYRO_RANDOM_WALK = 1.0e-5  # rad/s / sqrt(s)

    # --- 5. 液态神经网络 (LNN) 核心超参数 ---
    # 论文点：基于 LTC (Liquid Time-Constant) 的动力学模型
    LNN_INPUT_DIM = 6  # 输入维度: Acc(3) + Gyro(3)
    LNN_HIDDEN_DIM = 64  # 隐藏层神经元数量
    LNN_TAU_INIT = 0.5  # 时间常数初始值 (系统响应速度)
    LNN_LEARNING_RATE = 1e-3

    # --- 6. 优化算法超参数 (Manifold Optimization) ---

    # LATC (Lie-Algebraic Twist Correction) - 处理 SO(3) 轴向扭转
    LATC_ITERATIONS = 50
    LATC_LR = 0.02
    WEIGHT_GEODESIC = 1.0
    WEIGHT_TWIST_SMOOTH = 15.0  # 稍微增大平滑权重以抑制 LNN 推理初期的波动

    # PAKR (Phase-Adaptive Kinematic Refinement) - 轨迹细化
    PAKR_ITERATIONS = 150  # 增加迭代次数以保证 ODE 约束下收敛
    PAKR_LR = 0.05

    # 核心改进：取消硬性的 ENERGY_THRESH，改用 LNN 概率加权
    # 论文中称之为 "Probability-weighted Zero-Velocity Update (P-ZUPT)"
    WEIGHT_DATA_TERM = 0.1  # 相信视觉全局参考的比例 (Confidence in Vision)
    WEIGHT_ACC_SMOOTH = 1.0  # 物理加速度平滑约束 (Physics Consistency)
    WEIGHT_ZUPT_LNN = 5000.0  # LNN 引导的动态零速约束强度 (越高则足部滑动抑制越强)

    # 备用：用于 LNN 离散化求解的步长 (隐式欧拉法)
    ODE_SOLVER_STEPS = 2  # 每个 DT 内进行多少次内部积分
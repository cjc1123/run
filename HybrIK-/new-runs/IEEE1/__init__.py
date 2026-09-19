"""
IEEE Sensors Journal - IC-HybrIK Core Package
---------------------------------------------
Implementation of Tightly-Coupled Visual-Inertial Manifold Optimization.

Modules:
- geometry: Lie Algebra (SO3) operations with Taylor expansion for numerical stability.
- physics_sim: MEMS IMU simulation including Bias Instability and White Noise.
- solver: Two-stage optimization (LATC & PAKR) on the manifold.
"""

__version__ = '1.0.0'

# 从当前包导入核心组件
from .config import Config
from .physics_sim import PhysicsIMUSimulator
from .solve_liquid_pakr import ManifoldSolver
from .liquid_network import LiquidGaitObserver
from .data_bridge import DataBridge
from .hybrik_wrapper import run_hybrik_demo, run_hybrik_on_3dpw, generate_mock_data
from .geometry import (
    so3_exp_map,
    so3_log_map,
    geodesic_distance,
    skew_symmetric,
    batch_rodrigues # 确保此函数也被导出，部分模块可能需要
)

# 定义导出列表，保持命名空间整洁
__all__ = [
    'Config',
    'PhysicsIMUSimulator',
    'ManifoldSolver',
    'LiquidGaitObserver',
    'DataBridge',
    'run_hybrik_demo',
    'run_hybrik_on_3dpw',
    'generate_mock_data',
    'so3_exp_map',
    'so3_log_map',
    'geodesic_distance',
    'skew_symmetric',
    'batch_rodrigues'
]
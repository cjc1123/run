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
from .solver import ManifoldSolver
from .hybrik_wrapper import run_hybrik_demo
from .geometry import (
    so3_exp_map,
    so3_log_map,
    geodesic_distance,
    skew_symmetric
)

# 定义导出列表，保持命名空间整洁
__all__ = [
    'Config',
    'PhysicsIMUSimulator',
    'ManifoldSolver',
    'run_hybrik_demo',
    'so3_exp_map',
    'so3_log_map',
    'geodesic_distance',
    'skew_symmetric'
]
"""
DPG_track_ball.py

版本说明:
    - 该模块专门封装 DYNAMIC PREDICTION GRASPING 项目中的目标轨迹逻辑。
    - 本文件提供「小球 mocap 沿 x 轴移动」的轨迹生成器，可直接用于 MuJoCo scene 里的目标 body。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import mujoco
import numpy as np


class TrajectoryProvider(Protocol):
    """统一的轨迹接口，方便主循环替换不同的参考信号。"""

    def position(self, sim_time: float) -> np.ndarray:
        """返回当前时刻（仿真时间）的目标位置。"""

    def future_positions(self, sim_time: float, horizon: int, dt: float) -> np.ndarray:
        """返回未来 N 步的目标位置（从 t+dt 开始），shape=(N,3)。"""


@dataclass
class BaseTrajectory:
    """可选的基类，默认 future_positions 通过多次调用 position 构造。"""

    def position(self, sim_time: float) -> np.ndarray:  # pragma: no cover - 接口定义
        raise NotImplementedError

    def future_positions(self, sim_time: float, horizon: int, dt: float) -> np.ndarray:
        future = np.zeros((horizon, 3))
        for i in range(horizon):
            future[i] = self.position(sim_time + (i + 1) * dt)
        return future


@dataclass
class MocapBallTrajectory(BaseTrajectory):
    """
    小球 mocap 轨迹，输出为机械臂/底盘坐标系：
        - 世界系沿 x 轴做单向 S 曲线运动；
        - 返回的坐标 = 世界系坐标 - base_origin（底盘/机械臂基座标系），方便直接用于控制/绘图。
    """

    center: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.5, 1.2], dtype=float)
    )
    amplitude: float = 0.6
    extension_margin: float = 0.3
    period: float = 6.0
    base_origin: np.ndarray = field(
        default_factory=lambda: mujoco.MjModel.from_xml_path(
            "fetch_freight_mujoco/xml/scene.xml"
        ).body_pos[
            mujoco.mj_name2id(
                mujoco.MjModel.from_xml_path("fetch_freight_mujoco/xml/scene.xml"),
                mujoco.mjtObj.mjOBJ_BODY,
                "chassis",
            )
        ].copy()
    )

    def position(self, sim_time: float) -> np.ndarray:
        phase = min(sim_time / self.period, 1.0)
        progress = 0.5 * (1.0 - np.cos(np.pi * phase))
        x_start = self.center[0] - self.amplitude
        x_end = self.center[0] + self.amplitude + self.extension_margin
        x_pos = x_start + (x_end - x_start) * progress
        pos = self.center.copy()
        pos[0] = x_pos
        return pos - self.base_origin


def get_trajectory(object_track: str = "ball", **kwargs) -> TrajectoryProvider:
    """
    简单的轨迹选择入口:
        object_track="ball" -> MocapBallTrajectory（输出为底盘系）
        object_track="ball_in_robot" -> DPG_track_ball_in_robot 中的小球相对轨迹
    其他值会抛出异常。
    """
    name = object_track.lower()
    if name == "ball":
        return MocapBallTrajectory(**kwargs)
    if name == "ball_in_robot":
        from DPG_track_ball_in_robot import get_ball_in_robot_trajectory

        return get_ball_in_robot_trajectory(**kwargs)
    raise ValueError(f"未知轨迹类型: {object_track}")


__all__ = [
    "TrajectoryProvider",
    "BaseTrajectory",
    "MocapBallTrajectory",
    "get_trajectory",
]

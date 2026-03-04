"""
DPG_track_ball_in_robot.py

用途:
    - 在“小车移动、小球固定”的场景下，解析计算小球在机械臂/底盘坐标系下的轨迹。
    - 先在世界系生成底盘轨迹，再做世界系→底盘系的坐标变换，输出给控制/绘图使用。
接口:
    - BallInRobotFrameTrajectory: TrajectoryProvider，position(t) 返回 ball 在底盘系的 3D 坐标。
    - get_ball_in_robot_trajectory(**kwargs): 生成默认实例，便于 main/plot 调用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from DPG_track_ball import BaseTrajectory, TrajectoryProvider


@dataclass
class LinearBaseTrajectory(BaseTrajectory):
    """底盘直线轨迹（世界系），用于生成相对坐标。"""

    start: np.ndarray = field(default_factory=lambda: np.array([-0.8, 0.0, 0.0], dtype=float))
    end: np.ndarray = field(default_factory=lambda: np.array([0.50, 0.0, 0.0], dtype=float))
    speed: float = 0.3

    def __post_init__(self):
        # dataclass 默认值换成实际 ndarray，避免共享引用
        self.start = np.array(self.start, dtype=float).reshape(3)
        self.end = np.array(self.end, dtype=float).reshape(3)

    def _delta(self) -> np.ndarray:
        return self.end - self.start

    def length(self) -> float:
        return float(np.linalg.norm(self._delta()))

    def duration(self) -> float:
        return self.length() / max(float(self.speed), 1e-6)

    def position(self, sim_time: float) -> np.ndarray:
        total_len = self.length()
        if total_len < 1e-9:
            return self.start.copy()
        progress = min(max(sim_time, 0.0) * self.speed / total_len, 1.0)
        return self.start + self._delta() * progress


@dataclass
class BallInRobotFrameTrajectory(BaseTrajectory):
    """
    小球在机械臂/底盘坐标系下的相对轨迹：
        - 底盘沿 LinearBaseTrajectory 在世界系移动。
        - 小球世界坐标固定为 ball_world。
        - 输出 = ball_world - base_pos。
    """

    base_trajectory: BaseTrajectory
    ball_world: np.ndarray

    def __post_init__(self):
        self.ball_world = np.array(self.ball_world, dtype=float).reshape(3)

    def position(self, sim_time: float) -> np.ndarray:
        base_pos = self.base_trajectory.position(sim_time)
        return self.ball_world - base_pos


def get_ball_in_robot_trajectory(
    base_traj: Optional[TrajectoryProvider] = None, ball_world=None
) -> TrajectoryProvider:
    """
    生成小球在底盘系下的轨迹（基于解析计算）。
    参数:
        base_traj: 可传入自定义 BaseTrajectory 覆盖默认（默认使用 x:-0.8->0.5, speed=0.3 的直线）。
        ball_world: 小球世界坐标，可选覆盖默认 [0.25, 0.5, 1.2].
    """
    if base_traj is None:
        base_traj = LinearBaseTrajectory()
    bw = ball_world if ball_world is not None else np.array([0.25, 0.5, 1.2], dtype=float)
    return BallInRobotFrameTrajectory(base_traj, bw)


__all__ = ["BallInRobotFrameTrajectory", "get_ball_in_robot_trajectory", "LinearBaseTrajectory"]

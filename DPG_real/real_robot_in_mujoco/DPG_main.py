"""
DPG_main.py

简洁入口：
    - 选择轨迹（"ball" / "ball_in_robot"），默认为机械臂基坐标系下的轨迹。
    - 调用 MPCController 完成仿真：轨迹→世界坐标→MPC→雅可比分解→关节角目标（position 伺服）。
"""

from __future__ import annotations

from DPG_MPC import MPCController
from DPG_track_ball import get_trajectory
from DPG_track_ball_in_robot import get_ball_in_robot_trajectory
from real_base_udp_kf import RealTimeBaseTrajectory


def build_trajectory(name: str, use_real_base: bool = False):
    if name == "ball_in_robot":
        if use_real_base:
            base_traj = RealTimeBaseTrajectory()
            return get_ball_in_robot_trajectory(base_traj=base_traj)
        return get_ball_in_robot_trajectory()
    return get_trajectory("ball")


if __name__ == "__main__":
    # 只改这个参数即可切换轨迹: "ball" / "ball_in_robot"
    OBJECT_TRACK = "ball_in_robot"
    USE_REAL_BASE = True

    traj = build_trajectory(OBJECT_TRACK, use_real_base=USE_REAL_BASE)
    MPCController(trajectory=traj).run()

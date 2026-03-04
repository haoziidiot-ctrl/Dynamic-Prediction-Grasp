"""
DPG_main.py

简洁入口：
    - 选择轨迹（"ball" / "ball_in_robot"），默认为机械臂基坐标系下的轨迹。
    - 调用 MPCController 完成仿真：轨迹→世界坐标→MPC→雅可比分解→关节角目标（position 伺服）。
    - 启用漏斗约束：目标前方禁区仅保留中间通道，逼迫从指定通道接近。
"""

from __future__ import annotations

from DPG_MPC import MPCController
from DPG_track_ball import get_trajectory
from DPG_track_ball_in_robot import get_ball_in_robot_trajectory


def build_trajectory(name: str, kf_cfg=None):
    if name == "ball_in_robot":
        return get_ball_in_robot_trajectory(kf_cfg=kf_cfg)
    return get_trajectory("ball")


if __name__ == "__main__":
    # 只改这个参数即可切换轨迹: "ball" / "ball_in_robot"
    OBJECT_TRACK = "ball_in_robot"
    KF_CFG = {
        "poll_period_s": 0.01,
        "sigma_a": 0.08,
        "meas_noise": 0.005,
        "use_measurement_noise": True,
        "seed": 0,
    }
    # 约束版本默认关闭 RL 终端价值，专注约束接近策略
    USE_TERMINAL_VALUE = False
    TERMINAL_VALUE_DIM = 3

    # 漏斗约束（相对目标 target_world）：
    # y ∈ [y_t - 0.10, y_t] 且 x 在 [x_t - 0.05, x_t + 0.05] 之外为禁区
    ENABLE_FUNNEL_CONSTRAINT = True
    FUNNEL_DEPTH = 0.10
    FUNNEL_HALF_WIDTH = 0.05
    FUNNEL_MARGIN = 1e-3
    VISUALIZE_FUNNEL_ZONE = True
    FUNNEL_VIS_X_EXTENT = 1.2
    FUNNEL_VIS_Z_HALF = 1.0
    FUNNEL_VIS_RGBA = (1.0, 0.95, 0.45, 0.22)  # 淡黄色透明
    ENABLE_GRASP = True
    GRASP_TOL = 0.02
    GRASP_HOLD_STEPS = 10
    GRASP_HOLD_TIME_S = 1.0
    GRASP_ACTION = "none"  # "none" / "stop" / "attach"
    traj = build_trajectory(OBJECT_TRACK, kf_cfg=KF_CFG)
    MPCController(
        trajectory=traj,
        use_terminal_value=USE_TERMINAL_VALUE,
        terminal_value_dim=TERMINAL_VALUE_DIM,
        enable_funnel_constraint=ENABLE_FUNNEL_CONSTRAINT,
        funnel_depth=FUNNEL_DEPTH,
        funnel_half_width=FUNNEL_HALF_WIDTH,
        funnel_margin=FUNNEL_MARGIN,
        visualize_funnel_zone=VISUALIZE_FUNNEL_ZONE,
        funnel_vis_x_extent=FUNNEL_VIS_X_EXTENT,
        funnel_vis_z_half=FUNNEL_VIS_Z_HALF,
        funnel_vis_rgba=FUNNEL_VIS_RGBA,
        enable_grasp=ENABLE_GRASP,
        grasp_tol=GRASP_TOL,
        grasp_hold_steps=GRASP_HOLD_STEPS,
        grasp_hold_time_s=GRASP_HOLD_TIME_S,
        grasp_action=GRASP_ACTION,
    ).run()

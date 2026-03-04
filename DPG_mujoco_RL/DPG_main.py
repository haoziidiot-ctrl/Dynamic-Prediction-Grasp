"""
DPG_main.py

简洁入口：
    - 选择轨迹（"ball" / "ball_in_robot"），默认为机械臂基坐标系下的轨迹。
    - 调用 MPCController 完成仿真：轨迹→世界坐标→MPC→雅可比分解→关节角目标（position 伺服）。
    - 支持 RL 终端代价（Terminal Value）作为 MPC 终端项。
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
    # 如有训练好的价值网络，填入路径；为空则回退到二次型终端代价
    VALUE_MODEL_PATH = None
    TERMINAL_VALUE_SCALE = 0.2
    TERMINAL_FALLBACK_WEIGHT = 0.0
    TERMINAL_VALUE_DIM = 15  # [target_N(3), end_N(3), base_vel(3), q_current(6)]
    TERMINAL_ROT_SCALE = 1.0
    TERMINAL_SING_SCALE = 0.2
    TERMINAL_APPROACH_AXIS = "y"
    TERMINAL_APPROACH_DIR = None
    ENABLE_GRASP = True
    GRASP_TOL = 0.02
    GRASP_HOLD_STEPS = 10
    GRASP_HOLD_TIME_S = 1.0
    GRASP_ACTION = "none"  # "none" / "stop" / "attach"
    traj = build_trajectory(OBJECT_TRACK, kf_cfg=KF_CFG)
    MPCController(
        trajectory=traj,
        use_terminal_value=True,
        terminal_value_path=VALUE_MODEL_PATH,
        terminal_value_scale=TERMINAL_VALUE_SCALE,
        terminal_value_fallback=TERMINAL_FALLBACK_WEIGHT,
        terminal_value_dim=TERMINAL_VALUE_DIM,
        terminal_rot_scale=TERMINAL_ROT_SCALE,
        terminal_sing_scale=TERMINAL_SING_SCALE,
        terminal_approach_axis=TERMINAL_APPROACH_AXIS,
        terminal_approach_dir=TERMINAL_APPROACH_DIR,
        enable_grasp=ENABLE_GRASP,
        grasp_tol=GRASP_TOL,
        grasp_hold_steps=GRASP_HOLD_STEPS,
        grasp_hold_time_s=GRASP_HOLD_TIME_S,
        grasp_action=GRASP_ACTION,
    ).run()

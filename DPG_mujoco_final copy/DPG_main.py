"""
DPG_main.py

简洁入口：
    - 选择轨迹（"ball" / "ball_in_robot"），默认为机械臂基坐标系下的轨迹。
    - 调用 MPCController 完成仿真：轨迹→世界坐标→MPC→雅可比分解→关节角目标（position 伺服）。
    - 两阶段参考：先跟踪带 y 偏置的参考轨迹，满足阈值后切换为原始目标轨迹。
"""

from __future__ import annotations

import numpy as np

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
        # 随机 MPC 的辅助闭环协方差传播增益，由底盘 CV 模型的一维 LQR 自动生成
        "cov_feedback_q_pos": 20.0,
        "cov_feedback_q_vel": 2.0,
        "cov_feedback_r": 1.0,
        "seed": 0,
    }
    # 约束版本默认关闭 RL 终端价值，专注约束接近策略
    USE_TERMINAL_VALUE = False
    TERMINAL_VALUE_DIM = 3
    # 全程姿态参考固定朝向世界 +Y；位置项仍偏强，但不再像上一版那样过激
    POS_WEIGHT = 26.0
    ROT_WEIGHT = 0.10
    SMOOTH_WEIGHT = 1.5e-3
    TARGET_FACING_DIR = (0.0, 1.0, 0.0)
    # GUI 性能参数：60Hz 渲染 + 0.1s 一次简化日志
    RENDER_DT = 1.0 / 60.0
    PROFILE_PERIOD = 0.1

    # 约束MPC: 使用 OSQP 求解带硬约束的QP
    USE_CONSTRAINED_QP = True
    QP_SOLVER = "osqp"
    QP_INFEASIBLE_POLICY = "hold"
    QP_ENFORCE_JOINT_POS = True
    QP_ENFORCE_JOINT_VEL = True
    QP_ENFORCE_EE_X_UPPER = True
    QP_EE_X_MARGIN = 0.02
    QP_ENFORCE_EE_Y_UPPER = True
    QP_EE_Y_MARGIN = 0.0
    # 随机 MPC 工程实现：
    # 1) KF 协方差自适应缩放位置代价
    # 2) KF 协方差驱动 ee 约束动态收紧（chance-constraint backoff）
    # 3) 预留 tube 式执行层补偿接口。当前这套每拍重解的 receding-horizon 结构里，
    #    直接对“当前位置误差”再叠一层反馈会和 MPC 主反馈打架，默认关闭。
    USE_STOCHASTIC_MPC = True
    STOCHASTIC_RISK_ALPHA = 0.05
    STOCHASTIC_SIGMA_SCALE = 3.0
    STOCHASTIC_BACKOFF_MAX = 0.01
    STOCHASTIC_RELEASE_SLOWDOWN = 1.0
    STOCHASTIC_FF_SUPPRESSION = 18.0
    STOCHASTIC_ATTACH_POS_BOOST = 0.35
    STOCHASTIC_ATTACH_X_BOOST = 0.50
    STOCHASTIC_ATTACH_ROT_BOOST = 0.08
    STOCHASTIC_USE_TUBE_FEEDBACK = False
    STOCHASTIC_TUBE_GAIN_XY = 0.28
    STOCHASTIC_TUBE_GAIN_Z = 0.18
    STOCHASTIC_TUBE_MAX_LINEAR = 0.18

    # 两阶段切换：
    # 1) 第一阶段跟踪 target + [0, OFFSET_Y, 0]
    # 2) 当末端 x 到达 (target_x - OFFSET_SWITCH_X_FRONT) 后，
    #    连续命中 OFFSET_SWITCH_STEPS 个控制步即切到第二阶段（去掉偏置，直接跟踪原目标轨迹）
    ENABLE_PREGRASP = False
    PREGRASP_OFFSET = 0.0
    PREGRASP_DIR = (0.0, 1.0, 0.0)
    APPROACH_SPEED = 0.35
    ENABLE_PREDICTIVE_PHASE_SWITCH = False
    PHASE_USE_X_GATE_SWITCH = False
    PHASE_X_GATE_HALF_WIDTH = 0.03
    PHASE_X_GATE_HOLD_STEPS = 10
    PHASE_INSTANT_ATTACK = True
    PHASE_OPT_RADIUS_MIN = 0.40
    PHASE_OPT_RADIUS_MAX = 0.65
    PHASE_TRIGGER_INDEX = 6
    PHASE_CONFIRM_STEPS = 3
    PHASE_MIN_HOLD_S = 0.0
    PHASE_USE_PLANAR_DISTANCE = True

    USE_OFFSET_TRACKING = True
    OFFSET_Y = -0.13
    OFFSET_RELEASE_TIME_S = 0.25
    HOLD_POS_WEIGHT_SCALE = 1.0
    ATTACH_POS_WEIGHT_SCALE = 1.45
    HOLD_X_ERROR_GAIN = 1.0
    ATTACH_X_ERROR_GAIN = 1.35
    HOLD_ORIENTATION_GAIN = 1.0
    ATTACH_ORIENTATION_GAIN = 0.28
    OFFSET_SWITCH_TOL = 0.02  # 兼容保留
    OFFSET_SWITCH_STEPS = 8
    OFFSET_SWITCH_X_GATE_ENABLE = True
    OFFSET_SWITCH_X_FRONT = 0.20
    OFFSET_SWITCH_X_ALIGN_TOL = 0.04
    OFFSET_SWITCH_YZ_TOL = 0.09
    WARM_START_MAX = 0.0

    # 创新点1：不确定性感知自适应 MPC（KF 协方差 -> 误差项权重）
    ENABLE_UNCERTAINTY_AWARE = True
    UNCERTAINTY_BETA = 60.0
    UNCERTAINTY_MIN_SCALE = 0.85
    UNCERTAINTY_EMA = 0.15

    # 创新点3：操作度梯度引导（低操作度时给 MPC 一个远离奇异位的任务空间偏置）
    # 当前目标是更快进入 success，先关闭会引入末端偏置的操作度引导
    ENABLE_MANIP_GUIDANCE = False
    MANIP_LAMBDA = 0.025
    MANIP_W_THRESHOLD = 0.06
    MANIP_FD_DELTA = 0.004
    MANIP_GRAD_CLIP = 2.0
    MANIP_HORIZON_DECAY = 0.8
    MANIP_FIRST_STEP_ONLY = False

    # attach 阶段保留一定底盘运动补偿，但把动作幅度收回来
    BASE_FF_GAIN = 1.0
    EE_LINEAR_SPEED_LIMIT = 1.0

    ENABLE_GRASP = True
    # grasp success 判定：end_finger 到真实 target 距离需小于 0.02 m
    GRASP_TOL = 0.02
    GRASP_HOLD_STEPS = 10
    GRASP_HOLD_TIME_S = None  # 用 GRASP_HOLD_STEPS(10步)判定成功，避免1s停留导致错失动态抓取窗口
    GRASP_ACTION = "none"  # "none" / "stop" / "attach"
    traj = build_trajectory(OBJECT_TRACK, kf_cfg=KF_CFG)
    if hasattr(traj, "ball_world"):
        print(f"[target] ball_world(from trajectory) = {np.asarray(traj.ball_world, dtype=float)}")
        legacy = np.array([0.25, 0.5, 1.2], dtype=float)
        delta = np.asarray(traj.ball_world, dtype=float) - legacy
        print(f"[target] delta vs legacy [0.25,0.5,1.2] = {delta}")
    base = getattr(traj, "base_trajectory", None)
    base_src = getattr(base, "base_trajectory", base)
    if base_src is not None:
        lat_mean = getattr(base_src, "lateral_noise_mag_mean", 0.0)
        lat_std = getattr(base_src, "lateral_noise_mag_std", 0.0)
        lat_dt = getattr(base_src, "lateral_noise_dt", 0.0)
        print(
            f"[base traj] x linear + y step disturbance: mag_mean={lat_mean:.3f} m, "
            f"mag_std={lat_std:.3f} m, dt={lat_dt:.3f} s"
        )
    MPCController(
        trajectory=traj,
        warm_start_max=WARM_START_MAX,
        pos_weight=POS_WEIGHT,
        rot_weight=ROT_WEIGHT,
        smooth_weight=SMOOTH_WEIGHT,
        render_dt=RENDER_DT,
        profile_period=PROFILE_PERIOD,
        use_terminal_value=USE_TERMINAL_VALUE,
        terminal_value_dim=TERMINAL_VALUE_DIM,
        terminal_approach_dir=TARGET_FACING_DIR,
        # 由当前 MJCF 姿态可得：gripper/end_finger 局部 x 轴在初始时指向世界 +Z。
        # 因此若要“夹爪轴线”最终朝向世界 +Y，就应约束局部 x 轴 -> 世界 +Y。
        terminal_approach_axis="x",
        use_pregrasp=ENABLE_PREGRASP,
        pregrasp_offset=PREGRASP_OFFSET,
        pregrasp_dir=PREGRASP_DIR,
        approach_speed=APPROACH_SPEED,
        use_predictive_phase_switch=ENABLE_PREDICTIVE_PHASE_SWITCH,
        phase_opt_radius_min=PHASE_OPT_RADIUS_MIN,
        phase_opt_radius_max=PHASE_OPT_RADIUS_MAX,
        phase_trigger_index=PHASE_TRIGGER_INDEX,
        phase_confirm_steps=PHASE_CONFIRM_STEPS,
        phase_min_hold_s=PHASE_MIN_HOLD_S,
        phase_use_planar_distance=PHASE_USE_PLANAR_DISTANCE,
        phase_use_x_gate_switch=PHASE_USE_X_GATE_SWITCH,
        phase_x_gate_half_width=PHASE_X_GATE_HALF_WIDTH,
        phase_x_gate_hold_steps=PHASE_X_GATE_HOLD_STEPS,
        phase_instant_attack=PHASE_INSTANT_ATTACK,
        use_offset_tracking=USE_OFFSET_TRACKING,
        offset_y=OFFSET_Y,
        offset_release_time_s=OFFSET_RELEASE_TIME_S,
        hold_pos_weight_scale=HOLD_POS_WEIGHT_SCALE,
        attach_pos_weight_scale=ATTACH_POS_WEIGHT_SCALE,
        hold_x_error_gain=HOLD_X_ERROR_GAIN,
        attach_x_error_gain=ATTACH_X_ERROR_GAIN,
        hold_orientation_gain=HOLD_ORIENTATION_GAIN,
        attach_orientation_gain=ATTACH_ORIENTATION_GAIN,
        offset_trigger_tol=OFFSET_SWITCH_TOL,
        offset_trigger_steps=OFFSET_SWITCH_STEPS,
        offset_switch_x_gate_enable=OFFSET_SWITCH_X_GATE_ENABLE,
        offset_switch_x_front=OFFSET_SWITCH_X_FRONT,
        offset_switch_x_align_tol=OFFSET_SWITCH_X_ALIGN_TOL,
        offset_switch_yz_tol=OFFSET_SWITCH_YZ_TOL,
        use_uncertainty_aware_weighting=ENABLE_UNCERTAINTY_AWARE,
        uncertainty_beta=UNCERTAINTY_BETA,
        uncertainty_min_scale=UNCERTAINTY_MIN_SCALE,
        uncertainty_ema=UNCERTAINTY_EMA,
        use_manipulability_guidance=ENABLE_MANIP_GUIDANCE,
        manipulability_lambda=MANIP_LAMBDA,
        manipulability_w_threshold=MANIP_W_THRESHOLD,
        manipulability_fd_delta=MANIP_FD_DELTA,
        manipulability_grad_clip=MANIP_GRAD_CLIP,
        manipulability_horizon_decay=MANIP_HORIZON_DECAY,
        manipulability_first_step_only=MANIP_FIRST_STEP_ONLY,
        base_ff_gain=BASE_FF_GAIN,
        ee_linear_speed_limit=EE_LINEAR_SPEED_LIMIT,
        enable_grasp=ENABLE_GRASP,
        grasp_tol=GRASP_TOL,
        grasp_hold_steps=GRASP_HOLD_STEPS,
        grasp_hold_time_s=GRASP_HOLD_TIME_S,
        grasp_action=GRASP_ACTION,
        use_constrained_qp=USE_CONSTRAINED_QP,
        qp_solver=QP_SOLVER,
        qp_infeasible_policy=QP_INFEASIBLE_POLICY,
        qp_enforce_joint_pos=QP_ENFORCE_JOINT_POS,
        qp_enforce_joint_vel=QP_ENFORCE_JOINT_VEL,
        qp_enforce_ee_x_upper=QP_ENFORCE_EE_X_UPPER,
        qp_ee_x_margin=QP_EE_X_MARGIN,
        qp_enforce_ee_y_upper=QP_ENFORCE_EE_Y_UPPER,
        qp_ee_y_margin=QP_EE_Y_MARGIN,
        use_stochastic_mpc=USE_STOCHASTIC_MPC,
        stochastic_risk_alpha=STOCHASTIC_RISK_ALPHA,
        stochastic_sigma_scale=STOCHASTIC_SIGMA_SCALE,
        stochastic_backoff_max=STOCHASTIC_BACKOFF_MAX,
        stochastic_release_slowdown=STOCHASTIC_RELEASE_SLOWDOWN,
        stochastic_ff_suppression=STOCHASTIC_FF_SUPPRESSION,
        stochastic_attach_pos_boost=STOCHASTIC_ATTACH_POS_BOOST,
        stochastic_attach_x_boost=STOCHASTIC_ATTACH_X_BOOST,
        stochastic_attach_rot_boost=STOCHASTIC_ATTACH_ROT_BOOST,
        stochastic_use_tube_feedback=STOCHASTIC_USE_TUBE_FEEDBACK,
        stochastic_tube_gain_xy=STOCHASTIC_TUBE_GAIN_XY,
        stochastic_tube_gain_z=STOCHASTIC_TUBE_GAIN_Z,
        stochastic_tube_max_linear=STOCHASTIC_TUBE_MAX_LINEAR,
    ).run()

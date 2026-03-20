"""
DPG_MPC.py

封装 TaskSpace MPC 与机械臂控制逻辑：
    - TaskSpaceMPC: 无约束 QP（可选终端代价）。
    - MPCController: 接收轨迹（机械臂基坐标系），转换为世界坐标写入 mocap，并通过雅可比分解积分得到关节角目标，写入 position 伺服。
    - 预测分阶段策略: 基于未来视界判定进入“最佳作业区”的时机，执行蛰伏→出击切换。

轨迹约定：
    - Ball 轨迹: position 返回以机械臂基座标系为原点的目标坐标，需加回 base_origin 成为世界坐标。
    - Ball_in_robot 轨迹: position 返回相对底盘的坐标，需要先用底盘轨迹得到 base_pos，再折算世界坐标 = relative + base_pos。
"""

from __future__ import annotations

import sys
import time
from typing import Optional, Union

import mujoco
import mujoco.viewer
import numpy as np
import scipy.sparse as sp

try:
    import osqp
except Exception:  # pragma: no cover - 仅在启用约束QP时强依赖
    osqp = None

from DPG_track_ball import TrajectoryProvider, MocapBallTrajectory
from DPG_track_ball_in_robot import BallInRobotFrameTrajectory
try:
    from rl_value import TerminalValueModel
except Exception:
    class TerminalValueModel:  # pragma: no cover - 约束版默认不依赖 RL 文件
        def __init__(self, model_path=None, input_dim=3, scale=1.0, fallback_weight=20.0):
            self.input_dim = int(input_dim)
            self.scale = float(scale)
            self.fallback_weight = float(fallback_weight)

        def quadratic_approx(self, e: np.ndarray):
            e = np.asarray(e, dtype=float).reshape(-1)
            h = np.eye(e.shape[0], dtype=float) * float(self.fallback_weight) * float(self.scale)
            g = np.zeros(e.shape[0], dtype=float)
            v = 0.5 * float(e @ h @ e)
            return h, g, v


class TaskSpaceMPC:
    """无约束 QP（可插入 RL 终端代价近似）。"""

    def __init__(
        self,
        horizon: int,
        dt: float,
        pos_weight: float = 1.0,
        rot_weight: float = 0.0,
        smooth_weight: float = 1e-2,
        reg: float = 1e-6,
        terminal_value: Optional[TerminalValueModel] = None,
    ):
        self.horizon = horizon
        self.dt = dt
        self.task_dim = 6
        self.pos_dim = 3
        self.pos_weight = float(pos_weight)
        self.rot_weight = float(rot_weight)
        self.terminal_value = terminal_value
        self.terminal_dim = (
            int(getattr(terminal_value, "input_dim", self.pos_dim))
            if terminal_value is not None
            else self.pos_dim
        )
        if self.terminal_dim not in (3, 6, 7):
            raise ValueError("terminal_dim 仅支持 3 / 6 / 7")

        I6 = np.identity(self.task_dim)
        tril = np.tril(np.ones((horizon, horizon)))
        self.A = dt * np.kron(tril, I6)

        weight_block = np.diag([self.pos_weight] * 3 + [self.rot_weight] * 3)
        self.G = np.kron(np.identity(horizon), weight_block)

        self.B = self._build_B_matrix()
        self.Q = smooth_weight * np.identity(self.task_dim * horizon)
        self.reg_eye = reg * np.identity(self.task_dim * horizon)
        self.B_T_Q_B = self.B.T @ self.Q @ self.B
        self.A_T = self.A.T

        self.base_H = self.A_T @ self.G @ self.A + self.B_T_Q_B + self.reg_eye

        self.A_T_G = self.A_T @ self.G
        self.B_T_Q = self.B.T @ self.Q

        self.prev_twist = np.zeros(self.task_dim)
        self.full_solution = np.zeros(self.task_dim * self.horizon)

        # 终端状态映射：x_N = x_0 + C_term * u
        # 3维: 仅位置可控；6维: 位置+姿态可控；7维: 最后一维(奇异度)视作局部常量，不直接由 u 映射。
        if self.terminal_dim == 3:
            pick_lin = np.hstack([np.eye(self.pos_dim), np.zeros((self.pos_dim, 3))])
        elif self.terminal_dim == 6:
            pick_lin = np.eye(self.task_dim)
        else:
            pick_lin = np.vstack([np.eye(self.task_dim), np.zeros((1, self.task_dim))])
        self.C_term = self.dt * np.kron(np.ones((1, self.horizon)), pick_lin)
        self.x_prediction_matrix = self.A[:: self.task_dim, :].copy()
        self.y_prediction_matrix = self.A[1:: self.task_dim, :].copy()
        self.z_prediction_matrix = self.A[2:: self.task_dim, :].copy()
        self.last_solve_ok = True
        self.last_solve_status = "direct"
        self.last_solver = "direct"
        self._osqp_prob = None
        self._osqp_cache_sig = None

    def _build_B_matrix(self) -> np.ndarray:
        n = self.task_dim * self.horizon
        I = np.identity(n)
        upper = np.zeros((self.task_dim, n))
        lower = I[:-self.task_dim, :]
        stacked = np.vstack([upper, lower])
        return I - stacked

    def _build_stage_cost_matrix(self, pos_weight_scales: np.ndarray) -> np.ndarray:
        scales = np.asarray(pos_weight_scales, dtype=float).reshape(-1)
        if scales.shape[0] != self.horizon:
            raise ValueError("pos_weight_scales 长度必须等于 horizon")
        scales = np.clip(scales, 0.0, 10.0)
        diag = np.zeros(self.task_dim * self.horizon, dtype=float)
        for i, s in enumerate(scales):
            idx = i * self.task_dim
            diag[idx : idx + 3] = self.pos_weight * float(s)
            diag[idx + 3 : idx + 6] = self.rot_weight
        return np.diag(diag)

    def _solve_with_osqp_cached(
        self,
        p_csc: sp.csc_matrix,
        q: np.ndarray,
        a_csc: sp.csc_matrix,
        l: np.ndarray,
        u: np.ndarray,
    ):
        sig = (p_csc.shape, int(p_csc.nnz), a_csc.shape, int(a_csc.nnz))
        rebuild = self._osqp_prob is None or self._osqp_cache_sig != sig
        if rebuild:
            prob = osqp.OSQP()
            prob.setup(
                P=p_csc,
                q=q,
                A=a_csc,
                l=l,
                u=u,
                verbose=False,
                warm_start=True,
                polish=False,
                eps_abs=1e-5,
                eps_rel=1e-5,
                max_iter=4000,
            )
            self._osqp_prob = prob
            self._osqp_cache_sig = sig
        else:
            try:
                self._osqp_prob.update(q=q, l=l, u=u, Px=p_csc.data, Ax=a_csc.data)
            except Exception:
                prob = osqp.OSQP()
                prob.setup(
                    P=p_csc,
                    q=q,
                    A=a_csc,
                    l=l,
                    u=u,
                    verbose=False,
                    warm_start=True,
                    polish=False,
                    eps_abs=1e-5,
                    eps_rel=1e-5,
                    max_iter=4000,
                )
                self._osqp_prob = prob
                self._osqp_cache_sig = sig
        return self._osqp_prob.solve()

    def solve(
        self,
        error_vector: np.ndarray,
        *,
        current_state: Optional[np.ndarray] = None,
        desired_terminal: Optional[np.ndarray] = None,
        pos_weight_scales: Optional[np.ndarray] = None,
        extra_rhs: Optional[np.ndarray] = None,
        linear_constraint_matrix: Optional[np.ndarray] = None,
        linear_lower_bound: Optional[np.ndarray] = None,
        linear_upper_bound: Optional[np.ndarray] = None,
        qp_solver: str = "osqp",
    ) -> np.ndarray:
        if error_vector.shape[0] != self.task_dim * self.horizon:
            raise ValueError("error_vector 尺寸错误")

        u_anchor = np.zeros(self.task_dim * self.horizon)
        u_anchor[: self.task_dim] = self.prev_twist

        if pos_weight_scales is None:
            rhs = self.A_T_G @ error_vector + self.B_T_Q @ u_anchor
            H = self.base_H.copy()
        else:
            g_dyn = self._build_stage_cost_matrix(pos_weight_scales)
            a_t_g_dyn = self.A_T @ g_dyn
            rhs = a_t_g_dyn @ error_vector + self.B_T_Q @ u_anchor
            H = a_t_g_dyn @ self.A + self.B_T_Q_B + self.reg_eye

        if extra_rhs is not None:
            extra_rhs = np.asarray(extra_rhs, dtype=float).reshape(-1)
            if extra_rhs.shape[0] != self.task_dim * self.horizon:
                raise ValueError("extra_rhs 尺寸错误")
            rhs = rhs + extra_rhs

        if (
            self.terminal_value is not None
            and current_state is not None
            and desired_terminal is not None
        ):
            # 终端误差 e = x_N - x_target
            current_state = np.asarray(current_state, dtype=float).reshape(-1)
            desired_terminal = np.asarray(desired_terminal, dtype=float).reshape(-1)
            if current_state.shape[0] != self.terminal_dim:
                raise ValueError("current_state 维度与 terminal_dim 不一致")
            if desired_terminal.shape[0] != self.terminal_dim:
                raise ValueError("desired_terminal 维度与 terminal_dim 不一致")
            e0 = current_state - desired_terminal
            e_bar = e0 + self.C_term @ self.full_solution
            h_term, g_term, _ = self.terminal_value.quadratic_approx(e_bar)
            H += self.C_term.T @ h_term @ self.C_term
            rhs += self.C_term.T @ (h_term @ e0 + g_term)

        if linear_constraint_matrix is None:
            self.full_solution = np.linalg.solve(H, rhs)
            twist = self.full_solution[: self.task_dim]
            self.prev_twist = twist
            self.last_solve_ok = True
            self.last_solve_status = "solved"
            self.last_solver = "direct"
            return twist

        c_mat = np.asarray(linear_constraint_matrix, dtype=float)
        if c_mat.ndim != 2 or c_mat.shape[1] != self.task_dim * self.horizon:
            raise ValueError("linear_constraint_matrix 尺寸错误")
        if linear_lower_bound is None or linear_upper_bound is None:
            raise ValueError("使用约束QP时必须同时提供 linear_lower_bound / linear_upper_bound")
        c_low = np.asarray(linear_lower_bound, dtype=float).reshape(-1)
        c_up = np.asarray(linear_upper_bound, dtype=float).reshape(-1)
        if c_low.shape[0] != c_mat.shape[0] or c_up.shape[0] != c_mat.shape[0]:
            raise ValueError("约束上下界长度与 linear_constraint_matrix 行数不一致")
        if np.any(c_low > c_up):
            self.last_solve_ok = False
            self.last_solve_status = "invalid_bounds"
            self.last_solver = "osqp"
            return np.zeros(self.task_dim, dtype=float)

        if str(qp_solver).lower() != "osqp":
            raise ValueError("当前仅支持 qp_solver='osqp'")
        if osqp is None:
            raise RuntimeError("未安装 osqp，无法启用约束QP")

        # OSQP 形式: min 1/2 x^T P x + q^T x, s.t. l <= A x <= u
        p = 0.5 * (H + H.T)
        n = p.shape[0]
        p = p + 1e-9 * np.identity(n)
        q = -rhs
        p_csc = sp.csc_matrix(np.triu(p))
        a_csc = sp.csc_matrix(c_mat)
        try:
            res = self._solve_with_osqp_cached(p_csc, q, a_csc, c_low, c_up)
        except Exception as exc:
            self.last_solve_ok = False
            self.last_solve_status = f"osqp_setup_error:{type(exc).__name__}"
            self.last_solver = "osqp"
            return np.zeros(self.task_dim, dtype=float)
        status_val = int(getattr(res.info, "status_val", -1))
        status_str = str(getattr(res.info, "status", "unknown"))
        solved = status_val in (1, 2) and (res.x is not None)
        self.last_solve_ok = bool(solved)
        self.last_solve_status = status_str
        self.last_solver = "osqp"
        if not solved:
            return np.zeros(self.task_dim, dtype=float)

        sol = np.asarray(res.x, dtype=float).reshape(-1)
        self.full_solution = sol
        twist = sol[: self.task_dim]
        self.prev_twist = twist
        return twist


class MPCController:
    """负责轨迹取样、MPC 求解、雅可比分解、输出 position 伺服目标角。"""

    def __init__(
        self,
        model_xml: str = "fetch_freight_mujoco/xml/scene.xml",
        trajectory: Optional[TrajectoryProvider] = None,
        intercept_planner=None,
        horizon: int = 20,
        pos_weight: float = 12.0,
        rot_weight: float = 0.0,
        smooth_weight: float = 1e-3,
        render_dt: float = 0.02,
        control_dt: float = 0.01,
        profile_period: float = 0.1,
        warm_start_max: float = 6.0,
        warm_start_tol: float = 0.05,
        reach_max: float = 0.90,
        use_terminal_value: bool = True,
        terminal_value_path: Optional[str] = None,
        terminal_value_scale: float = 1.0,
        terminal_value_fallback: float = 20.0,
        terminal_value_dim: int = 3,
        terminal_rot_scale: float = 1.0,
        terminal_sing_scale: float = 0.2,
        terminal_approach_dir: Optional[np.ndarray] = None,
        terminal_approach_axis: Union[str, np.ndarray] = "y",
        use_pregrasp: bool = False,
        pregrasp_offset: float = 0.06,
        pregrasp_dir: Optional[np.ndarray] = None,
        pregrasp_tol: float = 0.02,
        pregrasp_hold_time_s: float = 0.5,
        approach_speed: float = 0.3,
        use_predictive_phase_switch: bool = False,
        phase_opt_radius_min: float = 0.40,
        phase_opt_radius_max: float = 0.65,
        phase_trigger_index: int = 6,
        phase_confirm_steps: int = 3,
        phase_min_hold_s: float = 0.0,
        phase_use_planar_distance: bool = True,
        phase_use_x_gate_switch: bool = False,
        phase_x_gate_half_width: float = 0.03,
        phase_x_gate_hold_steps: int = 10,
        phase_instant_attack: bool = True,
        use_offset_tracking: bool = False,
        offset_y: float = -0.1,
        offset_release_time_s: float = 0.25,
        hold_pos_weight_scale: float = 1.0,
        attach_pos_weight_scale: float = 1.0,
        hold_x_error_gain: float = 1.0,
        attach_x_error_gain: float = 1.0,
        hold_orientation_gain: float = 1.0,
        attach_orientation_gain: float = 1.0,
        offset_trigger_tol: float = 0.02,
        offset_trigger_steps: int = 10,
        offset_trigger_hold_time_s: float = 0.5,
        offset_switch_x_gate_enable: bool = False,
        offset_switch_x_front: float = 0.2,
        offset_switch_x_align_tol: float = 0.04,
        offset_switch_yz_tol: float = 0.10,
        enable_grasp: bool = True,
        grasp_tol: float = 0.03,
        grasp_hold_steps: int = 3,
        grasp_hold_time_s: Optional[float] = None,
        grasp_action: str = "stop",
        use_uncertainty_aware_weighting: bool = True,
        uncertainty_beta: float = 100.0,
        uncertainty_min_scale: float = 0.65,
        uncertainty_ema: float = 0.0,
        use_manipulability_guidance: bool = True,
        manipulability_lambda: float = 0.06,
        manipulability_w_threshold: float = 0.08,
        manipulability_fd_delta: float = 0.004,
        manipulability_grad_clip: float = 2.0,
        manipulability_horizon_decay: float = 0.8,
        manipulability_first_step_only: bool = False,
        base_ff_gain: float = 1.0,
        ee_linear_speed_limit: float = 0.8,
        use_constrained_qp: bool = False,
        qp_solver: str = "osqp",
        qp_infeasible_policy: str = "hold",
        qp_enforce_joint_pos: bool = True,
        qp_enforce_joint_vel: bool = True,
        qp_enforce_ee_x_upper: bool = False,
        qp_ee_x_margin: float = 0.0,
        qp_enforce_ee_y_upper: bool = False,
        qp_ee_y_margin: float = 0.0,
        qp_enforce_ee_z_lower: bool = False,
        qp_ee_z_margin: float = 0.0,
    ):
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data = mujoco.MjData(self.model)

        self.trajectory: TrajectoryProvider = trajectory or MocapBallTrajectory()
        self.intercept_planner = intercept_planner
        self.is_ball_rel = isinstance(self.trajectory, BallInRobotFrameTrajectory)
        self.base_traj_for_rel = self.trajectory.base_trajectory if self.is_ball_rel else None
        if self.is_ball_rel and self.base_traj_for_rel is None:
            from DPG_track_ball_in_robot import LinearBaseTrajectory

            self.base_traj_for_rel = LinearBaseTrajectory()
        if self.is_ball_rel and isinstance(self.trajectory, BallInRobotFrameTrajectory):
            if self.base_traj_for_rel is None:
                raise ValueError("ball_in_robot 需要 base_trajectory，但当前为 None")
            # 确保控制器与轨迹对象共享同一个底盘轨迹实例（含 KF 内部状态）
            self.trajectory.base_trajectory = self.base_traj_for_rel

        self.arm_joint_names = [f"XMS5-R800-B4G3B0C_joint_{i}" for i in range(1, 7)]
        self.arm_actuator_names = [f"joint_{i}" for i in range(1, 7)]

        initial_qpos = {
            "XMS5-R800-B4G3B0C_joint_1": 0.0,
            "XMS5-R800-B4G3B0C_joint_2": 0.0,
            "XMS5-R800-B4G3B0C_joint_3": 0.0,
            "XMS5-R800-B4G3B0C_joint_4": 0.0,
            "XMS5-R800-B4G3B0C_joint_5": 0.0,
            "XMS5-R800-B4G3B0C_joint_6": 0.0,
        }
        for joint_name, value in initial_qpos.items():
            joint_id = self.model.joint(joint_name).id
            adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[adr] = value
        mujoco.mj_forward(self.model, self.data)

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_finger")
        if self.ee_site_id < 0:
            raise ValueError("找不到 site=end_finger（请检查 SR5 xml 是否包含末端 site）")
        self.target_mocap_id = self.model.body("target").mocapid[0]

        self.arm_joint_ids = [self.model.joint(name).id for name in self.arm_joint_names]
        self.arm_joint_body_ids = np.array(
            [self.model.jnt_bodyid[j] for j in self.arm_joint_ids], dtype=int
        )
        self.arm_qpos_indices = np.array([self.model.jnt_qposadr[j] for j in self.arm_joint_ids])
        self.arm_dof_indices = np.array([self.model.jnt_dofadr[j] for j in self.arm_joint_ids])
        self.arm_dof = int(len(self.arm_joint_ids))
        joint_range = self.model.jnt_range[self.arm_joint_ids]
        self.arm_qpos_min = joint_range[:, 0].copy()
        self.arm_qpos_max = joint_range[:, 1].copy()

        self.q_nominal = self.data.qpos[self.arm_qpos_indices].copy()
        self.q_nominal_init = self.q_nominal.copy()

        self.arm_actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.arm_actuator_names
            ]
        )
        if np.any(self.arm_actuator_ids < 0):
            missing = [n for n, i in zip(self.arm_actuator_names, self.arm_actuator_ids) if i < 0]
            raise ValueError(f"找不到 SR5 执行器: {missing}（请检查 common.xml actuator 命名）")


        self.chassis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        # 底盘现在是 mocap 体，直接写 mocap_pos 驱动
        self.chassis_mocap_id = self.model.body("chassis").mocapid[0]
        # 使用机械臂第一段（joint_1 之后的 link1）作为 reach 参考点，避免用 sr5_link 原点导致“明明可达但被当成不可达”
        self.shoulder_body_id = self.model.body("XMS5-R800-B4G3B0C_link1").id
        self.shoulder_offset = (
            self.data.xpos[self.shoulder_body_id].copy()
            - self.data.xpos[self.chassis_body_id].copy()
        )

        self.dt = self.model.opt.timestep  # 物理步长
        self.control_dt = max(control_dt, self.dt)  # 控制刷新周期，避免低于物理步长
        self.control_every = max(1, int(round(self.control_dt / self.dt)))
        self.control_dt = self.control_every * self.dt  # 对齐为物理步长的整数倍
        self.render_every = max(1, int(round(render_dt / self.dt)))
        self.profile_period = profile_period
        self.warm_start_tol = warm_start_tol

        self.warm_start_max = warm_start_max
        if self.is_ball_rel and self.base_traj_for_rel is not None and (not use_offset_tracking):
            base_duration = getattr(self.base_traj_for_rel, "duration", None)
            if callable(base_duration):
                # 确保 warm start 覆盖到底盘走完全程，这样 q_nominal 会收敛到接近工作区的舒适姿态
                self.warm_start_max = max(self.warm_start_max, float(base_duration()) + 0.5)
        terminal_value = None
        if use_terminal_value:
            terminal_value = TerminalValueModel(
                model_path=terminal_value_path,
                input_dim=int(terminal_value_dim),
                scale=float(terminal_value_scale),
                fallback_weight=float(terminal_value_fallback),
            )

        self.mpc = TaskSpaceMPC(
            horizon=horizon,
            dt=self.control_dt,
            pos_weight=pos_weight,
            rot_weight=rot_weight,
            smooth_weight=smooth_weight,
            terminal_value=terminal_value,
        )
        self.use_orientation_task = rot_weight > 1e-9
        # 创新点1: 基于 KF 预测不确定性的自适应误差权重
        self.use_uncertainty_aware_weighting = bool(use_uncertainty_aware_weighting)
        self.uncertainty_beta = max(0.0, float(uncertainty_beta))
        self.uncertainty_min_scale = float(np.clip(uncertainty_min_scale, 0.0, 1.0))
        self.uncertainty_ema = float(np.clip(uncertainty_ema, 0.0, 0.99))
        self._uncertainty_scales_last = np.ones(self.mpc.horizon, dtype=float)
        self._uncertainty_sigma_last = np.zeros(self.mpc.horizon, dtype=float)
        # 创新点3：操作度梯度引导（低操作度时给末端速度一个“离奇异点更远”的偏置）
        self.use_manipulability_guidance = bool(use_manipulability_guidance)
        self.manipulability_lambda = max(0.0, float(manipulability_lambda))
        self.manipulability_w_threshold = max(1e-6, float(manipulability_w_threshold))
        self.manipulability_fd_delta = max(1e-5, float(manipulability_fd_delta))
        self.manipulability_grad_clip = max(1e-6, float(manipulability_grad_clip))
        self.manipulability_horizon_decay = float(np.clip(manipulability_horizon_decay, 0.0, 1.0))
        self.manipulability_first_step_only = bool(manipulability_first_step_only)
        self.manipulability_eps = 1e-6
        self._manip_last_w = 0.0
        self._manip_last_risk = 0.0

        # 终端价值函数输入维度与姿态偏好（用于 side-approach）
        self.terminal_value_dim = int(terminal_value_dim)
        if self.terminal_value_dim not in (3, 6, 7):
            raise ValueError("terminal_value_dim 仅支持 3 / 6 / 7")
        self.terminal_rot_scale = float(terminal_rot_scale)
        self.terminal_sing_scale = float(terminal_sing_scale)
        self.terminal_approach_dir = (
            None
            if terminal_approach_dir is None
            else np.array(terminal_approach_dir, dtype=float).reshape(3)
        )
        self.terminal_approach_axis = terminal_approach_axis
        self._approach_axis_local = self._parse_approach_axis(terminal_approach_axis)

        # 预抓取状态机（保持偏置 → 快速靠近）
        self.use_pregrasp = bool(use_pregrasp)
        self.pregrasp_offset = float(pregrasp_offset)
        self.pregrasp_dir = (
            np.array([0.0, 1.0, 0.0], dtype=float)
            if pregrasp_dir is None
            else np.array(pregrasp_dir, dtype=float).reshape(3)
        )
        dir_norm = float(np.linalg.norm(self.pregrasp_dir))
        if dir_norm > 1e-9:
            self.pregrasp_dir = self.pregrasp_dir / dir_norm
        self.pregrasp_tol = float(pregrasp_tol)
        self.pregrasp_hold_time_s = float(pregrasp_hold_time_s)
        self.approach_speed = float(approach_speed)
        self.approach_active = False
        self.approach_start_time = 0.0
        self._pregrasp_time = 0.0
        # 预测分阶段触发（创新点2）:
        # 在预测视界内检测“目标进入最佳作业半径区间”的时刻，触发由蛰伏到出击切换。
        self.use_predictive_phase_switch = bool(use_predictive_phase_switch)
        self.phase_opt_radius_min = max(0.05, float(phase_opt_radius_min))
        self.phase_opt_radius_max = max(self.phase_opt_radius_min + 1e-3, float(phase_opt_radius_max))
        self.phase_trigger_index = max(0, int(phase_trigger_index))
        self.phase_confirm_steps = max(1, int(phase_confirm_steps))
        self.phase_min_hold_s = max(0.0, float(phase_min_hold_s))
        self.phase_use_planar_distance = bool(phase_use_planar_distance)
        self.phase_use_x_gate_switch = bool(phase_use_x_gate_switch)
        self.phase_x_gate_half_width = max(1e-4, float(phase_x_gate_half_width))
        self.phase_x_gate_hold_steps = max(1, int(phase_x_gate_hold_steps))
        self.phase_instant_attack = bool(phase_instant_attack)
        self._phase_confirm_count = 0
        self._phase_last_enter_index = -1
        self._phase_x_hold_count = 0
        self._phase_last_x_err = np.nan

        # 侧向保持距离（y 偏置）→ 满足条件后切回原轨迹
        self.use_offset_tracking = bool(use_offset_tracking)
        self.offset_y = float(offset_y)
        self.offset_release_time_s = max(0.0, float(offset_release_time_s))
        self.hold_pos_weight_scale = max(1e-6, float(hold_pos_weight_scale))
        self.attach_pos_weight_scale = max(1e-6, float(attach_pos_weight_scale))
        self.hold_x_error_gain = max(1e-6, float(hold_x_error_gain))
        self.attach_x_error_gain = max(1e-6, float(attach_x_error_gain))
        self.hold_orientation_gain = float(hold_orientation_gain)
        self.attach_orientation_gain = float(attach_orientation_gain)
        self.offset_trigger_tol = float(offset_trigger_tol)
        self.offset_trigger_steps = max(1, int(offset_trigger_steps))
        self.offset_trigger_hold_time_s = float(offset_trigger_hold_time_s)
        self.offset_switch_x_gate_enable = bool(offset_switch_x_gate_enable)
        self.offset_switch_x_front = max(0.0, float(offset_switch_x_front))
        self.offset_switch_x_align_tol = max(0.0, float(offset_switch_x_align_tol))
        self.offset_switch_yz_tol = max(0.0, float(offset_switch_yz_tol))
        self.offset_active = bool(use_offset_tracking)
        self._offset_hit_count = 0
        self._offset_last_dist = np.nan
        self._offset_time = 0.0
        self._offset_release_start_time = np.nan
        self._offset_x_gate_ready = False
        self._offset_x_gate_threshold = np.nan
        self._offset_x_gate_delta = np.nan

        # 过大的关节速度会导致“乱甩”，默认给一个更保守的上限；需要更激进可在外部调大
        self.velocity_limit = np.deg2rad(240.0) * np.ones(self.arm_dof)
        self.damping = 0.02
        self.nullspace_gain = 0.6
        self.cond_threshold = 1e3
        self.min_twist_scale = 0.1
        self.drift_gain = 0.05
        self.nullspace_scale = np.deg2rad(45.0)
        self.time_offset = 0.0
        # ball_in_robot 场景底盘轨迹已显式预测，无需额外 preview_lead，避免过度前瞻带来的抖动
        self.preview_lead = 0.0 if self.is_ball_rel else 0.3
        self.last_qdot_cmd = np.zeros(self.arm_dof)
        self.accel_limit = np.deg2rad(1200.0) * np.ones(self.arm_dof)
        # 目标不可达时：先跟踪“最大可达点”，底盘靠近后自然过渡到真实目标
        self.reach_max = float(reach_max)
        self.nullspace_err_scale = 0.25
        # 平滑：在接近目标时保持足够反馈增益，避免 target_err 在 0.02m 附近“悬停”
        self.feedback_deadband = 0.005
        self.feedback_ramp = 0.03
        self.base_ff_gain = float(base_ff_gain)
        # 额外的任务空间限幅：避免 MPC 在大误差时给出过大的线速度，导致关节速度饱和→抖动
        self.ee_linear_speed_limit = float(ee_linear_speed_limit)

        # 抓取检测（逻辑触发）
        self.enable_grasp = bool(enable_grasp)
        self.grasp_tol = float(grasp_tol)
        self.grasp_hold_steps = max(1, int(grasp_hold_steps))
        self.grasp_hold_time_s = None if grasp_hold_time_s is None else float(grasp_hold_time_s)
        self.grasp_action = str(grasp_action)
        self.grasped = False
        self._grasp_counter = 0
        self._grasp_time = 0.0
        self._attach_target = False

        # 约束QP选项（默认关闭，保持兼容）
        self.use_constrained_qp = bool(use_constrained_qp)
        self.qp_solver = str(qp_solver).lower()
        if self.use_constrained_qp and self.qp_solver != "osqp":
            raise ValueError("use_constrained_qp=True 时仅支持 qp_solver='osqp'")
        if self.use_constrained_qp and osqp is None:
            raise RuntimeError("use_constrained_qp=True 需要安装 osqp")
        self.qp_infeasible_policy = str(qp_infeasible_policy).lower()
        if self.qp_infeasible_policy not in ("hold",):
            raise ValueError("qp_infeasible_policy 仅支持 'hold'")
        self.qp_enforce_joint_pos = bool(qp_enforce_joint_pos)
        self.qp_enforce_joint_vel = bool(qp_enforce_joint_vel)
        self.qp_enforce_ee_x_upper = bool(qp_enforce_ee_x_upper)
        self.qp_ee_x_margin = float(qp_ee_x_margin)
        self.qp_enforce_ee_y_upper = bool(qp_enforce_ee_y_upper)
        self.qp_ee_y_margin = float(qp_ee_y_margin)
        self.qp_enforce_ee_z_lower = bool(qp_enforce_ee_z_lower)
        self.qp_ee_z_margin = float(qp_ee_z_margin)
        self._qp_last_warn_time = -np.inf
        # 约束QP近目标降速：抑制 attach 段末端过冲，减小“冲过头再拉回”。
        self.qp_near_err = 0.08
        self.qp_stop_err = 0.02
        self.qp_min_qdot_scale = 0.12

    def _limit_twist_cmd(
        self, twist_cmd: np.ndarray, twist_fb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.ee_linear_speed_limit <= 0:
            return twist_cmd, twist_fb

        if self.use_orientation_task:
            lin = twist_cmd[:3]
            norm = float(np.linalg.norm(lin))
            if norm <= self.ee_linear_speed_limit or norm < 1e-12:
                return twist_cmd, twist_fb
            scale = self.ee_linear_speed_limit / norm
            return twist_cmd * scale, twist_fb * scale

        norm = float(np.linalg.norm(twist_cmd))
        if norm <= self.ee_linear_speed_limit or norm < 1e-12:
            return twist_cmd, twist_fb
        scale = self.ee_linear_speed_limit / norm
        return twist_cmd * scale, twist_fb * scale

    def _task_error(
        self,
        current_pos: np.ndarray,
        future_targets: np.ndarray,
        abs_time: Optional[float] = None,
    ) -> np.ndarray:
        e_hat = np.zeros(self.mpc.task_dim * self.mpc.horizon)
        ee_axis_world = self._approach_axis_world() if self.use_orientation_task else None
        if abs_time is None:
            abs_time = float(self.data.time)
        if self.use_offset_tracking:
            alpha = self._offset_release_progress(abs_time)
            orientation_gain = float(
                (1.0 - alpha) * self.hold_orientation_gain + alpha * self.attach_orientation_gain
            )
            x_error_gain = float(
                (1.0 - alpha) * self.hold_x_error_gain + alpha * self.attach_x_error_gain
            )
        else:
            orientation_gain = 1.0
            x_error_gain = 1.0
        for i, desired in enumerate(future_targets):
            idx = i * self.mpc.task_dim
            e_hat[idx : idx + 3] = desired - current_pos
            e_hat[idx] = x_error_gain * e_hat[idx]
            if self.use_orientation_task and ee_axis_world is not None:
                desired_abs_time = float(abs_time) + self.preview_lead + (i + 1) * self.control_dt
                desired_dir = self._desired_approach_dir(desired, desired_abs_time)
                e_hat[idx + 3 : idx + 6] = orientation_gain * np.cross(ee_axis_world, desired_dir)
        return e_hat

    def _task_jacobian(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        pos_jac = jacp[:, self.arm_dof_indices]
        rot_jac = jacr[:, self.arm_dof_indices]
        return np.vstack([pos_jac, rot_jac])

    def _jacobian_pinv(self, jac: np.ndarray) -> np.ndarray:
        task_dim = jac.shape[0]
        JJT = jac @ jac.T + (self.damping ** 2) * np.identity(task_dim)
        return jac.T @ np.linalg.inv(JJT)

    def _reachable_target(self, target_world: np.ndarray, mount_pos_world: np.ndarray) -> np.ndarray:
        vec = target_world - mount_pos_world
        dist = float(np.linalg.norm(vec))
        if dist < 1e-9 or dist <= self.reach_max:
            return target_world
        return mount_pos_world + (self.reach_max / dist) * vec

    def _limit_joint_accel(self, qdot_cmd: np.ndarray, qdot_prev: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0:
            return qdot_cmd
        max_delta = self.accel_limit * dt
        delta = qdot_cmd - qdot_prev
        delta = np.clip(delta, -max_delta, max_delta)
        return qdot_prev + delta

    def _qp_task_to_joint_step_map(self, jac_pinv: np.ndarray) -> np.ndarray:
        if self.use_orientation_task:
            step_map = np.asarray(jac_pinv, dtype=float)
            if step_map.shape != (self.arm_dof, self.mpc.task_dim):
                raise ValueError("jac_pinv 维度错误（姿态任务）")
            return step_map

        step_map = np.zeros((self.arm_dof, self.mpc.task_dim), dtype=float)
        jac_lin = np.asarray(jac_pinv, dtype=float)
        if jac_lin.shape != (self.arm_dof, 3):
            raise ValueError("jac_pinv 维度错误（位置任务）")
        step_map[:, :3] = jac_lin
        return step_map

    def _build_constrained_qp_bounds(
        self,
        *,
        jac_pinv: np.ndarray,
        q_ref: np.ndarray,
        current_pos_world: Optional[np.ndarray] = None,
        ee_x_upper_seq: Optional[np.ndarray] = None,
        ee_y_upper_seq: Optional[np.ndarray] = None,
        ee_z_lower_seq: Optional[np.ndarray] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.use_constrained_qp:
            return None, None, None

        a_blocks = []
        l_blocks = []
        u_blocks = []

        if self.qp_enforce_joint_pos or self.qp_enforce_joint_vel:
            step_map = self._qp_task_to_joint_step_map(jac_pinv)
            a_qdot = np.kron(np.identity(self.mpc.horizon), step_map)

            if self.qp_enforce_joint_vel:
                vel = np.tile(self.velocity_limit, self.mpc.horizon)
                a_blocks.append(a_qdot)
                l_blocks.append(-vel)
                u_blocks.append(vel)

            if self.qp_enforce_joint_pos:
                tril = np.tril(np.ones((self.mpc.horizon, self.mpc.horizon)))
                integ = np.kron(tril, np.identity(self.arm_dof))
                a_qpos = self.control_dt * (integ @ a_qdot)
                q_ref = np.asarray(q_ref, dtype=float).reshape(self.arm_dof)
                q_ref_stack = np.tile(q_ref, self.mpc.horizon)
                l_qpos = np.tile(self.arm_qpos_min, self.mpc.horizon) - q_ref_stack
                u_qpos = np.tile(self.arm_qpos_max, self.mpc.horizon) - q_ref_stack
                a_blocks.append(a_qpos)
                l_blocks.append(l_qpos)
                u_blocks.append(u_qpos)

        if self.qp_enforce_ee_x_upper:
            if current_pos_world is None or ee_x_upper_seq is None:
                raise ValueError("启用末端 x 上界约束时必须提供 current_pos_world / ee_x_upper_seq")
            ee_x_upper_seq = np.asarray(ee_x_upper_seq, dtype=float).reshape(-1)
            if ee_x_upper_seq.shape[0] != self.mpc.horizon:
                raise ValueError("ee_x_upper_seq 长度必须等于 horizon")
            a_ee_x = self.mpc.x_prediction_matrix.copy()
            current_x = float(np.asarray(current_pos_world, dtype=float).reshape(3)[0])
            l_ee_x = np.full(self.mpc.horizon, -np.inf, dtype=float)
            u_ee_x = ee_x_upper_seq - current_x
            a_blocks.append(a_ee_x)
            l_blocks.append(l_ee_x)
            u_blocks.append(u_ee_x)

        if self.qp_enforce_ee_y_upper:
            if current_pos_world is None or ee_y_upper_seq is None:
                raise ValueError("启用末端 y 上界约束时必须提供 current_pos_world / ee_y_upper_seq")
            ee_y_upper_seq = np.asarray(ee_y_upper_seq, dtype=float).reshape(-1)
            if ee_y_upper_seq.shape[0] != self.mpc.horizon:
                raise ValueError("ee_y_upper_seq 长度必须等于 horizon")
            a_ee_y = self.mpc.y_prediction_matrix.copy()
            current_y = float(np.asarray(current_pos_world, dtype=float).reshape(3)[1])
            l_ee_y = np.full(self.mpc.horizon, -np.inf, dtype=float)
            u_ee_y = ee_y_upper_seq - current_y
            a_blocks.append(a_ee_y)
            l_blocks.append(l_ee_y)
            u_blocks.append(u_ee_y)

        if self.qp_enforce_ee_z_lower:
            if current_pos_world is None or ee_z_lower_seq is None:
                raise ValueError("启用末端 z 下界约束时必须提供 current_pos_world / ee_z_lower_seq")
            ee_z_lower_seq = np.asarray(ee_z_lower_seq, dtype=float).reshape(-1)
            if ee_z_lower_seq.shape[0] != self.mpc.horizon:
                raise ValueError("ee_z_lower_seq 长度必须等于 horizon")
            a_ee_z = self.mpc.z_prediction_matrix.copy()
            current_z = float(np.asarray(current_pos_world, dtype=float).reshape(3)[2])
            l_ee_z = ee_z_lower_seq - current_z
            u_ee_z = np.full(self.mpc.horizon, np.inf, dtype=float)
            a_blocks.append(a_ee_z)
            l_blocks.append(l_ee_z)
            u_blocks.append(u_ee_z)

        if len(a_blocks) == 0:
            return None, None, None
        return np.vstack(a_blocks), np.concatenate(l_blocks), np.concatenate(u_blocks)

    def _qp_ee_x_upper_sequence(
        self,
        *,
        sim_time: float,
        target_world_now: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        margin = float(self.qp_ee_x_margin)
        if target_world_now is not None:
            x_now = float(np.asarray(target_world_now, dtype=float).reshape(3)[0]) + margin
            return np.full(self.mpc.horizon, x_now, dtype=float)
        future_world = self._world_future_targets(sim_time)
        return np.asarray(future_world[:, 0], dtype=float).reshape(self.mpc.horizon) + margin

    def _qp_ee_y_upper_sequence(
        self,
        *,
        sim_time: float,
        target_world_now: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        margin = float(self.qp_ee_y_margin)
        if target_world_now is not None:
            y_now = float(np.asarray(target_world_now, dtype=float).reshape(3)[1]) - margin
            return np.full(self.mpc.horizon, y_now, dtype=float)
        future_world = self._world_future_targets(sim_time)
        return np.asarray(future_world[:, 1], dtype=float).reshape(self.mpc.horizon) - margin

    def _qp_ee_z_lower_sequence(
        self,
        *,
        sim_time: float,
        target_world_now: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        margin = float(self.qp_ee_z_margin)
        if target_world_now is not None:
            z_now = float(np.asarray(target_world_now, dtype=float).reshape(3)[2]) + margin
            return np.full(self.mpc.horizon, z_now, dtype=float)
        future_world = self._world_future_targets(sim_time)
        return np.asarray(future_world[:, 2], dtype=float).reshape(self.mpc.horizon) + margin

    def _log_qp_infeasible(self, sim_time: float) -> None:
        if float(sim_time) - float(self._qp_last_warn_time) < 0.2:
            return
        self._qp_last_warn_time = float(sim_time)
        print(
            f"[qp] infeasible at t={float(self.data.time):.2f}s "
            f"(status={self.mpc.last_solve_status}), apply hold"
        )

    def _qp_qdot_scale(self, active_err: float) -> float:
        err = float(max(0.0, active_err))
        if err <= self.qp_stop_err:
            return self.qp_min_qdot_scale
        if err >= self.qp_near_err:
            return 1.0
        alpha = (err - self.qp_stop_err) / max(self.qp_near_err - self.qp_stop_err, 1e-9)
        return float(self.qp_min_qdot_scale + (1.0 - self.qp_min_qdot_scale) * alpha)

    def _feedback_gain(self, pos_err: float) -> float:
        if pos_err <= self.feedback_deadband:
            return 0.0
        return float(np.clip((pos_err - self.feedback_deadband) / self.feedback_ramp, 0.0, 1.0))

    def _project_qdot_to_axis_upper(
        self,
        qdot_cmd: np.ndarray,
        *,
        jac_axis_row: np.ndarray,
        current_value: float,
        upper_value: float,
    ) -> np.ndarray:
        a = np.asarray(jac_axis_row, dtype=float).reshape(self.arm_dof)
        denom = float(a @ a)
        if denom < 1e-12 or self.control_dt <= 1e-12:
            return qdot_cmd
        b = (float(upper_value) - float(current_value)) / self.control_dt
        violation = float(a @ qdot_cmd) - b
        if violation <= 1e-9:
            return qdot_cmd
        qdot_proj = np.asarray(qdot_cmd, dtype=float).reshape(self.arm_dof) - (violation / denom) * a
        return np.clip(qdot_proj, -self.velocity_limit, self.velocity_limit)

    def _project_qdot_to_axis_lower(
        self,
        qdot_cmd: np.ndarray,
        *,
        jac_axis_row: np.ndarray,
        current_value: float,
        lower_value: float,
    ) -> np.ndarray:
        a = np.asarray(jac_axis_row, dtype=float).reshape(self.arm_dof)
        denom = float(a @ a)
        if denom < 1e-12 or self.control_dt <= 1e-12:
            return qdot_cmd
        b = (float(lower_value) - float(current_value)) / self.control_dt
        violation = b - float(a @ qdot_cmd)
        if violation <= 1e-9:
            return qdot_cmd
        qdot_proj = np.asarray(qdot_cmd, dtype=float).reshape(self.arm_dof) + (violation / denom) * a
        return np.clip(qdot_proj, -self.velocity_limit, self.velocity_limit)

    def _offset_y_at(self, abs_time: float) -> float:
        if (not self.use_offset_tracking) or abs(self.offset_y) < 1e-12:
            return 0.0
        return float((1.0 - self._offset_release_progress(abs_time)) * self.offset_y)

    def _offset_release_progress(self, abs_time: float) -> float:
        if (not self.use_offset_tracking) or abs(self.offset_y) < 1e-12:
            return 1.0
        if self.offset_active:
            return 0.0
        if self.offset_release_time_s <= 1e-9:
            return 1.0
        if not np.isfinite(self._offset_release_start_time):
            return 1.0
        alpha = (float(abs_time) - float(self._offset_release_start_time)) / self.offset_release_time_s
        return float(np.clip(alpha, 0.0, 1.0))

    def _parse_approach_axis(self, axis: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(axis, str):
            key = axis.lower()
            mapping = {"x": np.array([1.0, 0.0, 0.0]),
                       "y": np.array([0.0, 1.0, 0.0]),
                       "z": np.array([0.0, 0.0, 1.0])}
            if key not in mapping:
                raise ValueError("terminal_approach_axis 仅支持 x/y/z 或 3 维向量")
            return mapping[key]
        vec = np.array(axis, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            raise ValueError("terminal_approach_axis 向量长度过小")
        return vec / norm

    def _base_pos_world(self, abs_time: float) -> np.ndarray:
        if self.is_ball_rel and self.base_traj_for_rel is not None:
            self._sync_base_traj(abs_time)
            return self.base_traj_for_rel.position(abs_time)
        if self.chassis_mocap_id >= 0:
            return self.data.mocap_pos[self.chassis_mocap_id].copy()
        return np.zeros(3, dtype=float)

    def _sync_base_traj(self, abs_time: float) -> None:
        if self.base_traj_for_rel is None:
            return
        sync = getattr(self.base_traj_for_rel, "sync", None)
        if callable(sync):
            sync(abs_time)

    def _ball_rel_traj_time(self, sim_time: float, abs_time: float) -> float:
        return float(sim_time) if self.use_offset_tracking else float(abs_time)

    def _ball_rel_target_world(self, traj_time: float) -> np.ndarray:
        if (not self.is_ball_rel) or (self.base_traj_for_rel is None):
            raise ValueError("ball_in_robot 目标恢复需要有效的 base_trajectory")
        self._sync_base_traj(traj_time)
        target_rel = np.asarray(self.trajectory.position(traj_time), dtype=float).reshape(3)
        base_pos = np.asarray(self.base_traj_for_rel.position(traj_time), dtype=float).reshape(3)
        return target_rel + base_pos

    def _ball_rel_future_targets_world(self, traj_time: float) -> np.ndarray:
        if (not self.is_ball_rel) or (self.base_traj_for_rel is None):
            raise ValueError("ball_in_robot 未来参考恢复需要有效的 base_trajectory")
        self._sync_base_traj(traj_time)
        rel_future = np.asarray(
            self.trajectory.future_positions(
                traj_time + self.preview_lead, self.mpc.horizon, self.control_dt
            ),
            dtype=float,
        ).reshape(self.mpc.horizon, 3)
        base_future = np.asarray(
            self.base_traj_for_rel.future_positions(
                traj_time + self.preview_lead, self.mpc.horizon, self.control_dt
            ),
            dtype=float,
        ).reshape(self.mpc.horizon, 3)
        return rel_future + base_future

    def _desired_approach_dir(self, target_world: np.ndarray, abs_time: float) -> np.ndarray:
        if self.terminal_approach_dir is not None:
            vec = self.terminal_approach_dir.copy()
        else:
            base_pos = self._base_pos_world(abs_time)
            vec = target_world - base_pos
            vec[2] = 0.0
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return np.array([0.0, 1.0, 0.0], dtype=float)
        return vec / norm

    def _approach_axis_world(self) -> np.ndarray:
        rot = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        return rot @ self._approach_axis_local

    def _singularity_metric(self) -> float:
        jac_full = self._task_jacobian()
        jac = jac_full if self.use_orientation_task else jac_full[:3]
        gram = jac @ jac.T + 1e-6 * np.eye(jac.shape[0])
        cond = float(np.linalg.cond(gram))
        if (not np.isfinite(cond)) or cond <= 0.0:
            cond = 1e6
        return float(np.clip(np.log(cond), 0.0, 20.0))

    def _terminal_error(
        self, current_pos_world: np.ndarray, target_world: np.ndarray, terminal_abs_time: float
    ) -> np.ndarray:
        pos_err = current_pos_world - target_world
        if self.terminal_value_dim <= 3:
            return pos_err
        desired_dir = self._desired_approach_dir(target_world, terminal_abs_time)
        ee_axis = self._approach_axis_world()
        rot_err = np.cross(ee_axis, desired_dir) * self.terminal_rot_scale
        if self.terminal_value_dim == 6:
            return np.concatenate([pos_err, rot_err])
        sing_err = np.array([self._singularity_metric() * self.terminal_sing_scale], dtype=float)
        return np.concatenate([pos_err, rot_err, sing_err])

    def _terminal_state(
        self, current_pos_world: np.ndarray, target_world: np.ndarray, terminal_abs_time: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.terminal_value_dim <= 3:
            return current_pos_world.copy(), target_world.copy()
        err = self._terminal_error(current_pos_world, target_world, terminal_abs_time)
        current_state = np.zeros(self.terminal_value_dim, dtype=float)
        desired_state = np.zeros(self.terminal_value_dim, dtype=float)
        current_state[:3] = current_pos_world
        current_state[3:] = err[3:]
        desired_state[:3] = target_world
        return current_state, desired_state

    def _base_velocity_world(self, abs_time: float) -> np.ndarray:
        if (not self.is_ball_rel) or (self.base_traj_for_rel is None):
            return np.zeros(3, dtype=float)
        self._sync_base_traj(abs_time)
        p0 = self.base_traj_for_rel.position(abs_time)
        p1 = self.base_traj_for_rel.position(abs_time + self.control_dt)
        return (p1 - p0) / self.control_dt

    def _base_stop_time(self) -> float:
        if (not self.is_ball_rel) or (self.base_traj_for_rel is None):
            return np.nan
        duration = getattr(self.base_traj_for_rel, "duration", None)
        if callable(duration):
            try:
                return float(duration())
            except Exception:
                return np.nan
        return np.nan

    def _future_with_base_drift_comp(
        self, future_world: np.ndarray, abs_time: float
    ) -> np.ndarray:
        """
        约束QP的参考补偿：在保持“在线QP硬约束”框架不变的前提下，
        通过参考前移抵消底盘平移扰动，缓解“底盘未停时末端贴不住目标”。
        """
        if (not self.is_ball_rel) or (self.base_traj_for_rel is None):
            return future_world
        if self.use_pregrasp:
            return future_world
        phase_gain = 1.0
        if self.use_offset_tracking:
            phase_gain = self._offset_release_progress(abs_time)
        if phase_gain <= 1e-9:
            return future_world
        self._sync_base_traj(abs_time)
        base_now = self.base_traj_for_rel.position(abs_time)
        base_future = self.base_traj_for_rel.future_positions(
            abs_time + self.preview_lead,
            self.mpc.horizon,
            self.control_dt,
        )
        drift = base_future - base_now.reshape(1, 3)
        return np.asarray(future_world, dtype=float) - (phase_gain * self.base_ff_gain) * drift

    def _future_pos_weight_scales(self, abs_time: float) -> Optional[np.ndarray]:
        if self.use_offset_tracking:
            alpha = self._offset_release_progress(abs_time)
            phase_scale = float(
                (1.0 - alpha) * self.hold_pos_weight_scale + alpha * self.attach_pos_weight_scale
            )
        else:
            phase_scale = 1.0

        scales = np.full(self.mpc.horizon, float(phase_scale), dtype=float)

        if not self.use_uncertainty_aware_weighting:
            return None if np.allclose(scales, 1.0) else scales
        if (not self.is_ball_rel) or (self.base_traj_for_rel is None):
            return None if np.allclose(scales, 1.0) else scales
        cov_fn = getattr(self.base_traj_for_rel, "future_covariances_xy", None)
        if not callable(cov_fn):
            return None if np.allclose(scales, 1.0) else scales
        self._sync_base_traj(abs_time)
        covs = cov_fn(self.mpc.horizon, self.control_dt)
        covs = np.asarray(covs, dtype=float)
        if covs.shape != (self.mpc.horizon, 2, 2):
            return None if np.allclose(scales, 1.0) else scales
        traces = np.clip(np.trace(covs, axis1=1, axis2=2), 0.0, None)
        sigma = np.sqrt(traces)
        uncertainty_scales = np.exp(-self.uncertainty_beta * sigma)
        uncertainty_scales = np.clip(uncertainty_scales, self.uncertainty_min_scale, 1.0)
        if self.uncertainty_ema > 1e-9:
            uncertainty_scales = (
                self.uncertainty_ema * self._uncertainty_scales_last
                + (1.0 - self.uncertainty_ema) * uncertainty_scales
            )
        self._uncertainty_sigma_last = sigma
        self._uncertainty_scales_last = uncertainty_scales
        scales = scales * uncertainty_scales
        return scales

    def _manipulability_value_from_jac_pos(self, jac_pos: np.ndarray) -> float:
        gram = jac_pos @ jac_pos.T + self.manipulability_eps * np.eye(3)
        det_val = float(np.linalg.det(gram))
        return float(np.sqrt(max(det_val, 0.0)))

    def _manipulability_value_at_q(self, q_arm: np.ndarray) -> float:
        q_backup = self.data.qpos[self.arm_qpos_indices].copy()
        self.data.qpos[self.arm_qpos_indices] = np.clip(
            np.asarray(q_arm, dtype=float).reshape(self.arm_dof),
            self.arm_qpos_min,
            self.arm_qpos_max,
        )
        mujoco.mj_forward(self.model, self.data)
        jac_full = self._task_jacobian()
        w = self._manipulability_value_from_jac_pos(jac_full[:3])
        self.data.qpos[self.arm_qpos_indices] = q_backup
        mujoco.mj_forward(self.model, self.data)
        return w

    def _manipulability_guidance_rhs(self) -> np.ndarray:
        n = self.mpc.task_dim * self.mpc.horizon
        rhs = np.zeros(n, dtype=float)
        if not self.use_manipulability_guidance:
            return rhs

        jac_full = self._task_jacobian()
        jac_pos = jac_full[:3]
        w_now = self._manipulability_value_from_jac_pos(jac_pos)
        risk = float(
            np.clip(
                (self.manipulability_w_threshold - w_now) / self.manipulability_w_threshold,
                0.0,
                1.0,
            )
        )
        self._manip_last_w = w_now
        self._manip_last_risk = risk
        if risk <= 1e-6:
            return rhs

        # 数值梯度：通过 J^+ 把笛卡尔微扰映射到关节微扰，再评估操作度变化。
        q_now = self.data.qpos[self.arm_qpos_indices].copy()
        j_pinv = np.linalg.pinv(jac_pos, rcond=1e-3)
        delta = self.manipulability_fd_delta
        grad = np.zeros(3, dtype=float)
        for i in range(3):
            dx = np.zeros(3, dtype=float)
            dx[i] = delta
            dq = j_pinv @ dx
            w_plus = self._manipulability_value_at_q(q_now + dq)
            w_minus = self._manipulability_value_at_q(q_now - dq)
            grad[i] = (w_plus - w_minus) / (2.0 * delta)

        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < 1e-9:
            return rhs
        grad_dir = grad / grad_norm
        if grad_norm > self.manipulability_grad_clip:
            grad = grad * (self.manipulability_grad_clip / grad_norm)
            grad_dir = grad / max(np.linalg.norm(grad), 1e-9)

        lam_eff = self.manipulability_lambda * risk
        if self.manipulability_first_step_only:
            rhs[:3] = lam_eff * grad_dir
            return rhs

        for i in range(self.mpc.horizon):
            idx = i * self.mpc.task_dim
            decay = self.manipulability_horizon_decay ** i
            rhs[idx : idx + 3] = lam_eff * decay * grad_dir
        return rhs

    def _control_target_world(self, sim_time: float, abs_time: float) -> np.ndarray:
        # hold 阶段保留可达裁剪，避免长时间追踪不可达前置点导致抖动；
        # attach 阶段直接追原相对轨迹恢复的参考，避免“已经到目标附近但仍被可达裁剪点卡住”。
        if self.is_ball_rel and self.use_offset_tracking:
            traj_time = self._ball_rel_traj_time(sim_time, abs_time)
            attach_target = self._ball_rel_target_world(traj_time)
            hold_target = attach_target + np.array([0.0, self.offset_y, 0.0], dtype=float)
            hold_target_clipped = hold_target.copy()
            if self.base_traj_for_rel is not None:
                self._sync_base_traj(traj_time)
                base_pos = self.base_traj_for_rel.position(traj_time)
                mount_pos = base_pos + self.shoulder_offset
                hold_target_clipped = self._reachable_target(hold_target, mount_pos)
            alpha = self._offset_release_progress(abs_time)
            return (1.0 - alpha) * hold_target_clipped + alpha * attach_target
        if not self.is_ball_rel:
            return self._world_target(sim_time)
        if self.base_traj_for_rel is None:
            return self._world_target(sim_time)
        traj_time = self._ball_rel_traj_time(sim_time, abs_time)
        self._sync_base_traj(traj_time)
        base_pos = self.base_traj_for_rel.position(traj_time)
        mount_pos = base_pos + self.shoulder_offset
        return self._reachable_target(self._ball_rel_target_world(traj_time), mount_pos)

    def _predict_opt_zone_entry_index(self, abs_time: float) -> Optional[int]:
        """
        在预测视界内估计目标进入最佳作业区（相对肩部距离区间）的位置索引。
        返回:
            - i (0-based): 第 i 个预测步进入区间
            - None: 预测视界内未进入
        """
        if self.mpc.horizon <= 0:
            return None

        def phase_distance(a: np.ndarray, b: np.ndarray) -> float:
            d = np.asarray(a, dtype=float).reshape(3) - np.asarray(b, dtype=float).reshape(3)
            if self.phase_use_planar_distance:
                d[2] = 0.0
            return float(np.linalg.norm(d))

        # ball_in_robot: 只使用相对轨迹 + 底盘预测恢复世界系参考。
        if self.is_ball_rel and self.base_traj_for_rel is not None:
            self._sync_base_traj(abs_time)
            base_future = np.asarray(
                self.base_traj_for_rel.future_positions(
                    abs_time + self.preview_lead,
                    self.mpc.horizon,
                    self.control_dt,
                ),
                dtype=float,
            ).reshape(self.mpc.horizon, 3)
            target_future = self._ball_rel_future_targets_world(abs_time)
            for i in range(base_future.shape[0]):
                shoulder_world = base_future[i] + self.shoulder_offset
                dist = phase_distance(target_future[i], shoulder_world)
                if self.phase_opt_radius_min <= dist <= self.phase_opt_radius_max:
                    return int(i)
            return None

        # 其它场景 fallback: 用当前肩部位置与世界系未来目标估计进入时刻。
        sim_time = max(0.0, float(abs_time - self.time_offset))
        future_world = self._world_future_targets(sim_time)
        shoulder_world = self.data.xpos[self.shoulder_body_id].copy()
        for i in range(future_world.shape[0]):
            dist = phase_distance(future_world[i], shoulder_world)
            if self.phase_opt_radius_min <= dist <= self.phase_opt_radius_max:
                return int(i)
        return None

    def _activate_attack_phase(self, sim_time: float, abs_time: float, reason: str) -> None:
        self.approach_active = True
        if self.phase_instant_attack and self.approach_speed > 1e-9:
            # 立即进入“贴近段”：将起始时间回拨，使当前 offset 直接衰减到 0。
            self.approach_start_time = float(abs_time) - self.pregrasp_offset / self.approach_speed - 1e-6
        else:
            self.approach_start_time = float(abs_time)
        self._phase_confirm_count = 0
        self._phase_x_hold_count = 0
        print(f"[phase] switch to attack at t={sim_time:.2f}s, reason={reason}")

    def _maybe_trigger_predictive_phase(
        self,
        sim_time: float,
        abs_time: float,
        current_pos_world: Optional[np.ndarray] = None,
        target_world: Optional[np.ndarray] = None,
    ) -> None:
        """
        蛰伏->出击触发:
            当预测进入最佳作业区的索引持续落在阈值内，且满足最小蛰伏时间时，开启 approach_active。
        """
        if (not self.use_pregrasp) or (not self.use_predictive_phase_switch) or self.approach_active:
            return

        if self.phase_use_x_gate_switch:
            ee = (
                np.asarray(current_pos_world, dtype=float).reshape(3)
                if current_pos_world is not None
                else self.data.site_xpos[self.ee_site_id].copy()
            )
            tgt = (
                np.asarray(target_world, dtype=float).reshape(3)
                if target_world is not None
                else np.asarray(self._world_target(max(0.0, float(sim_time))), dtype=float).reshape(3)
            )
            x_err = abs(float(ee[0]) - float(tgt[0]))
            self._phase_last_x_err = float(x_err)
            if x_err <= self.phase_x_gate_half_width:
                self._phase_x_hold_count += 1
            else:
                self._phase_x_hold_count = 0
            if (
                self._phase_x_hold_count >= self.phase_x_gate_hold_steps
                and float(sim_time) >= self.phase_min_hold_s
            ):
                self._activate_attack_phase(
                    sim_time=sim_time,
                    abs_time=abs_time,
                    reason=(
                        f"x_gate(|dx|={x_err:.4f}<= {self.phase_x_gate_half_width:.4f}, "
                        f"hold={self.phase_x_gate_hold_steps})"
                    ),
                )
            return

        enter_idx = self._predict_opt_zone_entry_index(abs_time)
        self._phase_last_enter_index = int(enter_idx) if enter_idx is not None else -1
        if enter_idx is not None and enter_idx <= self.phase_trigger_index:
            self._phase_confirm_count += 1
        else:
            self._phase_confirm_count = 0

        if (
            self._phase_confirm_count >= self.phase_confirm_steps
            and float(sim_time) >= self.phase_min_hold_s
        ):
            self._activate_attack_phase(
                sim_time=sim_time,
                abs_time=abs_time,
                reason=(
                    f"predictive(enter_idx={self._phase_last_enter_index}, "
                    f"zone=[{self.phase_opt_radius_min:.2f},{self.phase_opt_radius_max:.2f}]m, "
                    f"metric={'xy' if self.phase_use_planar_distance else 'xyz'})"
                ),
            )

    def _control_future_targets_world(
        self,
        sim_time: float,
        abs_time: float,
        current_pos_world: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # 两阶段偏置模式下，hold: target+offset, attach: target。
        # 仅 hold 阶段保留可达裁剪；attach 阶段直接给原相对轨迹恢复的参考，避免末端在目标附近悬停。
        if self.is_ball_rel and self.use_offset_tracking:
            traj_time = self._ball_rel_traj_time(sim_time, abs_time)
            attach_future = self._ball_rel_future_targets_world(traj_time)
            offset_vals = np.array(
                [
                    self._offset_y_at(
                        float(abs_time) + self.preview_lead + (i + 1) * self.control_dt
                    )
                    for i in range(self.mpc.horizon)
                ],
                dtype=float,
            ).reshape(self.mpc.horizon, 1)
            hold_future = attach_future + np.hstack(
                [
                    np.zeros((self.mpc.horizon, 1), dtype=float),
                    offset_vals,
                    np.zeros((self.mpc.horizon, 1), dtype=float),
                ]
            )
            hold_future_clipped = hold_future.copy()
            if self.base_traj_for_rel is not None:
                self._sync_base_traj(traj_time)
                base_future = self.base_traj_for_rel.future_positions(
                    traj_time + self.preview_lead,
                    self.mpc.horizon,
                    self.control_dt,
                )
                hold_future_clipped = np.zeros((self.mpc.horizon, 3), dtype=float)
                for i in range(self.mpc.horizon):
                    mount_pos = base_future[i] + self.shoulder_offset
                    hold_future_clipped[i] = self._reachable_target(hold_future[i], mount_pos)
            blend_vals = np.array(
                [
                    self._offset_release_progress(
                        float(abs_time) + self.preview_lead + (i + 1) * self.control_dt
                    )
                    for i in range(self.mpc.horizon)
                ],
                dtype=float,
            ).reshape(self.mpc.horizon, 1)
            return (1.0 - blend_vals) * hold_future_clipped + blend_vals * attach_future
        if self.intercept_planner is not None and current_pos_world is not None:
            return self.intercept_planner.plan(
                current_pos_world=current_pos_world,
                abs_time=abs_time,
                horizon=self.mpc.horizon,
                control_dt=self.control_dt,
            )
        if self.use_pregrasp and self.is_ball_rel and self.base_traj_for_rel is not None:
            self._sync_base_traj(abs_time)
            enter_idx = None
            if self.use_predictive_phase_switch and (not self.approach_active):
                enter_idx = self._predict_opt_zone_entry_index(abs_time)
                self._phase_last_enter_index = int(enter_idx) if enter_idx is not None else -1
            base_future = self.base_traj_for_rel.future_positions(
                abs_time + self.preview_lead,
                self.mpc.horizon,
                self.control_dt,
            )
            future = np.zeros((self.mpc.horizon, 3), dtype=float)
            rel_future = np.asarray(
                self.trajectory.future_positions(
                    abs_time + self.preview_lead, self.mpc.horizon, self.control_dt
                ),
                dtype=float,
            ).reshape(self.mpc.horizon, 3)
            for i in range(self.mpc.horizon):
                base_pos = base_future[i]
                t_abs = abs_time + self.preview_lead + (i + 1) * self.control_dt
                if self.approach_active:
                    offset = self._pregrasp_offset_at(t_abs)
                elif enter_idx is not None and i >= enter_idx:
                    # 在预测视界内平滑启动“出击段”，避免硬切导致的抖动。
                    span = max(1, self.mpc.horizon - int(enter_idx))
                    alpha = float(i - int(enter_idx) + 1) / float(span)
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    offset = (1.0 - alpha) * self.pregrasp_offset
                else:
                    offset = self.pregrasp_offset
                target_rel = rel_future[i]
                desired_rel = target_rel - offset * self.pregrasp_dir
                future[i] = desired_rel + base_pos
            return future
        if not self.is_ball_rel:
            future = self._world_future_targets(sim_time)
            offset_y = self._offset_y_at(abs_time)
            if abs(offset_y) > 1e-12:
                future = future + np.array([0.0, offset_y, 0.0], dtype=float)
            return future
        if self.base_traj_for_rel is None:
            future = self._world_future_targets(sim_time)
            offset_y = self._offset_y_at(abs_time)
            if abs(offset_y) > 1e-12:
                future = future + np.array([0.0, offset_y, 0.0], dtype=float)
            return future
        traj_time = self._ball_rel_traj_time(sim_time, abs_time)
        self._sync_base_traj(traj_time)
        base_future = self.base_traj_for_rel.future_positions(
            traj_time + self.preview_lead,
            self.mpc.horizon,
            self.control_dt,
        )
        target_future = self._ball_rel_future_targets_world(traj_time)
        future = np.zeros((self.mpc.horizon, 3), dtype=float)
        for i in range(self.mpc.horizon):
            base_pos = base_future[i]
            mount_pos = base_pos + self.shoulder_offset
            future[i] = self._reachable_target(target_future[i], mount_pos)
        offset_y = self._offset_y_at(abs_time)
        if abs(offset_y) > 1e-12:
            future = future + np.array([0.0, offset_y, 0.0], dtype=float)
        return future

    def _apply_joint_position(self, q_des: np.ndarray):
        if q_des.shape != (self.arm_dof,):
            raise ValueError("q_des 尺寸错误")
        q_des = np.clip(q_des, self.arm_qpos_min, self.arm_qpos_max)
        self.data.ctrl[self.arm_actuator_ids] = q_des

    def _world_target(self, sim_time: float) -> np.ndarray:
        if self.is_ball_rel:
            # ball_in_robot: 通过相对轨迹 + 底盘预测恢复等价世界系参考。
            if self.base_traj_for_rel is None:
                return np.asarray(self.trajectory.position(sim_time), dtype=float).reshape(3)
            return self._ball_rel_target_world(sim_time)
        base_origin = getattr(self.trajectory, "base_origin", None)
        if base_origin is not None:
            return self.trajectory.position(sim_time) + base_origin
        return self.trajectory.position(sim_time)

    def _world_future_targets(self, sim_time: float) -> np.ndarray:
        if self.is_ball_rel:
            # ball_in_robot: 未来参考同样由相对轨迹与底盘预测共同恢复。
            if self.base_traj_for_rel is None:
                rel_future = self.trajectory.future_positions(
                    sim_time + self.preview_lead, self.mpc.horizon, self.control_dt
                )
                return np.asarray(rel_future, dtype=float).reshape(self.mpc.horizon, 3)
            return self._ball_rel_future_targets_world(sim_time)
        future_rel = self.trajectory.future_positions(
            sim_time + self.preview_lead, self.mpc.horizon, self.control_dt
        )
        base_origin = getattr(self.trajectory, "base_origin", None)
        if base_origin is not None:
            return future_rel + base_origin
        return future_rel

    def _pregrasp_offset_at(self, abs_time: float) -> float:
        if not self.use_pregrasp:
            return 0.0
        if not self.approach_active:
            return self.pregrasp_offset
        dt = max(0.0, float(abs_time) - float(self.approach_start_time))
        offset = max(self.pregrasp_offset - self.approach_speed * dt, 0.0)
        return offset

    def _warm_start_to_pose(
        self,
        target_pos_world: np.ndarray,
        max_duration: float = 2.5,
        tol: float = 0.05,
        viewer=None,
    ):
        steps = int(max_duration / self.dt)
        self.data.mocap_pos[self.target_mocap_id] = target_pos_world
        qdot_cmd = np.zeros(self.arm_dof)
        q_des = self.data.qpos[self.arm_qpos_indices].copy()
        last_dist = np.inf

        if steps <= 0:
            last_dist = float(
                np.linalg.norm(self.data.site_xpos[self.ee_site_id] - target_pos_world)
            )
            self.last_qdot_cmd = np.zeros_like(qdot_cmd)
            self.mpc.prev_twist = np.zeros_like(self.mpc.prev_twist)
            self.time_offset = self.data.time
            return last_dist

        for k in range(steps):
            step_start = time.time()
            if viewer and (not viewer.is_running()):
                break

            if self.is_ball_rel and self.base_traj_for_rel is not None and self.chassis_mocap_id >= 0:
                # 两阶段偏置模式: warm start 时固定底盘，避免预热阶段“偷走”动态轨迹时间。
                if self.use_offset_tracking:
                    abs_time = 0.0
                else:
                    # 其它模式沿用原逻辑：warm start 阶段按仿真步推进底盘。
                    abs_time = k * self.dt
                self._sync_base_traj(abs_time)
                base_pos = self.base_traj_for_rel.position(abs_time)
                self.data.mocap_pos[self.chassis_mocap_id] = base_pos
                self.data.mocap_quat[self.chassis_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
                mujoco.mj_forward(self.model, self.data)

            if k % self.control_every == 0:
                sim_time = self.data.time
                abs_time = self.data.time
                current_pos_world = self.data.site_xpos[self.ee_site_id].copy()
                current_pos = current_pos_world
                future_targets = self._control_future_targets_world(
                    sim_time=sim_time,
                    abs_time=abs_time,
                    current_pos_world=current_pos_world,
                )
                future_for_opt = (
                    self._future_with_base_drift_comp(future_targets, abs_time=abs_time)
                    if self.use_constrained_qp
                    else future_targets
                )
                e_hat = self._task_error(current_pos, future_for_opt, abs_time=abs_time)
                terminal_time = abs_time + self.preview_lead + self.mpc.horizon * self.control_dt
                current_state, desired_state = self._terminal_state(
                    current_pos, future_for_opt[-1], terminal_time
                )
                pos_weight_scales = self._future_pos_weight_scales(abs_time)
                manip_rhs = self._manipulability_guidance_rhs()

                jac_full = self._task_jacobian()
                jac = jac_full if self.use_orientation_task else jac_full[:3]
                jac_pinv = self._jacobian_pinv(jac)
                if self.use_constrained_qp:
                    q_now = self.data.qpos[self.arm_qpos_indices].copy()
                    ee_x_upper_seq = self._qp_ee_x_upper_sequence(
                        sim_time=sim_time,
                        target_world_now=target_pos_world,
                    )
                    ee_y_upper_seq = self._qp_ee_y_upper_sequence(
                        sim_time=sim_time,
                        target_world_now=target_pos_world,
                    )
                    ee_z_lower_seq = self._qp_ee_z_lower_sequence(
                        sim_time=sim_time,
                        target_world_now=target_pos_world,
                    )
                    a_cons, l_cons, u_cons = self._build_constrained_qp_bounds(
                        jac_pinv=jac_pinv,
                        q_ref=q_now,
                        current_pos_world=current_pos_world,
                        ee_x_upper_seq=ee_x_upper_seq,
                        ee_y_upper_seq=ee_y_upper_seq,
                        ee_z_lower_seq=ee_z_lower_seq,
                    )
                    twist = self.mpc.solve(
                        e_hat,
                        current_state=current_state,
                        desired_terminal=desired_state,
                        pos_weight_scales=pos_weight_scales,
                        extra_rhs=manip_rhs,
                        linear_constraint_matrix=a_cons,
                        linear_lower_bound=l_cons,
                        linear_upper_bound=u_cons,
                        qp_solver=self.qp_solver,
                    )
                    if self.mpc.last_solve_ok:
                        active_err = float(np.linalg.norm(future_targets[0] - current_pos))
                        if self.use_offset_tracking and self.offset_active:
                            qp_scale = 1.0
                        else:
                            qp_scale = self._qp_qdot_scale(active_err)
                        if self.use_orientation_task:
                            twist_cmd = twist.copy() * qp_scale
                            self.mpc.prev_twist = twist_cmd.copy()
                        else:
                            twist_cmd = twist[:3].copy() * qp_scale
                            self.mpc.prev_twist[:3] = twist_cmd
                            self.mpc.prev_twist[3:] = 0.0
                        qdot_raw = jac_pinv @ twist_cmd
                        qdot_cmd = self._limit_joint_accel(qdot_raw, qdot_cmd, self.control_dt)
                        if self.qp_enforce_joint_vel:
                            qdot_cmd = np.clip(qdot_cmd, -self.velocity_limit, self.velocity_limit)
                        if self.qp_enforce_ee_x_upper:
                            qdot_cmd = self._project_qdot_to_axis_upper(
                                qdot_cmd,
                                jac_axis_row=jac_full[0],
                                current_value=float(current_pos_world[0]),
                                upper_value=float(ee_x_upper_seq[0]),
                            )
                        if self.qp_enforce_ee_y_upper:
                            qdot_cmd = self._project_qdot_to_axis_upper(
                                qdot_cmd,
                                jac_axis_row=jac_full[1],
                                current_value=float(current_pos_world[1]),
                                upper_value=float(ee_y_upper_seq[0]),
                            )
                        if self.qp_enforce_ee_z_lower:
                            qdot_cmd = self._project_qdot_to_axis_lower(
                                qdot_cmd,
                                jac_axis_row=jac_full[2],
                                current_value=float(current_pos_world[2]),
                                lower_value=float(ee_z_lower_seq[0]),
                            )
                        q_des = q_now + qdot_cmd * self.control_dt
                        if self.qp_enforce_joint_pos:
                            q_des = np.clip(q_des, self.arm_qpos_min, self.arm_qpos_max)
                    else:
                        if self.qp_infeasible_policy == "hold":
                            self._log_qp_infeasible(sim_time=float(sim_time))
                            qdot_cmd = np.zeros(self.arm_dof, dtype=float)
                            self.mpc.prev_twist[:] = 0.0
                        else:
                            raise RuntimeError(
                                f"未支持的 qp_infeasible_policy={self.qp_infeasible_policy}"
                            )
                else:
                    cond_val = np.linalg.cond(jac @ jac.T)
                    if self.intercept_planner is not None:
                        ball_err = float(np.linalg.norm(future_targets[0] - current_pos))
                    else:
                        ball_err = float(np.linalg.norm(target_pos_world - current_pos))
                    fb_gain = self._feedback_gain(ball_err)

                    twist = self.mpc.solve(
                        e_hat,
                        current_state=current_state,
                        desired_terminal=desired_state,
                        pos_weight_scales=pos_weight_scales,
                        extra_rhs=manip_rhs,
                    )
                    twist_fb = (twist.copy() if self.use_orientation_task else twist[:3].copy()) * fb_gain
                    twist_ff = np.zeros_like(twist_fb)
                    if (
                        self.intercept_planner is None
                        and (not self.use_offset_tracking)
                        and self.is_ball_rel
                        and self.base_traj_for_rel is not None
                    ):
                        desired_now = self._control_target_world(sim_time=sim_time, abs_time=abs_time)
                        desired_next = future_targets[0]
                        desired_vel = (desired_next - desired_now) / self.control_dt
                        base_vel = self._base_velocity_world(abs_time)
                        ff_xyz = self.base_ff_gain * (desired_vel - base_vel)
                        if self.use_orientation_task:
                            twist_ff[:3] = ff_xyz
                        else:
                            twist_ff = ff_xyz
                    twist_cmd = twist_fb + twist_ff

                    if cond_val > self.cond_threshold:
                        scale = np.clip(self.cond_threshold / cond_val, self.min_twist_scale, 1.0)
                        twist_cmd = twist_cmd * scale
                        twist_fb = twist_fb * scale
                    twist_cmd, twist_fb = self._limit_twist_cmd(twist_cmd, twist_fb)

                    # 让 MPC 的内部状态与实际执行的 feedback 保持一致，避免 warm_start→control 的“带速”抖动
                    if self.use_orientation_task:
                        self.mpc.prev_twist = twist_fb.copy()
                    else:
                        self.mpc.prev_twist[:3] = twist_fb
                        self.mpc.prev_twist[3:] = 0.0

                    qdot_task = jac_pinv @ twist_cmd
                    vel_ratio_task = np.max(np.abs(qdot_task) / self.velocity_limit)
                    task_budget = float(np.clip(1.0 - vel_ratio_task, 0.0, 1.0))

                    null_w = 1.0 / (1.0 + (ball_err / self.nullspace_err_scale) ** 2)
                    joint_offset = self.q_nominal - self.data.qpos[self.arm_qpos_indices]
                    dyn_gain = (self.nullspace_gain * null_w * task_budget) * (
                        1.0 + np.clip(np.linalg.norm(joint_offset) / self.nullspace_scale, 0.0, 3.0)
                    )
                    nullspace_term = dyn_gain * joint_offset
                    qdot_cmd_raw = qdot_task + (
                        (np.identity(self.arm_dof) - jac_pinv @ jac) @ nullspace_term
                    )
                    vel_ratio = np.max(np.abs(qdot_cmd_raw) / self.velocity_limit)
                    if vel_ratio > 1.0:
                        qdot_cmd_raw = qdot_cmd_raw / vel_ratio
                    qdot_cmd = self._limit_joint_accel(qdot_cmd_raw, qdot_cmd, self.control_dt)
                    qdot_cmd = np.clip(qdot_cmd, -self.velocity_limit, self.velocity_limit)
                    joint_bias = self.drift_gain * joint_offset * self.control_dt
                    # position 伺服：用内部的 q_des 积分（而不是用当前 qpos），这样当机械臂跟不上时误差会累积，伺服会自动“加力追上”
                    q_des = q_des + qdot_cmd * self.control_dt + joint_bias
                    q_des = np.clip(q_des, self.arm_qpos_min, self.arm_qpos_max)

            self._apply_joint_position(q_des)

            mujoco.mj_step(self.model, self.data)
            if viewer and (k % self.render_every == 0):
                viewer.sync()
            last_dist = float(
                np.linalg.norm(self.data.site_xpos[self.ee_site_id] - target_pos_world)
            )
            if last_dist <= tol:
                break

            if viewer:
                compute_elapsed = time.time() - step_start
                if self.dt - compute_elapsed > 0:
                    time.sleep(self.dt - compute_elapsed)
        # warm_start 结束后不要把速度/扭量“带入”主控制回路，避免开头来回抖动
        self.last_qdot_cmd = np.zeros_like(qdot_cmd)
        self.mpc.prev_twist = np.zeros_like(self.mpc.prev_twist)
        self.time_offset = self.data.time
        return last_dist

    def _control_loop(
        self,
        viewer=None,
        max_time: float = 20.0,
        callback=None,
        step_callback=None,
        realtime_sync: bool = True,
    ):
        last_log_time = -np.inf
        step_count = 0
        wall_start = time.time()
        qdot_cmd = self.last_qdot_cmd.copy()
        q_des = self.data.qpos[self.arm_qpos_indices].copy()
        qdot_pre_clip = np.zeros_like(qdot_cmd)
        cond_last = np.nan
        while True:
            step_start = time.time()
            sim_time = self.data.time - self.time_offset

            if self.is_ball_rel and self.base_traj_for_rel is not None and self.chassis_mocap_id >= 0:
                # 偏置两阶段模式使用控制相对时间，保证“按开始键后”从轨迹起点开始动态抓取。
                base_traj_time = sim_time if self.use_offset_tracking else self.data.time
                self._sync_base_traj(base_traj_time)
                base_pos = self.base_traj_for_rel.position(base_traj_time)
                self.data.mocap_pos[self.chassis_mocap_id] = base_pos
                self.data.mocap_quat[self.chassis_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
                mujoco.mj_forward(self.model, self.data)

            if viewer:
                if not viewer.is_running():
                    break
            else:
                if sim_time >= max_time:
                    break

            target_world_ref = self._world_target(sim_time)
            current_pos_world = self.data.site_xpos[self.ee_site_id].copy()
            target_world = current_pos_world if self._attach_target else target_world_ref
            self.data.mocap_pos[self.target_mocap_id] = target_world

            current_pos = current_pos_world.copy()
            future_world = self._control_future_targets_world(
                sim_time=sim_time,
                abs_time=self.data.time,
                current_pos_world=current_pos_world,
            )
            e_hat = None

            if step_count % self.control_every == 0:
                need_recompute_future = False
                if self.use_offset_tracking and self.offset_active:
                    # 第一阶段: 跟踪 y 偏置轨迹；连续命中后切到原轨迹（第二阶段）。
                    # 切换判据：末端 x 到达阈值后，连续命中若干控制步再切换。
                    hold_ref_switch = target_world_ref + np.array([0.0, self.offset_y, 0.0], dtype=float)
                    offset_dist = float(np.linalg.norm(current_pos_world - hold_ref_switch))
                    self._offset_last_dist = offset_dist
                    target_x = float(target_world_ref[0])
                    x_gate_threshold = target_x - self.offset_switch_x_front
                    if self.offset_switch_x_gate_enable:
                        x_gate_ready = float(current_pos_world[0]) >= x_gate_threshold
                    else:
                        x_gate_ready = True
                    self._offset_x_gate_ready = bool(x_gate_ready)
                    self._offset_x_gate_threshold = float(x_gate_threshold)
                    self._offset_x_gate_delta = float(current_pos_world[0] - x_gate_threshold)

                    hold_ref_switch = target_world_ref + np.array(
                        [0.0, self.offset_y, 0.0], dtype=float
                    )
                    x_align_err = abs(float(current_pos_world[0]) - float(hold_ref_switch[0]))
                    yz_hold_err = float(
                        np.linalg.norm(
                            np.asarray(current_pos_world[1:3], dtype=float)
                            - np.asarray(hold_ref_switch[1:3], dtype=float)
                        )
                    )
                    hold_ready = x_gate_ready
                    if self.offset_switch_x_align_tol > 1e-9:
                        hold_ready = hold_ready and (x_align_err <= self.offset_switch_x_align_tol)
                    if self.offset_switch_yz_tol > 1e-9:
                        hold_ready = hold_ready and (yz_hold_err <= self.offset_switch_yz_tol)

                    if hold_ready:
                        self._offset_hit_count += 1
                    else:
                        self._offset_hit_count = 0

                    if self._offset_hit_count >= self.offset_trigger_steps:
                        self.offset_active = False
                        self._offset_release_start_time = float(self.data.time)
                        self._offset_hit_count = 0
                        need_recompute_future = True

                was_attack = bool(self.approach_active)
                self._maybe_trigger_predictive_phase(
                    sim_time=sim_time,
                    abs_time=float(self.data.time),
                    current_pos_world=current_pos_world,
                    target_world=target_world_ref,
                )
                if (not was_attack) and self.approach_active:
                    # 切相位的这一拍立即重算参考，避免继续使用 hold 轨迹一个控制周期。
                    need_recompute_future = True
                if need_recompute_future:
                    future_world = self._control_future_targets_world(
                        sim_time=sim_time,
                        abs_time=self.data.time,
                        current_pos_world=current_pos_world,
                    )
                future_for_opt = (
                    self._future_with_base_drift_comp(future_world, abs_time=float(self.data.time))
                    if self.use_constrained_qp
                    else future_world
                )
                e_hat = self._task_error(current_pos, future_for_opt, abs_time=float(self.data.time))

                terminal_time = (
                    self.data.time + self.preview_lead + self.mpc.horizon * self.control_dt
                )
                current_state, desired_state = self._terminal_state(
                    current_pos, future_for_opt[-1], terminal_time
                )
                pos_weight_scales = self._future_pos_weight_scales(float(self.data.time))
                manip_rhs = self._manipulability_guidance_rhs()
                jac_full = self._task_jacobian()
                jac = jac_full if self.use_orientation_task else jac_full[:3]
                jac_pinv = self._jacobian_pinv(jac)
                cond_val = np.linalg.cond(jac @ jac.T)
                cond_last = float(cond_val)
                if self.use_constrained_qp:
                    q_now = self.data.qpos[self.arm_qpos_indices].copy()
                    ee_x_upper_seq = self._qp_ee_x_upper_sequence(
                        sim_time=sim_time,
                        target_world_now=target_world_ref,
                    )
                    ee_y_upper_seq = self._qp_ee_y_upper_sequence(
                        sim_time=sim_time,
                        target_world_now=target_world_ref,
                    )
                    ee_z_lower_seq = self._qp_ee_z_lower_sequence(
                        sim_time=sim_time,
                        target_world_now=target_world_ref,
                    )
                    a_cons, l_cons, u_cons = self._build_constrained_qp_bounds(
                        jac_pinv=jac_pinv,
                        q_ref=q_now,
                        current_pos_world=current_pos_world,
                        ee_x_upper_seq=ee_x_upper_seq,
                        ee_y_upper_seq=ee_y_upper_seq,
                        ee_z_lower_seq=ee_z_lower_seq,
                    )
                    twist = self.mpc.solve(
                        e_hat,
                        current_state=current_state,
                        desired_terminal=desired_state,
                        pos_weight_scales=pos_weight_scales,
                        extra_rhs=manip_rhs,
                        linear_constraint_matrix=a_cons,
                        linear_lower_bound=l_cons,
                        linear_upper_bound=u_cons,
                        qp_solver=self.qp_solver,
                    )
                    if self.mpc.last_solve_ok:
                        active_err = float(np.linalg.norm(future_world[0] - current_pos))
                        if self.use_offset_tracking and self.offset_active:
                            qp_scale = 1.0
                        else:
                            qp_scale = self._qp_qdot_scale(active_err)
                        if self.use_orientation_task:
                            twist_cmd = twist.copy() * qp_scale
                            self.mpc.prev_twist = twist_cmd.copy()
                        else:
                            twist_cmd = twist[:3].copy() * qp_scale
                            self.mpc.prev_twist[:3] = twist_cmd
                            self.mpc.prev_twist[3:] = 0.0
                        qdot_raw = jac_pinv @ twist_cmd
                        qdot_pre_clip = qdot_raw.copy()
                        qdot_cmd = self._limit_joint_accel(qdot_raw, self.last_qdot_cmd, self.control_dt)
                        if self.qp_enforce_joint_vel:
                            qdot_cmd = np.clip(qdot_cmd, -self.velocity_limit, self.velocity_limit)
                        if self.qp_enforce_ee_x_upper:
                            qdot_cmd = self._project_qdot_to_axis_upper(
                                qdot_cmd,
                                jac_axis_row=jac_full[0],
                                current_value=float(current_pos_world[0]),
                                upper_value=float(ee_x_upper_seq[0]),
                            )
                        if self.qp_enforce_ee_y_upper:
                            qdot_cmd = self._project_qdot_to_axis_upper(
                                qdot_cmd,
                                jac_axis_row=jac_full[1],
                                current_value=float(current_pos_world[1]),
                                upper_value=float(ee_y_upper_seq[0]),
                            )
                        if self.qp_enforce_ee_z_lower:
                            qdot_cmd = self._project_qdot_to_axis_lower(
                                qdot_cmd,
                                jac_axis_row=jac_full[2],
                                current_value=float(current_pos_world[2]),
                                lower_value=float(ee_z_lower_seq[0]),
                            )
                        self.last_qdot_cmd = qdot_cmd.copy()
                        q_des = q_now + qdot_cmd * self.control_dt
                        if self.qp_enforce_joint_pos:
                            q_des = np.clip(q_des, self.arm_qpos_min, self.arm_qpos_max)
                    else:
                        if self.qp_infeasible_policy == "hold":
                            self._log_qp_infeasible(sim_time=float(sim_time))
                            qdot_cmd = np.zeros(self.arm_dof, dtype=float)
                            qdot_pre_clip = qdot_cmd.copy()
                            self.last_qdot_cmd = qdot_cmd.copy()
                            self.mpc.prev_twist[:] = 0.0
                        else:
                            raise RuntimeError(
                                f"未支持的 qp_infeasible_policy={self.qp_infeasible_policy}"
                            )
                else:
                    twist = self.mpc.solve(
                        e_hat,
                        current_state=current_state,
                        desired_terminal=desired_state,
                        pos_weight_scales=pos_weight_scales,
                        extra_rhs=manip_rhs,
                    )
                    if self.intercept_planner is not None or self.use_pregrasp or self.use_offset_tracking:
                        ball_err = float(np.linalg.norm(future_world[0] - current_pos))
                    else:
                        ball_err = float(np.linalg.norm(target_world - current_pos))
                    fb_gain = self._feedback_gain(ball_err)

                    twist_fb = (twist.copy() if self.use_orientation_task else twist[:3].copy()) * fb_gain
                    twist_ff = np.zeros_like(twist_fb)
                    enable_base_ff = (
                        self.intercept_planner is None
                        and (not self.use_pregrasp)
                        and self.is_ball_rel
                        and self.base_traj_for_rel is not None
                        and ((not self.use_offset_tracking) or (self.use_offset_tracking and (not self.offset_active)))
                    )
                    if enable_base_ff:
                        ff_phase_gain = (
                            self._offset_release_progress(self.data.time)
                            if self.use_offset_tracking
                            else 1.0
                        )
                        desired_now = self._control_target_world(sim_time=sim_time, abs_time=self.data.time)
                        desired_next = future_world[0]
                        desired_vel = (desired_next - desired_now) / self.control_dt
                        base_vel = self._base_velocity_world(self.data.time)
                        ff_xyz = ff_phase_gain * self.base_ff_gain * (desired_vel - base_vel)
                        if self.use_orientation_task:
                            twist_ff[:3] = ff_xyz
                        else:
                            twist_ff = ff_xyz
                    twist_cmd = twist_fb + twist_ff

                    if cond_val > self.cond_threshold:
                        scale = np.clip(self.cond_threshold / cond_val, self.min_twist_scale, 1.0)
                        twist_cmd = twist_cmd * scale
                        twist_fb = twist_fb * scale
                    twist_cmd, twist_fb = self._limit_twist_cmd(twist_cmd, twist_fb)

                    if self.use_orientation_task:
                        self.mpc.prev_twist = twist_fb.copy()
                    else:
                        self.mpc.prev_twist[:3] = twist_fb
                        self.mpc.prev_twist[3:] = 0.0

                    qdot_task = jac_pinv @ twist_cmd
                    vel_ratio_task = np.max(np.abs(qdot_task) / self.velocity_limit)
                    task_budget = float(np.clip(1.0 - vel_ratio_task, 0.0, 1.0))

                    null_w = 1.0 / (1.0 + (ball_err / self.nullspace_err_scale) ** 2)
                    joint_offset = self.q_nominal - self.data.qpos[self.arm_qpos_indices]
                    dyn_gain = (self.nullspace_gain * null_w * task_budget) * (
                        1.0 + np.clip(np.linalg.norm(joint_offset) / self.nullspace_scale, 0.0, 3.0)
                    )
                    nullspace_term = dyn_gain * joint_offset
                    qdot_cmd = qdot_task + (
                        (np.identity(self.arm_dof) - jac_pinv @ jac) @ nullspace_term
                    )
                    qdot_pre_clip = qdot_cmd.copy()
                    vel_ratio = np.max(np.abs(qdot_cmd) / self.velocity_limit)
                    if vel_ratio > 1.0:
                        qdot_cmd = qdot_cmd / vel_ratio
                    qdot_cmd = self._limit_joint_accel(qdot_cmd, self.last_qdot_cmd, self.control_dt)
                    qdot_cmd = np.clip(qdot_cmd, -self.velocity_limit, self.velocity_limit)
                    self.last_qdot_cmd = qdot_cmd.copy()
                    joint_bias = self.drift_gain * joint_offset * self.control_dt
                    q_des = q_des + qdot_cmd * self.control_dt + joint_bias
                    q_des = np.clip(q_des, self.arm_qpos_min, self.arm_qpos_max)

                if callback is not None:
                    terminal_err = self._terminal_error(
                        current_pos, future_world[-1], terminal_time
                    )
                    uncertainty_scale_first = (
                        float(self._uncertainty_scales_last[0])
                        if self._uncertainty_scales_last.size > 0
                        else 1.0
                    )
                    callback(
                        {
                            "time": float(sim_time),
                            "abs_time": float(self.data.time),
                            "current_pos": current_pos.copy(),
                            "target_pos": future_world[0].copy(),
                            "terminal_pos": future_world[-1].copy(),
                            "terminal_error": terminal_err.copy(),
                            "q": self.data.qpos[self.arm_qpos_indices].copy(),
                            "qdot": qdot_cmd.copy(),
                            "cond": float(cond_val),
                            "manip_w": float(self._manip_last_w),
                            "manip_risk": float(self._manip_last_risk),
                            "uncertainty_scale_first": uncertainty_scale_first,
                            "qp_enabled": bool(self.use_constrained_qp),
                            "qp_ok": bool(self.mpc.last_solve_ok),
                            "qp_status": str(self.mpc.last_solve_status),
                        }
                    )

            self._apply_joint_position(q_des)

            mujoco.mj_step(self.model, self.data)
            if viewer and (step_count % self.render_every == 0):
                viewer.sync()
            step_count += 1

            compute_elapsed = time.time() - step_start

            # 统一使用 step 后的实时状态做评估/判定，避免与可视化看到的位置错一拍。
            sim_time_now = self.data.time - self.time_offset
            target_world_ref_now = self._world_target(sim_time_now)
            current_pos_world_now = self.data.site_xpos[self.ee_site_id].copy()
            target_err_now = float(np.linalg.norm(current_pos_world_now - target_world_ref_now))
            log_due = sim_time - last_log_time >= self.profile_period
            phase_now = "hold" if (self.use_offset_tracking and self.offset_active) else "attach"
            hold_ref_now = None
            hold_err_now = np.nan

            if step_callback is not None:
                hold_ref_now = target_world_ref_now + np.array([0.0, self.offset_y, 0.0], dtype=float)
                hold_err_now = float(np.linalg.norm(current_pos_world_now - hold_ref_now))
                active_ref_now = self._control_target_world(
                    sim_time=sim_time_now,
                    abs_time=float(self.data.time),
                )
                active_err_now = float(np.linalg.norm(current_pos_world_now - active_ref_now))
                base_traj_time_now = float(
                    sim_time_now if self.use_offset_tracking else self.data.time
                )
                base_pos_world_now = self._base_pos_world(base_traj_time_now)
                base_vel_world_now = self._base_velocity_world(base_traj_time_now)
                base_speed_norm_now = float(np.linalg.norm(base_vel_world_now))
                base_stop_time_now = self._base_stop_time()
                if np.isfinite(base_stop_time_now):
                    base_time_to_stop_now = float(base_stop_time_now - base_traj_time_now)
                else:
                    base_time_to_stop_now = np.nan
                delta_hold_now = current_pos_world_now - hold_ref_now
                delta_target_now = current_pos_world_now - target_world_ref_now
                offset_dist_now = (
                    float(
                        hold_err_now
                        if (self.use_offset_tracking and self.offset_active)
                        else np.linalg.norm(current_pos_world_now - active_ref_now)
                    )
                    if self.use_offset_tracking
                    else np.nan
                )
                x_gate_ready_now = bool(self._offset_x_gate_ready)
                x_gate_threshold_now = float(self._offset_x_gate_threshold)
                x_gate_delta_now = float(self._offset_x_gate_delta)
                step_callback(
                    {
                        "step": int(step_count),
                        "time": float(sim_time_now),
                        "abs_time": float(self.data.time),
                        "phase": phase_now,
                        "target_err": target_err_now,
                        "hold_err": hold_err_now,
                        "active_err": active_err_now,
                        "attach_err": target_err_now,
                        "ee_pos": current_pos_world_now.copy(),
                        "target_ref": target_world_ref_now.copy(),
                        "hold_ref": hold_ref_now.copy(),
                        "active_ref": active_ref_now.copy(),
                        "base_pos_world": base_pos_world_now.copy(),
                        "base_vel_world": base_vel_world_now.copy(),
                        "base_speed_norm": base_speed_norm_now,
                        "base_stop_time": float(base_stop_time_now),
                        "base_time_to_stop": float(base_time_to_stop_now),
                        "dx_hold": float(delta_hold_now[0]),
                        "dy_hold": float(delta_hold_now[1]),
                        "dz_hold": float(delta_hold_now[2]),
                        "dx_target": float(delta_target_now[0]),
                        "dy_target": float(delta_target_now[1]),
                        "dz_target": float(delta_target_now[2]),
                        "offset_dist_now": float(offset_dist_now),
                        "offset_hit_count": int(self._offset_hit_count),
                        "offset_trigger_steps": int(self.offset_trigger_steps),
                        "offset_x_gate_enable": bool(self.offset_switch_x_gate_enable),
                        "offset_x_gate_ready": bool(x_gate_ready_now),
                        "offset_x_gate_threshold": float(x_gate_threshold_now),
                        "offset_x_gate_delta": float(x_gate_delta_now),
                        "qp_ok": bool(self.mpc.last_solve_ok),
                        "qp_status": str(self.mpc.last_solve_status),
                        "qdot_norm": float(np.linalg.norm(self.last_qdot_cmd)),
                        "jac_cond": float(cond_last),
                        "grasped": bool(self.grasped),
                    }
                )

            if log_due:
                if phase_now == "hold":
                    if hold_ref_now is None:
                        hold_ref_now = target_world_ref_now + np.array(
                            [0.0, self.offset_y, 0.0], dtype=float
                        )
                        hold_err_now = float(np.linalg.norm(current_pos_world_now - hold_ref_now))
                    phase_target_now = hold_ref_now
                    phase_err_now = hold_err_now
                else:
                    phase_target_now = target_world_ref_now
                    phase_err_now = target_err_now

                ee_fmt = (
                    f"[{current_pos_world_now[0]:+.3f},{current_pos_world_now[1]:+.3f},"
                    f"{current_pos_world_now[2]:+.3f}]"
                )
                phase_target_fmt = (
                    f"[{phase_target_now[0]:+.3f},{phase_target_now[1]:+.3f},"
                    f"{phase_target_now[2]:+.3f}]"
                )
                phase_label = "grasp success" if self.grasped else phase_now
                print(
                    f"[{phase_label}] t={self.data.time:.2f}s, "
                    f"err={phase_err_now:.4f} m, "
                    f"ee={ee_fmt}, ref={phase_target_fmt}"
                )
                last_log_time = sim_time

            if self.use_pregrasp and (not self.approach_active) and (not self.use_predictive_phase_switch):
                pregrasp_dist = float(np.linalg.norm(current_pos_world - future_world[0]))
                if pregrasp_dist <= self.pregrasp_tol:
                    self._pregrasp_time += self.dt
                else:
                    self._pregrasp_time = 0.0
                if self._pregrasp_time >= self.pregrasp_hold_time_s:
                    self.approach_active = True
                    self.approach_start_time = float(self.data.time)
                    print(
                        f"[approach] start at t={sim_time:.2f}s, "
                        f"pregrasp_dist={pregrasp_dist:.4f} m"
                    )

            if self.enable_grasp and (not self.grasped):
                grasp_dist = float(np.linalg.norm(current_pos_world_now - target_world_ref_now))
                if grasp_dist <= self.grasp_tol:
                    if self.grasp_hold_time_s is None:
                        self._grasp_counter += 1
                    else:
                        self._grasp_time += self.dt
                else:
                    self._grasp_counter = 0
                    self._grasp_time = 0.0
                hold_ok = False
                if self.grasp_hold_time_s is None:
                    hold_ok = self._grasp_counter >= self.grasp_hold_steps
                else:
                    hold_ok = self._grasp_time >= self.grasp_hold_time_s
                if hold_ok:
                    self.grasped = True
                    ee_now = self.data.site_xpos[self.ee_site_id].copy()
                    tgt_now = self._world_target(self.data.time - self.time_offset)
                    ee_fmt = f"[{ee_now[0]:+.3f},{ee_now[1]:+.3f},{ee_now[2]:+.3f}]"
                    tgt_fmt = f"[{tgt_now[0]:+.3f},{tgt_now[1]:+.3f},{tgt_now[2]:+.3f}]"
                    print(
                        f"[grasp success] t={self.data.time:.2f}s, "
                        f"err={grasp_dist:.4f} m, ee={ee_fmt}, ref={tgt_fmt}"
                    )
                    if self.grasp_action == "attach":
                        self._attach_target = True
                    elif self.grasp_action == "stop":
                        break

            if realtime_sync and (self.dt - compute_elapsed > 0):
                time.sleep(self.dt - compute_elapsed)

    def run_headless(
        self,
        max_time: float = 15.0,
        *,
        step_callback=None,
        control_callback=None,
        realtime_sync: bool = False,
    ) -> dict:
        """无头运行一次完整流程（warm_start + control），返回回合摘要。"""
        if self.is_ball_rel and self.base_traj_for_rel is not None and self.chassis_mocap_id >= 0:
            self._sync_base_traj(0.0)
            base0 = self.base_traj_for_rel.position(0.0)
            self.data.mocap_pos[self.chassis_mocap_id] = base0
            self.data.mocap_quat[self.chassis_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
            mujoco.mj_forward(self.model, self.data)

        start_world = self._world_target(0.0)
        final_dist = self._warm_start_to_pose(
            start_world, max_duration=self.warm_start_max, tol=self.warm_start_tol, viewer=None
        )
        ee_after = self.data.site_xpos[self.ee_site_id].copy()
        ball_dist_after = float(np.linalg.norm(ee_after - start_world))
        desired_after = self._control_target_world(sim_time=0.0, abs_time=self.data.time)
        dist_des_after = float(np.linalg.norm(ee_after - desired_after))
        clamp_active = float(np.linalg.norm(desired_after - start_world)) > 1e-6
        jac_after = self._task_jacobian()
        jac_task_after = jac_after if self.use_orientation_task else jac_after[:3]
        cond_after = np.linalg.cond(jac_task_after @ jac_task_after.T)
        settle_tol = max(self.warm_start_tol, 0.08)
        nominal_cond_max = 1e6
        should_update_nominal = (
            (not clamp_active and ball_dist_after <= settle_tol)
            or (clamp_active and dist_des_after <= settle_tol)
        )
        if should_update_nominal and cond_after <= nominal_cond_max:
            self.q_nominal = self.data.qpos[self.arm_qpos_indices].copy()
            nominal_note = "updated"
        else:
            self.q_nominal = self.q_nominal_init.copy()
            nominal_note = "kept_init"

        self._control_loop(
            viewer=None,
            max_time=float(max_time),
            callback=control_callback,
            step_callback=step_callback,
            realtime_sync=realtime_sync,
        )
        sim_time_now = float(self.data.time - self.time_offset)
        target_world_now = self._world_target(sim_time_now)
        ee_now = self.data.site_xpos[self.ee_site_id].copy()
        final_target_err = float(np.linalg.norm(ee_now - target_world_now))
        return {
            "grasped": bool(self.grasped),
            "sim_time": sim_time_now,
            "final_target_err": final_target_err,
            "warm_start_dist_ball": float(ball_dist_after),
            "warm_start_dist_des": float(dist_des_after),
            "warm_start_last_dist": float(final_dist),
            "warm_start_cond": float(cond_after),
            "q_nominal": nominal_note,
        }

    def run(self):
        print("Launching MPC追踪仿真...")
        if self.is_ball_rel and self.base_traj_for_rel is not None and self.chassis_mocap_id >= 0:
            self._sync_base_traj(0.0)
            base0 = self.base_traj_for_rel.position(0.0)
            self.data.mocap_pos[self.chassis_mocap_id] = base0
            self.data.mocap_quat[self.chassis_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
            mujoco.mj_forward(self.model, self.data)

        start_world = self._world_target(0.0)
        ee_init = self.data.site_xpos[self.ee_site_id].copy()
        dist_init = np.linalg.norm(ee_init - start_world)
        base_init = (
            self.base_traj_for_rel.position(0.0) if self.is_ball_rel and self.base_traj_for_rel else None
        )
        print(
            f"[warm_start] initial_ee={ee_init}, target0={start_world}, "
            f"dist={dist_init:.3f} m, base0={base_init if base_init is not None else 'static'}, "
            f"max_dur={self.warm_start_max}s, tol={self.warm_start_tol}m"
        )
        is_available = getattr(mujoco.viewer, "is_available", lambda: True)
        if is_available():
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 0.0
                viewer.cam.elevation = -25.0
                viewer.sync()
                if sys.stdin is not None and sys.stdin.isatty():
                    print("[start] 视角可先调整；准备好后在终端按回车开始仿真...")
                    try:
                        input()
                    except EOFError:
                        print("[start] 未检测到可交互输入，跳过等待并立即开始。")
                else:
                    print("[start] 当前非交互终端，跳过回车等待并立即开始。")

                final_dist = self._warm_start_to_pose(
                    start_world, max_duration=self.warm_start_max, tol=self.warm_start_tol, viewer=viewer
                )
                ee_after = self.data.site_xpos[self.ee_site_id].copy()
                ball_dist_after = float(np.linalg.norm(ee_after - start_world))
                desired_after = self._control_target_world(sim_time=0.0, abs_time=self.data.time)
                dist_des_after = float(np.linalg.norm(ee_after - desired_after))
                clamp_active = float(np.linalg.norm(desired_after - start_world)) > 1e-6
                jac_after = self._task_jacobian()
                jac_task_after = jac_after if self.use_orientation_task else jac_after[:3]
                cond_after = np.linalg.cond(jac_task_after @ jac_task_after.T)
                settle_tol = max(self.warm_start_tol, 0.08)
                nominal_cond_max = 1e6
                should_update_nominal = (
                    (not clamp_active and ball_dist_after <= settle_tol)
                    or (clamp_active and dist_des_after <= settle_tol)
                )
                if should_update_nominal and cond_after <= nominal_cond_max:
                    self.q_nominal = self.data.qpos[self.arm_qpos_indices].copy()
                    nominal_note = "updated"
                else:
                    self.q_nominal = self.q_nominal_init.copy()
                    nominal_note = "kept_init"
                print(
                    f"[warm_start] after settle dist_des={dist_des_after:.3f} m (last_ball={final_dist:.3f}), "
                    f"dist_ball={ball_dist_after:.3f} m, "
                    f"ee={ee_after}, cond={cond_after:.1e}, sim_t={self.data.time:.2f}s, "
                    f"q_nominal={nominal_note} (settle_tol={settle_tol:.2f}, cond_max={nominal_cond_max:.1e})"
                )
                self._control_loop(viewer=viewer, realtime_sync=True)
        else:
            print("mujoco.viewer 不可用，使用 headless 模式运行以获取误差日志。")
            final_dist = self._warm_start_to_pose(
                start_world, max_duration=self.warm_start_max, tol=self.warm_start_tol, viewer=None
            )
            ee_after = self.data.site_xpos[self.ee_site_id].copy()
            ball_dist_after = float(np.linalg.norm(ee_after - start_world))
            desired_after = self._control_target_world(sim_time=0.0, abs_time=self.data.time)
            dist_des_after = float(np.linalg.norm(ee_after - desired_after))
            clamp_active = float(np.linalg.norm(desired_after - start_world)) > 1e-6
            jac_after = self._task_jacobian()
            jac_task_after = jac_after if self.use_orientation_task else jac_after[:3]
            cond_after = np.linalg.cond(jac_task_after @ jac_task_after.T)
            settle_tol = max(self.warm_start_tol, 0.08)
            nominal_cond_max = 1e6
            should_update_nominal = (
                (not clamp_active and ball_dist_after <= settle_tol)
                or (clamp_active and dist_des_after <= settle_tol)
            )
            if should_update_nominal and cond_after <= nominal_cond_max:
                self.q_nominal = self.data.qpos[self.arm_qpos_indices].copy()
                nominal_note = "updated"
            else:
                self.q_nominal = self.q_nominal_init.copy()
                nominal_note = "kept_init"
            print(
                f"[warm_start] after settle dist_des={dist_des_after:.3f} m (last_ball={final_dist:.3f}), "
                f"dist_ball={ball_dist_after:.3f} m, "
                f"ee={ee_after}, cond={cond_after:.1e}, sim_t={self.data.time:.2f}s, "
                f"q_nominal={nominal_note} (settle_tol={settle_tol:.2f}, cond_max={nominal_cond_max:.1e})"
            )
            self._control_loop(viewer=None, max_time=15.0)


__all__ = ["TaskSpaceMPC", "MPCController"]

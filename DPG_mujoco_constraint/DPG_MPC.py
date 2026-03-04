"""
DPG_MPC.py

封装 TaskSpace MPC 与机械臂控制逻辑：
    - TaskSpaceMPC: 无约束 QP（可选终端代价）。
    - MPCController: 接收轨迹（机械臂基坐标系），转换为世界坐标写入 mocap，并通过雅可比分解积分得到关节角目标，写入 position 伺服。
    - Funnel 约束层: 在目标前方禁区加入离散安全过滤，限制末端只能经过漏斗通道靠近目标。

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

        weight_block = np.diag([pos_weight] * 3 + [rot_weight] * 3)
        self.G = np.kron(np.identity(horizon), weight_block)

        self.B = self._build_B_matrix()
        self.Q = smooth_weight * np.identity(self.task_dim * horizon)

        self.base_H = self.A.T @ self.G @ self.A + self.B.T @ self.Q @ self.B + reg * np.identity(
            self.task_dim * horizon
        )

        self.A_T_G = self.A.T @ self.G
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

    def _build_B_matrix(self) -> np.ndarray:
        n = self.task_dim * self.horizon
        I = np.identity(n)
        upper = np.zeros((self.task_dim, n))
        lower = I[:-self.task_dim, :]
        stacked = np.vstack([upper, lower])
        return I - stacked

    def solve(
        self,
        error_vector: np.ndarray,
        *,
        current_state: Optional[np.ndarray] = None,
        desired_terminal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if error_vector.shape[0] != self.task_dim * self.horizon:
            raise ValueError("error_vector 尺寸错误")

        u_anchor = np.zeros(self.task_dim * self.horizon)
        u_anchor[: self.task_dim] = self.prev_twist

        rhs = self.A_T_G @ error_vector + self.B_T_Q @ u_anchor
        H = self.base_H.copy()

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

        self.full_solution = np.linalg.solve(H, rhs)
        twist = self.full_solution[: self.task_dim]
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
        use_offset_tracking: bool = False,
        offset_y: float = -0.1,
        offset_trigger_tol: float = 0.02,
        offset_trigger_hold_time_s: float = 0.5,
        enable_grasp: bool = True,
        grasp_tol: float = 0.03,
        grasp_hold_steps: int = 3,
        grasp_hold_time_s: Optional[float] = None,
        grasp_action: str = "stop",
        enable_funnel_constraint: bool = True,
        funnel_depth: float = 0.10,
        funnel_half_width: float = 0.05,
        funnel_margin: float = 1e-3,
        visualize_funnel_zone: bool = True,
        funnel_vis_x_extent: float = 1.2,
        funnel_vis_z_half: float = 1.0,
        funnel_vis_rgba: Optional[np.ndarray] = None,
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
        if self.is_ball_rel and self.base_traj_for_rel is not None:
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

        # 侧向保持距离（y 偏置）→ 满足条件后切回原轨迹
        self.use_offset_tracking = bool(use_offset_tracking)
        self.offset_y = float(offset_y)
        self.offset_trigger_tol = float(offset_trigger_tol)
        self.offset_trigger_hold_time_s = float(offset_trigger_hold_time_s)
        self.offset_active = bool(use_offset_tracking)
        self._offset_time = 0.0

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
        # 平滑：小误差时逐步减小 task feedback，减少抖动；底盘运动用前馈抵消
        self.feedback_deadband = 0.01
        self.feedback_ramp = 0.08
        self.base_ff_gain = 1.0
        # 额外的任务空间限幅：避免 MPC 在大误差时给出过大的线速度，导致关节速度饱和→抖动
        self.ee_linear_speed_limit = 0.8  # m/s

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

        # 漏斗约束: 在目标前方 [y_t-depth, y_t] 里，仅允许 x 落在 [x_t-half_width, x_t+half_width]
        self.enable_funnel_constraint = bool(enable_funnel_constraint)
        self.funnel_depth = max(0.0, float(funnel_depth))
        self.funnel_half_width = max(1e-6, float(funnel_half_width))
        self.funnel_margin = max(1e-6, float(funnel_margin))
        self._funnel_target_adjust_count = 0
        self._funnel_twist_adjust_count = 0
        self.visualize_funnel_zone = bool(visualize_funnel_zone)
        self.funnel_vis_x_extent = max(0.05, float(funnel_vis_x_extent))
        self.funnel_vis_z_half = max(0.05, float(funnel_vis_z_half))
        if funnel_vis_rgba is None:
            self.funnel_vis_rgba = np.array([1.0, 0.95, 0.45, 0.20], dtype=np.float32)
        else:
            rgba = np.asarray(funnel_vis_rgba, dtype=np.float32).reshape(-1)
            if rgba.size != 4:
                raise ValueError("funnel_vis_rgba 需要 4 维 RGBA")
            self.funnel_vis_rgba = np.clip(rgba, 0.0, 1.0)

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

    def _task_error(self, current_pos: np.ndarray, future_targets: np.ndarray) -> np.ndarray:
        e_hat = np.zeros(self.mpc.task_dim * self.mpc.horizon)
        for i, desired in enumerate(future_targets):
            idx = i * self.mpc.task_dim
            e_hat[idx : idx + 3] = desired - current_pos
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

    def _funnel_anchor_world(
        self, sim_time: float, future_targets: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self.is_ball_rel and hasattr(self.trajectory, "ball_world"):
            return np.asarray(self.trajectory.ball_world, dtype=float).reshape(3)
        if future_targets is not None and future_targets.shape[0] > 0:
            return np.asarray(future_targets[-1], dtype=float).reshape(3)
        return np.asarray(self._world_target(sim_time), dtype=float).reshape(3)

    def _funnel_bounds(self, target_world: np.ndarray) -> tuple[float, float, float, float]:
        x_t = float(target_world[0])
        y_t = float(target_world[1])
        return (
            x_t - self.funnel_half_width,
            x_t + self.funnel_half_width,
            y_t - self.funnel_depth,
            y_t,
        )

    def _is_forbidden_by_funnel(self, pos_world: np.ndarray, target_world: np.ndarray) -> bool:
        if (not self.enable_funnel_constraint) or self.funnel_depth <= 0.0:
            return False
        x = float(pos_world[0])
        y = float(pos_world[1])
        x_l, x_r, y_l, y_h = self._funnel_bounds(target_world)
        if y < y_l or y > y_h:
            return False
        return (x <= x_l) or (x >= x_r)

    def _project_out_of_funnel(self, pos_world: np.ndarray, target_world: np.ndarray) -> tuple[np.ndarray, bool]:
        """若点落入禁区，投影到最近可行边界（离散安全层）。"""
        p = np.asarray(pos_world, dtype=float).reshape(3)
        if not self._is_forbidden_by_funnel(p, target_world):
            return p, False

        x_l, x_r, y_l, y_h = self._funnel_bounds(target_world)
        eps = self.funnel_margin
        x_inner_l = x_l + eps
        x_inner_r = x_r - eps
        if x_inner_r <= x_inner_l:
            mid = 0.5 * (x_l + x_r)
            x_inner_l = mid - 1e-6
            x_inner_r = mid + 1e-6

        candidates = []
        c_back = p.copy()
        c_back[1] = y_l - eps
        candidates.append(c_back)

        c_front = p.copy()
        c_front[1] = y_h + eps
        candidates.append(c_front)

        c_corridor = p.copy()
        c_corridor[0] = np.clip(c_corridor[0], x_inner_l, x_inner_r)
        candidates.append(c_corridor)

        dists = [float(np.linalg.norm(c - p)) for c in candidates]
        best = candidates[int(np.argmin(dists))]
        return best, True

    def _apply_funnel_to_future_targets(
        self, future_targets: np.ndarray, target_world: np.ndarray
    ) -> np.ndarray:
        if (not self.enable_funnel_constraint) or future_targets.size == 0:
            return future_targets
        adjusted = np.asarray(future_targets, dtype=float).copy()
        changed = 0
        for i in range(adjusted.shape[0]):
            p_new, is_changed = self._project_out_of_funnel(adjusted[i], target_world)
            adjusted[i] = p_new
            changed += int(is_changed)
        if changed > 0:
            self._funnel_target_adjust_count += int(changed)
        return adjusted

    def _apply_funnel_to_twist(
        self,
        current_pos_world: np.ndarray,
        target_world: np.ndarray,
        twist_cmd: np.ndarray,
        twist_fb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.enable_funnel_constraint:
            return twist_cmd, twist_fb

        twist_cmd_out = np.asarray(twist_cmd, dtype=float).copy()
        twist_fb_out = np.asarray(twist_fb, dtype=float).copy()
        lin_cmd = twist_cmd_out[:3] if self.use_orientation_task else twist_cmd_out
        next_pos = np.asarray(current_pos_world, dtype=float).reshape(3) + lin_cmd * self.control_dt
        next_safe, changed = self._project_out_of_funnel(next_pos, target_world)
        if not changed:
            return twist_cmd_out, twist_fb_out

        safe_lin = (next_safe - current_pos_world) / max(self.control_dt, 1e-9)
        if self.use_orientation_task:
            delta = safe_lin - twist_cmd_out[:3]
            twist_cmd_out[:3] = safe_lin
            twist_fb_out[:3] = twist_fb_out[:3] + delta
        else:
            delta = safe_lin - twist_cmd_out
            twist_cmd_out = safe_lin
            twist_fb_out = twist_fb_out + delta
        self._funnel_twist_adjust_count += 1
        return twist_cmd_out, twist_fb_out

    def _draw_funnel_marker(self, viewer, target_world: np.ndarray) -> None:
        """在 viewer 中绘制漏斗禁区（纯可视化，不参与碰撞）。"""
        if (not self.enable_funnel_constraint) or (not self.visualize_funnel_zone):
            return
        if viewer is None:
            return
        if not hasattr(viewer, "user_scn"):
            return

        x_t, y_t, z_t = [float(v) for v in np.asarray(target_world, dtype=float).reshape(3)]
        x_l = x_t - self.funnel_half_width
        x_r = x_t + self.funnel_half_width
        y_center = y_t - 0.5 * self.funnel_depth
        y_half = 0.5 * self.funnel_depth
        eps = self.funnel_margin
        x_extent = self.funnel_vis_x_extent

        left_half_x = max(0.01, x_extent)
        right_half_x = max(0.01, x_extent)
        left_center_x = x_l - left_half_x - eps
        right_center_x = x_r + right_half_x + eps

        # 两个禁区盒: 左侧和右侧（中间通道留空）
        boxes = [
            (np.array([left_center_x, y_center, z_t], dtype=np.float64),
             np.array([left_half_x, y_half, self.funnel_vis_z_half], dtype=np.float32)),
            (np.array([right_center_x, y_center, z_t], dtype=np.float64),
             np.array([right_half_x, y_half, self.funnel_vis_z_half], dtype=np.float32)),
        ]

        mat = np.eye(3, dtype=np.float32).reshape(-1)
        rgba = self.funnel_vis_rgba.copy()
        with viewer.lock():
            scn = viewer.user_scn
            scn.ngeom = 0
            for pos, size in boxes:
                if scn.ngeom >= len(scn.geoms):
                    break
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=size,
                    pos=pos,
                    mat=mat,
                    rgba=rgba,
                )
                scn.ngeom += 1

    def _limit_joint_accel(self, qdot_cmd: np.ndarray, qdot_prev: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0:
            return qdot_cmd
        max_delta = self.accel_limit * dt
        delta = qdot_cmd - qdot_prev
        delta = np.clip(delta, -max_delta, max_delta)
        return qdot_prev + delta

    def _feedback_gain(self, pos_err: float) -> float:
        if pos_err <= self.feedback_deadband:
            return 0.0
        return float(np.clip((pos_err - self.feedback_deadband) / self.feedback_ramp, 0.0, 1.0))

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

    def _control_target_world(self, sim_time: float, abs_time: float) -> np.ndarray:
        if not self.is_ball_rel:
            return self._world_target(sim_time)
        if self.base_traj_for_rel is None:
            return self.trajectory.ball_world
        self._sync_base_traj(abs_time)
        base_pos = self.base_traj_for_rel.position(abs_time)
        mount_pos = base_pos + self.shoulder_offset
        return self._reachable_target(self.trajectory.ball_world, mount_pos)

    def _control_future_targets_world(
        self,
        sim_time: float,
        abs_time: float,
        current_pos_world: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        funnel_anchor = self._funnel_anchor_world(sim_time=sim_time)
        if self.intercept_planner is not None and current_pos_world is not None:
            future = self.intercept_planner.plan(
                current_pos_world=current_pos_world,
                abs_time=abs_time,
                horizon=self.mpc.horizon,
                control_dt=self.control_dt,
            )
            return self._apply_funnel_to_future_targets(future, funnel_anchor)
        if self.use_pregrasp and self.is_ball_rel and self.base_traj_for_rel is not None:
            self._sync_base_traj(abs_time)
            base_future = self.base_traj_for_rel.future_positions(
                abs_time + self.preview_lead,
                self.mpc.horizon,
                self.control_dt,
            )
            future = np.zeros((self.mpc.horizon, 3), dtype=float)
            for i in range(self.mpc.horizon):
                base_pos = base_future[i]
                t_abs = abs_time + self.preview_lead + (i + 1) * self.control_dt
                offset = self._pregrasp_offset_at(t_abs)
                target_rel = self.trajectory.ball_world - base_pos
                desired_rel = target_rel - offset * self.pregrasp_dir
                future[i] = desired_rel + base_pos
            return self._apply_funnel_to_future_targets(future, funnel_anchor)
        if not self.is_ball_rel:
            future = self._world_future_targets(sim_time)
            if self.use_offset_tracking and self.offset_active:
                future = future + np.array([0.0, self.offset_y, 0.0], dtype=float)
            return self._apply_funnel_to_future_targets(future, funnel_anchor)
        if self.base_traj_for_rel is None:
            future = np.tile(self.trajectory.ball_world, (self.mpc.horizon, 1))
            if self.use_offset_tracking and self.offset_active:
                future = future + np.array([0.0, self.offset_y, 0.0], dtype=float)
            return self._apply_funnel_to_future_targets(future, funnel_anchor)
        self._sync_base_traj(abs_time)
        base_future = self.base_traj_for_rel.future_positions(
            abs_time + self.preview_lead,
            self.mpc.horizon,
            self.control_dt,
        )
        future = np.zeros((self.mpc.horizon, 3), dtype=float)
        for i in range(self.mpc.horizon):
            base_pos = base_future[i]
            mount_pos = base_pos + self.shoulder_offset
            future[i] = self._reachable_target(self.trajectory.ball_world, mount_pos)
        if self.use_offset_tracking and self.offset_active:
            future = future + np.array([0.0, self.offset_y, 0.0], dtype=float)
        return self._apply_funnel_to_future_targets(future, funnel_anchor)

    def _apply_joint_position(self, q_des: np.ndarray):
        if q_des.shape != (self.arm_dof,):
            raise ValueError("q_des 尺寸错误")
        q_des = np.clip(q_des, self.arm_qpos_min, self.arm_qpos_max)
        self.data.ctrl[self.arm_actuator_ids] = q_des

    def _world_target(self, sim_time: float) -> np.ndarray:
        if self.is_ball_rel:
            # ball_in_robot: 世界中小球静止，直接返回 ball_world
            return self.trajectory.ball_world
        base_origin = getattr(self.trajectory, "base_origin", None)
        if base_origin is not None:
            return self.trajectory.position(sim_time) + base_origin
        return self.trajectory.position(sim_time)

    def _world_future_targets(self, sim_time: float) -> np.ndarray:
        if self.is_ball_rel:
            # 未来点也是同一个世界系位置
            return np.tile(self.trajectory.ball_world, (self.mpc.horizon, 1))
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

        for k in range(steps):
            step_start = time.time()
            if viewer and (not viewer.is_running()):
                break

            if self.is_ball_rel and self.base_traj_for_rel is not None and self.chassis_mocap_id >= 0:
                # warm start 阶段使用从 0 开始的绝对时间推进底盘
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
                e_hat = self._task_error(current_pos, future_targets)
                terminal_time = abs_time + self.preview_lead + self.mpc.horizon * self.control_dt
                current_state, desired_state = self._terminal_state(
                    current_pos, future_targets[-1], terminal_time
                )
                twist = self.mpc.solve(
                    e_hat,
                    current_state=current_state,
                    desired_terminal=desired_state,
                )

                jac_full = self._task_jacobian()
                jac = jac_full if self.use_orientation_task else jac_full[:3]
                jac_pinv = self._jacobian_pinv(jac)
                cond_val = np.linalg.cond(jac @ jac.T)
                if self.intercept_planner is not None:
                    ball_err = float(np.linalg.norm(future_targets[0] - current_pos))
                else:
                    ball_err = float(np.linalg.norm(target_pos_world - current_pos))
                fb_gain = self._feedback_gain(ball_err)

                twist_fb = (twist.copy() if self.use_orientation_task else twist[:3].copy()) * fb_gain
                twist_ff = np.zeros_like(twist_fb)
                if (
                    self.intercept_planner is None
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
                funnel_anchor = self._funnel_anchor_world(sim_time=sim_time, future_targets=future_targets)
                twist_cmd, twist_fb = self._apply_funnel_to_twist(
                    current_pos_world=current_pos,
                    target_world=funnel_anchor,
                    twist_cmd=twist_cmd,
                    twist_fb=twist_fb,
                )

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
                self._draw_funnel_marker(viewer, target_pos_world)
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

    def _control_loop(self, viewer=None, max_time: float = 20.0, callback=None):
        last_log_time = -np.inf
        step_count = 0
        wall_start = time.time()
        qdot_cmd = self.last_qdot_cmd.copy()
        q_des = self.data.qpos[self.arm_qpos_indices].copy()
        qdot_pre_clip = np.zeros_like(qdot_cmd)
        while True:
            step_start = time.time()
            sim_time = self.data.time - self.time_offset

            if self.is_ball_rel and self.base_traj_for_rel is not None and self.chassis_mocap_id >= 0:
                # 底盘轨迹需要使用真实仿真时间，不能减去 time_offset，否则 warm_start 结束后会跳回起点
                self._sync_base_traj(self.data.time)
                base_pos = self.base_traj_for_rel.position(self.data.time)
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
                e_hat = self._task_error(current_pos, future_world)

                terminal_time = (
                    self.data.time + self.preview_lead + self.mpc.horizon * self.control_dt
                )
                current_state, desired_state = self._terminal_state(
                    current_pos, future_world[-1], terminal_time
                )
                twist = self.mpc.solve(
                    e_hat,
                    current_state=current_state,
                    desired_terminal=desired_state,
                )
                jac_full = self._task_jacobian()
                jac = jac_full if self.use_orientation_task else jac_full[:3]
                jac_pinv = self._jacobian_pinv(jac)

                cond_val = np.linalg.cond(jac @ jac.T)
                if self.intercept_planner is not None or self.use_pregrasp or self.use_offset_tracking:
                    ball_err = float(np.linalg.norm(future_world[0] - current_pos))
                else:
                    ball_err = float(np.linalg.norm(target_world - current_pos))
                fb_gain = self._feedback_gain(ball_err)

                twist_fb = (twist.copy() if self.use_orientation_task else twist[:3].copy()) * fb_gain
                twist_ff = np.zeros_like(twist_fb)
                if (
                    self.intercept_planner is None
                    and (not self.use_pregrasp)
                    and self.is_ball_rel
                    and self.base_traj_for_rel is not None
                ):
                    desired_now = self._control_target_world(sim_time=sim_time, abs_time=self.data.time)
                    desired_next = future_world[0]
                    desired_vel = (desired_next - desired_now) / self.control_dt
                    base_vel = self._base_velocity_world(self.data.time)
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
                funnel_anchor = self._funnel_anchor_world(sim_time=sim_time, future_targets=future_world)
                twist_cmd, twist_fb = self._apply_funnel_to_twist(
                    current_pos_world=current_pos,
                    target_world=funnel_anchor,
                    twist_cmd=twist_cmd,
                    twist_fb=twist_fb,
                )

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
                        }
                    )

            self._apply_joint_position(q_des)

            mujoco.mj_step(self.model, self.data)
            if viewer and (step_count % self.render_every == 0):
                self._draw_funnel_marker(viewer, target_world_ref)
                viewer.sync()
            step_count += 1

            compute_elapsed = time.time() - step_start

            if sim_time - last_log_time >= self.profile_period:
                pos_err = np.linalg.norm(current_pos_world - target_world_ref)
                joint_dev = np.linalg.norm(self.data.qpos[self.arm_qpos_indices] - self.q_nominal)
                jac_full = self._task_jacobian()
                jac = jac_full if self.use_orientation_task else jac_full[:3]
                jac_condition = np.linalg.cond(jac @ jac.T)
                clip_hits = np.count_nonzero(
                    np.abs(qdot_pre_clip) >= (self.velocity_limit - np.deg2rad(1.0))
                )
                clip_fraction = clip_hits / len(qdot_cmd)
                rt_factor = sim_time / max(time.time() - wall_start, 1e-6)
                base_info = (
                    f", base_x={self.data.mocap_pos[self.chassis_mocap_id][0]:.2f}"
                    if self.is_ball_rel and self.chassis_mocap_id >= 0
                    else ""
                )
                funnel_info = (
                    f", funnel_ref_adj={self._funnel_target_adjust_count}, "
                    f"funnel_twist_adj={self._funnel_twist_adjust_count}"
                    if self.enable_funnel_constraint
                    else ""
                )
                prefix = "[grasp success] " if self.grasped else ""
                print(
                    f"{prefix}time={sim_time:.2f}s, pos_err={pos_err:.4f} m, "
                    f"joint_dev={joint_dev:.3f} rad, cond={jac_condition:.1e}, "
                    f"clip_frac={clip_fraction:.2f}, step_ms={compute_elapsed*1e3:.2f}, "
                    f"rt={rt_factor:.2f}x, control_every={self.control_every}{base_info}{funnel_info}"
                )
                last_log_time = sim_time

            if self.use_offset_tracking and self.offset_active:
                compensated = current_pos_world.copy()
                compensated[1] += -self.offset_y
                offset_dist = float(np.linalg.norm(compensated - target_world_ref))
                if offset_dist <= self.offset_trigger_tol:
                    self._offset_time += self.dt
                else:
                    self._offset_time = 0.0
                if self._offset_time >= self.offset_trigger_hold_time_s:
                    self.offset_active = False
                    print(
                        f"[approach] switch to target at t={sim_time:.2f}s, "
                        f"comp_dist={offset_dist:.4f} m"
                    )

            if self.use_pregrasp and (not self.approach_active):
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
                grasp_dist = float(np.linalg.norm(current_pos_world - target_world_ref))
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
                    print(
                        f"[grasp] success at t={sim_time:.2f}s, "
                        f"dist={grasp_dist:.4f} m (tol={self.grasp_tol:.3f})"
                    )
                    if self.grasp_action == "attach":
                        self._attach_target = True
                    elif self.grasp_action == "stop":
                        break

            if self.dt - compute_elapsed > 0:
                time.sleep(self.dt - compute_elapsed)

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
                self._draw_funnel_marker(viewer, start_world)
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
                self._control_loop(viewer=viewer)
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

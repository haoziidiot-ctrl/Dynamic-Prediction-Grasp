"""
DPG_MPC.py

封装 TaskSpace MPC 与机械臂控制逻辑：
    - TaskSpaceMPC: 无约束解析解（位置误差 + 平滑项）。
    - MPCController: 接收轨迹（机械臂基坐标系），转换为世界坐标写入 mocap，并通过雅可比分解积分得到关节角目标，写入 position 伺服。

轨迹约定：
    - Ball 轨迹: position 返回以机械臂基座标系为原点的目标坐标，需加回 base_origin 成为世界坐标。
    - Ball_in_robot 轨迹: position 返回相对底盘的坐标，需要先用底盘轨迹得到 base_pos，再折算世界坐标 = relative + base_pos。
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from DPG_track_ball import TrajectoryProvider, MocapBallTrajectory
from DPG_track_ball_in_robot import BallInRobotFrameTrajectory


class TaskSpaceMPC:
    """实现无约束 QP 解析解，仅考虑末端线位置误差。"""

    def __init__(
        self,
        horizon: int,
        dt: float,
        pos_weight: float = 1.0,
        rot_weight: float = 0.0,
        smooth_weight: float = 1e-2,
        reg: float = 1e-6,
    ):
        self.horizon = horizon
        self.dt = dt
        self.task_dim = 6

        I6 = np.identity(self.task_dim)
        tril = np.tril(np.ones((horizon, horizon)))
        self.A = dt * np.kron(tril, I6)

        weight_block = np.diag([pos_weight] * 3 + [rot_weight] * 3)
        self.G = np.kron(np.identity(horizon), weight_block)

        self.B = self._build_B_matrix()
        self.Q = smooth_weight * np.identity(self.task_dim * horizon)

        H = self.A.T @ self.G @ self.A + self.B.T @ self.Q @ self.B + reg * np.identity(
            self.task_dim * horizon
        )
        self.H_inv = np.linalg.inv(H)

        self.A_T_G = self.A.T @ self.G
        self.B_T_Q = self.B.T @ self.Q

        self.prev_twist = np.zeros(self.task_dim)
        self.full_solution = np.zeros(self.task_dim * self.horizon)

    def _build_B_matrix(self) -> np.ndarray:
        n = self.task_dim * self.horizon
        I = np.identity(n)
        upper = np.zeros((self.task_dim, n))
        lower = I[:-self.task_dim, :]
        stacked = np.vstack([upper, lower])
        return I - stacked

    def solve(self, error_vector: np.ndarray) -> np.ndarray:
        if error_vector.shape[0] != self.task_dim * self.horizon:
            raise ValueError("error_vector 尺寸错误")

        u_anchor = np.zeros(self.task_dim * self.horizon)
        u_anchor[: self.task_dim] = self.prev_twist

        rhs = self.A_T_G @ error_vector + self.B_T_Q @ u_anchor
        self.full_solution = self.H_inv @ rhs
        twist = self.full_solution[: self.task_dim]
        self.prev_twist = twist
        return twist


class MPCController:
    """负责轨迹取样、MPC 求解、雅可比分解、输出 position 伺服目标角。"""

    def __init__(
        self,
        model_xml: str = "fetch_freight_mujoco/xml/scene.xml",
        trajectory: Optional[TrajectoryProvider] = None,
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
    ):
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data = mujoco.MjData(self.model)

        self.trajectory: TrajectoryProvider = trajectory or MocapBallTrajectory()
        self.is_ball_rel = isinstance(self.trajectory, BallInRobotFrameTrajectory)
        self.base_traj_for_rel = self.trajectory.base_trajectory if self.is_ball_rel else None
        if self.is_ball_rel and self.base_traj_for_rel is None:
            from DPG_track_ball_in_robot import LinearBaseTrajectory

            self.base_traj_for_rel = LinearBaseTrajectory()
        if self.is_ball_rel and isinstance(self.trajectory, BallInRobotFrameTrajectory):
            # 确保 ball_in_robot 轨迹内部也拿到同一个 base_trajectory（后续会直接用 trajectory.future_positions）
            if self.base_traj_for_rel is None:
                raise ValueError("ball_in_robot 需要 base_trajectory，但当前为 None")
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
        self.mpc = TaskSpaceMPC(
            horizon=horizon,
            dt=self.control_dt,
            pos_weight=pos_weight,
            rot_weight=rot_weight,
            smooth_weight=smooth_weight,
        )
        self.use_orientation_task = rot_weight > 1e-9

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

    def _sync_base_traj(self, abs_time: float) -> None:
        if self.base_traj_for_rel is None:
            return
        sync = getattr(self.base_traj_for_rel, "sync", None)
        if callable(sync):
            sync(abs_time)

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

    def _control_future_targets_world(self, sim_time: float, abs_time: float) -> np.ndarray:
        if not self.is_ball_rel:
            return self._world_future_targets(sim_time)
        # ball_in_robot：MPC 直接在「底盘/机械臂相对坐标系」下工作（只做平移补偿）。
        # 轨迹输出 = ball_world - base_pos(t)，因此未来 N 帧的参考轨迹就是未来 N 帧的相对坐标。
        self._sync_base_traj(abs_time)
        return self.trajectory.future_positions(
            abs_time + self.preview_lead, self.mpc.horizon, self.control_dt
        )

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
                if self.is_ball_rel and self.base_traj_for_rel is not None:
                    self._sync_base_traj(abs_time)
                    base_pos_now = self.base_traj_for_rel.position(abs_time)
                    current_pos = current_pos_world - base_pos_now
                else:
                    current_pos = current_pos_world
                future_targets = self._control_future_targets_world(sim_time=sim_time, abs_time=abs_time)
                e_hat = self._task_error(current_pos, future_targets)
                twist = self.mpc.solve(e_hat)

                jac_full = self._task_jacobian()
                jac = jac_full if self.use_orientation_task else jac_full[:3]
                jac_pinv = self._jacobian_pinv(jac)
                cond_val = np.linalg.cond(jac @ jac.T)
                if self.is_ball_rel and self.base_traj_for_rel is not None:
                    desired_now = self.trajectory.position(abs_time)
                    ball_err = float(np.linalg.norm(desired_now - current_pos))
                else:
                    ball_err = float(np.linalg.norm(target_pos_world - current_pos))
                fb_gain = self._feedback_gain(ball_err)

                twist_fb = (twist.copy() if self.use_orientation_task else twist[:3].copy()) * fb_gain
                twist_ff = np.zeros_like(twist_fb)
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

    def _control_loop(self, viewer=None, max_time: float = 20.0):
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

            target_world = self._world_target(sim_time)
            self.data.mocap_pos[self.target_mocap_id] = target_world

            current_pos_world = self.data.site_xpos[self.ee_site_id].copy()
            if self.is_ball_rel and self.base_traj_for_rel is not None:
                self._sync_base_traj(self.data.time)
                base_pos_now = self.base_traj_for_rel.position(self.data.time)
                current_pos = current_pos_world - base_pos_now
            else:
                current_pos = current_pos_world
            future_world = self._control_future_targets_world(sim_time=sim_time, abs_time=self.data.time)
            e_hat = None

            if step_count % self.control_every == 0:
                e_hat = self._task_error(current_pos, future_world)

                twist = self.mpc.solve(e_hat)
                jac_full = self._task_jacobian()
                jac = jac_full if self.use_orientation_task else jac_full[:3]
                jac_pinv = self._jacobian_pinv(jac)

                cond_val = np.linalg.cond(jac @ jac.T)
                if self.is_ball_rel and self.base_traj_for_rel is not None:
                    desired_now = self.trajectory.position(self.data.time)
                    ball_err = float(np.linalg.norm(desired_now - current_pos))
                else:
                    ball_err = float(np.linalg.norm(target_world - current_pos))
                fb_gain = self._feedback_gain(ball_err)

                twist_fb = (twist.copy() if self.use_orientation_task else twist[:3].copy()) * fb_gain
                twist_ff = np.zeros_like(twist_fb)
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

            self._apply_joint_position(q_des)

            mujoco.mj_step(self.model, self.data)
            if viewer and (step_count % self.render_every == 0):
                viewer.sync()
            step_count += 1

            compute_elapsed = time.time() - step_start

            if sim_time - last_log_time >= self.profile_period:
                pos_err = np.linalg.norm(current_pos - target_world)
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
                print(
                    f"time={sim_time:.2f}s, pos_err={pos_err:.4f} m, "
                    f"joint_dev={joint_dev:.3f} rad, cond={jac_condition:.1e}, "
                    f"clip_frac={clip_fraction:.2f}, step_ms={compute_elapsed*1e3:.2f}, "
                    f"rt={rt_factor:.2f}x, control_every={self.control_every}{base_info}"
                )
                last_log_time = sim_time

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

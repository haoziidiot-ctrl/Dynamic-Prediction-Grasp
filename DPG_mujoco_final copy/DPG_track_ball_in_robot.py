"""
DPG_track_ball_in_robot.py

用途:
    - 在“小车移动、小球固定”的场景下，输出小球在机械臂/底盘坐标系下的轨迹。
    - 支持两种底盘轨迹来源:
        1) 解析直线轨迹（LinearBaseTrajectory）
        2) KF 估计 + 外推轨迹（KFPredictiveBaseTrajectory）

接口:
    - BallInRobotFrameTrajectory: position(t) 返回 ball 在底盘系的 3D 坐标。
    - get_ball_in_robot_trajectory(...): 生成默认实例，便于 main/plot 调用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.linalg import solve_discrete_are

from DPG_track_ball import BaseTrajectory, TrajectoryProvider


@dataclass
class LinearBaseTrajectory(BaseTrajectory):
    """底盘直线轨迹（世界系），用于生成相对坐标。"""

    start: np.ndarray = field(default_factory=lambda: np.array([-0.8, 0.0, 0.0], dtype=float))
    end: np.ndarray = field(default_factory=lambda: np.array([0.50, 0.0, 0.0], dtype=float))
    speed: float = 0.3
    # 每步横向扰动（左右偏移）：
    # 幅值 ~ N(mean,std)，再随机赋予正负号，保证有左右抖动。
    lateral_noise_dt: float = 0.01
    lateral_noise_mag_mean: float = 0.005
    lateral_noise_mag_std: float = 0.005
    lateral_noise_seed: Optional[int] = 0

    def __post_init__(self):
        self.start = np.array(self.start, dtype=float).reshape(3)
        self.end = np.array(self.end, dtype=float).reshape(3)
        self.lateral_noise_dt = max(float(self.lateral_noise_dt), 1e-6)
        self.lateral_noise_mag_mean = max(float(self.lateral_noise_mag_mean), 0.0)
        self.lateral_noise_mag_std = max(float(self.lateral_noise_mag_std), 0.0)
        self._noise_rng = np.random.default_rng(self.lateral_noise_seed)
        self._lateral_noise_cache: Dict[int, float] = {}

    def _delta(self) -> np.ndarray:
        return self.end - self.start

    def length(self) -> float:
        return float(np.linalg.norm(self._delta()))

    def duration(self) -> float:
        return self.length() / max(float(self.speed), 1e-6)

    def _lateral_noise(self, sim_time: float) -> float:
        if self.lateral_noise_mag_mean <= 0.0 and self.lateral_noise_mag_std <= 0.0:
            return 0.0
        idx = int(np.floor(max(float(sim_time), 0.0) / self.lateral_noise_dt + 1e-12))
        val = self._lateral_noise_cache.get(idx)
        if val is None:
            amp = float(self._noise_rng.normal(self.lateral_noise_mag_mean, self.lateral_noise_mag_std))
            amp = max(0.0, amp)
            sign = -1.0 if float(self._noise_rng.random()) < 0.5 else 1.0
            val = sign * amp
            self._lateral_noise_cache[idx] = val
        return float(val)

    def position(self, sim_time: float) -> np.ndarray:
        total_len = self.length()
        if total_len < 1e-9:
            return self.start.copy()
        dur = self.duration()
        progress = min(max(sim_time, 0.0) * self.speed / total_len, 1.0)
        pos = self.start + self._delta() * progress
        if float(sim_time) < dur - 1e-12:
            pos = pos.copy()
            pos[1] += self._lateral_noise(sim_time)
        return pos


def _make_kf_matrices(dt: float, sigma_a: float, meas_noise: float):
    f = np.array(
        [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    q11 = 0.25 * dt**4
    q12 = 0.5 * dt**3
    q22 = dt**2
    q_1d = np.array([[q11, q12], [q12, q22]], dtype=float) * (sigma_a**2)
    q = np.zeros((4, 4), dtype=float)
    q[0:2, 0:2] = q_1d
    q[2:4, 2:4] = q_1d
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=float)
    r = np.diag([meas_noise**2, meas_noise**2]).astype(float)
    return f, q, h, r


@dataclass
class KFPredictiveBaseTrajectory(BaseTrajectory):
    """
    底盘 KF 轨迹包装器（仿真版）：
        - 真值来自 base_trajectory.position(t)
        - 观测 = 真值 + 噪声
        - KF 估计 [x, vx, y, vy]
        - position(t) 基于最近 sync 时刻的状态做外推（不看未来观测）
    """

    base_trajectory: TrajectoryProvider
    poll_period_s: float = 0.01
    sigma_a: float = 0.08
    meas_noise: float = 0.005
    use_measurement_noise: bool = True
    seed_velocity_from_first_two_measurements: bool = True
    lock_after_stop: bool = True
    cov_feedback_q_pos: float = 20.0
    cov_feedback_q_vel: float = 2.0
    cov_feedback_r: float = 1.0
    seed: Optional[int] = 0

    def __post_init__(self):
        if self.poll_period_s <= 0:
            raise ValueError("poll_period_s must be > 0")
        self._rng = np.random.default_rng(self.seed)
        self._f, self._q, self._h, self._r = _make_kf_matrices(
            dt=float(self.poll_period_s),
            sigma_a=float(self.sigma_a),
            meas_noise=float(self.meas_noise),
        )
        self._a_1d = np.array(
            [[1.0, float(self.poll_period_s)], [0.0, 1.0]],
            dtype=float,
        )
        self._b_1d = np.array(
            [[0.5 * float(self.poll_period_s) ** 2], [float(self.poll_period_s)]],
            dtype=float,
        )
        self._x = np.zeros(4, dtype=float)
        self._p = np.diag([1.0, 5.0, 1.0, 5.0]).astype(float)
        self._initialized = False
        self._last_filter_t = 0.0
        self._anchor_t = 0.0
        self._anchor_x = self._x.copy()
        self._anchor_p = self._p.copy()
        self._z_world = 0.0
        self._prev_meas_z: Optional[np.ndarray] = None
        self._prev_meas_t: Optional[float] = None
        self._vel_seeded = (not bool(self.seed_velocity_from_first_two_measurements))
        self._cov_feedback_k_1d = self._solve_cov_feedback_gain_1d()
        a_cl = self._a_1d - self._b_1d @ self._cov_feedback_k_1d
        self._f_cl = np.zeros((4, 4), dtype=float)
        self._f_cl[0:2, 0:2] = a_cl
        self._f_cl[2:4, 2:4] = a_cl

    def _solve_cov_feedback_gain_1d(self) -> np.ndarray:
        q_pos = max(float(self.cov_feedback_q_pos), 1e-9)
        q_vel = max(float(self.cov_feedback_q_vel), 1e-9)
        r = max(float(self.cov_feedback_r), 1e-9)
        q_mat = np.diag([q_pos, q_vel]).astype(float)
        r_mat = np.array([[r]], dtype=float)
        try:
            p = solve_discrete_are(self._a_1d, self._b_1d, q_mat, r_mat)
            k = np.linalg.solve(self._b_1d.T @ p @ self._b_1d + r_mat, self._b_1d.T @ p @ self._a_1d)
            return np.asarray(k, dtype=float).reshape(1, 2)
        except Exception:
            # 回退到一个稳定的经验增益，避免因数值问题阻塞整条仿真链。
            return np.array([[4.0, 3.0]], dtype=float)

    def _base_stop_time(self) -> Optional[float]:
        d = getattr(self.base_trajectory, "duration", None)
        if callable(d):
            try:
                return float(d())
            except Exception:
                return None
        return None

    def _is_base_stopped(self, sim_time: float) -> bool:
        if not self.lock_after_stop:
            return False
        stop_t = self._base_stop_time()
        if stop_t is None:
            return False
        return float(sim_time) >= (stop_t - 1e-12)

    def _lock_to_base_stop(self, sim_time: float) -> None:
        base_stop = np.asarray(self.base_trajectory.position(sim_time), dtype=float).reshape(3)
        self._z_world = float(base_stop[2])
        self._x = np.array([float(base_stop[0]), 0.0, float(base_stop[1]), 0.0], dtype=float)
        # 锁止后给很小协方差，抑制终点抖动
        self._p = np.diag([1e-8, 1e-8, 1e-8, 1e-8]).astype(float)
        self._prev_meas_z = base_stop[:2].copy()
        self._prev_meas_t = float(sim_time)
        self._vel_seeded = True

    def _kf_predict(self, x: np.ndarray, p: np.ndarray):
        x_pred = self._f @ x
        p_pred = self._f @ p @ self._f.T + self._q
        return x_pred, p_pred

    def _kf_update(self, x_pred: np.ndarray, p_pred: np.ndarray, z: np.ndarray):
        s = self._h @ p_pred @ self._h.T + self._r
        k = p_pred @ self._h.T @ np.linalg.inv(s)
        y = z - (self._h @ x_pred)
        x_new = x_pred + k @ y
        p_new = (np.eye(p_pred.shape[0]) - k @ self._h) @ p_pred
        return x_new, p_new

    def _observe_xy(self, sim_time: float) -> np.ndarray:
        base_true = np.asarray(self.base_trajectory.position(sim_time), dtype=float).reshape(3)
        self._z_world = float(base_true[2])
        z = base_true[:2].copy()
        if (
            self.use_measurement_noise
            and self.meas_noise > 0.0
            and (not self._is_base_stopped(sim_time))
        ):
            z += self._rng.normal(0.0, self.meas_noise, size=2)
        return z

    def _ensure_initialized(self, sim_time: float) -> None:
        if self._initialized:
            return
        z = self._observe_xy(sim_time)
        self._x = np.array([float(z[0]), 0.0, float(z[1]), 0.0], dtype=float)
        self._p = np.diag([1.0, 5.0, 1.0, 5.0]).astype(float)
        self._initialized = True
        self._last_filter_t = float(sim_time)
        self._anchor_t = float(sim_time)
        self._anchor_x = self._x.copy()
        self._anchor_p = self._p.copy()
        self._prev_meas_z = z.copy()
        self._prev_meas_t = float(sim_time)

    def _advance_filter_to(self, sim_time_now: float) -> None:
        self._ensure_initialized(sim_time_now)
        t_target = float(sim_time_now)
        step = float(self.poll_period_s)
        while self._last_filter_t + step <= t_target + 1e-12:
            t_k = self._last_filter_t + step
            if self._is_base_stopped(t_k):
                self._lock_to_base_stop(t_k)
                self._last_filter_t = float(t_k)
                continue
            x_pred, p_pred = self._kf_predict(self._x, self._p)
            z = self._observe_xy(t_k)
            if (not self._vel_seeded) and (self._prev_meas_z is not None) and (self._prev_meas_t is not None):
                dt_meas = max(float(t_k) - float(self._prev_meas_t), 1e-6)
                vx0 = float(z[0] - self._prev_meas_z[0]) / dt_meas
                vy0 = float(z[1] - self._prev_meas_z[1]) / dt_meas
                self._x = np.array([float(z[0]), vx0, float(z[1]), vy0], dtype=float)
                self._p = np.diag([1.0, 5.0, 1.0, 5.0]).astype(float)
                self._vel_seeded = True
            else:
                self._x, self._p = self._kf_update(x_pred, p_pred, z)
            self._prev_meas_z = z.copy()
            self._prev_meas_t = float(t_k)
            self._last_filter_t = t_k
        if self._is_base_stopped(t_target):
            self._lock_to_base_stop(t_target)
            self._last_filter_t = float(t_target)

    def sync(self, sim_time_now: float) -> None:
        """
        用当前“可见观测”更新过滤器，并冻结一个锚点状态。
        后续 position(t>now) 只做外推，不使用未来观测。
        """
        self._advance_filter_to(float(sim_time_now))
        self._anchor_t = float(sim_time_now)
        self._anchor_x = self._x.copy()
        self._anchor_p = self._p.copy()

    def _predict_from_anchor(self, query_time: float) -> np.ndarray:
        if not self._initialized:
            self.sync(query_time)
        dt = max(float(query_time) - float(self._anchor_t), 0.0)
        xq = float(self._anchor_x[0]) + float(self._anchor_x[1]) * dt
        yq = float(self._anchor_x[2]) + float(self._anchor_x[3]) * dt
        return np.array([xq, yq, float(self._z_world)], dtype=float)

    def anchor_cov_xy(self) -> np.ndarray:
        if not self._initialized:
            self.sync(0.0)
        idx = np.ix_([0, 2], [0, 2])
        return self._anchor_p[idx].copy()

    def future_covariances_xy(self, horizon: int, dt: float) -> np.ndarray:
        if horizon <= 0:
            return np.zeros((0, 2, 2), dtype=float)
        if not self._initialized:
            self.sync(0.0)
        step_dt = max(float(dt), 1e-9)
        substeps = max(1, int(round(step_dt / self.poll_period_s)))
        out = np.zeros((int(horizon), 2, 2), dtype=float)
        p = self._anchor_p.copy()
        for i in range(int(horizon)):
            t_future = float(self._anchor_t) + (i + 1) * step_dt
            if self._is_base_stopped(t_future):
                p = np.diag([1e-8, 1e-8, 1e-8, 1e-8]).astype(float)
            else:
                for _ in range(substeps):
                    p = self._f @ p @ self._f.T + self._q
            out[i] = p[np.ix_([0, 2], [0, 2])]
        return out

    def future_covariances_xy_closed_loop(self, horizon: int, dt: float) -> np.ndarray:
        """
        显式闭环协方差传播：
            Σ_{k+1} = F_cl Σ_k F_cl^T + Q
        其中 F_cl = A - B K，K 由一维双积分器 LQR 自动求出，并在 x/y 两轴上复用。
        """
        if horizon <= 0:
            return np.zeros((0, 2, 2), dtype=float)
        if not self._initialized:
            self.sync(0.0)
        step_dt = max(float(dt), 1e-9)
        substeps = max(1, int(round(step_dt / self.poll_period_s)))
        out = np.zeros((int(horizon), 2, 2), dtype=float)
        p = self._anchor_p.copy()
        for i in range(int(horizon)):
            t_future = float(self._anchor_t) + (i + 1) * step_dt
            if self._is_base_stopped(t_future):
                p = np.diag([1e-8, 1e-8, 1e-8, 1e-8]).astype(float)
            else:
                for _ in range(substeps):
                    p = self._f_cl @ p @ self._f_cl.T + self._q
            out[i] = p[np.ix_([0, 2], [0, 2])]
        return out

    def position(self, sim_time: float) -> np.ndarray:
        # 若外部未显式 sync，至少在首个查询时初始化。
        if not self._initialized:
            self.sync(sim_time)
        return self._predict_from_anchor(sim_time)

    def future_positions(self, sim_time: float, horizon: int, dt: float) -> np.ndarray:
        if not self._initialized:
            self.sync(sim_time)
        future = np.zeros((horizon, 3), dtype=float)
        for i in range(horizon):
            future[i] = self._predict_from_anchor(float(sim_time) + (i + 1) * float(dt))
        return future

    def duration(self) -> float:
        d = getattr(self.base_trajectory, "duration", None)
        if callable(d):
            return float(d())
        return 0.0


@dataclass
class BallInRobotFrameTrajectory(BaseTrajectory):
    """
    小球在机械臂/底盘坐标系下的相对轨迹：
        - 底盘沿 base_trajectory 在世界系移动（可由 KF 预测）。
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

    def future_positions(self, sim_time: float, horizon: int, dt: float) -> np.ndarray:
        base_future = self.base_trajectory.future_positions(sim_time, horizon, dt)
        return self.ball_world.reshape(1, 3) - base_future


def get_ball_in_robot_trajectory(
    base_traj: Optional[TrajectoryProvider] = None,
    ball_world=None,
    kf_cfg: Optional[Dict] = None,
) -> TrajectoryProvider:
    """
    生成小球在底盘系下的轨迹。
    参数:
        base_traj: 可传入自定义 BaseTrajectory 覆盖默认（默认 x:-0.8->0.5, speed=0.3 直线）。
        ball_world: 小球世界坐标，可选覆盖默认 [0.25, 0.5, 1.2]。
        默认总是将底盘轨迹包装为 KF 估计+外推。
        kf_cfg: KFPredictiveBaseTrajectory 的参数字典。
    """
    if base_traj is None:
        base_traj = LinearBaseTrajectory()
    if not isinstance(base_traj, KFPredictiveBaseTrajectory):
        base_traj = KFPredictiveBaseTrajectory(base_trajectory=base_traj, **(kf_cfg or {}))
    bw = ball_world if ball_world is not None else np.array([0.25, 0.5, 1.2], dtype=float)
    return BallInRobotFrameTrajectory(base_traj, bw)


__all__ = [
    "BallInRobotFrameTrajectory",
    "get_ball_in_robot_trajectory",
    "LinearBaseTrajectory",
    "KFPredictiveBaseTrajectory",
]

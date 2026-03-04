"""
real_base_udp_kf.py

把真实底盘的 UDP(0x17) 状态接到 MuJoCo 里作为「底盘轨迹」使用。

目标：
    - 后台线程按 real/get_robot_status.py 里的配置轮询 0x17，应答中取 (x, y)。
    - 使用线性离散卡尔曼滤波（CV 模型）估计 [x, vx, y, vy]。
    - 提供 BaseTrajectory 接口：
        - sync(sim_time_now): 用于对齐仿真时间与 wall time（保证未来采样的时间基一致）。
        - position(sim_time): 返回世界系下底盘位置 (x, y, z)。

说明：
    - 这里只用 (x, y) 做观测；角度/角速度暂不参与（后续可扩展 CTRV/EKF）。
    - 默认会把「第一次收到的 (x,y)」作为 real 原点，把它对齐到 MuJoCo 里 chassis 的初始位置，
      使得仿真中的底盘轨迹是“相对位移”而不是直接用 map 的绝对坐标。
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from DPG_track_ball import BaseTrajectory


def _import_udp_module():
    try:
        # 从仓库根目录运行时：real 作为 namespace package 可直接 import
        from real import get_robot_status as udp  # type: ignore

        return udp
    except Exception:
        # 从 real/real_robot_in_mujoco 目录运行时：把上一级 real 加进 sys.path
        import os
        import sys

        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        import get_robot_status as udp  # type: ignore

        return udp


udp = _import_udp_module()


def _default_sim_origin() -> np.ndarray:
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path("fetch_freight_mujoco/xml/scene.xml")
        chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        if chassis_id >= 0:
            return model.body_pos[chassis_id].copy()
    except Exception:
        pass
    return np.array([-0.8, 0.0, 0.0], dtype=float)


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


def _kf_predict(x: np.ndarray, p: np.ndarray, f: np.ndarray, q: np.ndarray):
    x_pred = f @ x
    p_pred = f @ p @ f.T + q
    return x_pred, p_pred


def _kf_update(x_pred: np.ndarray, p_pred: np.ndarray, z: np.ndarray, h: np.ndarray, r: np.ndarray):
    s = h @ p_pred @ h.T + r
    k = p_pred @ h.T @ np.linalg.inv(s)
    y = z - (h @ x_pred)
    x_new = x_pred + k @ y
    p_new = (np.eye(p_pred.shape[0]) - k @ h) @ p_pred
    return x_new, p_new


@dataclass
class RealTimeBaseTrajectory(BaseTrajectory):
    """
    实时底盘轨迹（世界系）：
        - position(sim_time) 用于 MuJoCo 里设置 chassis mocap_pos；
        - 同时也可被 MPC 用于采样未来底盘位置（position(t+Δt)）。
    """

    poll_period_s: float = field(default_factory=lambda: float(udp.POLL_PERIOD_S))
    socket_timeout_s: float = 0.05
    sigma_a: float = 0.05
    meas_noise: float = 0.002
    seed_velocity_from_first_two_measurements: bool = True

    sim_origin: Optional[np.ndarray] = None
    use_relative_origin: bool = True
    swap_xy: bool = False
    sign_x: float = 1.0
    sign_y: float = 1.0

    robot_ip: str = field(default_factory=lambda: str(udp.ROBOT_IP))
    robot_port: int = field(default_factory=lambda: int(udp.ROBOT_PORT))

    def __post_init__(self):
        if self.poll_period_s <= 0:
            raise ValueError(f"poll_period_s must be > 0, got {self.poll_period_s}")

        self.sim_origin = (
            _default_sim_origin() if self.sim_origin is None else np.array(self.sim_origin, dtype=float)
        ).reshape(3)

        dt = float(self.poll_period_s)
        self._f, self._q, self._h, self._r = _make_kf_matrices(
            dt=dt, sigma_a=float(self.sigma_a), meas_noise=float(self.meas_noise)
        )

        self._x = np.zeros(4, dtype=float)
        self._p = np.diag([1.0, 5.0, 1.0, 5.0]).astype(float)
        self._initialized = False
        self._t_est_wall: Optional[float] = None
        self._wall_sim_offset: Optional[float] = None
        self._real_origin_xy: Optional[np.ndarray] = None

        self.ok_count = 0
        self.timeout_count = 0
        self.parse_fail_count = 0

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(float(self.socket_timeout_s))

        self._seq = 0
        self._thread = threading.Thread(target=self._worker, name="RealTimeBaseTrajectoryUDP", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        try:
            self._thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass

    def sync(self, sim_time_now: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._wall_sim_offset = now - float(sim_time_now)

    def _map_real_to_sim_xy(self, x_real: float, y_real: float) -> np.ndarray:
        if self.use_relative_origin:
            if self._real_origin_xy is None:
                self._real_origin_xy = np.array([x_real, y_real], dtype=float)
                dx, dy = 0.0, 0.0
            else:
                dx = x_real - float(self._real_origin_xy[0])
                dy = y_real - float(self._real_origin_xy[1])
        else:
            dx, dy = x_real, y_real

        if self.swap_xy:
            dx, dy = dy, dx

        dx *= float(self.sign_x)
        dy *= float(self.sign_y)

        return np.array([float(self.sim_origin[0]) + dx, float(self.sim_origin[1]) + dy], dtype=float)

    def _worker(self):
        dt = float(self.poll_period_s)
        seq = int(self._seq)
        x = self._x.copy()
        p = self._p.copy()
        initialized = bool(self._initialized)
        vel_seeded = False
        prev_meas_z: Optional[np.ndarray] = None
        prev_meas_wall: Optional[float] = None

        while not self._stop.is_set():
            cycle_start = time.monotonic()

            x_pred, p_pred = _kf_predict(x, p, self._f, self._q)
            z: Optional[np.ndarray] = None
            meas_wall: Optional[float] = None

            try:
                packet = udp.build_0x17_command(seq)
                self._sock.sendto(packet, (self.robot_ip, int(self.robot_port)))
            except Exception:
                z = None

            try:
                data, _ = self._sock.recvfrom(2048)
                meas_wall = time.monotonic()
                parsed, err = udp.parse_0x17_response(data, expected_seq=seq)
                if parsed is None:
                    self.parse_fail_count += 1
                    z = None
                else:
                    z = self._map_real_to_sim_xy(float(parsed["x"]), float(parsed["y"]))
            except socket.timeout:
                self.timeout_count += 1
                z = None
            except Exception:
                self.parse_fail_count += 1
                z = None

            if z is not None:
                self.ok_count += 1
                if not initialized:
                    x = np.array([float(z[0]), 0.0, float(z[1]), 0.0], dtype=float)
                    initialized = True
                    p = self._p.copy()
                    vel_seeded = not bool(self.seed_velocity_from_first_two_measurements)
                elif (
                    (not vel_seeded)
                    and bool(self.seed_velocity_from_first_two_measurements)
                    and (prev_meas_z is not None)
                    and (prev_meas_wall is not None)
                    and (meas_wall is not None)
                ):
                    dt_meas = float(meas_wall) - float(prev_meas_wall)
                    dt_meas = max(dt_meas, 1e-6)
                    vx0 = float(z[0] - prev_meas_z[0]) / dt_meas
                    vy0 = float(z[1] - prev_meas_z[1]) / dt_meas
                    x = np.array([float(z[0]), vx0, float(z[1]), vy0], dtype=float)
                    p = self._p.copy()
                    vel_seeded = True
                else:
                    x, p = _kf_update(x_pred, p_pred, z, self._h, self._r)

                prev_meas_z = z.copy()
                prev_meas_wall = float(meas_wall) if meas_wall is not None else time.monotonic()
            else:
                x, p = x_pred, p_pred

            t_est_wall = time.monotonic()
            with self._lock:
                self._x = x.copy()
                self._p = p.copy()
                self._initialized = initialized
                self._t_est_wall = float(t_est_wall)

            seq = (seq + 1) % 65536

            elapsed = time.monotonic() - cycle_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    def position(self, sim_time: float) -> np.ndarray:
        with self._lock:
            initialized = bool(self._initialized)
            x = self._x.copy()
            t_est_wall = self._t_est_wall
            offset = self._wall_sim_offset
            z0 = float(self.sim_origin[2])

        if (not initialized) or (t_est_wall is None) or (offset is None):
            return np.array([float(self.sim_origin[0]), float(self.sim_origin[1]), z0], dtype=float)

        wall_query = float(offset) + float(sim_time)
        delta = wall_query - float(t_est_wall)
        if delta < 0.0:
            delta = 0.0

        xq = float(x[0]) + float(x[1]) * delta
        yq = float(x[2]) + float(x[3]) * delta
        return np.array([xq, yq, z0], dtype=float)

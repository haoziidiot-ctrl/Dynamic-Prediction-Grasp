"""
real/kalman_filter_robot.py

从底盘 UDP(0x17) 实时获取 (x, y) 位置观测，使用线性离散卡尔曼滤波（CV 模型）
估计状态，并做未来 5s 的轨迹预测。

观测：z_k = [x_k, y_k]^T
状态：X_k = [x_k, vx_k, y_k, vy_k]^T

离散模型（dt = real/get_robot_status.py 里的 POLL_PERIOD_S）：
    X_{k|k-1} = F X_{k-1|k-1} + w_k
    z_k       = H X_{k|k-1} + v_k

其中：
    F = [[1, dt, 0,  0],
         [0,  1, 0,  0],
         [0,  0, 1, dt],
         [0,  0, 0,  1]]
    H = [[1, 0, 0, 0],
         [0, 0, 1, 0]]
"""

from __future__ import annotations

import socket
import time
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


try:
    # 作为包导入（python -m real.kalman_filter_robot）
    from . import get_robot_status as udp
except ImportError:
    # 作为脚本运行（python real/kalman_filter_robot.py）
    import get_robot_status as udp


def make_kf_matrices(dt: float, sigma_a: float, meas_noise: float):
    f = np.array(
        [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # CV 模型的离散过程噪声（基于白噪声加速度）
    q11 = 0.25 * dt**4
    q12 = 0.5 * dt**3
    q22 = dt**2
    q_1d = np.array([[q11, q12], [q12, q22]], dtype=float) * (sigma_a**2)

    q = np.zeros((4, 4), dtype=float)
    q[0:2, 0:2] = q_1d
    q[2:4, 2:4] = q_1d

    h = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    r = np.diag([meas_noise**2, meas_noise**2]).astype(float)
    return f, q, h, r


def kf_predict(x: np.ndarray, p: np.ndarray, f: np.ndarray, q: np.ndarray):
    x_pred = f @ x
    p_pred = f @ p @ f.T + q
    return x_pred, p_pred


def kf_update(
    x_pred: np.ndarray,
    p_pred: np.ndarray,
    z: np.ndarray,
    h: np.ndarray,
    r: np.ndarray,
):
    s = h @ p_pred @ h.T + r
    k = p_pred @ h.T @ np.linalg.inv(s)
    y = z - (h @ x_pred)
    x_new = x_pred + k @ y
    i = np.eye(p_pred.shape[0])
    p_new = (i - k @ h) @ p_pred
    return x_new, p_new


def recv_xy_measurement(sock: socket.socket, seq: int) -> tuple[Optional[np.ndarray], str]:
    """
    单次：收一帧应答并解析 (x, y)。
    返回 (z, msg)，z 为 None 表示失败；msg 用于打印提示。
    """
    try:
        data, _ = sock.recvfrom(2048)
    except socket.timeout:
        return None, "TIMEOUT"

    parsed, err = udp.parse_0x17_response(data, expected_seq=seq)
    if parsed is None:
        return None, f"PARSE_FAIL: {err}"

    return np.array([parsed["x"], parsed["y"]], dtype=float), "OK"


def main():
    dt = float(udp.POLL_PERIOD_S)
    if dt <= 0:
        raise ValueError(f"POLL_PERIOD_S must be > 0, got {dt}")

    # 预测未来 5s（离散步数由 dt 决定）
    pred_horizon_s = 5.0
    pred_steps = max(1, int(round(pred_horizon_s / dt)))

    # 先沿用你 v4 的噪声参数（后续可根据实测调参）
    sigma_a = 0.05
    meas_noise = 0.002

    f, q, h, r = make_kf_matrices(dt=dt, sigma_a=sigma_a, meas_noise=meas_noise)

    x = np.zeros(4, dtype=float)  # [x, vx, y, vy]
    p = np.diag([1.0, 5.0, 1.0, 5.0]).astype(float)
    initialized = False
    vel_seeded = False
    prev_z: Optional[np.ndarray] = None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(float(udp.SOCKET_TIMEOUT_S))

    plt.ion()
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_x.set_title("Kalman Filter X with 5s Prediction")
    ax_x.set_xlabel("Time (s)")
    ax_x.set_ylabel("X (m)")
    ax_x.grid(True)
    line_meas_x, = ax_x.plot([], [], label="Meas X", color="blue", linewidth=1.5)
    line_est_x, = ax_x.plot([], [], label="KF X", color="red", linewidth=1.2)
    line_pred_x, = ax_x.plot([], [], label="Pred X (+5s)", color="green", linestyle="--", linewidth=1.2)
    ax_x.legend(loc="best")

    ax_y.set_title("Kalman Filter Y with 5s Prediction")
    ax_y.set_xlabel("Time (s)")
    ax_y.set_ylabel("Y (m)")
    ax_y.grid(True)
    line_meas_y, = ax_y.plot([], [], label="Meas Y", color="blue", linewidth=1.5)
    line_est_y, = ax_y.plot([], [], label="KF Y", color="red", linewidth=1.2)
    line_pred_y, = ax_y.plot([], [], label="Pred Y (+5s)", color="green", linestyle="--", linewidth=1.2)
    ax_y.legend(loc="best")

    ts: list[float] = []
    x_meas_hist: list[float] = []
    y_meas_hist: list[float] = []
    x_est_hist: list[float] = []
    y_est_hist: list[float] = []

    seq = 0
    step = 0
    last_plot_t = time.monotonic()

    sent_count = 0
    ok_count = 0
    miss_count = 0

    print(f"UDP target: {udp.ROBOT_IP}:{udp.ROBOT_PORT}")
    print(f"dt = POLL_PERIOD_S = {dt:.6f}s, pred_horizon = {pred_horizon_s:.1f}s ({pred_steps} steps)")
    print("按 Ctrl+C 停止程序\n")

    try:
        while True:
            # 1) 发 0x17 请求
            packet = udp.build_0x17_command(seq)
            sock.sendto(packet, (udp.ROBOT_IP, udp.ROBOT_PORT))
            sent_count += 1

            # 2) 收并解析观测 z=[x,y]
            z, status = recv_xy_measurement(sock, seq=seq)
            if z is None:
                miss_count += 1
            else:
                ok_count += 1

            # 3) 离散卡尔曼滤波：先预测，再（有观测才）更新
            x_pred, p_pred = kf_predict(x, p, f, q)

            if z is not None:
                if not initialized:
                    x = np.array([z[0], 0.0, z[1], 0.0], dtype=float)
                    initialized = True
                    vel_seeded = False
                    prev_z = z.copy()
                elif (not vel_seeded) and (prev_z is not None):
                    vx0 = float(z[0] - prev_z[0]) / dt
                    vy0 = float(z[1] - prev_z[1]) / dt
                    x = np.array([z[0], vx0, z[1], vy0], dtype=float)
                    p = np.diag([1.0, 5.0, 1.0, 5.0]).astype(float)
                    vel_seeded = True
                    prev_z = z.copy()
                else:
                    x, p = kf_update(x_pred, p_pred, z, h, r)
                    prev_z = z.copy()
            else:
                x, p = x_pred, p_pred

            # 4) 记录历史 & 计算未来 5s 预测（CV：x(t)=x+vx*t, y(t)=y+vy*t）
            t_curr = step * dt
            ts.append(float(t_curr))
            if z is not None:
                x_meas_hist.append(float(z[0]))
                y_meas_hist.append(float(z[1]))
            else:
                x_meas_hist.append(float("nan"))
                y_meas_hist.append(float("nan"))

            x_est_hist.append(float(x[0]))
            y_est_hist.append(float(x[2]))

            vx, vy = float(x[1]), float(x[3])
            t_future = (np.arange(0, pred_steps + 1, dtype=float) * dt)
            t_pred = t_curr + t_future
            x_pred_future = float(x[0]) + vx * t_future
            y_pred_future = float(x[2]) + vy * t_future

            # 5) 打印状态（失败时也提示，便于你调频率）
            if status != "OK":
                print(
                    f"[Seq:{seq}] t={t_curr:.3f}s | {status} | sent={sent_count} ok={ok_count} miss={miss_count}"
                )
            else:
                print(
                    f"[Seq:{seq}] t={t_curr:.3f}s | z=({z[0]:.3f},{z[1]:.3f}) | "
                    f"est=({x[0]:.3f},{x[2]:.3f}) | v=({vx:.3f},{vy:.3f})"
                )

            # 6) 更新绘图（避免每帧都重绘）
            now_plot = time.monotonic()
            if (now_plot - last_plot_t) > 0.05:
                line_meas_x.set_data(ts, x_meas_hist)
                line_est_x.set_data(ts, x_est_hist)
                line_pred_x.set_data(t_pred, x_pred_future)

                line_meas_y.set_data(ts, y_meas_hist)
                line_est_y.set_data(ts, y_est_hist)
                line_pred_y.set_data(t_pred, y_pred_future)

                ax_x.relim()
                ax_x.autoscale_view()
                ax_y.relim()
                ax_y.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                last_plot_t = now_plot

            # 下一步
            seq = (seq + 1) % 65536
            step += 1
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n程序已停止。")
    finally:
        sock.close()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()

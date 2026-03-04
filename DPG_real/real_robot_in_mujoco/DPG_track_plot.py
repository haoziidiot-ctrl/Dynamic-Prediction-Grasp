"""
DPG_track_plot.py

功能:
    - 统一的轨迹可视化工具，支持任何实现了 TrajectoryProvider 接口的轨迹（小球/小车等）。
    - 一次调用即可绘制空间轨迹（x-y 投影）和时间轴上的坐标变化，便于快速检查参考信号。

用法示例:
    from DPG_track_ball import MocapBallTrajectory
    from DPG_track_plot import plot_trajectory

    plot_trajectory(MocapBallTrajectory(), duration=6.0, dt=0.02)
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import time

from DPG_track_ball import TrajectoryProvider, get_trajectory
from DPG_track_ball_in_robot import BallInRobotFrameTrajectory, get_ball_in_robot_trajectory


def _sample_trajectory(
    traj: TrajectoryProvider, duration: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    if duration <= 0:
        raise ValueError("duration 必须大于 0")
    if dt <= 0:
        raise ValueError("dt 必须大于 0")
    times = np.arange(0.0, duration + 1e-9, dt)
    points = np.vstack([traj.position(t) for t in times])
    return times, points


def plot_trajectory(
    duration: float,
    traj: Optional[TrajectoryProvider] = None,
    object_track: str = "ball",
    dt: float = 0.02,
    title: Optional[str] = None,
    show: bool = True,
):
    """
    绘制轨迹:
        - 左侧子图: x-y 平面投影，颜色随时间渐变。
        - 右侧子图: x(t)、y(t) 随时间变化曲线（含时间轴）。
    参数:
        traj:    轨迹提供器（不传则根据 object_track 自动生成）
        object_track: "ball" / "ball_in_robot"（传入 traj 时忽略）
        duration: 采样总时长（秒）
        dt:      采样时间步长
        title:   可选标题
        show:    是否立即调用 plt.show(block=False)
    返回:
        (fig, axes) 便于调用方自行调整或保存。
    """
    if traj is not None:
        _traj = traj
    elif object_track == "ball_in_robot":
        _traj = get_ball_in_robot_trajectory()
    else:
        _traj = get_trajectory(object_track)
    times, points = _sample_trajectory(_traj, duration, dt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if title:
        fig.suptitle(title, fontsize=13)

    # x-y 投影，颜色映射时间
    sc = axes[0].scatter(points[:, 0], points[:, 1], c=times, cmap="viridis", s=18)
    axes[0].plot(points[:, 0], points[:, 1], color="C0", linewidth=1.0)
    axes[0].set_aspect("equal", "box")
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Y [m]")
    axes[0].set_title("Trajectory XY projection")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    cb = fig.colorbar(sc, ax=axes[0])
    cb.set_label("Time [s]")

    # 时间轴：x(t), y(t)
    axes[1].plot(times, points[:, 0], label="X(t)", color="C1")
    axes[1].plot(times, points[:, 1], label="Y(t)", color="C2")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Position [m]")
    axes[1].set_title("Position over time")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    if show:
        plt.show(block=False)
        plt.pause(0.1)
    return fig, axes


def run_sim_demo(
    traj: TrajectoryProvider,
    duration: float,
    dt: float = 0.02,
    model_xml: str = "fetch_freight_mujoco/xml/scene.xml",
    hold_after: bool = True,
):
    """在 MuJoCo 中播放目标轨迹，方便可视化场景与目标运动。

    - 默认移动 "target" mocap（小球）；若传入 LinearChassisTrajectory 则直接移动底盘 body。
    """
    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)
    target_mocap_id = model.body("target").mocapid[0]
    chassis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    chassis_mocap_id = model.body_mocapid[chassis_body_id]

    times = np.arange(0.0, duration + 1e-9, dt)
    # 如果调用方给出的 dt 与模型步长不一致，会使用传入 dt 做实时节拍
    effective_dt = dt if dt > 0 else float(model.opt.timestep)
    # 判断轨迹类型
    is_ball_rel = isinstance(traj, BallInRobotFrameTrajectory)
    base_traj_for_rel = traj.base_trajectory if is_ball_rel else None
    base_origin = getattr(traj, "base_origin", None) if not is_ball_rel else None

    # 先把初始位姿写入模型，避免一开始就“跳到”后续位置
    init_pos_rel = traj.position(0.0)
    if is_ball_rel and base_traj_for_rel is not None:
        base_pos = base_traj_for_rel.position(0.0)
        if chassis_mocap_id >= 0:
            data.mocap_pos[chassis_mocap_id] = base_pos
            data.mocap_quat[chassis_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
        data.mocap_pos[target_mocap_id] = traj.ball_world
    else:
        world_init = init_pos_rel + base_origin if base_origin is not None else init_pos_rel
        data.mocap_pos[target_mocap_id] = world_init
    mujoco.mj_forward(model, data)

    is_available = getattr(mujoco.viewer, "is_available", lambda: True)
    if is_available():
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 0.0
            viewer.cam.elevation = -25.0
            for t in times:
                loop_start = time.perf_counter()
                target_pos_rel = traj.position(t)
                if is_ball_rel and base_traj_for_rel is not None:
                    base_pos = base_traj_for_rel.position(t)
                    if chassis_mocap_id >= 0:
                        data.mocap_pos[chassis_mocap_id] = base_pos
                        data.mocap_quat[chassis_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
                    data.mocap_pos[target_mocap_id] = traj.ball_world
                else:
                    world_pos = target_pos_rel + base_origin if base_origin is not None else target_pos_rel
                    data.mocap_pos[target_mocap_id] = world_pos
                mujoco.mj_step(model, data)
                viewer.sync()
                elapsed = time.perf_counter() - loop_start
                if effective_dt - elapsed > 0:
                    time.sleep(effective_dt - elapsed)
            if hold_after:
                # 让窗口保持，直到用户手动关闭
                while viewer.is_running():
                    time.sleep(0.05)
    else:
        print("mujoco.viewer 不可用，使用 headless 模式打印位置。")
        for t in times:
            loop_start = time.perf_counter()
            target_pos = traj.position(t)
            if is_ball_rel and base_traj_for_rel is not None:
                base_pos = base_traj_for_rel.position(t)
                if chassis_mocap_id >= 0:
                    data.mocap_pos[chassis_mocap_id] = base_pos
                    data.mocap_quat[chassis_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
                data.mocap_pos[target_mocap_id] = traj.ball_world
            else:
                world_pos = target_pos + base_origin if base_origin is not None else target_pos
                data.mocap_pos[target_mocap_id] = world_pos
            mujoco.mj_step(model, data)
            if int(t * 10) % 10 == 0:
                note = "chassis" if is_ball_rel else "mocap"
                val = model.body_pos[chassis_body_id] if is_ball_rel else data.mocap_pos[target_mocap_id]
                print(f"t={t:.2f}s, {note}={val}")
            elapsed = time.perf_counter() - loop_start
            if effective_dt - elapsed > 0:
                time.sleep(effective_dt - elapsed)


__all__ = ["plot_trajectory", "run_sim_demo"]


if __name__ == "__main__":
    # 只改这个变量即可切换轨迹: "ball" / "ball_in_robot" / 自定义 TrajectoryProvider
    OBJECT_TRACK = "ball_in_robot"
    DURATION = 6.0  # 采样时长（秒）
    DT = 0.02
    SHOW_SIM = True  # 改为 False 则只画图不播放仿真

    if OBJECT_TRACK == "ball_in_robot":
        _traj = get_ball_in_robot_trajectory()
    else:
        _traj = get_trajectory(OBJECT_TRACK)
    plot_trajectory(duration=DURATION, traj=_traj, object_track=OBJECT_TRACK, dt=DT, title=f"Track: {OBJECT_TRACK}", show=True)
    if SHOW_SIM:
        run_sim_demo(traj=_traj, duration=DURATION, dt=DT, hold_after=True)
    # 防止脚本退出过快，可按需保持窗口
    import matplotlib.pyplot as _plt

    _plt.show()

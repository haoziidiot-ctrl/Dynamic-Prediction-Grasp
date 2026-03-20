"""
Generate rollout trajectory plots for DPG_mujoco_final.

Outputs:
    - plots/base_traj_world_xy.png
    - plots/ee_traj_world.png
    - plots/ee_traj_base_frame.png

Usage:
    python DPG_mujoco_final/DPG_generate_plots.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from DPG_MPC import MPCController
from DPG_track_ball import get_trajectory
from DPG_track_ball_in_robot import get_ball_in_robot_trajectory


def build_trajectory(name: str, kf_cfg: Optional[Dict] = None):
    if name == "ball_in_robot":
        return get_ball_in_robot_trajectory(kf_cfg=kf_cfg)
    return get_trajectory("ball")


def _resolve_duration_seconds(obj) -> Optional[float]:
    if obj is None:
        return None
    duration_attr = getattr(obj, "duration", None)
    if callable(duration_attr):
        try:
            val = float(duration_attr())
            if np.isfinite(val) and val > 0.0:
                return val
            return None
        except Exception:
            return None
    if duration_attr is not None:
        try:
            val = float(duration_attr)
            if np.isfinite(val) and val > 0.0:
                return val
            return None
        except Exception:
            return None
    return None


def _infer_max_time(traj, extra_time: float, fallback: float) -> float:
    base = getattr(traj, "base_trajectory", None)
    dur = _resolve_duration_seconds(base)
    if dur is None:
        base_src = getattr(base, "base_trajectory", None) if base is not None else None
        dur = _resolve_duration_seconds(base_src)
    if dur is None:
        return float(fallback)
    return float(dur + extra_time)


def _build_controller(traj) -> MPCController:
    # Keep consistent with historical DPG_mujoco_final plot rollout config.
    return MPCController(
        trajectory=traj,
        warm_start_max=0.0,
        pos_weight=26.0,
        rot_weight=0.1,
        smooth_weight=1.5e-3,
        render_dt=1.0 / 60.0,
        profile_period=1.0e9,
        use_terminal_value=False,
        terminal_value_dim=3,
        terminal_approach_dir=(0.0, 1.0, 0.0),
        terminal_approach_axis="x",
        use_pregrasp=False,
        pregrasp_offset=0.0,
        pregrasp_dir=(0.0, 1.0, 0.0),
        approach_speed=0.35,
        use_predictive_phase_switch=False,
        phase_opt_radius_min=0.40,
        phase_opt_radius_max=0.65,
        phase_trigger_index=6,
        phase_confirm_steps=3,
        phase_min_hold_s=0.0,
        phase_use_planar_distance=True,
        phase_use_x_gate_switch=False,
        phase_x_gate_half_width=0.03,
        phase_x_gate_hold_steps=10,
        phase_instant_attack=True,
        use_offset_tracking=True,
        offset_y=-0.13,
        offset_release_time_s=0.25,
        hold_pos_weight_scale=1.0,
        attach_pos_weight_scale=1.45,
        hold_x_error_gain=1.0,
        attach_x_error_gain=1.35,
        hold_orientation_gain=1.0,
        attach_orientation_gain=0.28,
        offset_trigger_tol=0.02,
        offset_trigger_steps=8,
        offset_switch_x_gate_enable=True,
        offset_switch_x_front=0.2,
        offset_switch_x_align_tol=0.04,
        offset_switch_yz_tol=0.09,
        use_uncertainty_aware_weighting=True,
        uncertainty_beta=60.0,
        uncertainty_min_scale=0.85,
        uncertainty_ema=0.15,
        use_manipulability_guidance=False,
        manipulability_lambda=0.025,
        manipulability_w_threshold=0.06,
        manipulability_fd_delta=0.004,
        manipulability_grad_clip=2.0,
        manipulability_horizon_decay=0.8,
        manipulability_first_step_only=False,
        base_ff_gain=1.0,
        ee_linear_speed_limit=1.0,
        enable_grasp=True,
        grasp_tol=0.02,
        grasp_hold_steps=10,
        grasp_hold_time_s=None,
        grasp_action="none",
        use_constrained_qp=True,
        qp_solver="osqp",
        qp_infeasible_policy="hold",
        qp_enforce_joint_pos=True,
        qp_enforce_joint_vel=True,
        qp_enforce_ee_x_upper=True,
        qp_ee_x_margin=0.02,
        qp_enforce_ee_y_upper=True,
        qp_ee_y_margin=0.0,
    )


def _plot_base_world_xy(
    base_world: np.ndarray,
    out_path: Path,
    *,
    base_kf_pred_world: Optional[np.ndarray] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    main_label = "base"
    if base_kf_pred_world is not None:
        main_label = "base (ground truth)"
    ax.plot(base_world[:, 0], base_world[:, 1], color="C0", linewidth=2.0, label=main_label)

    if base_kf_pred_world is not None:
        ax.plot(
            base_kf_pred_world[:, 0],
            base_kf_pred_world[:, 1],
            color="C3",
            linewidth=1.8,
            linestyle="--",
            label="base (KF predicted)",
        )

    ax.scatter(base_world[0, 0], base_world[0, 1], c="C2", s=80, marker="o", label="start")
    ax.scatter(base_world[-1, 0], base_world[-1, 1], c="C3", s=80, marker="X", label="end")
    ax.set_title("Base trajectory in world frame (XY)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Readability-first y-axis scaling: avoid compression caused by equal aspect.
    y_all = [np.asarray(base_world[:, 1], dtype=float).reshape(-1)]
    if base_kf_pred_world is not None:
        y_all.append(np.asarray(base_kf_pred_world[:, 1], dtype=float).reshape(-1))
    y_stack = np.concatenate(y_all) if len(y_all) > 1 else y_all[0]
    y_min = float(np.min(y_stack))
    y_max = float(np.max(y_stack))
    y_center = 0.5 * (y_min + y_max)
    y_half_span = max(0.08, 0.55 * (y_max - y_min))
    ax.set_ylim(y_center - y_half_span, y_center + y_half_span)
    ax.set_aspect("auto")

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_ee_world(times: np.ndarray, ee_world: np.ndarray, target_world: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ee_world[:, 0], ee_world[:, 1], color="C0", linewidth=2.0, label="ee world")
    axes[0].plot(target_world[:, 0], target_world[:, 1], color="C1", linewidth=1.8, label="target world")
    axes[0].scatter(ee_world[0, 0], ee_world[0, 1], c="C2", s=70, marker="o", label="ee start")
    axes[0].scatter(ee_world[-1, 0], ee_world[-1, 1], c="C3", s=70, marker="X", label="ee end")
    axes[0].set_title("End-effector trajectory in world frame (XY)")
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Y [m]")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="best")

    axes[1].plot(times, ee_world[:, 0], color="C0", label="ee_x")
    axes[1].plot(times, ee_world[:, 1], color="C1", label="ee_y")
    axes[1].plot(times, ee_world[:, 2], color="C2", label="ee_z")
    axes[1].plot(times, target_world[:, 0], color="C0", linestyle="--", alpha=0.7, label="target_x")
    axes[1].plot(times, target_world[:, 1], color="C1", linestyle="--", alpha=0.7, label="target_y")
    axes[1].plot(times, target_world[:, 2], color="C2", linestyle="--", alpha=0.7, label="target_z")
    axes[1].set_title("World coordinates over time")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Position [m]")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="best", ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_ee_base(times: np.ndarray, ee_base: np.ndarray, target_base: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ee_base[:, 0], ee_base[:, 1], color="C0", linewidth=2.0, label="ee in base")
    axes[0].plot(target_base[:, 0], target_base[:, 1], color="C1", linewidth=1.8, label="target in base")
    axes[0].scatter(ee_base[0, 0], ee_base[0, 1], c="C2", s=70, marker="o", label="ee start")
    axes[0].scatter(ee_base[-1, 0], ee_base[-1, 1], c="C3", s=70, marker="X", label="ee end")
    axes[0].set_title("End-effector trajectory in base frame (XY)")
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Y [m]")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="best")

    axes[1].plot(times, ee_base[:, 0], color="C0", label="ee_x_base")
    axes[1].plot(times, ee_base[:, 1], color="C1", label="ee_y_base")
    axes[1].plot(times, ee_base[:, 2], color="C2", label="ee_z_base")
    axes[1].plot(times, target_base[:, 0], color="C0", linestyle="--", alpha=0.7, label="target_x_base")
    axes[1].plot(times, target_base[:, 1], color="C1", linestyle="--", alpha=0.7, label="target_y_base")
    axes[1].plot(times, target_base[:, 2], color="C2", linestyle="--", alpha=0.7, label="target_z_base")
    axes[1].set_title("Base-frame coordinates over time")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Position [m]")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="best", ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Run one rollout and save trajectory plots.")
    parser.add_argument(
        "--object-track",
        choices=["ball", "ball_in_robot"],
        default="ball_in_robot",
        help="trajectory mode",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="DPG_mujoco_final/plots",
        help="output directory for png files",
    )
    parser.add_argument(
        "--extra-time",
        type=float,
        default=4.0,
        help="extra simulated seconds after base trajectory duration",
    )
    parser.add_argument(
        "--fallback-max-time",
        type=float,
        default=15.0,
        help="fallback horizon when base duration is unavailable",
    )
    parser.add_argument(
        "--kf-seed",
        type=int,
        default=0,
        help="KF trajectory seed (same meaning as DPG_main.py)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kf_cfg = {
        "poll_period_s": 0.01,
        "sigma_a": 0.08,
        "meas_noise": 0.005,
        "use_measurement_noise": True,
        "seed": int(args.kf_seed),
    }

    traj = build_trajectory(args.object_track, kf_cfg=kf_cfg)
    ctrl = _build_controller(traj)
    max_time = _infer_max_time(traj, extra_time=float(args.extra_time), fallback=float(args.fallback_max_time))

    records: List[Dict] = []

    def on_step(stats: Dict):
        records.append(
            {
                "time": float(stats["time"]),
                "abs_time": float(stats["abs_time"]),
                "ee_pos": np.asarray(stats["ee_pos"], dtype=float).reshape(3),
                "target_ref": np.asarray(stats["target_ref"], dtype=float).reshape(3),
                "base_pos_world": np.asarray(stats["base_pos_world"], dtype=float).reshape(3),
            }
        )

    summary = ctrl.run_headless(max_time=max_time, step_callback=on_step, realtime_sync=False)
    if len(records) == 0:
        raise RuntimeError("No rollout samples were collected; plotting aborted.")

    times = np.asarray([r["time"] for r in records], dtype=float)
    abs_times = np.asarray([r["abs_time"] for r in records], dtype=float)
    ee_world = np.vstack([r["ee_pos"] for r in records])
    target_world = np.vstack([r["target_ref"] for r in records])
    base_world_kf = np.vstack([r["base_pos_world"] for r in records])

    base_query_times = times if bool(getattr(ctrl, "use_offset_tracking", False)) else abs_times
    base_world_true = None
    base_obj = getattr(ctrl, "base_traj_for_rel", None)
    if base_obj is not None:
        base_src = getattr(base_obj, "base_trajectory", None)
        if base_src is not None:
            base_world_true = np.vstack(
                [
                    np.asarray(base_src.position(float(t)), dtype=float).reshape(3)
                    for t in base_query_times
                ]
            )

    base_world_main = base_world_true if base_world_true is not None else base_world_kf
    ee_base = ee_world - base_world_kf
    target_base = target_world - base_world_kf

    p_base = out_dir / "base_traj_world_xy.png"
    p_world = out_dir / "ee_traj_world.png"
    p_base_frame = out_dir / "ee_traj_base_frame.png"

    _plot_base_world_xy(
        base_world_main,
        p_base,
        base_kf_pred_world=(base_world_kf if base_world_true is not None else None),
    )
    _plot_ee_world(times, ee_world, target_world, p_world)
    _plot_ee_base(times, ee_base, target_base, p_base_frame)

    print("[done] rollout summary:")
    print(
        f"  grasped={summary['grasped']}, sim_time={summary['sim_time']:.3f}s, "
        f"final_target_err={summary['final_target_err']:.4f} m"
    )
    print("[done] saved plots:")
    print(f"  {p_base}")
    print(f"  {p_world}")
    print(f"  {p_base_frame}")


if __name__ == "__main__":
    main()


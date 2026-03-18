from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from DPG_main import build_trajectory
from DPG_MPC import MPCController


SEEDS = (0, 1, 2)
MAX_TIME = 6.0


@dataclass
class CaseResult:
    mode: str
    seed: int
    grasped: bool
    success_time: float
    attach_time: float
    min_target_err: float
    final_target_err: float
    qp_infeasible_count: int
    max_backoff_y: float


def build_controller(*, seed: int, use_stochastic: bool) -> MPCController:
    traj = build_trajectory(
        "ball_in_robot",
        kf_cfg={
            "poll_period_s": 0.01,
            "sigma_a": 0.08,
            "meas_noise": 0.005,
            "use_measurement_noise": True,
            "cov_feedback_q_pos": 20.0,
            "cov_feedback_q_vel": 2.0,
            "cov_feedback_r": 1.0,
            "seed": int(seed),
        },
    )
    return MPCController(
        trajectory=traj,
        warm_start_max=0.0,
        use_terminal_value=False,
        pos_weight=26.0,
        rot_weight=0.10,
        smooth_weight=1.5e-3,
        render_dt=1.0 / 60.0,
        profile_period=1000.0,
        terminal_approach_dir=(0.0, 1.0, 0.0),
        terminal_approach_axis="x",
        use_offset_tracking=True,
        offset_y=-0.13,
        offset_release_time_s=0.35,
        hold_pos_weight_scale=1.0,
        attach_pos_weight_scale=1.45,
        hold_x_error_gain=1.0,
        attach_x_error_gain=1.35,
        hold_orientation_gain=1.0,
        attach_orientation_gain=0.28,
        offset_trigger_tol=0.02,
        offset_trigger_steps=8,
        offset_switch_x_gate_enable=True,
        offset_switch_x_front=0.20,
        offset_switch_x_align_tol=0.04,
        offset_switch_yz_tol=0.09,
        use_uncertainty_aware_weighting=True,
        uncertainty_beta=60.0,
        uncertainty_min_scale=0.85,
        uncertainty_ema=0.15,
        use_manipulability_guidance=False,
        base_ff_gain=1.0,
        ee_linear_speed_limit=1.0,
        enable_grasp=True,
        grasp_tol=0.02,
        grasp_hold_steps=10,
        grasp_action="stop",
        use_constrained_qp=True,
        qp_solver="osqp",
        qp_infeasible_policy="hold",
        qp_enforce_joint_pos=True,
        qp_enforce_joint_vel=True,
        qp_enforce_ee_x_upper=True,
        qp_ee_x_margin=0.02,
        qp_enforce_ee_y_upper=True,
        qp_ee_y_margin=0.0,
        use_stochastic_mpc=bool(use_stochastic),
        stochastic_risk_alpha=0.05,
        stochastic_sigma_scale=3.0,
        stochastic_backoff_max=0.01,
        stochastic_release_slowdown=1.0,
        stochastic_ff_suppression=18.0,
        stochastic_attach_pos_boost=0.35,
        stochastic_attach_x_boost=0.50,
        stochastic_attach_rot_boost=0.08,
        stochastic_use_tube_feedback=False,
        stochastic_tube_gain_xy=0.28,
        stochastic_tube_gain_z=0.18,
        stochastic_tube_max_linear=0.18,
    )


def run_case(*, seed: int, use_stochastic: bool) -> CaseResult:
    ctrl = build_controller(seed=seed, use_stochastic=use_stochastic)
    result = ctrl.run_headless(max_time=MAX_TIME, realtime_sync=False)
    success_time = float(result["grasp_time"]) if bool(result["grasped"]) else math.nan
    return CaseResult(
        mode="stochastic" if use_stochastic else "basic",
        seed=int(seed),
        grasped=bool(result["grasped"]),
        success_time=success_time,
        attach_time=float(result["attach_time"]),
        min_target_err=float(result["best_target_err"]),
        final_target_err=float(result["final_target_err"]),
        qp_infeasible_count=int(result["qp_infeasible_count"]),
        max_backoff_y=float(result["max_backoff_y"]),
    )


def summarize(results: list[CaseResult], mode: str) -> None:
    rows = [r for r in results if r.mode == mode]
    success_rate = np.mean([1.0 if r.grasped else 0.0 for r in rows])
    success_times = [r.success_time for r in rows if np.isfinite(r.success_time)]
    attach_times = [r.attach_time for r in rows if np.isfinite(r.attach_time)]
    min_errs = [r.min_target_err for r in rows]
    final_errs = [r.final_target_err for r in rows]
    infeasible = [r.qp_infeasible_count for r in rows]
    backoff = [r.max_backoff_y for r in rows]
    print(
        f"[summary:{mode}] success_rate={success_rate:.2f}, "
        f"success_time_mean={np.mean(success_times) if success_times else float('nan'):.3f}s, "
        f"attach_time_mean={np.mean(attach_times) if attach_times else float('nan'):.3f}s, "
        f"min_target_err_mean={np.mean(min_errs):.4f}m, "
        f"final_target_err_mean={np.mean(final_errs):.4f}m, "
        f"qp_infeasible_mean={np.mean(infeasible):.2f}, "
        f"max_backoff_y_mean={np.mean(backoff):.4f}m"
    )


def main() -> None:
    results: list[CaseResult] = []
    for seed in SEEDS:
        for use_stochastic in (False, True):
            row = run_case(seed=seed, use_stochastic=use_stochastic)
            results.append(row)
            print(
                f"[case] mode={row.mode}, seed={row.seed}, grasped={row.grasped}, "
                f"success_time={row.success_time:.3f}s, attach_time={row.attach_time:.3f}s, "
                f"min_target_err={row.min_target_err:.4f}m, final_target_err={row.final_target_err:.4f}m, "
                f"qp_infeasible={row.qp_infeasible_count}, max_backoff_y={row.max_backoff_y:.4f}m"
            )
    summarize(results, "basic")
    summarize(results, "stochastic")


if __name__ == "__main__":
    main()

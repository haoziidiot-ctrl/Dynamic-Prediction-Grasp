"""
DPG_ablation_random_mpc.py

消融实验2："random_mpc" 开/关对比。

实现映射（analytic 框架内可直接切换）：
- random_mpc_on = 1 -> use_constrained_qp=True  （OSQP 约束QP链路）
- random_mpc_on = 0 -> use_constrained_qp=False （解析链路）

其余统计口径与 uncertainty 消融保持一致。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from DPG_MPC import MPCController
from DPG_track_ball_in_robot import (
    BallInRobotFrameTrajectory,
    KFPredictiveBaseTrajectory,
    LinearBaseTrajectory,
)


BALL_WORLD = np.array([0.25, 0.5, 1.2], dtype=float)
ZONE_X_HALF_WIDTH = 0.05
ZONE_Y_BACK = 0.10


@dataclass
class EpisodeRandomParams:
    x_start: float
    x_end: float
    speed: float
    y_noise_mean: float
    y_noise_std: float
    noise_seed: int
    kf_seed: int


def _sample_episode_params(rng: np.random.Generator, y_noise_std: float) -> EpisodeRandomParams:
    return EpisodeRandomParams(
        x_start=float(rng.uniform(-1.1, -0.5)),
        x_end=float(rng.uniform(0.2, 0.8)),
        speed=float(rng.uniform(0.1, 0.5)),
        y_noise_mean=float(rng.uniform(0.005, 0.02)),
        y_noise_std=float(y_noise_std),
        noise_seed=int(rng.integers(0, 2**31 - 1)),
        kf_seed=int(rng.integers(0, 2**31 - 1)),
    )


def _make_random_trajectory(params: EpisodeRandomParams) -> Tuple[BallInRobotFrameTrajectory, LinearBaseTrajectory]:
    base = LinearBaseTrajectory(
        start=np.array([params.x_start, 0.0, 0.0], dtype=float),
        end=np.array([params.x_end, 0.0, 0.0], dtype=float),
        speed=params.speed,
        lateral_noise_dt=0.01,
        lateral_noise_mag_mean=params.y_noise_mean,
        lateral_noise_mag_std=params.y_noise_std,
        lateral_noise_seed=params.noise_seed,
    )
    kf = KFPredictiveBaseTrajectory(
        base_trajectory=base,
        poll_period_s=0.01,
        sigma_a=0.08,
        meas_noise=0.005,
        use_measurement_noise=True,
        seed=params.kf_seed,
    )
    traj = BallInRobotFrameTrajectory(base_trajectory=kf, ball_world=BALL_WORLD.copy())
    return traj, base


def _build_controller(
    traj: BallInRobotFrameTrajectory,
    *,
    random_mpc_on: bool,
    uncertainty_on: bool,
) -> MPCController:
    # 与 DPG_mujoco_analytic/DPG_main.py 当前版本对齐；
    # 唯一区别是 random_mpc_on 仅切 use_constrained_qp 开关。
    return MPCController(
        trajectory=traj,
        horizon=6,
        control_dt=0.03,
        warm_start_max=0.0,
        pos_weight=26.0,
        rot_weight=0.25,
        smooth_weight=1.5e-3,
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
        offset_release_time_s=0.45,
        hold_pos_weight_scale=1.0,
        attach_pos_weight_scale=1.15,
        hold_x_error_gain=1.0,
        attach_x_error_gain=1.0,
        hold_orientation_gain=1.0,
        attach_orientation_gain=0.28,
        offset_trigger_tol=0.03,
        offset_trigger_steps=8,
        offset_switch_x_gate_enable=True,
        offset_switch_x_front=0.05,
        offset_switch_x_align_tol=0.04,
        offset_switch_yz_tol=0.09,
        use_uncertainty_aware_weighting=bool(uncertainty_on),
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
        ee_linear_speed_limit=0.75,
        enable_grasp=True,
        grasp_tol=0.02,
        grasp_hold_steps=10,
        grasp_hold_time_s=None,
        grasp_action="attach",
        use_constrained_qp=bool(random_mpc_on),
        qp_solver="osqp",
        qp_infeasible_policy="hold",
        qp_enforce_joint_pos=True,
        qp_enforce_joint_vel=True,
        qp_enforce_ee_x_upper=True,
        qp_ee_x_margin=-0.015,
        qp_enforce_ee_y_upper=True,
        qp_ee_y_margin=0.0,
        qp_enforce_ee_z_lower=True,
        qp_ee_z_margin=-0.010,
        profile_period=1e9,
    )


def _safe_mean(values: List[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_p95(values: List[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), 95.0))


def _diff_rms_scalar(values: List[float]) -> float:
    if len(values) < 2:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    d = np.diff(arr)
    return float(np.sqrt(np.mean(np.square(d))))


def _diff_rms_vector(values: List[np.ndarray]) -> float:
    if len(values) < 2:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    d = np.diff(arr, axis=0)
    d_norm = np.linalg.norm(d, axis=1)
    return float(np.sqrt(np.mean(np.square(d_norm))))


def _nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _zone_hit(points_world: np.ndarray, target_world: np.ndarray) -> bool:
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    tar = np.asarray(target_world, dtype=float).reshape(3)

    x_low = float(tar[0] - ZONE_X_HALF_WIDTH)
    x_high = float(tar[0] + ZONE_X_HALF_WIDTH)
    y_low = float(tar[1] - ZONE_Y_BACK)
    y_high = float(tar[1])

    x_ok = (pts[:, 0] <= x_low) | (pts[:, 0] >= x_high)
    y_ok = (pts[:, 1] >= y_low) & (pts[:, 1] <= y_high)
    return bool(np.any(x_ok & y_ok))


def _win_lower_is_better(a: float, b: float) -> float:
    if (not np.isfinite(a)) or (not np.isfinite(b)):
        return float("nan")
    return float(1.0 if a < b else 0.0)


def _run_one_mode(
    params: EpisodeRandomParams,
    *,
    random_mpc_on: bool,
    uncertainty_on: bool,
    extra_time: float,
) -> Dict:
    traj, base = _make_random_trajectory(params)
    ctrl = _build_controller(
        traj,
        random_mpc_on=random_mpc_on,
        uncertainty_on=uncertainty_on,
    )
    base_duration = float(base.duration())
    max_time = float(base_duration + extra_time)

    target_err_list: List[float] = []
    hold_err_list: List[float] = []
    attach_err_list: List[float] = []
    qdot_list: List[np.ndarray] = []

    step_count = 0
    zone_steps = 0
    zone_entries = 0
    prev_in_zone = False

    def on_control(stats: Dict):
        qdot = np.asarray(stats.get("qdot", np.zeros(ctrl.arm_dof)), dtype=float).reshape(-1)
        qdot_list.append(qdot)

    def on_step(stats: Dict):
        nonlocal step_count, zone_steps, zone_entries, prev_in_zone
        step_count += 1

        phase = str(stats["phase"])
        target_err = float(stats["target_err"])
        hold_err = float(stats["hold_err"])
        attach_err = float(stats["attach_err"])

        target_err_list.append(target_err)
        if phase == "hold":
            hold_err_list.append(hold_err)
        else:
            attach_err_list.append(attach_err)

        target_ref = np.asarray(stats["target_ref"], dtype=float).reshape(3)
        body_pts = np.asarray(ctrl.data.xpos[ctrl.arm_joint_body_ids], dtype=float).reshape(-1, 3)
        ee_pt = np.asarray(ctrl.data.site_xpos[ctrl.ee_site_id], dtype=float).reshape(1, 3)
        all_pts = np.vstack([body_pts, ee_pt])
        in_zone = _zone_hit(all_pts, target_ref)

        if in_zone:
            zone_steps += 1
        if in_zone and (not prev_in_zone):
            zone_entries += 1
        prev_in_zone = in_zone

    run_info = ctrl.run_headless(
        max_time=max_time,
        step_callback=on_step,
        control_callback=on_control,
        realtime_sync=False,
    )

    zone_time_s = float(zone_steps * ctrl.dt)
    summary = {
        "random_mpc_on": int(bool(random_mpc_on)),
        "base_duration_s": base_duration,
        "max_time_s": max_time,
        "steps_total": int(step_count),
        "sim_time_s": float(run_info["sim_time"]),
        "success": int(bool(run_info["grasped"])),
        "zone_steps": int(zone_steps),
        "zone_time_s": zone_time_s,
        "zone_ratio": float(zone_steps / max(step_count, 1)),
        "zone_entries": int(zone_entries),
        "target_err_mean": _safe_mean(target_err_list),
        "target_err_min": float(np.min(target_err_list)) if len(target_err_list) > 0 else np.nan,
        "target_err_p95": _safe_p95(target_err_list),
        "hold_steps": int(len(hold_err_list)),
        "hold_err_mean": _safe_mean(hold_err_list),
        "hold_err_p95": _safe_p95(hold_err_list),
        "attach_steps": int(len(attach_err_list)),
        "attach_err_mean": _safe_mean(attach_err_list),
        "attach_err_p95": _safe_p95(attach_err_list),
        "hf_target_err_diff_rms": _diff_rms_scalar(target_err_list),
        "hf_attach_err_diff_rms": _diff_rms_scalar(attach_err_list),
        "hf_qdot_diff_rms": _diff_rms_vector(qdot_list),
    }
    return summary


def _summary_for_mode(rows: List[Dict], random_mpc_on: int) -> Dict:
    sub = [r for r in rows if int(r["random_mpc_on"]) == int(random_mpc_on)]
    n = len(sub)
    return {
        "random_mpc_on": int(random_mpc_on),
        "episodes": n,
        "success_rate": _nanmean([r["success"] for r in sub]),
        "zone_time_mean_s": _nanmean([r["zone_time_s"] for r in sub]),
        "zone_ratio_mean": _nanmean([r["zone_ratio"] for r in sub]),
        "zone_entries_mean": _nanmean([r["zone_entries"] for r in sub]),
        "target_err_mean": _nanmean([r["target_err_mean"] for r in sub]),
        "target_err_p95_mean": _nanmean([r["target_err_p95"] for r in sub]),
        "hold_err_mean": _nanmean([r["hold_err_mean"] for r in sub]),
        "attach_err_mean": _nanmean([r["attach_err_mean"] for r in sub]),
        "hf_target_err_diff_rms_mean": _nanmean([r["hf_target_err_diff_rms"] for r in sub]),
        "hf_attach_err_diff_rms_mean": _nanmean([r["hf_attach_err_diff_rms"] for r in sub]),
        "hf_qdot_diff_rms_mean": _nanmean([r["hf_qdot_diff_rms"] for r in sub]),
    }


def run_ablation(
    episodes: int,
    seed: int,
    y_noise_std: float,
    out_dir: Path,
    extra_time: float,
    uncertainty_on: bool,
) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    per_episode_mode_path = out_dir / "per_episode_mode.csv"
    per_delta_path = out_dir / "per_episode_delta_on_minus_off.csv"
    summary_mode_path = out_dir / "summary_by_mode.csv"
    summary_delta_path = out_dir / "summary_delta.csv"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "episodes": episodes,
                "seed": seed,
                "ball_world": BALL_WORLD.tolist(),
                "ablation": "random_mpc on/off",
                "switch_mapping": {
                    "random_mpc_on": {
                        "use_constrained_qp": True,
                        "qp_constraints": ["ee_x_upper", "ee_y_upper", "ee_z_lower"],
                        "qp_margins": {"x": -0.015, "y": 0.0, "z": -0.010},
                    },
                    "random_mpc_off": {
                        "use_constrained_qp": False,
                        "path": "default analytic mpc",
                    },
                },
                "fixed_switches": {
                    "use_uncertainty_aware_weighting": bool(uncertainty_on),
                    "use_manipulability_guidance": False,
                },
                "random_ranges": {
                    "x_start": [-1.1, -0.5],
                    "x_end": [0.2, 0.8],
                    "speed": [0.1, 0.5],
                    "y_noise_mean": [0.005, 0.02],
                    "y_noise_std_fixed": y_noise_std,
                },
                "zone_rule": {
                    "x_exclusion_half_width": ZONE_X_HALF_WIDTH,
                    "y_window_back": ZONE_Y_BACK,
                    "points": "6 arm joint bodies + end_finger site",
                },
                "fairness": "paired same-random-params per episode",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    random_params = [_sample_episode_params(rng, y_noise_std=y_noise_std) for _ in range(episodes)]
    all_rows: List[Dict] = []
    delta_rows: List[Dict] = []

    for ep_idx, params in enumerate(random_params, start=1):
        row_on = _run_one_mode(
            params,
            random_mpc_on=True,
            uncertainty_on=uncertainty_on,
            extra_time=extra_time,
        )
        row_off = _run_one_mode(
            params,
            random_mpc_on=False,
            uncertainty_on=uncertainty_on,
            extra_time=extra_time,
        )

        common = {
            "episode": ep_idx,
            "x_start": params.x_start,
            "x_end": params.x_end,
            "speed": params.speed,
            "y_noise_mean": params.y_noise_mean,
            "y_noise_std": params.y_noise_std,
            "noise_seed": params.noise_seed,
            "kf_seed": params.kf_seed,
        }
        row_on.update(common)
        row_off.update(common)
        all_rows.extend([row_on, row_off])

        delta_row = {
            "episode": ep_idx,
            "success_delta_on_minus_off": float(row_on["success"] - row_off["success"]),
            "zone_time_delta_on_minus_off": float(row_on["zone_time_s"] - row_off["zone_time_s"]),
            "zone_ratio_delta_on_minus_off": float(row_on["zone_ratio"] - row_off["zone_ratio"]),
            "target_err_mean_delta_on_minus_off": float(
                row_on["target_err_mean"] - row_off["target_err_mean"]
            ),
            "attach_err_mean_delta_on_minus_off": float(
                row_on["attach_err_mean"] - row_off["attach_err_mean"]
            ),
            "hf_target_err_diff_rms_delta_on_minus_off": float(
                row_on["hf_target_err_diff_rms"] - row_off["hf_target_err_diff_rms"]
            ),
            "hf_attach_err_diff_rms_delta_on_minus_off": float(
                row_on["hf_attach_err_diff_rms"] - row_off["hf_attach_err_diff_rms"]
            ),
            "hf_qdot_diff_rms_delta_on_minus_off": float(
                row_on["hf_qdot_diff_rms"] - row_off["hf_qdot_diff_rms"]
            ),
            "win_zone_time_on": _win_lower_is_better(row_on["zone_time_s"], row_off["zone_time_s"]),
            "win_attach_err_on": _win_lower_is_better(
                row_on["attach_err_mean"], row_off["attach_err_mean"]
            ),
            "win_hf_attach_on": _win_lower_is_better(
                row_on["hf_attach_err_diff_rms"], row_off["hf_attach_err_diff_rms"]
            ),
            "win_hf_qdot_on": _win_lower_is_better(
                row_on["hf_qdot_diff_rms"], row_off["hf_qdot_diff_rms"]
            ),
        }
        delta_rows.append(delta_row)

        print(
            f"[episode {ep_idx:03d}/{episodes}] "
            f"ON(success={row_on['success']}, zone={row_on['zone_time_s']:.3f}s, attach={row_on['attach_err_mean']:.4f}) | "
            f"OFF(success={row_off['success']}, zone={row_off['zone_time_s']:.3f}s, attach={row_off['attach_err_mean']:.4f})"
        )

    per_episode_fields = [
        "episode",
        "random_mpc_on",
        "x_start",
        "x_end",
        "speed",
        "y_noise_mean",
        "y_noise_std",
        "noise_seed",
        "kf_seed",
        "base_duration_s",
        "max_time_s",
        "steps_total",
        "sim_time_s",
        "success",
        "zone_steps",
        "zone_time_s",
        "zone_ratio",
        "zone_entries",
        "target_err_mean",
        "target_err_min",
        "target_err_p95",
        "hold_steps",
        "hold_err_mean",
        "hold_err_p95",
        "attach_steps",
        "attach_err_mean",
        "attach_err_p95",
        "hf_target_err_diff_rms",
        "hf_attach_err_diff_rms",
        "hf_qdot_diff_rms",
    ]
    with open(per_episode_mode_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_episode_fields)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in per_episode_fields})

    delta_fields = [
        "episode",
        "success_delta_on_minus_off",
        "zone_time_delta_on_minus_off",
        "zone_ratio_delta_on_minus_off",
        "target_err_mean_delta_on_minus_off",
        "attach_err_mean_delta_on_minus_off",
        "hf_target_err_diff_rms_delta_on_minus_off",
        "hf_attach_err_diff_rms_delta_on_minus_off",
        "hf_qdot_diff_rms_delta_on_minus_off",
        "win_zone_time_on",
        "win_attach_err_on",
        "win_hf_attach_on",
        "win_hf_qdot_on",
    ]
    with open(per_delta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=delta_fields)
        writer.writeheader()
        for r in delta_rows:
            writer.writerow({k: r.get(k, "") for k in delta_fields})

    summary_on = _summary_for_mode(all_rows, random_mpc_on=1)
    summary_off = _summary_for_mode(all_rows, random_mpc_on=0)
    with open(summary_mode_path, "w", newline="", encoding="utf-8") as f:
        fields = list(summary_on.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(summary_on)
        writer.writerow(summary_off)

    def _mean_delta(key: str) -> float:
        return _nanmean([float(r[key]) for r in delta_rows])

    summary_delta = {
        "episodes": int(episodes),
        "success_delta_on_minus_off_mean": _mean_delta("success_delta_on_minus_off"),
        "zone_time_delta_on_minus_off_mean": _mean_delta("zone_time_delta_on_minus_off"),
        "zone_ratio_delta_on_minus_off_mean": _mean_delta("zone_ratio_delta_on_minus_off"),
        "target_err_mean_delta_on_minus_off_mean": _mean_delta("target_err_mean_delta_on_minus_off"),
        "attach_err_mean_delta_on_minus_off_mean": _mean_delta("attach_err_mean_delta_on_minus_off"),
        "hf_target_err_diff_rms_delta_on_minus_off_mean": _mean_delta(
            "hf_target_err_diff_rms_delta_on_minus_off"
        ),
        "hf_attach_err_diff_rms_delta_on_minus_off_mean": _mean_delta(
            "hf_attach_err_diff_rms_delta_on_minus_off"
        ),
        "hf_qdot_diff_rms_delta_on_minus_off_mean": _mean_delta("hf_qdot_diff_rms_delta_on_minus_off"),
        "win_zone_time_on_rate": _nanmean([float(r["win_zone_time_on"]) for r in delta_rows]),
        "win_attach_err_on_rate": _nanmean([float(r["win_attach_err_on"]) for r in delta_rows]),
        "win_hf_attach_on_rate": _nanmean([float(r["win_hf_attach_on"]) for r in delta_rows]),
        "win_hf_qdot_on_rate": _nanmean([float(r["win_hf_qdot_on"]) for r in delta_rows]),
    }
    with open(summary_delta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_delta.keys()))
        writer.writeheader()
        writer.writerow(summary_delta)

    print("\n[done] random_mpc ablation finished.")
    print(f"  per-episode : {per_episode_mode_path}")
    print(f"  per-delta   : {per_delta_path}")
    print(f"  summary-mode: {summary_mode_path}")
    print(f"  summary-dlt : {summary_delta_path}")
    print(f"  config      : {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation: random_mpc ON vs OFF under paired random episodes"
    )
    parser.add_argument("--episodes", type=int, default=200, help="number of paired episodes")
    parser.add_argument("--seed", type=int, default=20260306, help="global RNG seed")
    parser.add_argument(
        "--y-noise-std",
        type=float,
        default=0.005,
        help="fixed lateral disturbance std (mean randomized in [0.005, 0.02])",
    )
    parser.add_argument(
        "--extra-time",
        type=float,
        default=6.0,
        help="extra simulation time added to each episode after base duration",
    )
    parser.add_argument(
        "--uncertainty-on",
        type=int,
        default=1,
        choices=[0, 1],
        help="keep uncertainty-aware weighting fixed on/off for both groups",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="output directory (default: DPG_mujoco_analytic/experiment_results/ablation_random_mpc_<timestamp>)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be > 0")
    if args.y_noise_std < 0.0:
        raise ValueError("y-noise-std must be >= 0")

    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parent / "experiment_results" / f"ablation_random_mpc_{stamp}"
    else:
        out_dir = Path(args.out_dir).resolve()

    run_ablation(
        episodes=int(args.episodes),
        seed=int(args.seed),
        y_noise_std=float(args.y_noise_std),
        out_dir=out_dir,
        extra_time=float(args.extra_time),
        uncertainty_on=bool(int(args.uncertainty_on)),
    )


if __name__ == "__main__":
    main()

"""
DPG_batch_experiment.py

无头批量综合实验入口（默认200回合）：
1) 成功率（grasp success）
2) 机械臂任一点进入禁区时间（任一关节或末端进入即计时）
3) 分阶段跟踪误差（hold/attach 每步误差）
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from DPG_MPC import MPCController
from DPG_track_ball_in_robot import (
    BallInRobotFrameTrajectory,
    KFPredictiveBaseTrajectory,
    LinearBaseTrajectory,
)


BALL_WORLD = np.array([0.25, 0.5, 1.2], dtype=float)


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


def _build_controller(traj: BallInRobotFrameTrajectory) -> MPCController:
    return MPCController(
        trajectory=traj,
        warm_start_max=0.0,
        pos_weight=12.0,
        rot_weight=0.3,
        use_terminal_value=False,
        terminal_value_dim=3,
        terminal_approach_dir=(0.0, 1.0, 0.0),
        terminal_approach_axis="x",
        enable_funnel_constraint=False,
        funnel_depth=0.10,
        funnel_half_width=0.05,
        funnel_margin=1e-3,
        visualize_funnel_zone=False,
        use_pregrasp=False,
        use_predictive_phase_switch=False,
        use_offset_tracking=True,
        offset_y=-0.13,
        offset_trigger_tol=0.03,
        offset_trigger_steps=6,
        use_uncertainty_aware_weighting=True,
        uncertainty_beta=100.0,
        uncertainty_min_scale=0.65,
        uncertainty_ema=0.2,
        use_manipulability_guidance=True,
        manipulability_lambda=0.06,
        manipulability_w_threshold=0.08,
        manipulability_fd_delta=0.004,
        manipulability_grad_clip=2.0,
        manipulability_horizon_decay=0.8,
        manipulability_first_step_only=False,
        enable_grasp=True,
        grasp_tol=0.02,
        grasp_hold_steps=10,
        grasp_hold_time_s=None,
        grasp_action="none",
        profile_period=1e9,  # 批量实验不输出实时 hold/attach 日志
    )


def _safe_mean(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_p95(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), 95.0))


def run_batch(episodes: int, seed: int, y_noise_std: float, out_dir: Path, extra_time: float) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_episode_path = out_dir / "per_episode.csv"
    per_step_path = out_dir / "per_step.csv"
    summary_path = out_dir / "summary.csv"
    config_path = out_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "episodes": episodes,
                "seed": seed,
                "ball_world": BALL_WORLD.tolist(),
                "random_ranges": {
                    "x_start": [-1.1, -0.5],
                    "x_end": [0.2, 0.8],
                    "speed": [0.1, 0.5],
                    "y_noise_mean": [0.005, 0.02],
                    "y_noise_std_fixed": y_noise_std,
                },
                "forbidden_zone": {
                    "x": "(-inf, x_t-0.05] U [x_t+0.05, +inf)",
                    "y": "[y_t-0.10, y_t]",
                    "z": "all",
                    "rule": "any arm joint body or end_finger in zone => in_funnel=True",
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    ep_fields = [
        "episode",
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
        "target_err_mean",
        "target_err_min",
        "target_err_p95",
        "hold_steps",
        "hold_err_mean",
        "hold_err_p95",
        "attach_steps",
        "attach_err_mean",
        "attach_err_p95",
    ]
    step_fields = [
        "episode",
        "step",
        "time_s",
        "phase",
        "target_err",
        "hold_err",
        "attach_err",
        "in_funnel",
        "grasped",
    ]

    success_count = 0
    global_steps = 0
    global_zone_steps = 0
    global_zone_time_s = 0.0
    global_hold_err_sum = 0.0
    global_hold_steps = 0
    global_attach_err_sum = 0.0
    global_attach_steps = 0

    with open(per_episode_path, "w", newline="", encoding="utf-8") as ep_f, open(
        per_step_path, "w", newline="", encoding="utf-8"
    ) as step_f:
        ep_writer = csv.DictWriter(ep_f, fieldnames=ep_fields)
        step_writer = csv.DictWriter(step_f, fieldnames=step_fields)
        ep_writer.writeheader()
        step_writer.writeheader()

        for ep in range(1, episodes + 1):
            params = _sample_episode_params(rng, y_noise_std=y_noise_std)
            traj, base = _make_random_trajectory(params)
            ctrl = _build_controller(traj)
            base_duration = float(base.duration())
            max_time = float(base_duration + extra_time)

            target_err_list: list[float] = []
            hold_err_list: list[float] = []
            attach_err_list: list[float] = []
            zone_steps = 0
            step_count = 0

            def on_step(stats: Dict):
                nonlocal zone_steps, step_count
                step_count += 1
                in_funnel = bool(stats["in_funnel"])
                phase = str(stats["phase"])
                target_err = float(stats["target_err"])
                hold_err = float(stats["hold_err"])
                attach_err = float(stats["attach_err"])

                target_err_list.append(target_err)
                if phase == "hold":
                    hold_err_list.append(hold_err)
                else:
                    attach_err_list.append(attach_err)
                if in_funnel:
                    zone_steps += 1

                step_writer.writerow(
                    {
                        "episode": ep,
                        "step": int(stats["step"]),
                        "time_s": float(stats["time"]),
                        "phase": phase,
                        "target_err": target_err,
                        "hold_err": hold_err if phase == "hold" else "",
                        "attach_err": attach_err if phase == "attach" else "",
                        "in_funnel": int(in_funnel),
                        "grasped": int(bool(stats["grasped"])),
                    }
                )

            run_info = ctrl.run_headless(
                max_time=max_time,
                step_callback=on_step,
                realtime_sync=False,
            )

            success = int(bool(run_info["grasped"]))
            zone_time_s = float(zone_steps * ctrl.dt)
            zone_ratio = float(zone_steps / max(step_count, 1))

            success_count += success
            global_steps += step_count
            global_zone_steps += zone_steps
            global_zone_time_s += zone_time_s
            global_hold_steps += len(hold_err_list)
            global_attach_steps += len(attach_err_list)
            global_hold_err_sum += float(np.sum(hold_err_list)) if len(hold_err_list) > 0 else 0.0
            global_attach_err_sum += (
                float(np.sum(attach_err_list)) if len(attach_err_list) > 0 else 0.0
            )

            ep_writer.writerow(
                {
                    "episode": ep,
                    "x_start": params.x_start,
                    "x_end": params.x_end,
                    "speed": params.speed,
                    "y_noise_mean": params.y_noise_mean,
                    "y_noise_std": params.y_noise_std,
                    "noise_seed": params.noise_seed,
                    "kf_seed": params.kf_seed,
                    "base_duration_s": base_duration,
                    "max_time_s": max_time,
                    "steps_total": step_count,
                    "sim_time_s": float(run_info["sim_time"]),
                    "success": success,
                    "zone_steps": zone_steps,
                    "zone_time_s": zone_time_s,
                    "zone_ratio": zone_ratio,
                    "target_err_mean": _safe_mean(target_err_list),
                    "target_err_min": float(np.min(target_err_list)) if len(target_err_list) > 0 else np.nan,
                    "target_err_p95": _safe_p95(target_err_list),
                    "hold_steps": len(hold_err_list),
                    "hold_err_mean": _safe_mean(hold_err_list),
                    "hold_err_p95": _safe_p95(hold_err_list),
                    "attach_steps": len(attach_err_list),
                    "attach_err_mean": _safe_mean(attach_err_list),
                    "attach_err_p95": _safe_p95(attach_err_list),
                }
            )
            ep_f.flush()
            step_f.flush()
            print(
                f"[episode {ep:03d}/{episodes}] success={success}, "
                f"zone_time={zone_time_s:.3f}s, steps={step_count}, "
                f"mean_target_err={_safe_mean(target_err_list):.4f}m"
            )

    summary_row = {
        "episodes": episodes,
        "success_count": success_count,
        "success_rate": float(success_count / max(episodes, 1)),
        "total_steps": global_steps,
        "zone_steps_total": global_zone_steps,
        "zone_time_total_s": float(global_zone_time_s),
        "zone_time_mean_s_per_episode": float(global_zone_time_s / max(episodes, 1)),
        "zone_step_ratio": float(global_zone_steps / max(global_steps, 1)),
        "hold_steps_total": global_hold_steps,
        "hold_err_mean_global": float(global_hold_err_sum / max(global_hold_steps, 1)),
        "attach_steps_total": global_attach_steps,
        "attach_err_mean_global": float(global_attach_err_sum / max(global_attach_steps, 1)),
    }

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    print("\n[done] batch experiment finished.")
    print(f"  per-episode: {per_episode_path}")
    print(f"  per-step   : {per_step_path}")
    print(f"  summary    : {summary_path}")
    print(f"  config     : {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Headless random batch experiments for DPG_mujoco_final")
    parser.add_argument("--episodes", type=int, default=200, help="number of episodes")
    parser.add_argument("--seed", type=int, default=20260306, help="global RNG seed")
    parser.add_argument(
        "--y-noise-std",
        type=float,
        default=0.005,
        help="fixed lateral noise std (mean is randomized in [0.005, 0.02])",
    )
    parser.add_argument(
        "--extra-time",
        type=float,
        default=6.0,
        help="extra seconds added to base trajectory duration for each episode",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="output directory (default: DPG_mujoco_final/experiment_results/<timestamp>)",
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
        out_dir = Path(__file__).resolve().parent / "experiment_results" / f"batch_{stamp}"
    else:
        out_dir = Path(args.out_dir).resolve()

    run_batch(
        episodes=int(args.episodes),
        seed=int(args.seed),
        y_noise_std=float(args.y_noise_std),
        out_dir=out_dir,
        extra_time=float(args.extra_time),
    )


if __name__ == "__main__":
    main()

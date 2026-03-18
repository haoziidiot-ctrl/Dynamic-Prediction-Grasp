"""
DPG_ablation_uncertainty.py

消融实验1：自适应权重（uncertainty-aware weighting）对高频扰动抑制效果

设计：
- 同一随机回合参数（轨迹起终点、速度、扰动均值、噪声种子、KF种子）下，
  分别运行：
    1) uncertainty_on
    2) uncertainty_off
- 默认 200 回合（共 400 次仿真），保证公平对照。
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
    uncertainty_on: bool,
) -> MPCController:
    return MPCController(
        trajectory=traj,
        warm_start_max=0.0,
        pos_weight=12.0,
        rot_weight=0.3,
        use_terminal_value=False,
        terminal_value_dim=3,
        terminal_approach_dir=(0.0, 1.0, 0.0),
        terminal_approach_axis="x",
        use_pregrasp=False,
        use_predictive_phase_switch=False,
        use_offset_tracking=True,
        offset_y=-0.13,
        offset_trigger_tol=0.03,
        offset_trigger_steps=6,
        use_uncertainty_aware_weighting=bool(uncertainty_on),
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
        grasp_action="stop",
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


def _run_one_mode(
    params: EpisodeRandomParams,
    *,
    uncertainty_on: bool,
    extra_time: float,
) -> Tuple[Dict, List[Dict]]:
    traj, base = _make_random_trajectory(params)
    ctrl = _build_controller(traj, uncertainty_on=uncertainty_on)
    base_duration = float(base.duration())
    max_time = float(base_duration + extra_time)

    target_err_list: List[float] = []
    hold_err_list: List[float] = []
    attach_err_list: List[float] = []
    step_count = 0
    step_rows: List[Dict] = []

    def on_step(stats: Dict):
        nonlocal step_count
        step_count += 1
        phase = str(stats["phase"])
        t = float(stats["time"])
        step_i = int(stats["step"])
        target_err = float(stats["target_err"])
        hold_err = float(stats["hold_err"])
        attach_err = float(stats["attach_err"])

        target_err_list.append(target_err)
        if phase == "hold":
            hold_err_list.append(hold_err)
        else:
            attach_err_list.append(attach_err)

        step_rows.append(
            {
                "step": step_i,
                "time_s": t,
                "phase": phase,
                "target_err": target_err,
                "hold_err": hold_err if phase == "hold" else "",
                "attach_err": attach_err if phase == "attach" else "",
            }
        )

    run_info = ctrl.run_headless(
        max_time=max_time,
        step_callback=on_step,
        realtime_sync=False,
    )

    summary = {
        "uncertainty_on": int(bool(uncertainty_on)),
        "base_duration_s": base_duration,
        "max_time_s": max_time,
        "steps_total": int(step_count),
        "sim_time_s": float(run_info["sim_time"]),
        "target_err_mean": _safe_mean(target_err_list),
        "target_err_min": float(np.min(target_err_list)) if len(target_err_list) > 0 else np.nan,
        "target_err_p95": _safe_p95(target_err_list),
        "hold_steps": int(len(hold_err_list)),
        "hold_err_mean": _safe_mean(hold_err_list),
        "hold_err_p95": _safe_p95(hold_err_list),
        "attach_steps": int(len(attach_err_list)),
        "attach_err_mean": _safe_mean(attach_err_list),
        "attach_err_p95": _safe_p95(attach_err_list),
    }
    return summary, step_rows


def _nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _summary_for_mode(rows: List[Dict], uncertainty_on: int) -> Dict:
    sub = [r for r in rows if int(r["uncertainty_on"]) == int(uncertainty_on)]
    n = len(sub)
    return {
        "uncertainty_on": int(uncertainty_on),
        "episodes": n,
        "target_err_mean": _nanmean([r["target_err_mean"] for r in sub]),
        "target_err_p95_mean": _nanmean([r["target_err_p95"] for r in sub]),
        "hold_err_mean": _nanmean([r["hold_err_mean"] for r in sub]),
        "attach_err_mean": _nanmean([r["attach_err_mean"] for r in sub]),
    }


def run_ablation(episodes: int, seed: int, y_noise_std: float, out_dir: Path, extra_time: float) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    per_episode_on_path = out_dir / "per_episode_on.csv"
    per_episode_off_path = out_dir / "per_episode_off.csv"
    per_step_on_path = out_dir / "per_step_on.csv"
    per_step_off_path = out_dir / "per_step_off.csv"
    per_delta_path = out_dir / "per_episode_delta_on_minus_off.csv"
    summary_on_path = out_dir / "summary_on.csv"
    summary_off_path = out_dir / "summary_off.csv"
    summary_delta_path = out_dir / "summary_delta.csv"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "episodes": episodes,
                "seed": seed,
                "ball_world": BALL_WORLD.tolist(),
                "ablation": "uncertainty_aware_weighting on/off",
                "random_ranges": {
                    "x_start": [-1.1, -0.5],
                    "x_end": [0.2, 0.8],
                    "speed": [0.1, 0.5],
                    "y_noise_mean": [0.005, 0.02],
                    "y_noise_std_fixed": y_noise_std,
                },
                "fairness": "paired same-random-params per episode",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    step_fields = ["episode", "step", "time_s", "phase", "target_err", "hold_err", "attach_err"]
    random_params = [_sample_episode_params(rng, y_noise_std=y_noise_std) for _ in range(episodes)]
    all_rows: List[Dict] = []
    delta_rows: List[Dict] = []
    with open(per_step_on_path, "w", newline="", encoding="utf-8") as step_on_f, open(
        per_step_off_path, "w", newline="", encoding="utf-8"
    ) as step_off_f:
        step_on_writer = csv.DictWriter(step_on_f, fieldnames=step_fields)
        step_off_writer = csv.DictWriter(step_off_f, fieldnames=step_fields)
        step_on_writer.writeheader()
        step_off_writer.writeheader()
        for ep_idx, params in enumerate(random_params, start=1):
            row_on, step_rows_on = _run_one_mode(params, uncertainty_on=True, extra_time=extra_time)
            row_off, step_rows_off = _run_one_mode(params, uncertainty_on=False, extra_time=extra_time)

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
            for s in step_rows_on:
                step_on_writer.writerow({"episode": ep_idx, **s})
            for s in step_rows_off:
                step_off_writer.writerow({"episode": ep_idx, **s})

            delta = {
                "episode": ep_idx,
                "target_err_mean_delta_on_minus_off": row_on["target_err_mean"] - row_off["target_err_mean"],
                "attach_err_mean_delta_on_minus_off": row_on["attach_err_mean"] - row_off["attach_err_mean"],
                "win_attach_err_on": int(row_on["attach_err_mean"] < row_off["attach_err_mean"]),
            }
            delta_rows.append(delta)

            print(
                f"[episode {ep_idx:03d}/{episodes}] "
                f"ON(target_err={row_on['target_err_mean']:.4f}, attach_err={row_on['attach_err_mean']:.4f}) | "
                f"OFF(target_err={row_off['target_err_mean']:.4f}, attach_err={row_off['attach_err_mean']:.4f})"
            )

    episode_fields = [
        "episode",
        "target_err_mean",
        "target_err_min",
        "target_err_p95",
        "hold_err_mean",
        "hold_err_p95",
        "attach_err_mean",
        "attach_err_p95",
    ]
    rows_on = [r for r in all_rows if int(r["uncertainty_on"]) == 1]
    rows_off = [r for r in all_rows if int(r["uncertainty_on"]) == 0]
    with open(per_episode_on_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=episode_fields)
        writer.writeheader()
        for r in rows_on:
            out = {k: r[k] for k in episode_fields}
            writer.writerow(out)
    with open(per_episode_off_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=episode_fields)
        writer.writeheader()
        for r in rows_off:
            out = {k: r[k] for k in episode_fields}
            writer.writerow(out)

    delta_fields = list(delta_rows[0].keys()) if len(delta_rows) > 0 else ["episode"]
    with open(per_delta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=delta_fields)
        writer.writeheader()
        for r in delta_rows:
            writer.writerow(r)

    summary_on = _summary_for_mode(all_rows, uncertainty_on=1)
    summary_off = _summary_for_mode(all_rows, uncertainty_on=0)
    summary_on_out = {k: v for k, v in summary_on.items() if k != "uncertainty_on"}
    summary_off_out = {k: v for k, v in summary_off.items() if k != "uncertainty_on"}
    with open(summary_on_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_on_out.keys()))
        writer.writeheader()
        writer.writerow(summary_on_out)
    with open(summary_off_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_off_out.keys()))
        writer.writeheader()
        writer.writerow(summary_off_out)

    def _mean_delta(key: str) -> float:
        vals = [float(r[key]) for r in delta_rows]
        return _nanmean(vals)

    summary_delta = {
        "episodes": episodes,
        "target_err_mean_delta_on_minus_off_mean": _mean_delta("target_err_mean_delta_on_minus_off"),
        "attach_err_mean_delta_on_minus_off_mean": _mean_delta("attach_err_mean_delta_on_minus_off"),
        "win_attach_err_on_rate": _nanmean([float(r["win_attach_err_on"]) for r in delta_rows]),
    }
    with open(summary_delta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_delta.keys()))
        writer.writeheader()
        writer.writerow(summary_delta)

    print("\n[done] uncertainty ablation finished.")
    print(f"  per-ep-on  : {per_episode_on_path}")
    print(f"  per-ep-off : {per_episode_off_path}")
    print(f"  step-on    : {per_step_on_path}")
    print(f"  step-off   : {per_step_off_path}")
    print(f"  per-delta  : {per_delta_path}")
    print(f"  summary-on : {summary_on_path}")
    print(f"  summary-off: {summary_off_path}")
    print(f"  delta-sum  : {summary_delta_path}")
    print(f"  config     : {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation: uncertainty-aware weighting ON vs OFF under paired random episodes"
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
        default=4.0,
        help="extra simulation time added to each episode after base duration",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="output directory (default: DPG_mujoco_final/experiment_results/ablation_uncertainty_<timestamp>)",
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
        out_dir = (
            Path(__file__).resolve().parent
            / "experiment_results"
            / f"ablation_uncertainty_{stamp}"
        )
    else:
        out_dir = Path(args.out_dir).resolve()

    run_ablation(
        episodes=int(args.episodes),
        seed=int(args.seed),
        y_noise_std=float(args.y_noise_std),
        out_dir=out_dir,
        extra_time=float(args.extra_time),
    )


if __name__ == "__main__":
    main()

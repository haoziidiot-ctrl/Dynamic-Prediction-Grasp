from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from DPG_MPC import MPCController
from DPG_track_ball_in_robot import get_ball_in_robot_trajectory


@dataclass(frozen=True)
class Candidate:
    pos_weight: float
    rot_weight: float
    offset_y: float
    offset_tol: float
    offset_steps: int
    grasp_tol: float
    grasp_steps: int


@dataclass
class Result:
    candidate: Candidate
    success: bool
    dynamic_success: bool
    attach_time: float
    grasp_time: float
    min_hold_err: float
    min_target_err: float
    mean_attach_target_err: float
    ori_dot_y_at_grasp: float
    score: float


def build_controller(c: Candidate) -> MPCController:
    traj = get_ball_in_robot_trajectory(
        kf_cfg={
            "poll_period_s": 0.01,
            "sigma_a": 0.08,
            "meas_noise": 0.005,
            "use_measurement_noise": True,
            "seed": 0,
        }
    )
    ctrl = MPCController(
        model_xml="../fetch_freight_mujoco/xml/scene.xml",
        trajectory=traj,
        warm_start_max=0.0,
        pos_weight=c.pos_weight,
        rot_weight=c.rot_weight,
        use_terminal_value=False,
        terminal_value_dim=3,
        terminal_approach_dir=(0.0, 1.0, 0.0),
        terminal_approach_axis="x",
        use_pregrasp=False,
        use_predictive_phase_switch=False,
        use_offset_tracking=True,
        offset_y=c.offset_y,
        offset_trigger_tol=c.offset_tol,
        offset_trigger_steps=c.offset_steps,
        enable_grasp=True,
        grasp_tol=c.grasp_tol,
        grasp_hold_steps=c.grasp_steps,
        grasp_hold_time_s=None,
        grasp_action="none",
    )
    if ctrl.is_ball_rel and ctrl.base_traj_for_rel is not None and ctrl.chassis_mocap_id >= 0:
        ctrl._sync_base_traj(0.0)
        base0 = ctrl.base_traj_for_rel.position(0.0)
        ctrl.data.mocap_pos[ctrl.chassis_mocap_id] = base0
        ctrl.data.mocap_quat[ctrl.chassis_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(ctrl.model, ctrl.data)
    return ctrl


def run_once(
    c: Candidate,
    max_time: float = 6.2,
    sample_dt: float = 0.02,
) -> tuple[Result, list[dict[str, float | str]]]:
    ctrl = build_controller(c)
    # No warm-start rollout; just align time offset.
    _ = ctrl._warm_start_to_pose(ctrl._world_target(0.0), max_duration=0.0, tol=0.08, viewer=None)

    q_des = ctrl.data.qpos[ctrl.arm_qpos_indices].copy()
    step_count = 0
    rows: list[dict[str, float | str]] = []
    last_sample_t = -1e9

    attach_time = np.inf
    grasp_time = np.inf
    min_hold_err = np.inf
    min_target_err = np.inf
    attach_errs: list[float] = []
    ori_dot_y_at_grasp = np.nan
    prev_phase = "hold"

    while True:
        sim_time = ctrl.data.time - ctrl.time_offset
        if sim_time >= max_time:
            break

        if ctrl.is_ball_rel and ctrl.base_traj_for_rel is not None and ctrl.chassis_mocap_id >= 0:
            base_traj_time = sim_time
            ctrl._sync_base_traj(base_traj_time)
            base_pos = ctrl.base_traj_for_rel.position(base_traj_time)
            ctrl.data.mocap_pos[ctrl.chassis_mocap_id] = base_pos
            ctrl.data.mocap_quat[ctrl.chassis_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
            mujoco.mj_forward(ctrl.model, ctrl.data)

        target_world_ref = ctrl._world_target(sim_time)
        current_pos_world = ctrl.data.site_xpos[ctrl.ee_site_id].copy()
        target_world = current_pos_world if ctrl._attach_target else target_world_ref
        ctrl.data.mocap_pos[ctrl.target_mocap_id] = target_world

        future_world = ctrl._control_future_targets_world(
            sim_time=sim_time,
            abs_time=ctrl.data.time,
            current_pos_world=current_pos_world,
        )

        if step_count % ctrl.control_every == 0:
            if ctrl.use_offset_tracking and ctrl.offset_active:
                offset_target = future_world[0].copy()
                offset_dist = float(np.linalg.norm(current_pos_world - offset_target))
                ctrl._offset_last_dist = offset_dist
                if offset_dist <= ctrl.offset_trigger_tol:
                    ctrl._offset_hit_count += 1
                else:
                    ctrl._offset_hit_count = 0
                if ctrl._offset_hit_count >= ctrl.offset_trigger_steps:
                    ctrl.offset_active = False
                    ctrl._offset_hit_count = 0
                    future_world = ctrl._control_future_targets_world(
                        sim_time=sim_time,
                        abs_time=ctrl.data.time,
                        current_pos_world=current_pos_world,
                    )

            e_hat = ctrl._task_error(
                current_pos_world.copy(),
                future_world,
                abs_time=float(ctrl.data.time),
            )
            terminal_time = ctrl.data.time + ctrl.preview_lead + ctrl.mpc.horizon * ctrl.control_dt
            current_state, desired_state = ctrl._terminal_state(
                current_pos_world.copy(),
                future_world[-1],
                terminal_time,
            )
            twist = ctrl.mpc.solve(
                e_hat,
                current_state=current_state,
                desired_terminal=desired_state,
            )
            jac_full = ctrl._task_jacobian()
            jac = jac_full if ctrl.use_orientation_task else jac_full[:3]
            jac_pinv = ctrl._jacobian_pinv(jac)
            ball_err = float(np.linalg.norm(future_world[0] - current_pos_world))
            fb_gain = ctrl._feedback_gain(ball_err)
            twist_fb = (twist.copy() if ctrl.use_orientation_task else twist[:3].copy()) * fb_gain
            twist_cmd = twist_fb
            cond_val = np.linalg.cond(jac @ jac.T)
            if cond_val > ctrl.cond_threshold:
                scale = np.clip(ctrl.cond_threshold / cond_val, ctrl.min_twist_scale, 1.0)
                twist_cmd *= scale
                twist_fb *= scale
            twist_cmd, twist_fb = ctrl._limit_twist_cmd(twist_cmd, twist_fb)

            qdot_task = jac_pinv @ twist_cmd
            joint_offset = ctrl.q_nominal - ctrl.data.qpos[ctrl.arm_qpos_indices]
            null_w = 1.0 / (1.0 + (ball_err / ctrl.nullspace_err_scale) ** 2)
            vel_ratio_task = np.max(np.abs(qdot_task) / ctrl.velocity_limit)
            task_budget = float(np.clip(1.0 - vel_ratio_task, 0.0, 1.0))
            dyn_gain = (ctrl.nullspace_gain * null_w * task_budget) * (
                1.0 + np.clip(np.linalg.norm(joint_offset) / ctrl.nullspace_scale, 0.0, 3.0)
            )
            nullspace_term = dyn_gain * joint_offset
            qdot_cmd = qdot_task + ((np.identity(ctrl.arm_dof) - jac_pinv @ jac) @ nullspace_term)
            vel_ratio = np.max(np.abs(qdot_cmd) / ctrl.velocity_limit)
            if vel_ratio > 1.0:
                qdot_cmd = qdot_cmd / vel_ratio
            qdot_cmd = ctrl._limit_joint_accel(qdot_cmd, ctrl.last_qdot_cmd, ctrl.control_dt)
            qdot_cmd = np.clip(qdot_cmd, -ctrl.velocity_limit, ctrl.velocity_limit)
            ctrl.last_qdot_cmd = qdot_cmd.copy()
            q_des = q_des + qdot_cmd * ctrl.control_dt + ctrl.drift_gain * joint_offset * ctrl.control_dt
            q_des = np.clip(q_des, ctrl.arm_qpos_min, ctrl.arm_qpos_max)

        ctrl._apply_joint_position(q_des)
        mujoco.mj_step(ctrl.model, ctrl.data)
        step_count += 1

        target_err = float(np.linalg.norm(current_pos_world - target_world_ref))
        hold_err = float(np.linalg.norm(current_pos_world - future_world[0]))
        min_target_err = min(min_target_err, target_err)
        if ctrl.offset_active:
            min_hold_err = min(min_hold_err, hold_err)

        if ctrl.enable_grasp and (not ctrl.grasped):
            if target_err <= ctrl.grasp_tol:
                ctrl._grasp_counter += 1
            else:
                ctrl._grasp_counter = 0
            if ctrl._grasp_counter >= ctrl.grasp_hold_steps:
                ctrl.grasped = True

        phase = "hold" if ctrl.offset_active else ("grasp success" if ctrl.grasped else "attach")
        if phase == "attach":
            attach_errs.append(target_err)
        if prev_phase == "hold" and phase == "attach" and not np.isfinite(attach_time):
            attach_time = float(ctrl.data.time)
        if prev_phase != "grasp success" and phase == "grasp success" and not np.isfinite(grasp_time):
            grasp_time = float(ctrl.data.time)
            ori_dot_y_at_grasp = float(
                np.dot(ctrl._approach_axis_world(), np.array([0.0, 1.0, 0.0], dtype=float))
            )
        prev_phase = phase

        if ctrl.data.time - last_sample_t >= sample_dt:
            rows.append(
                {
                    "time": float(ctrl.data.time),
                    "phase": phase,
                    "hold_err": hold_err,
                    "target_err": target_err,
                    "offset_hit": int(ctrl._offset_hit_count),
                    "ori_dotY": float(
                        np.dot(ctrl._approach_axis_world(), np.array([0.0, 1.0, 0.0], dtype=float))
                    ),
                    "base_x": float(ctrl.data.mocap_pos[ctrl.chassis_mocap_id][0]),
                }
            )
            last_sample_t = ctrl.data.time

    success = np.isfinite(grasp_time)
    # base trajectory dynamic phase duration: (0.50 - (-0.8)) / 0.3 = 4.333...
    dynamic_window_end = 4.3333333333
    dynamic_success = bool(success and grasp_time <= dynamic_window_end)
    mean_attach_target_err = float(np.mean(attach_errs)) if attach_errs else np.inf

    score = 0.0
    if not success:
        score += 1e6
    else:
        score += grasp_time
    if not dynamic_success:
        score += 100.0
    score += 10.0 * max(0.0, mean_attach_target_err - 0.02)
    score += 2.0 * max(0.0, 0.98 - (ori_dot_y_at_grasp if np.isfinite(ori_dot_y_at_grasp) else 0.0))

    result = Result(
        candidate=c,
        success=success,
        dynamic_success=dynamic_success,
        attach_time=float(attach_time) if np.isfinite(attach_time) else np.inf,
        grasp_time=float(grasp_time) if np.isfinite(grasp_time) else np.inf,
        min_hold_err=float(min_hold_err) if np.isfinite(min_hold_err) else np.inf,
        min_target_err=float(min_target_err) if np.isfinite(min_target_err) else np.inf,
        mean_attach_target_err=mean_attach_target_err,
        ori_dot_y_at_grasp=float(ori_dot_y_at_grasp) if np.isfinite(ori_dot_y_at_grasp) else np.nan,
        score=float(score),
    )
    return result, rows


def candidate_grid() -> Iterable[Candidate]:
    for rot_weight in (0.1, 0.15, 0.2, 0.25):
        for offset_tol in (0.018, 0.02, 0.022):
            for offset_steps in (8, 10):
                yield Candidate(
                    pos_weight=12.0,
                    rot_weight=rot_weight,
                    offset_y=-0.13,
                    offset_tol=offset_tol,
                    offset_steps=offset_steps,
                    grasp_tol=0.02,
                    grasp_steps=10,
                )


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[Result] = []
    best_rows: list[dict[str, float | str]] = []
    best: Result | None = None

    for idx, c in enumerate(candidate_grid(), start=1):
        result, rows = run_once(c)
        results.append(result)
        print(
            f"[{idx:02d}] rot={c.rot_weight:.3f}, tol={c.offset_tol:.3f}, steps={c.offset_steps}, "
            f"success={int(result.success)}, dynamic={int(result.dynamic_success)}, "
            f"attach_t={result.attach_time:.3f}, grasp_t={result.grasp_time:.3f}, "
            f"min_hold={result.min_hold_err:.4f}, min_target={result.min_target_err:.4f}, "
            f"oriY={result.ori_dot_y_at_grasp:.4f}, score={result.score:.3f}"
        )
        if best is None or result.score < best.score:
            best = result
            best_rows = rows

    if best is None:
        raise RuntimeError("No tuning result produced.")

    summary_path = out_dir / "tune_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pos_weight",
                "rot_weight",
                "offset_y",
                "offset_tol",
                "offset_steps",
                "grasp_tol",
                "grasp_steps",
                "success",
                "dynamic_success",
                "attach_time",
                "grasp_time",
                "min_hold_err",
                "min_target_err",
                "mean_attach_target_err",
                "ori_dot_y_at_grasp",
                "score",
            ]
        )
        for r in results:
            c = r.candidate
            writer.writerow(
                [
                    c.pos_weight,
                    c.rot_weight,
                    c.offset_y,
                    c.offset_tol,
                    c.offset_steps,
                    c.grasp_tol,
                    c.grasp_steps,
                    int(r.success),
                    int(r.dynamic_success),
                    r.attach_time,
                    r.grasp_time,
                    r.min_hold_err,
                    r.min_target_err,
                    r.mean_attach_target_err,
                    r.ori_dot_y_at_grasp,
                    r.score,
                ]
            )

    best_path = out_dir / "best_run.csv"
    if best_rows:
        with best_path.open("w", newline="") as f:
            fieldnames = list(best_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(best_rows)

    best_cfg_path = out_dir / "best_config.txt"
    with best_cfg_path.open("w") as f:
        c = best.candidate
        f.write(
            "\n".join(
                [
                    f"POS_WEIGHT={c.pos_weight}",
                    f"ROT_WEIGHT={c.rot_weight}",
                    f"OFFSET_Y={c.offset_y}",
                    f"OFFSET_SWITCH_TOL={c.offset_tol}",
                    f"OFFSET_SWITCH_STEPS={c.offset_steps}",
                    f"GRASP_TOL={c.grasp_tol}",
                    f"GRASP_HOLD_STEPS={c.grasp_steps}",
                    f"attach_time={best.attach_time}",
                    f"grasp_time={best.grasp_time}",
                    f"dynamic_success={int(best.dynamic_success)}",
                    f"min_hold_err={best.min_hold_err}",
                    f"min_target_err={best.min_target_err}",
                    f"ori_dot_y_at_grasp={best.ori_dot_y_at_grasp}",
                    f"score={best.score}",
                ]
            )
        )

    c = best.candidate
    print(
        "\nBest config: "
        f"rot={c.rot_weight:.3f}, tol={c.offset_tol:.3f}, steps={c.offset_steps}, "
        f"attach_t={best.attach_time:.3f}, grasp_t={best.grasp_time:.3f}, "
        f"dynamic={int(best.dynamic_success)}, min_target={best.min_target_err:.4f}, "
        f"oriY={best.ori_dot_y_at_grasp:.4f}"
    )
    print(f"Saved: {summary_path}")
    print(f"Saved: {best_path}")
    print(f"Saved: {best_cfg_path}")


if __name__ == "__main__":
    main()

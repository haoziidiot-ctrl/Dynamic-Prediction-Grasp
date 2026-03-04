"""
train_value.py

使用现有 MPC 作为行为策略，采样轨迹并训练终端价值函数 V(e)。

核心定义:
    - 终端状态 e: 默认 6D = [位置误差(3), 姿态对齐误差(3)]
    - 网络输出 V(e): cost-to-go（累计代价，越小越好）
    - 部署时 MPC 在每步对 V 做二次近似并写入 QP 终端项

支持两种训练方式:
    - mc: Monte-Carlo 回归目标 C_t = -sum(gamma^k * r_{t+k})
    - td0: Bellman 目标 C_t = c_t + gamma * V_target(e_{t+1})

示例:
    python DPG_RL_mujoco/train_value.py --method td0 --episodes 2000 --epochs 50 --save value_net.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from datetime import datetime
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import mujoco
import numpy as np
import torch
from torch import nn

from DPG_MPC import MPCController
from DPG_track_ball_in_robot import (
    LinearBaseTrajectory,
    get_ball_in_robot_trajectory,
)
from rl_value import ValueNet

try:
    import wandb
except Exception:  # pragma: no cover - wandb 可选依赖
    wandb = None


class JitteredLineTrajectory(LinearBaseTrajectory):
    """直线底盘轨迹 + 每步 y 方向扰动。"""

    def __init__(
        self,
        start,
        end,
        speed,
        y_jitter_amp=0.01,
        jitter_dt=0.01,
        seed=None,
    ):
        super().__init__(start=start, end=end, speed=speed)
        self.y_jitter_amp = float(y_jitter_amp)
        self.jitter_dt = float(jitter_dt)
        self._rng = np.random.default_rng(seed)
        self._jitter_cache: Dict[int, float] = {}

    def base_position(self, sim_time: float) -> np.ndarray:
        return super().position(sim_time)

    def _jitter_at(self, sim_time: float) -> float:
        if self.y_jitter_amp <= 0.0 or self.jitter_dt <= 0.0:
            return 0.0
        step = int(np.floor(sim_time / self.jitter_dt + 1e-9))
        if step not in self._jitter_cache:
            self._jitter_cache[step] = float(
                self._rng.uniform(-self.y_jitter_amp, self.y_jitter_amp)
            )
        return self._jitter_cache[step]

    def position(self, sim_time: float) -> np.ndarray:
        base_pos = self.base_position(sim_time)
        base_pos[1] += self._jitter_at(sim_time)
        return base_pos


@dataclass
class RewardConfig:
    dist_weight: float = 1.0
    rot_weight: float = 0.2
    sing_weight: float = 0.1
    energy_weight: float = 1e-3
    success_tol: float = 0.02
    success_bonus: float = 1.0


@dataclass
class TrainConfig:
    method: str = "td0"  # "mc" or "td0"
    gamma: float = 0.98
    batch_size: int = 256
    epochs: int = 40
    lr: float = 1e-3
    td_target_tau: float = 0.02
    log_interval: int = 1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_proxy_env_for_wandb():
    """统一代理环境变量，避免大小写冲突导致 wandb 连接到错误代理。"""
    pairs = [
        ("HTTP_PROXY", "http_proxy"),
        ("HTTPS_PROXY", "https_proxy"),
        ("ALL_PROXY", "all_proxy"),
        ("NO_PROXY", "no_proxy"),
    ]
    for upper, lower in pairs:
        up_val = os.environ.get(upper, "").strip()
        low_val = os.environ.get(lower, "").strip()
        chosen = up_val or low_val
        if up_val and low_val and up_val != low_val:
            # 如果存在冲突，优先使用本机代理（127.0.0.1 / localhost），否则使用大写变量。
            up_local = ("127.0.0.1" in up_val) or ("localhost" in up_val)
            low_local = ("127.0.0.1" in low_val) or ("localhost" in low_val)
            if up_local and (not low_local):
                chosen = up_val
            elif low_local and (not up_local):
                chosen = low_val
            else:
                chosen = up_val
            print(f"[wandb] proxy conflict for {upper}/{lower}, using: {chosen}")
        if chosen:
            os.environ[upper] = chosen
            os.environ[lower] = chosen


def write_csv(path: str, rows: List[Dict[str, float]]):
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_loss_plot(csv_path: str, png_path: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    epochs = []
    losses = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
    if len(epochs) == 0:
        return False

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Value Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()
    return True


def compute_rewards(logs, cfg: RewardConfig) -> List[float]:
    rewards = []
    success = False
    for item in logs:
        e = np.asarray(item["terminal_error"], dtype=float)
        dist = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:])) if e.shape[0] > 3 else 0.0
        sing_err = float(abs(e[6])) if e.shape[0] > 6 else 0.0
        qdot = item["qdot"]
        r = (
            -cfg.dist_weight * dist
            - cfg.rot_weight * rot_err
            - cfg.sing_weight * sing_err
            - cfg.energy_weight * float(np.linalg.norm(qdot))
        )
        if (not success) and dist <= cfg.success_tol:
            r += cfg.success_bonus
            success = True
        rewards.append(float(r))
    return rewards


def discounted_returns(rewards: List[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=float)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns


def base_trajectory_with_y_jitter(
    y_range=(-0.2, 0.2),
    y_jitter_amp=0.01,
    jitter_dt=0.01,
    x_start_range=(-1.0, -0.3),
    x_end_range=(0.2, 0.8),
    speed_range=(0.15, 0.45),
):
    """直线轨迹，start/end 同 y；每步附加随机 y 扰动。"""
    y_offset = random.uniform(*y_range)
    x0 = random.uniform(*x_start_range)
    x1 = random.uniform(*x_end_range)
    speed = random.uniform(*speed_range)
    return JitteredLineTrajectory(
        start=[x0, y_offset, 0.0],
        end=[x1, y_offset, 0.0],
        speed=speed,
        y_jitter_amp=y_jitter_amp,
        jitter_dt=jitter_dt,
    )


def _min_distance_to_path(ball_world, base_traj, shoulder_offset, steps=80):
    duration = float(base_traj.duration())
    min_dist = 1e9
    for i in range(steps + 1):
        t = duration * i / steps
        base_pos = (
            base_traj.base_position(t)
            if hasattr(base_traj, "base_position")
            else base_traj.position(t)
        )
        mount_pos = base_pos + shoulder_offset
        dist = float(np.linalg.norm(ball_world - mount_pos))
        if dist < min_dist:
            min_dist = dist
    return min_dist


def sample_ball_world_on_line(
    base_traj,
    shoulder_offset,
    reach_max,
    attempts=50,
    y_range=(0.35, 0.65),
    z_range=(1.0, 1.4),
    x_margin=0.05,
    reach_margin=0.95,
):
    """在底盘轨迹附近采样一个静止目标，尽量保证可达。"""
    start = float(base_traj.start[0])
    end = float(base_traj.end[0])
    x_min, x_max = (start, end) if start <= end else (end, start)
    x_min -= x_margin
    x_max += x_margin
    for _ in range(attempts):
        x = random.uniform(x_min, x_max)
        y = random.uniform(*y_range)
        z = random.uniform(*z_range)
        ball = np.array([x, y, z], dtype=float)
        min_dist = _min_distance_to_path(ball, base_traj, shoulder_offset)
        if min_dist <= reach_max * reach_margin:
            return ball
    return np.array([0.25, 0.5, 1.2], dtype=float)


def rollout_episode(
    max_time: float,
    profile_period: float,
    reward_cfg: RewardConfig,
    gamma: float,
    reach_params: Tuple[np.ndarray, float],
    terminal_value_dim: int,
    terminal_rot_scale: float,
    terminal_sing_scale: float,
    kf_poll_period: float,
    kf_sigma_a: float,
    kf_meas_noise: float,
    kf_use_noise: bool,
    kf_seed: int,
) -> Dict[str, np.ndarray]:
    base_traj_truth = base_trajectory_with_y_jitter()
    shoulder_offset, reach_max = reach_params
    ball_world = sample_ball_world_on_line(base_traj_truth, shoulder_offset, reach_max)
    traj = get_ball_in_robot_trajectory(
        base_traj=base_traj_truth,
        ball_world=ball_world,
        kf_cfg={
            "poll_period_s": float(kf_poll_period),
            "sigma_a": float(kf_sigma_a),
            "meas_noise": float(kf_meas_noise),
            "use_measurement_noise": bool(kf_use_noise),
            "seed": int(kf_seed),
        },
    )

    controller = MPCController(
        trajectory=traj,
        profile_period=profile_period,
        use_terminal_value=False,
        enable_grasp=False,
        terminal_value_dim=int(terminal_value_dim),
        terminal_rot_scale=float(terminal_rot_scale),
        terminal_sing_scale=float(terminal_sing_scale),
    )

    if controller.is_ball_rel and controller.base_traj_for_rel is not None and controller.chassis_mocap_id >= 0:
        base0 = controller.base_traj_for_rel.position(0.0)
        controller.data.mocap_pos[controller.chassis_mocap_id] = base0
        controller.data.mocap_quat[controller.chassis_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(controller.model, controller.data)

    start_world = controller._world_target(0.0)
    controller._warm_start_to_pose(
        start_world,
        max_duration=controller.warm_start_max,
        tol=controller.warm_start_tol,
        viewer=None,
    )

    logs = []

    def callback(item):
        logs.append(item)

    controller._control_loop(viewer=None, max_time=max_time, callback=callback)
    if len(logs) < 2:
        return {
            "states": np.zeros((0, terminal_value_dim), dtype=float),
            "next_states": np.zeros((0, terminal_value_dim), dtype=float),
            "rewards": np.zeros((0,), dtype=float),
            "returns": np.zeros((0,), dtype=float),
            "dones": np.zeros((0,), dtype=float),
            "metrics": {
                "steps": 0,
                "reward_sum": 0.0,
                "cost_sum": 0.0,
                "cost_to_go0": 0.0,
                "min_dist": 0.0,
                "final_dist": 0.0,
                "min_rot": 0.0,
                "final_rot": 0.0,
                "mean_sing": 0.0,
                "success": 0.0,
            },
        }

    rewards = np.asarray(compute_rewards(logs, reward_cfg), dtype=float)
    returns = discounted_returns(rewards.tolist(), gamma=float(gamma))
    states = np.asarray([item["terminal_error"] for item in logs], dtype=float)
    next_states = np.vstack([states[1:], states[-1:]])
    dones = np.zeros((states.shape[0],), dtype=float)
    dones[-1] = 1.0
    pos_dist = np.linalg.norm(states[:, :3], axis=1)
    if states.shape[1] >= 6:
        rot_dist = np.linalg.norm(states[:, 3:6], axis=1)
    else:
        rot_dist = np.zeros_like(pos_dist)
    if states.shape[1] >= 7:
        sing_abs = np.abs(states[:, 6])
    else:
        sing_abs = np.zeros_like(pos_dist)
    metrics = {
        "steps": int(states.shape[0]),
        "reward_sum": float(np.sum(rewards)),
        "cost_sum": float(np.sum(-rewards)),
        "cost_to_go0": float(-returns[0]),
        "min_dist": float(np.min(pos_dist)),
        "final_dist": float(pos_dist[-1]),
        "min_rot": float(np.min(rot_dist)),
        "final_rot": float(rot_dist[-1]),
        "mean_sing": float(np.mean(sing_abs)),
        "success": float(np.min(pos_dist) <= reward_cfg.success_tol),
    }
    return {
        "states": states,
        "next_states": next_states,
        "rewards": rewards,
        "returns": returns,
        "dones": dones,
        "metrics": metrics,
    }


def collect_episodes(
    episodes: int,
    max_time: float,
    reward_cfg: RewardConfig,
    gamma: float,
    terminal_value_dim: int,
    terminal_rot_scale: float,
    terminal_sing_scale: float,
    kf_poll_period: float,
    kf_sigma_a: float,
    kf_meas_noise: float,
    kf_use_noise: bool,
    kf_seed: int,
    episode_hook=None,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, float]]]:
    probe = MPCController(
        trajectory=get_ball_in_robot_trajectory(
            base_traj=base_trajectory_with_y_jitter(
                y_range=(0.0, 0.0),
                y_jitter_amp=0.0,
                jitter_dt=0.01,
                x_start_range=(-0.8, -0.8),
                x_end_range=(0.5, 0.5),
                speed_range=(0.3, 0.3),
            )
        ),
        use_terminal_value=False,
        enable_grasp=False,
        terminal_value_dim=int(terminal_value_dim),
        terminal_rot_scale=float(terminal_rot_scale),
        terminal_sing_scale=float(terminal_sing_scale),
    )
    reach_params = (probe.shoulder_offset.copy(), float(probe.reach_max))

    episodes_data = []
    episode_rows = []
    for ep in range(episodes):
        ep_data = rollout_episode(
            max_time=max_time,
            profile_period=1e9,
            reward_cfg=reward_cfg,
            gamma=gamma,
            reach_params=reach_params,
            terminal_value_dim=int(terminal_value_dim),
            terminal_rot_scale=float(terminal_rot_scale),
            terminal_sing_scale=float(terminal_sing_scale),
            kf_poll_period=float(kf_poll_period),
            kf_sigma_a=float(kf_sigma_a),
            kf_meas_noise=float(kf_meas_noise),
            kf_use_noise=bool(kf_use_noise),
            kf_seed=int(kf_seed + ep),
        )
        if ep_data["states"].shape[0] > 0:
            episodes_data.append(ep_data)
        metrics = dict(ep_data["metrics"])
        metrics["episode"] = int(ep + 1)
        episode_rows.append(metrics)
        if episode_hook is not None:
            episode_hook(metrics)
        print(
            f"[collect] episode {ep+1}/{episodes}, "
            f"steps={int(metrics['steps'])}, min_dist={metrics['min_dist']:.4f}, "
            f"cost_to_go0={metrics['cost_to_go0']:.4f}, success={int(metrics['success'])}"
        )
    return episodes_data, episode_rows


def stack_mc_dataset(episodes_data: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.vstack([d["states"] for d in episodes_data])
    y = -np.concatenate([d["returns"] for d in episodes_data])  # cost-to-go
    return x, y


def stack_td_dataset(
    episodes_data: List[Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = np.vstack([d["states"] for d in episodes_data])
    r = np.concatenate([d["rewards"] for d in episodes_data])
    s_next = np.vstack([d["next_states"] for d in episodes_data])
    done = np.concatenate([d["dones"] for d in episodes_data])
    c = -r  # step cost
    return s, c, s_next, done


def _tensor(data: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32, device=device)


def train_mc(
    model: ValueNet,
    x: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    device: str,
    epoch_hook=None,
) -> List[float]:
    x_t = _tensor(x, device)
    y_t = _tensor(y, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    n = x_t.shape[0]
    losses = []
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        n_batch = 0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            pred = model(x_t[idx])
            loss = loss_fn(pred, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n_batch += 1
        epoch_loss = total / max(n_batch, 1)
        losses.append(epoch_loss)
        if epoch_hook is not None:
            epoch_hook({"epoch": int(epoch + 1), "loss": float(epoch_loss)})
        if (epoch + 1) % cfg.log_interval == 0:
            print(f"[train-mc] epoch {epoch+1}/{cfg.epochs} loss={epoch_loss:.6f}")
    return losses


def soft_update(target_model: nn.Module, model: nn.Module, tau: float):
    with torch.no_grad():
        for tp, p in zip(target_model.parameters(), model.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


def train_td0(
    model: ValueNet,
    s: np.ndarray,
    c: np.ndarray,
    s_next: np.ndarray,
    done: np.ndarray,
    cfg: TrainConfig,
    device: str,
    epoch_hook=None,
) -> List[float]:
    s_t = _tensor(s, device)
    c_t = _tensor(c, device)
    s_next_t = _tensor(s_next, device)
    done_t = _tensor(done, device)

    target_model = copy.deepcopy(model).to(device)
    target_model.eval()

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    n = s_t.shape[0]
    losses = []
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        n_batch = 0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            s_b = s_t[idx]
            c_b = c_t[idx]
            s_next_b = s_next_t[idx]
            done_b = done_t[idx]
            with torch.no_grad():
                target_v_next = target_model(s_next_b)
                y_b = c_b + cfg.gamma * (1.0 - done_b) * target_v_next
            pred = model(s_b)
            loss = loss_fn(pred, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            soft_update(target_model, model, cfg.td_target_tau)
            total += float(loss.item())
            n_batch += 1
        epoch_loss = total / max(n_batch, 1)
        losses.append(epoch_loss)
        if epoch_hook is not None:
            epoch_hook({"epoch": int(epoch + 1), "loss": float(epoch_loss)})
        if (epoch + 1) % cfg.log_interval == 0:
            print(f"[train-td0] epoch {epoch+1}/{cfg.epochs} loss={epoch_loss:.6f}")
    return losses


def train_value(
    episodes: int,
    save_path: str,
    device: str,
    max_time: float,
    terminal_value_dim: int,
    terminal_rot_scale: float,
    terminal_sing_scale: float,
    kf_poll_period: float,
    kf_sigma_a: float,
    kf_meas_noise: float,
    kf_use_noise: bool,
    kf_seed: int,
    reward_cfg: RewardConfig,
    train_cfg: TrainConfig,
    seed: int,
    out_dir: str,
    run_name: str,
    plot: bool,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_mode: str,
    wandb_init_timeout: float,
):
    set_seed(seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name if run_name else f"{train_cfg.method}_{ts}"
    run_dir = os.path.join(out_dir, run_id)
    ensure_dir(run_dir)

    wandb_run = None
    if use_wandb:
        if wandb is None:
            print("[warn] wandb 未安装，跳过 wandb 日志。")
        else:
            normalize_proxy_env_for_wandb()
            wandb_kwargs = {
                "project": wandb_project,
                "entity": wandb_entity if wandb_entity else None,
                "name": run_id,
                "config": {
                    "episodes": episodes,
                    "device": device,
                    "max_time": max_time,
                    "terminal_value_dim": terminal_value_dim,
                    "terminal_rot_scale": terminal_rot_scale,
                    "terminal_sing_scale": terminal_sing_scale,
                    "kf_poll_period": kf_poll_period,
                    "kf_sigma_a": kf_sigma_a,
                    "kf_meas_noise": kf_meas_noise,
                    "kf_use_noise": kf_use_noise,
                    "reward_cfg": asdict(reward_cfg),
                    "train_cfg": asdict(train_cfg),
                    "seed": seed,
                },
                "settings": wandb.Settings(init_timeout=float(wandb_init_timeout)),
            }
            try:
                wandb_run = wandb.init(mode=wandb_mode, **wandb_kwargs)
            except Exception as e:
                print(f"[warn] wandb init 失败(mode={wandb_mode}): {e}")
                if wandb_mode == "online":
                    raise
                else:
                    print("[warn] 继续无 wandb 日志。")

    def on_episode(metrics: Dict[str, float]):
        if wandb_run is not None:
            wandb_run.log(
                {
                    "collect/episode": int(metrics["episode"]),
                    "collect/steps": int(metrics["steps"]),
                    "collect/min_dist": float(metrics["min_dist"]),
                    "collect/final_dist": float(metrics["final_dist"]),
                    "collect/cost_to_go0": float(metrics["cost_to_go0"]),
                    "collect/success": float(metrics["success"]),
                }
            )

    episodes_data, episode_rows = collect_episodes(
        episodes=episodes,
        max_time=max_time,
        reward_cfg=reward_cfg,
        gamma=float(train_cfg.gamma),
        terminal_value_dim=terminal_value_dim,
        terminal_rot_scale=terminal_rot_scale,
        terminal_sing_scale=terminal_sing_scale,
        kf_poll_period=kf_poll_period,
        kf_sigma_a=kf_sigma_a,
        kf_meas_noise=kf_meas_noise,
        kf_use_noise=kf_use_noise,
        kf_seed=kf_seed,
        episode_hook=on_episode,
    )
    if len(episodes_data) == 0:
        raise RuntimeError("未采到有效数据，请增加 episodes 或放宽轨迹/奖励配置。")

    input_dim = int(episodes_data[0]["states"].shape[1])
    model = ValueNet(input_dim=input_dim).to(device)
    epoch_rows: List[Dict[str, float]] = []

    def on_epoch(payload: Dict[str, float]):
        row = {"epoch": int(payload["epoch"]), "loss": float(payload["loss"])}
        epoch_rows.append(row)
        if wandb_run is not None:
            wandb_run.log({"train/epoch": row["epoch"], "train/loss": row["loss"]})

    if train_cfg.method == "mc":
        x, y = stack_mc_dataset(episodes_data)
        losses = train_mc(model, x, y, train_cfg, device=device, epoch_hook=on_epoch)
        dataset_size = int(x.shape[0])
    elif train_cfg.method == "td0":
        s, c, s_next, done = stack_td_dataset(episodes_data)
        losses = train_td0(model, s, c, s_next, done, train_cfg, device=device, epoch_hook=on_epoch)
        dataset_size = int(s.shape[0])
    else:
        raise ValueError("method 仅支持 mc / td0")

    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "method": train_cfg.method,
        "gamma": train_cfg.gamma,
        "reward_config": asdict(reward_cfg),
        "train_config": asdict(train_cfg),
        "terminal_rot_scale": float(terminal_rot_scale),
        "terminal_sing_scale": float(terminal_sing_scale),
        "kf_poll_period": float(kf_poll_period),
        "kf_sigma_a": float(kf_sigma_a),
        "kf_meas_noise": float(kf_meas_noise),
        "kf_use_noise": bool(kf_use_noise),
        "dataset_size": dataset_size,
        "episodes": int(episodes),
        "seed": int(seed),
        "final_loss": float(losses[-1] if len(losses) else 0.0),
    }
    model_path = save_path if os.path.isabs(save_path) else os.path.join(run_dir, save_path)
    torch.save(ckpt, model_path)

    episode_csv = os.path.join(run_dir, "episode_metrics.csv")
    epoch_csv = os.path.join(run_dir, "train_metrics.csv")
    write_csv(episode_csv, episode_rows)
    write_csv(epoch_csv, epoch_rows)

    summary = {
        "run_id": run_id,
        "run_dir": run_dir,
        "model_path": model_path,
        "method": train_cfg.method,
        "input_dim": input_dim,
        "dataset_size": dataset_size,
        "episodes": episodes,
        "final_loss": float(losses[-1] if len(losses) else 0.0),
        "success_rate": float(np.mean([r["success"] for r in episode_rows])) if episode_rows else 0.0,
        "mean_min_dist": float(np.mean([r["min_dist"] for r in episode_rows])) if episode_rows else 0.0,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_path = os.path.join(run_dir, "train_loss.png")
    if plot:
        has_plot = save_loss_plot(epoch_csv, plot_path)
        if (not has_plot) and (wandb_run is not None):
            wandb_run.log({"warn/no_matplotlib": 1})

    if wandb_run is not None:
        wandb_run.summary["final_loss"] = summary["final_loss"]
        wandb_run.summary["success_rate"] = summary["success_rate"]
        wandb_run.summary["mean_min_dist"] = summary["mean_min_dist"]
        wandb_run.summary["dataset_size"] = summary["dataset_size"]
        wandb_run.finish()
    print(
        f"[done] model={model_path}, method={train_cfg.method}, "
        f"input_dim={input_dim}, dataset={dataset_size}, final_loss={ckpt['final_loss']:.6f}, "
        f"run_dir={run_dir}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="td0", choices=["mc", "td0"])
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--save", type=str, default="value_net.pt")
    parser.add_argument("--out-dir", type=str, default="DPG_RL_mujoco/runs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-time", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=True)
    parser.add_argument("--wandb-project", type=str, default="dpg-rl-mpc")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-init-timeout", type=float, default=90.0)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--td-target-tau", type=float, default=0.02)
    parser.add_argument("--log-interval", type=int, default=1)

    parser.add_argument("--terminal-value-dim", type=int, default=7, choices=[3, 6, 7])
    parser.add_argument("--terminal-rot-scale", type=float, default=1.0)
    parser.add_argument("--terminal-sing-scale", type=float, default=0.2)
    parser.add_argument("--kf-poll-period", type=float, default=0.01)
    parser.add_argument("--kf-sigma-a", type=float, default=0.08)
    parser.add_argument("--kf-meas-noise", type=float, default=0.005)
    parser.add_argument("--kf-use-noise", dest="kf_use_noise", action="store_true")
    parser.add_argument("--no-kf-use-noise", dest="kf_use_noise", action="store_false")
    parser.add_argument("--kf-seed", type=int, default=0)
    parser.set_defaults(kf_use_noise=True)

    parser.add_argument("--dist-weight", type=float, default=1.0)
    parser.add_argument("--rot-weight", type=float, default=0.2)
    parser.add_argument("--sing-weight", type=float, default=0.1)
    parser.add_argument("--energy-weight", type=float, default=1e-3)
    parser.add_argument("--success-tol", type=float, default=0.02)
    parser.add_argument("--success-bonus", type=float, default=1.0)

    args = parser.parse_args()

    reward_cfg = RewardConfig(
        dist_weight=float(args.dist_weight),
        rot_weight=float(args.rot_weight),
        sing_weight=float(args.sing_weight),
        energy_weight=float(args.energy_weight),
        success_tol=float(args.success_tol),
        success_bonus=float(args.success_bonus),
    )
    train_cfg = TrainConfig(
        method=str(args.method),
        gamma=float(args.gamma),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        td_target_tau=float(args.td_target_tau),
        log_interval=int(args.log_interval),
    )

    train_value(
        episodes=int(args.episodes),
        save_path=str(args.save),
        device=str(args.device),
        max_time=float(args.max_time),
        terminal_value_dim=int(args.terminal_value_dim),
        terminal_rot_scale=float(args.terminal_rot_scale),
        terminal_sing_scale=float(args.terminal_sing_scale),
        kf_poll_period=float(args.kf_poll_period),
        kf_sigma_a=float(args.kf_sigma_a),
        kf_meas_noise=float(args.kf_meas_noise),
        kf_use_noise=bool(args.kf_use_noise),
        kf_seed=int(args.kf_seed),
        reward_cfg=reward_cfg,
        train_cfg=train_cfg,
        seed=int(args.seed),
        out_dir=str(args.out_dir),
        run_name=str(args.run_name),
        plot=bool(args.plot),
        use_wandb=bool(args.use_wandb),
        wandb_project=str(args.wandb_project),
        wandb_entity=str(args.wandb_entity),
        wandb_mode=str(args.wandb_mode),
        wandb_init_timeout=float(args.wandb_init_timeout),
    )


if __name__ == "__main__":
    main()

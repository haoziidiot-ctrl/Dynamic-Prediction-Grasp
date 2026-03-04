"""
train_value.py

使用 MPC + 当前价值函数共同采样轨迹，并在线训练终端价值函数 V(s)。

核心定义:
    - 终端状态 s_N: 默认 15D = [target_N(3), end_N(3), base_vel(3), q_current(6)]
      （由控制器内部做尺度归一化后送入价值网络）
    - 网络输出 V(s_N): cost-to-go（累计代价，越小越好）
    - 奖励: 距离惩罚 + 成功奖励 + 奇异/速度爆炸重罚 +
            禁区入区轻罚 + 停留线性/二次递增重罚 + 离开奖励 + 分数下限终止
    - 训练时每个 episode 都用“当前 V”参与 MPC 终端项，随后立刻更新 V
    - 训练轨迹固定为与非 RL 版本一致的底盘直线轨迹（便于可比）
    - 终端价值项带 scale / Hessian 上限 / 梯度裁剪，避免数值失衡

支持两种训练方式:
    - mc: Monte-Carlo 回归目标 C_t = -sum(gamma^k * r_{t+k})
    - td0: Bellman 目标 C_t = c_t + gamma * V_target(e_{t+1})

示例:
    python DPG_mujoco_RL/train_value.py --method td0 --episodes 1000 --epochs 50 --save value_net.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import socket
from datetime import datetime
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import mujoco
import numpy as np
import torch
from torch import nn

from DPG_MPC import MPCController
from DPG_track_ball_in_robot import (
    LinearBaseTrajectory,
    get_ball_in_robot_trajectory,
)
from rl_value import TerminalValueModel, ValueNet

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
    # 基础距离惩罚：距离越远扣分越多（RL 只负责评分，不替代 MPC 跟踪）
    distance_step_weight: float = 3.0
    # 成功奖励：同时满足终端距离和终端速度误差阈值
    success_dist_tol: float = 0.02
    success_speed_tol: float = 0.05
    success_bonus: float = 10.0
    stop_on_success: bool = False
    # 禁区惩罚：入区轻罚 + 停留递增重罚（防止“进了就赖着不走”）
    forbidden_entry_penalty: float = 2.0
    forbidden_step_penalty: float = 0.3
    forbidden_streak_penalty: float = 0.002
    forbidden_streak_quad_penalty: float = 0.000005
    forbidden_exit_bonus: float = 3.0
    # 奇异惩罚：一次性重罚（通过分数下限触发终止）
    singularity_metric_thresh: float = 9.0
    manipulability_thresh: float = 0.008
    singularity_hit_penalty: float = 15000.0
    # 关节速度爆炸惩罚：一次性重罚（通过分数下限触发终止）
    qdot_ratio_thresh: float = 1.0
    qdot_hit_penalty: float = 15000.0
    # 回合总分下限，触发终止
    episode_score_floor: float = -12000.0


@dataclass
class TrainConfig:
    method: str = "td0"  # "mc" or "td0"
    gamma: float = 0.98
    batch_size: int = 256
    epochs: int = 40  # 在线模式下作为 updates_per_episode 的默认回退值
    lr: float = 1e-3
    td_target_tau: float = 0.02
    log_interval: int = 1
    updates_per_episode: int = 4  # 每个 episode 采样后执行多少次梯度更新
    min_replay_size: int = 1024
    replay_capacity: int = 200000


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


def _proxy_host_port(proxy_url: str) -> Optional[Tuple[str, int]]:
    if not proxy_url:
        return None
    try:
        raw = proxy_url.strip()
        if "://" not in raw:
            raw = f"http://{raw}"
        parsed = urlparse(raw)
        host = parsed.hostname
        if not host:
            return None
        if parsed.port is not None:
            port = int(parsed.port)
        else:
            scheme = (parsed.scheme or "http").lower()
            if scheme in ("https",):
                port = 443
            elif scheme in ("socks5", "socks5h", "socks4"):
                port = 1080
            else:
                port = 80
        return host, port
    except Exception:
        return None


def _is_reachable_tcp(host: str, port: int, timeout_s: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False


def disable_unreachable_local_proxies(timeout_s: float = 0.25) -> List[str]:
    """
    清理不可达的本机代理（127.0.0.1/localhost），避免 wandb 一直卡 init 超时。
    返回被清理的环境变量名列表。
    """
    proxy_keys = ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]
    cleared: List[str] = []
    for key in proxy_keys:
        val = os.environ.get(key, "").strip()
        if not val:
            continue
        hp = _proxy_host_port(val)
        if hp is None:
            continue
        host, port = hp
        host_lower = host.lower()
        is_local = host_lower in ("127.0.0.1", "localhost", "::1")
        if is_local and (not _is_reachable_tcp(host, port, timeout_s=timeout_s)):
            os.environ.pop(key, None)
            cleared.append(key)
    return cleared


def clear_all_proxy_env() -> List[str]:
    proxy_keys = ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]
    removed: List[str] = []
    for key in proxy_keys:
        if os.environ.get(key):
            os.environ.pop(key, None)
            removed.append(key)
    return removed


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


def _score_step(item: Dict[str, np.ndarray], cfg: RewardConfig, state: Dict[str, float]):
    dist = float(item.get("terminal_dist", 1e9))
    speed_err = float(item.get("terminal_speed_error", 1e9))
    sing = float(item.get("singularity_metric", 0.0))
    manip = float(item.get("manipulability", 0.0))
    qdot_ratio = float(item.get("qdot_raw_ratio", item.get("qdot_limit_ratio", 0.0)))
    forbidden_count = float(item.get("forbidden_count", 0.0))

    reward = 0.0
    reason = []

    # 距离惩罚：终端距离越远，扣分越多
    reward -= float(cfg.distance_step_weight) * dist

    if "forbidden_streak" not in state:
        state["forbidden_streak"] = 0.0
    if "in_forbidden_prev" not in state:
        state["in_forbidden_prev"] = False

    # 禁区惩罚：首次进入轻罚；停留越久，线性+二次递增惩罚越大
    in_forbidden = forbidden_count > 0.0
    if in_forbidden:
        if not bool(state.get("in_forbidden_prev", False)):
            reward -= float(cfg.forbidden_entry_penalty)
        state["forbidden_streak"] = float(state.get("forbidden_streak", 0.0)) + 1.0
        streak = float(state["forbidden_streak"])
        occupancy = 1.0 + 0.25 * max(forbidden_count - 1.0, 0.0)
        reward -= float(cfg.forbidden_step_penalty) * occupancy
        reward -= float(cfg.forbidden_streak_penalty) * streak * occupancy
        reward -= float(cfg.forbidden_streak_quad_penalty) * (streak ** 2) * occupancy
    else:
        if bool(state.get("in_forbidden_prev", False)):
            reward += float(cfg.forbidden_exit_bonus)
        state["forbidden_streak"] = 0.0
    state["in_forbidden_prev"] = bool(in_forbidden)

    # 成功奖励：距离 + 速度同时接近
    success_hit = (
        (not bool(state.get("success", False)))
        and dist <= float(cfg.success_dist_tol)
        and speed_err <= float(cfg.success_speed_tol)
    )
    if success_hit:
        reward += float(cfg.success_bonus)
        state["success"] = True
        reason.append("success")

    # 奇异/爆速：一次性重罚，依赖分数下限终止（而不是显式事件终止）
    singularity_hit = (sing >= float(cfg.singularity_metric_thresh)) or (
        manip > 0.0 and manip <= float(cfg.manipulability_thresh)
    )
    if singularity_hit:
        reward -= float(cfg.singularity_hit_penalty)
        reason.append("singularity_hit")

    qdot_hit = qdot_ratio >= float(cfg.qdot_ratio_thresh)
    if qdot_hit:
        reward -= float(cfg.qdot_hit_penalty)
        reason.append("qdot_hit")

    score = float(state.get("score", 0.0)) + reward
    state["score"] = score
    floor_hit = score <= float(cfg.episode_score_floor)
    if floor_hit:
        reason.append("score_floor")

    done = floor_hit or (bool(cfg.stop_on_success) and success_hit)
    return float(reward), bool(done), "|".join(reason), dist, speed_err


def compute_rewards_and_dones(logs, cfg: RewardConfig):
    rewards = []
    dones = []
    reasons = []
    state = {"score": 0.0, "success": False, "forbidden_streak": 0.0, "in_forbidden_prev": False}
    for item in logs:
        r, done, reason, _, _ = _score_step(item, cfg, state)
        rewards.append(float(r))
        dones.append(1.0 if done else 0.0)
        reasons.append(reason)
        if done:
            break
    return np.asarray(rewards, dtype=float), np.asarray(dones, dtype=float), reasons, state


def discounted_returns(rewards: List[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=float)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns


# 与非 RL 版本对齐的固定底盘轨迹 + 默认目标点
FIXED_BASE_START = np.array([-0.8, 0.0, 0.0], dtype=float)
FIXED_BASE_END = np.array([0.50, 0.0, 0.0], dtype=float)
FIXED_BASE_SPEED = 0.3
FIXED_BALL_WORLD = np.array([0.25, 0.5, 1.2], dtype=float)


class ReplayBuffer:
    """简单环形回放池，存储 TD/MC 所需转移。"""

    def __init__(self, capacity: int, state_dim: int):
        if capacity <= 0:
            raise ValueError("replay capacity 必须 > 0")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.costs = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self._ptr = 0
        self.size = 0

    def add_batch(
        self,
        states: np.ndarray,
        costs: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ):
        n = int(states.shape[0])
        for i in range(n):
            idx = self._ptr
            self.states[idx] = states[i]
            self.costs[idx] = costs[i]
            self.next_states[idx] = next_states[i]
            self.dones[idx] = dones[i]
            self._ptr = (self._ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.size <= 0:
            raise RuntimeError("replay 为空，无法采样")
        n = min(int(batch_size), self.size)
        idx = np.random.randint(0, self.size, size=n)
        return self.states[idx], self.costs[idx], self.next_states[idx], self.dones[idx]


def build_terminal_value_from_model(
    model: ValueNet,
    input_dim: int,
    device: str,
    scale: float,
    hessian_max: float,
    grad_clip: float,
) -> TerminalValueModel:
    """将当前训练中的 ValueNet 直接注入 MPC 终端项（无需落盘再加载）。"""
    tv = TerminalValueModel(
        model_path=None,
        input_dim=int(input_dim),
        device=str(device),
        scale=float(scale),
        hessian_max=float(hessian_max),
        grad_clip=float(grad_clip),
    )
    tv._model = model.to(device)  # 训练过程内共享当前网络参数
    tv._model.eval()
    return tv


def base_trajectory_with_y_jitter(
    y_range=(-0.2, 0.2),
    y_jitter_amp=0.01,
    jitter_dt=0.01,
    x_start_range=(-1.0, -0.3),
    x_end_range=(0.2, 0.8),
    speed_range=(0.15, 0.45),
):
    """历史工具：随机轨迹采样（直线 + y 扰动），当前默认训练不使用。"""
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
    """历史工具：在底盘轨迹附近随机采样目标，当前默认训练使用固定目标。"""
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
    terminal_value_dim: int,
    terminal_rot_scale: float,
    terminal_sing_scale: float,
    kf_poll_period: float,
    kf_sigma_a: float,
    kf_meas_noise: float,
    kf_use_noise: bool,
    kf_seed: int,
    use_terminal_value: bool,
    terminal_value_model: Optional[TerminalValueModel],
    base_start: Optional[np.ndarray] = None,
    base_end: Optional[np.ndarray] = None,
    base_speed: float = FIXED_BASE_SPEED,
    ball_world: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    单回合 rollout：
    - 使用固定轨迹/目标构建环境；
    - MPC 每步都带当前终端价值函数；
    - 回调里按奖励规则累积分数，触发分数下限后提前终止回合。
    """
    if base_start is None:
        base_start = FIXED_BASE_START
    if base_end is None:
        base_end = FIXED_BASE_END
    if ball_world is None:
        ball_world = FIXED_BALL_WORLD
    base_traj_truth = LinearBaseTrajectory(
        start=np.asarray(base_start, dtype=float).reshape(3),
        end=np.asarray(base_end, dtype=float).reshape(3),
        speed=float(base_speed),
    )
    traj = get_ball_in_robot_trajectory(
        base_traj=base_traj_truth,
        ball_world=np.asarray(ball_world, dtype=float).reshape(3),
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
        use_terminal_value=bool(use_terminal_value),
        terminal_value_model=terminal_value_model,
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
    online_state = {
        "score": 0.0,
        "success": False,
        "forbidden_streak": 0.0,
        "in_forbidden_prev": False,
    }

    def callback(item):
        r, done, reason, dist, speed_err = _score_step(item, reward_cfg, online_state)
        item["step_reward"] = float(r)
        item["done_flag"] = 1.0 if done else 0.0
        item["done_reason"] = reason
        item["terminal_dist"] = float(dist)
        item["terminal_speed_error"] = float(speed_err)
        item["episode_score"] = float(online_state["score"])
        item["forbidden_streak"] = float(online_state.get("forbidden_streak", 0.0))
        logs.append(item)
        return bool(done)

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
                "min_speed_err": 0.0,
                "final_speed_err": 0.0,
                "min_rot": 0.0,
                "final_rot": 0.0,
                "mean_sing": 0.0,
                "mean_manip": 0.0,
                "max_qdot_ratio": 0.0,
                "mean_forbidden": 0.0,
                "max_forbidden": 0.0,
                "max_forbidden_streak": 0.0,
                "mean_mpc_cost_track": 0.0,
                "mean_mpc_cost_smooth": 0.0,
                "mean_mpc_cost_terminal": 0.0,
                "mean_mpc_cost_total": 0.0,
                "final_mpc_cost_track": 0.0,
                "final_mpc_cost_smooth": 0.0,
                "final_mpc_cost_terminal": 0.0,
                "final_mpc_cost_total": 0.0,
                "success": 0.0,
                "final_score": 0.0,
                "done_reason": "",
            },
        }

    rewards, dones, done_reasons, final_state = compute_rewards_and_dones(logs, reward_cfg)
    valid_len = int(rewards.shape[0])
    logs = logs[:valid_len]
    returns = discounted_returns(rewards.tolist(), gamma=float(gamma))
    states = np.asarray(
        [np.asarray(item.get("terminal_state", item["terminal_error"]), dtype=float) for item in logs],
        dtype=float,
    )
    next_states = np.vstack([states[1:], states[-1:]])
    pos_dist = np.asarray([float(item.get("terminal_dist", 0.0)) for item in logs], dtype=float)
    speed_err = np.asarray([float(item.get("terminal_speed_error", 0.0)) for item in logs], dtype=float)
    rot_dist = np.asarray(
        [
            float(np.linalg.norm(np.asarray(item.get("terminal_error", np.zeros(3)))[3:6]))
            if np.asarray(item.get("terminal_error", np.zeros(3))).shape[0] >= 6
            else 0.0
            for item in logs
        ],
        dtype=float,
    )
    sing_abs = np.asarray(
        [float(item.get("singularity_metric", 0.0)) for item in logs], dtype=float
    )
    manip_vals = np.asarray(
        [float(item.get("manipulability", 0.0)) for item in logs], dtype=float
    )
    qdot_ratio_vals = np.asarray(
        [float(item.get("qdot_limit_ratio", 0.0)) for item in logs], dtype=float
    )
    forbidden_vals = np.asarray(
        [float(item.get("forbidden_count", 0.0)) for item in logs], dtype=float
    )
    forbidden_streak_vals = np.asarray(
        [float(item.get("forbidden_streak", 0.0)) for item in logs], dtype=float
    )
    mpc_track_vals = np.asarray(
        [float(item.get("mpc_cost_track", 0.0)) for item in logs], dtype=float
    )
    mpc_smooth_vals = np.asarray(
        [float(item.get("mpc_cost_smooth", 0.0)) for item in logs], dtype=float
    )
    mpc_terminal_vals = np.asarray(
        [float(item.get("mpc_cost_terminal", 0.0)) for item in logs], dtype=float
    )
    mpc_total_vals = np.asarray(
        [float(item.get("mpc_cost_total", 0.0)) for item in logs], dtype=float
    )
    metrics = {
        "steps": int(states.shape[0]),
        "reward_sum": float(np.sum(rewards)),
        "cost_sum": float(np.sum(-rewards)),
        "cost_to_go0": float(-returns[0]),
        "min_dist": float(np.min(pos_dist)),
        "final_dist": float(pos_dist[-1]),
        "min_speed_err": float(np.min(speed_err)),
        "final_speed_err": float(speed_err[-1]),
        "min_rot": float(np.min(rot_dist)),
        "final_rot": float(rot_dist[-1]),
        "mean_sing": float(np.mean(sing_abs)),
        "mean_manip": float(np.mean(manip_vals)),
        "max_qdot_ratio": float(np.max(qdot_ratio_vals)),
        "mean_forbidden": float(np.mean(forbidden_vals)),
        "max_forbidden": float(np.max(forbidden_vals)),
        "max_forbidden_streak": float(np.max(forbidden_streak_vals)),
        "mean_mpc_cost_track": float(np.mean(mpc_track_vals)),
        "mean_mpc_cost_smooth": float(np.mean(mpc_smooth_vals)),
        "mean_mpc_cost_terminal": float(np.mean(mpc_terminal_vals)),
        "mean_mpc_cost_total": float(np.mean(mpc_total_vals)),
        "final_mpc_cost_track": float(mpc_track_vals[-1]),
        "final_mpc_cost_smooth": float(mpc_smooth_vals[-1]),
        "final_mpc_cost_terminal": float(mpc_terminal_vals[-1]),
        "final_mpc_cost_total": float(mpc_total_vals[-1]),
        "success": float(bool(final_state.get("success", False))),
        "final_score": float(final_state.get("score", 0.0)),
        "done_reason": done_reasons[-1] if len(done_reasons) else "",
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
    """离线采样工具（当前在线训练主流程默认不走这里）。"""
    episodes_data = []
    episode_rows = []
    for ep in range(episodes):
        ep_data = rollout_episode(
            max_time=max_time,
            profile_period=1e9,
            reward_cfg=reward_cfg,
            gamma=gamma,
            terminal_value_dim=int(terminal_value_dim),
            terminal_rot_scale=float(terminal_rot_scale),
            terminal_sing_scale=float(terminal_sing_scale),
            kf_poll_period=float(kf_poll_period),
            kf_sigma_a=float(kf_sigma_a),
            kf_meas_noise=float(kf_meas_noise),
            kf_use_noise=bool(kf_use_noise),
            kf_seed=int(kf_seed + ep),
            use_terminal_value=False,
            terminal_value_model=None,
            base_start=FIXED_BASE_START,
            base_end=FIXED_BASE_END,
            base_speed=float(FIXED_BASE_SPEED),
            ball_world=FIXED_BALL_WORLD,
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
    """离线 MC 训练入口（保留用于对照实验）。"""
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
    """离线 TD0 训练入口（保留用于对照实验）。"""
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


def td0_online_update(
    model: ValueNet,
    target_model: ValueNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    cfg: TrainConfig,
    device: str,
) -> float:
    """在线 TD0：每个 episode 采样后，从 replay 做若干次小批量更新。"""
    if replay.size < max(int(cfg.min_replay_size), 2):
        return float("nan")
    model.train()
    target_model.eval()
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(max(int(cfg.updates_per_episode), 1)):
        s_b, c_b, s_next_b, done_b = replay.sample(int(cfg.batch_size))
        s_t = _tensor(s_b, device)
        c_t = _tensor(c_b, device)
        s_next_t = _tensor(s_next_b, device)
        done_t = _tensor(done_b, device)
        with torch.no_grad():
            target_v_next = target_model(s_next_t)
            y = c_t + float(cfg.gamma) * (1.0 - done_t) * target_v_next
        pred = model(s_t)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        soft_update(target_model, model, float(cfg.td_target_tau))
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def mc_online_update(
    model: ValueNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    cfg: TrainConfig,
    device: str,
) -> float:
    """在线 MC：每个 episode 采样后，从 replay 做若干次回归更新。"""
    if replay.size < max(int(cfg.min_replay_size), 2):
        return float("nan")
    model.train()
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(max(int(cfg.updates_per_episode), 1)):
        s_b, y_b, _, _ = replay.sample(int(cfg.batch_size))
        s_t = _tensor(s_b, device)
        y_t = _tensor(y_b, device)
        pred = model(s_t)
        loss = loss_fn(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_value(
    episodes: int,
    save_path: str,
    device: str,
    max_time: float,
    terminal_value_dim: int,
    terminal_value_scale: float,
    terminal_hessian_max: float,
    terminal_grad_clip: float,
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
    """
    在线训练主流程：
    1) 用当前 ValueNet 参与 MPC rollout；
    2) 将新样本写入 replay；
    3) 立刻进行参数更新；
    4) 进入下一 episode。
    """
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
            dropped_local = disable_unreachable_local_proxies(timeout_s=0.25)
            if dropped_local:
                print(f"[wandb] dropped unreachable local proxies: {', '.join(dropped_local)}")
            wandb_kwargs = {
                "project": wandb_project,
                "entity": wandb_entity if wandb_entity else None,
                "name": run_id,
                "config": {
                    "episodes": episodes,
                    "device": device,
                    "max_time": max_time,
                    "terminal_value_dim": terminal_value_dim,
                    "terminal_value_scale": terminal_value_scale,
                    "terminal_hessian_max": terminal_hessian_max,
                    "terminal_grad_clip": terminal_grad_clip,
                    "terminal_rot_scale": terminal_rot_scale,
                    "terminal_sing_scale": terminal_sing_scale,
                    "kf_poll_period": kf_poll_period,
                    "kf_sigma_a": kf_sigma_a,
                    "kf_meas_noise": kf_meas_noise,
                    "kf_use_noise": kf_use_noise,
                    "base_start": FIXED_BASE_START.tolist(),
                    "base_end": FIXED_BASE_END.tolist(),
                    "base_speed": float(FIXED_BASE_SPEED),
                    "ball_world": FIXED_BALL_WORLD.tolist(),
                    "reward_cfg": asdict(reward_cfg),
                    "train_cfg": asdict(train_cfg),
                    "seed": seed,
                },
                "settings": wandb.Settings(init_timeout=float(wandb_init_timeout)),
            }
            init_err = None
            try:
                wandb_run = wandb.init(mode=wandb_mode, **wandb_kwargs)
            except Exception as e:
                init_err = e
                print(f"[warn] wandb init 失败(mode={wandb_mode}): {e}")

            if (wandb_run is None) and (wandb_mode == "online"):
                removed = clear_all_proxy_env()
                if removed:
                    print(f"[wandb] retry init without proxy env: {', '.join(removed)}")
                    try:
                        wandb_run = wandb.init(mode=wandb_mode, **wandb_kwargs)
                        init_err = None
                    except Exception as e:
                        init_err = e
                        print(f"[warn] wandb retry 失败(mode={wandb_mode}): {e}")

            if (wandb_run is None) and (wandb_mode == "online"):
                raise RuntimeError(
                    "wandb online init failed after proxy checks/retry. "
                    "请检查网络或代理可达性，或用 --no-wandb 先本地训练。"
                ) from init_err
            if (wandb_run is None) and (wandb_mode != "online"):
                print("[warn] 继续无 wandb 日志。")

    input_dim = int(terminal_value_dim)
    model = ValueNet(input_dim=input_dim).to(device)
    target_model = copy.deepcopy(model).to(device)
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.lr))

    replay = ReplayBuffer(
        capacity=int(train_cfg.replay_capacity),
        state_dim=input_dim,
    )
    episode_rows: List[Dict[str, float]] = []
    epoch_rows: List[Dict[str, float]] = []
    losses: List[float] = []

    for ep in range(int(episodes)):
        model.eval()
        terminal_value_model = build_terminal_value_from_model(
            model=model,
            input_dim=input_dim,
            device=device,
            scale=float(terminal_value_scale),
            hessian_max=float(terminal_hessian_max),
            grad_clip=float(terminal_grad_clip),
        )
        ep_data = rollout_episode(
            max_time=max_time,
            profile_period=1e9,
            reward_cfg=reward_cfg,
            gamma=float(train_cfg.gamma),
            terminal_value_dim=terminal_value_dim,
            terminal_rot_scale=terminal_rot_scale,
            terminal_sing_scale=terminal_sing_scale,
            kf_poll_period=kf_poll_period,
            kf_sigma_a=kf_sigma_a,
            kf_meas_noise=kf_meas_noise,
            kf_use_noise=kf_use_noise,
            kf_seed=int(kf_seed + ep),
            use_terminal_value=True,
            terminal_value_model=terminal_value_model,
            base_start=FIXED_BASE_START,
            base_end=FIXED_BASE_END,
            base_speed=float(FIXED_BASE_SPEED),
            ball_world=FIXED_BALL_WORLD,
        )

        if ep_data["states"].shape[0] > 0:
            if train_cfg.method == "td0":
                costs = (-ep_data["rewards"]).astype(np.float32)
                next_states = ep_data["next_states"].astype(np.float32)
                dones = ep_data["dones"].astype(np.float32)
            else:
                costs = (-ep_data["returns"]).astype(np.float32)
                next_states = ep_data["states"].astype(np.float32)
                dones = np.ones_like(costs, dtype=np.float32)
            replay.add_batch(
                states=ep_data["states"].astype(np.float32),
                costs=costs,
                next_states=next_states,
                dones=dones,
            )

        if train_cfg.method == "td0":
            loss = td0_online_update(
                model=model,
                target_model=target_model,
                optimizer=optimizer,
                replay=replay,
                cfg=train_cfg,
                device=device,
            )
        elif train_cfg.method == "mc":
            loss = mc_online_update(
                model=model,
                optimizer=optimizer,
                replay=replay,
                cfg=train_cfg,
                device=device,
            )
        else:
            raise ValueError("method 仅支持 mc / td0")

        losses.append(float(loss))
        epoch_rows.append({"epoch": int(ep + 1), "loss": float(loss)})

        metrics = dict(ep_data["metrics"])
        metrics["episode"] = int(ep + 1)
        metrics["replay_size"] = int(replay.size)
        metrics["train_loss"] = float(loss)
        episode_rows.append(metrics)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "collect/episode": int(metrics["episode"]),
                    "collect/steps": int(metrics["steps"]),
                    "collect/min_dist": float(metrics["min_dist"]),
                    "collect/final_dist": float(metrics["final_dist"]),
                    "collect/final_speed_err": float(metrics.get("final_speed_err", 0.0)),
                    "collect/cost_to_go0": float(metrics["cost_to_go0"]),
                    "collect/mean_manip": float(metrics.get("mean_manip", 0.0)),
                    "collect/max_qdot_ratio": float(metrics.get("max_qdot_ratio", 0.0)),
                    "collect/mean_forbidden": float(metrics.get("mean_forbidden", 0.0)),
                    "collect/max_forbidden": float(metrics.get("max_forbidden", 0.0)),
                    "collect/max_forbidden_streak": float(metrics.get("max_forbidden_streak", 0.0)),
                    "collect/mean_mpc_cost_track": float(metrics.get("mean_mpc_cost_track", 0.0)),
                    "collect/mean_mpc_cost_smooth": float(metrics.get("mean_mpc_cost_smooth", 0.0)),
                    "collect/mean_mpc_cost_terminal": float(metrics.get("mean_mpc_cost_terminal", 0.0)),
                    "collect/mean_mpc_cost_total": float(metrics.get("mean_mpc_cost_total", 0.0)),
                    "collect/final_score": float(metrics.get("final_score", 0.0)),
                    "collect/success": float(metrics["success"]),
                    "train/loss": float(loss),
                    "train/replay_size": int(replay.size),
                }
            )

        loss_text = "nan" if not np.isfinite(loss) else f"{loss:.6f}"
        print(
            f"[online] episode {ep+1}/{episodes}, "
            f"steps={int(metrics['steps'])}, min_dist={metrics['min_dist']:.4f}, "
            f"cost_to_go0={metrics['cost_to_go0']:.4f}, "
            f"mean_forbidden={metrics.get('mean_forbidden', 0.0):.4f}, "
            f"mpc(track/smooth/term/total)="
            f"{metrics.get('mean_mpc_cost_track', 0.0):.3e}/"
            f"{metrics.get('mean_mpc_cost_smooth', 0.0):.3e}/"
            f"{metrics.get('mean_mpc_cost_terminal', 0.0):.3e}/"
            f"{metrics.get('mean_mpc_cost_total', 0.0):.3e}, "
            f"final_score={metrics.get('final_score', 0.0):.4f}, "
            f"done={metrics.get('done_reason', '')}, "
            f"success={int(metrics['success'])}, replay={int(replay.size)}, loss={loss_text}"
        )

    dataset_size = int(replay.size)
    if dataset_size <= 0:
        raise RuntimeError("在线训练未采到有效样本，请检查 max-time / 奖励配置。")

    finite_losses = [float(v) for v in losses if np.isfinite(v)]
    final_loss = float(finite_losses[-1]) if finite_losses else float("nan")

    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "method": train_cfg.method,
        "gamma": train_cfg.gamma,
        "reward_config": asdict(reward_cfg),
        "train_config": asdict(train_cfg),
        "terminal_rot_scale": float(terminal_rot_scale),
        "terminal_sing_scale": float(terminal_sing_scale),
        "terminal_value_scale": float(terminal_value_scale),
        "terminal_hessian_max": float(terminal_hessian_max),
        "terminal_grad_clip": float(terminal_grad_clip),
        "kf_poll_period": float(kf_poll_period),
        "kf_sigma_a": float(kf_sigma_a),
        "kf_meas_noise": float(kf_meas_noise),
        "kf_use_noise": bool(kf_use_noise),
        "base_start": FIXED_BASE_START.tolist(),
        "base_end": FIXED_BASE_END.tolist(),
        "base_speed": float(FIXED_BASE_SPEED),
        "ball_world": FIXED_BALL_WORLD.tolist(),
        "dataset_size": dataset_size,
        "episodes": int(episodes),
        "seed": int(seed),
        "final_loss": float(final_loss),
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
        "terminal_value_scale": float(terminal_value_scale),
        "terminal_hessian_max": float(terminal_hessian_max),
        "terminal_grad_clip": float(terminal_grad_clip),
        "dataset_size": dataset_size,
        "episodes": episodes,
        "base_start": FIXED_BASE_START.tolist(),
        "base_end": FIXED_BASE_END.tolist(),
        "base_speed": float(FIXED_BASE_SPEED),
        "ball_world": FIXED_BALL_WORLD.tolist(),
        "final_loss": float(final_loss),
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
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--save", type=str, default="value_net.pt")
    parser.add_argument("--out-dir", type=str, default="DPG_mujoco_RL/runs")
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
    parser.add_argument("--wandb-init-timeout", type=float, default=30.0)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--td-target-tau", type=float, default=0.02)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--updates-per-episode", type=int, default=None)
    parser.add_argument("--min-replay-size", type=int, default=1024)
    parser.add_argument("--replay-capacity", type=int, default=200000)

    parser.add_argument("--terminal-value-dim", type=int, default=15, choices=[3, 6, 7, 15])
    parser.add_argument("--terminal-value-scale", type=float, default=0.008)
    parser.add_argument("--terminal-hessian-max", type=float, default=80.0)
    parser.add_argument("--terminal-grad-clip", type=float, default=120.0)
    parser.add_argument("--terminal-rot-scale", type=float, default=1.0)
    parser.add_argument("--terminal-sing-scale", type=float, default=0.2)
    parser.add_argument("--kf-poll-period", type=float, default=0.01)
    parser.add_argument("--kf-sigma-a", type=float, default=0.08)
    parser.add_argument("--kf-meas-noise", type=float, default=0.005)
    parser.add_argument("--kf-use-noise", dest="kf_use_noise", action="store_true")
    parser.add_argument("--no-kf-use-noise", dest="kf_use_noise", action="store_false")
    parser.add_argument("--kf-seed", type=int, default=0)
    parser.set_defaults(kf_use_noise=True)

    parser.add_argument("--distance-step-weight", type=float, default=3.0)
    parser.add_argument("--success-dist-tol", type=float, default=0.02)
    parser.add_argument("--success-speed-tol", type=float, default=0.05)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--stop-on-success", dest="stop_on_success", action="store_true")
    parser.add_argument("--no-stop-on-success", dest="stop_on_success", action="store_false")
    parser.set_defaults(stop_on_success=False)
    parser.add_argument("--forbidden-step-penalty", type=float, default=0.3)
    parser.add_argument("--forbidden-streak-penalty", type=float, default=0.002)
    parser.add_argument("--forbidden-entry-penalty", type=float, default=2.0)
    parser.add_argument("--forbidden-streak-quad-penalty", type=float, default=0.000005)
    parser.add_argument("--forbidden-exit-bonus", type=float, default=3.0)
    parser.add_argument("--singularity-metric-thresh", type=float, default=9.0)
    parser.add_argument("--manipulability-thresh", type=float, default=0.008)
    parser.add_argument("--singularity-hit-penalty", type=float, default=15000.0)
    parser.add_argument("--qdot-ratio-thresh", type=float, default=1.0)
    parser.add_argument("--qdot-hit-penalty", type=float, default=15000.0)
    parser.add_argument("--episode-score-floor", type=float, default=-12000.0)

    args = parser.parse_args()

    reward_cfg = RewardConfig(
        distance_step_weight=float(args.distance_step_weight),
        success_dist_tol=float(args.success_dist_tol),
        success_speed_tol=float(args.success_speed_tol),
        success_bonus=float(args.success_bonus),
        stop_on_success=bool(args.stop_on_success),
        forbidden_entry_penalty=float(args.forbidden_entry_penalty),
        forbidden_step_penalty=float(args.forbidden_step_penalty),
        forbidden_streak_penalty=float(args.forbidden_streak_penalty),
        forbidden_streak_quad_penalty=float(args.forbidden_streak_quad_penalty),
        forbidden_exit_bonus=float(args.forbidden_exit_bonus),
        singularity_metric_thresh=float(args.singularity_metric_thresh),
        manipulability_thresh=float(args.manipulability_thresh),
        singularity_hit_penalty=float(args.singularity_hit_penalty),
        qdot_ratio_thresh=float(args.qdot_ratio_thresh),
        qdot_hit_penalty=float(args.qdot_hit_penalty),
        episode_score_floor=float(args.episode_score_floor),
    )
    train_cfg = TrainConfig(
        method=str(args.method),
        gamma=float(args.gamma),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        td_target_tau=float(args.td_target_tau),
        log_interval=int(args.log_interval),
        updates_per_episode=(
            int(args.updates_per_episode)
            if args.updates_per_episode is not None
            else int(args.epochs)
        ),
        min_replay_size=int(args.min_replay_size),
        replay_capacity=int(args.replay_capacity),
    )

    train_value(
        episodes=int(args.episodes),
        save_path=str(args.save),
        device=str(args.device),
        max_time=float(args.max_time),
        terminal_value_dim=int(args.terminal_value_dim),
        terminal_value_scale=float(args.terminal_value_scale),
        terminal_hessian_max=float(args.terminal_hessian_max),
        terminal_grad_clip=float(args.terminal_grad_clip),
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

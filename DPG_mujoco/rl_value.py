"""
rl_value.py

终端价值函数 (Value Function) 的轻量封装：
    - 支持加载 PyTorch MLP 模型。
    - 计算 V(e) 在给定 e 处的一阶/二阶导数，用于构造终端代价二次近似。
    - 若未提供模型或加载失败，回退为简单二次型（等价于传统终端代价）。

说明:
    - 默认把终端状态简化为 3D 位置误差 e = (x_N - x_target)。
    - 可扩展到 6D: [位置误差, 姿态对齐误差]，但终端代价映射需同步修改。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch 不可用时
    torch = None
    nn = None


def _project_psd(h: np.ndarray, min_eig: float = 1e-6, max_eig: float = 1e3) -> np.ndarray:
    """对称化并将 Hessian 投影到 PSD，避免 QP 不稳定。"""
    h = 0.5 * (h + h.T)
    w, v = np.linalg.eigh(h)
    w = np.clip(w, min_eig, max_eig)
    return v @ np.diag(w) @ v.T


class ValueNet(nn.Module):
    """简单 MLP 价值网络：输入 e，输出标量 V(e)。"""

    def __init__(self, input_dim: int = 3, hidden=(64, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class TerminalValueModel:
    """封装价值函数 + Hessian/Gradient 计算。"""

    model_path: Optional[str] = None
    input_dim: int = 3
    device: str = "cpu"
    fallback_weight: float = 20.0
    hessian_eps: float = 1e-6
    hessian_max: float = 1e3
    scale: float = 1.0

    def __post_init__(self):
        self._use_torch = torch is not None
        self._model = None
        if self._use_torch and self.model_path:
            self._model = ValueNet(input_dim=self.input_dim).to(self.device)
            try:
                state = torch.load(self.model_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self._model.load_state_dict(state)
                self._model.eval()
            except Exception:
                # 加载失败时回退到二次型
                self._model = None

    def quadratic_approx(self, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        返回 (H, g, v):
            V(e) ≈ 0.5 e^T H e + g^T e + const
        """
        e = np.asarray(e, dtype=np.float32).reshape(-1)
        if (not self._use_torch) or (self._model is None):
            h = np.eye(self.input_dim, dtype=float) * float(self.fallback_weight)
            g = np.zeros(self.input_dim, dtype=float)
            v = 0.5 * float(self.fallback_weight) * float(e @ e)
            return h, g, v

        with torch.enable_grad():
            x = torch.tensor(e, dtype=torch.float32, requires_grad=True, device=self.device)
            v = self._model(x)
            g = torch.autograd.grad(v, x, create_graph=True)[0]
            # Hessian: input_dim 小时直接算；若维度大建议替换为近似
            h = torch.autograd.functional.hessian(lambda y: self._model(y), x)
            h = h.detach().cpu().numpy()
            g = g.detach().cpu().numpy()
            v = float(v.detach().cpu().item())

        h = _project_psd(h, min_eig=self.hessian_eps, max_eig=self.hessian_max)
        h *= float(self.scale)
        g *= float(self.scale)
        v *= float(self.scale)
        return h, g, v

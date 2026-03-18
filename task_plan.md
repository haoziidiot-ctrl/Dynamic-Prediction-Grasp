# Task Plan

## Goal
将 `DPG_mujoco_final copy` 从当前的工程化 stochastic layer 推进到更完整的随机 MPC 版本：加入显式闭环协方差传播、保持现有 chance-constraint/backoff 与可选 tube-style 执行层，并完成 headless 对照实验（stochastic vs basic MPC），同时保证 stochastic 关闭时基础 MPC 行为不被破坏。

## Phases
- [x] Phase 1: 接入 stochastic 参数、chance backoff、tube-style feedback
- [x] Phase 2: 加入显式闭环协方差传播 `(A+BK) Σ (A+BK)^T + W`
- [x] Phase 3: 跑 headless 对照实验并汇总指标
- [x] Phase 4: 给出结果和下一步建议

## Conclusion
- 已完成随机 MPC 工程版落地：代价自适应 + chance backoff + 显式闭环协方差传播，并保留可选 tube-style 执行补偿接口。
- 额外排查发现 `copy` 版曾错误地改变了基础 constrained-QP 路径：
  - 在 QP 分支里无条件调用 `_limit_twist_cmd(...)`
  - stochastic 关闭时仍改变协方差查询的时间基准
- 这两点已修复，基础 MPC 已恢复到与 `DPG_mujoco_final` 一致的 headless 表现。
- 当前场景下，chance backoff 本身不会破坏跟踪；真正会导致大误差和大量 infeasible 的是 tube-style 位置补偿，因此已改为默认关闭。

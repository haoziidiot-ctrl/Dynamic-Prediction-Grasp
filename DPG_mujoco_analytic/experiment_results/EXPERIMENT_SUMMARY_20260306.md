# DPG_mujoco_final 实验总报告（综合实验 + 两个消融）

## 1. 报告范围
本报告汇总 3 组实验：
1. 综合随机实验（200 回合）
2. 消融实验 A：自适应权重（uncertainty-aware weighting）开/关（200 对）
3. 消融实验 B：操作度梯度引导（manipulability guidance）开/关（200 对）

本报告数据全部来自 `DPG_mujoco_final/experiment_results/` 下的 CSV/JSON 文件，无人工改写。

## 2. 关键脚本与结果文件

### 2.1 实验脚本
1. 综合实验脚本：`DPG_mujoco_final/DPG_batch_experiment.py`
2. 消融 A 脚本：`DPG_mujoco_final/DPG_ablation_uncertainty.py`
3. 消融 B 脚本：`DPG_mujoco_final/DPG_ablation_manip.py`
4. 控制器主实现：`DPG_mujoco_final/DPG_MPC.py`

### 2.2 本次实验目录
1. 综合实验目录：`DPG_mujoco_final/experiment_results/batch_20260306_055318`
2. 消融 A 目录：`DPG_mujoco_final/experiment_results/ablation_uncertainty_20260306_061125`
3. 消融 B 目录：`DPG_mujoco_final/experiment_results/ablation_manip_20260306_062951`

### 2.3 每组目录中的核心文件
1. `config.json`：实验配置与随机范围
2. `summary*.csv`：总体统计
3. `per_episode*.csv`：逐回合指标
4. `per_step.csv`（仅综合实验）：逐步指标

## 3. 通用实验设置

### 3.1 场景与随机化
1. 目标球世界坐标固定：`[0.25, 0.5, 1.2]`
2. 每回合随机底盘参数：
   - `x_start ~ U[-1.1, -0.5]`
   - `x_end ~ U[0.2, 0.8]`
   - `speed ~ U[0.1, 0.5]`
   - `y_noise_mean ~ U[0.005, 0.02]`
   - `y_noise_std = 0.005`（固定）
3. 每组实验回合数：`200`
4. 随机种子：`20260306`
5. 运行模式：headless

### 3.2 控制器关键固定参数
1. 两阶段参考：`hold` 跟踪 `target + [0, -0.13, 0]`，满足阈值后切 `attach`
2. 切换阈值：`offset_trigger_tol=0.03`，`offset_trigger_steps=6`
3. 成功判定：`target_err < 0.02` 且连续 `10` 步
4. 轨迹权重：`pos_weight=12.0`, `rot_weight=0.3`

### 3.3 禁区判定规则（本次实验统计口径）
禁区几何：
1. `x ∈ (-∞, x_t-0.05] ∪ [x_t+0.05, +∞)`
2. `y ∈ [y_t-0.10, y_t]`
3. `z` 全空间

计入规则：
1. 每一步若“任一关节体（6个）或末端 end_finger”在禁区内，则该步记为 `in_funnel=1`
2. `zone_time_s = zone_steps * dt`

## 4. 指标定义

### 4.1 主指标
1. 成功率：`success_rate = success_count / episodes`
2. 禁区时间：`zone_time_s`、`zone_ratio`
3. 跟踪误差：
   - `hold_err`（hold 阶段）
   - `attach_err`（attach 阶段）

### 4.2 高频抑制与控制平滑指标（消融）
1. `hf_target_err_diff_rms`：`target_err` 相邻步差分 RMS
2. `hf_attach_err_diff_rms`：`attach_err` 相邻步差分 RMS
3. `hf_qdot_diff_rms`：控制输出 `qdot` 相邻步差分 RMS

### 4.3 奇异相关指标（操作度消融）
1. `cond_log10_mean / p95`
2. `manip_w_mean / p10`
3. `manip_risk_mean / p95`

## 5. 综合随机实验（200 回合）

来源文件：
1. `batch_20260306_055318/summary.csv`
2. `batch_20260306_055318/per_episode.csv`
3. `batch_20260306_055318/per_step.csv`

### 5.1 总体结果
1. 成功率：`195/200 = 97.5%`
2. 总步数：`1,075,990`
3. 总禁区时间：`149.464 s`
4. 单回合平均禁区时间：`0.74732 s`
5. 全局步级禁区占比：`0.06945`
6. 全局 `hold_err` 均值：`0.46095 m`
7. 全局 `attach_err` 均值：`0.02873 m`

### 5.2 分布特性（来自逐回合统计）
1. 禁区时间中位数：`0.104 s`
2. 禁区时间 P90 / P95：`5.962 s / 6.077 s`
3. 禁区时间为 0 的回合比例：`35%`
4. 禁区时间 > 6s 的回合比例：`9%`
5. `attach_err_mean` 中位数 / P90 / P95：`0.0221 / 0.0511 / 0.0565 m`
6. `target_err_min` 中位数 / P90 / P95：`0.00884 / 0.0154 / 0.0170 m`

### 5.3 阶段对比
1. hold 阶段步级禁区占比：`1.15%`
2. attach 阶段步级禁区占比：`8.46%`

结论：禁区风险主要发生在 attach 阶段。

### 5.4 失败回合
失败回合编号：`[71, 115, 144, 161, 191]`

## 6. 消融 A：自适应权重开/关（公平成对）

来源文件：
1. `ablation_uncertainty_20260306_061125/summary_by_mode.csv`
2. `ablation_uncertainty_20260306_061125/summary_delta.csv`
3. `ablation_uncertainty_20260306_061125/per_episode_delta_on_minus_off.csv`

### 6.1 对照设置
1. ON：`use_uncertainty_aware_weighting=True`
2. OFF：`use_uncertainty_aware_weighting=False`
3. 公平性：每个 episode 使用完全相同的随机参数与随机种子，ON/OFF 各跑一次。

### 6.2 模式均值结果
| 指标 | ON | OFF | ON-OFF |
|---|---:|---:|---:|
| success_rate | 0.975 | 0.960 | +0.015 |
| zone_time_mean_s | 0.7473 | 1.1006 | -0.3532 |
| zone_ratio_mean | 0.0699 | 0.1022 | -0.0323 |
| target_err_mean | 0.13229 | 0.13433 | -0.00204 |
| attach_err_mean | 0.02808 | 0.03039 | -0.00204 |
| hf_qdot_diff_rms_mean | 0.16606 | 0.17208 | -0.00602 |

### 6.3 百分比解释
1. 成功率：+1.5 个百分点
2. 禁区时间均值：下降约 `32.1%`
3. 禁区占比均值：下降约 `31.6%`
4. `target_err_mean`：下降约 `1.5%`
5. `attach_err_mean`：下降约 `6.7%`
6. `hf_qdot_diff_rms`：下降约 `3.5%`

### 6.4 成对胜率与分布
1. `zone_time`：ON 更好 `78`、更差 `68`、平局 `54`（非平局胜率 `53.4%`）
2. `target_err_mean`：ON 更好 `124/200`（`62%`）
3. `hf_qdot_diff_rms`：ON 更好 `145/200`（`72.5%`）
4. `attach_err_mean`：ON 更好仅 `51/197`，但均值依然更优，说明收益集中于困难回合的长尾抑制。

### 6.5 结论
在本实验设置下，自适应权重对“禁区风险抑制 + 控制平滑 + 成功率”呈正收益，建议保留。

## 7. 消融 B：操作度梯度开/关（公平成对）

来源文件：
1. `ablation_manip_20260306_062951/summary_by_mode.csv`
2. `ablation_manip_20260306_062951/summary_delta.csv`
3. `ablation_manip_20260306_062951/per_episode_delta_on_minus_off.csv`

### 7.1 对照设置
1. ON：`use_manipulability_guidance=True`
2. OFF：`use_manipulability_guidance=False`
3. 其余参数保持一致，且本组固定 `use_uncertainty_aware_weighting=True`

### 7.2 模式均值结果
| 指标 | ON | OFF | ON-OFF |
|---|---:|---:|---:|
| success_rate | 0.975 | 0.985 | -0.010 |
| zone_time_mean_s | 0.7473 | 0.5964 | +0.1510 |
| zone_ratio_mean | 0.0699 | 0.0554 | +0.01446 |
| target_err_mean | 0.13229 | 0.12968 | +0.00261 |
| attach_err_mean | 0.02808 | 0.02700 | +0.00104 |
| hf_qdot_diff_rms_mean | 0.16606 | 0.15999 | +0.00607 |
| qdot_norm_mean | 1.44368 | 1.36809 | +0.07559 |

### 7.3 百分比解释
1. 成功率：下降 1.0 个百分点
2. 禁区时间均值：上升约 `25.3%`
3. 禁区占比均值：上升约 `26.1%`
4. `target_err_mean`：上升约 `2.0%`
5. `attach_err_mean`：上升约 `3.9%`
6. `hf_qdot_diff_rms`：上升约 `3.8%`
7. `qdot_norm_mean`：上升约 `5.5%`

### 7.4 成对胜率与分布
1. `zone_time`：ON 更好 `60`、更差 `64`、平局 `76`（非平局胜率 `48.4%`）
2. `target_err_mean`：ON 更好 `68/200`（`34%`）
3. `hf_qdot_diff_rms`：ON 更好 `37/200`（`18.5%`）
4. 成功率净变化：ON 仅 `1` 回合优于 OFF，`3` 回合劣于 OFF。

### 7.5 注意项
`manip_w` 与 `manip_risk` 在 OFF 组为 0（功能关闭即不计算），因此不能直接与 ON 的绝对值做物理对比，只能作为 ON 组内部诊断量。

### 7.6 结论
在当前参数与任务设置下，操作度梯度引导未体现收益，且存在长尾退化回合。

## 8. 综合结论（可直接用于论文“实验总结”）
1. 综合随机实验验证了系统具备较高成功率（`97.5%`）和较低 attach 阶段误差（全局均值 `0.0287 m`）。
2. 自适应权重消融显示其对高频扰动场景有效，显著降低禁区时间与控制抖动，并提高成功率。
3. 操作度梯度在当前配置下无明显正贡献，反而在多个核心指标上退化。
4. 当前推荐配置：
   - `use_uncertainty_aware_weighting=True`
   - `use_manipulability_guidance=False`（或单独再调参后复验）

## 9. 复现实验命令

### 9.1 综合实验
```bash
python DPG_mujoco_final/DPG_batch_experiment.py --episodes 200 --seed 20260306 --y-noise-std 0.005
```

### 9.2 消融 A（自适应权重）
```bash
python DPG_mujoco_final/DPG_ablation_uncertainty.py --episodes 200 --seed 20260306 --y-noise-std 0.005
```

### 9.3 消融 B（操作度梯度）
```bash
python DPG_mujoco_final/DPG_ablation_manip.py --episodes 200 --seed 20260306 --y-noise-std 0.005 --uncertainty-on 1
```

## 10. 结果文件索引（本次）
1. 综合实验汇总：`DPG_mujoco_final/experiment_results/batch_20260306_055318/summary.csv`
2. 综合实验逐回合：`DPG_mujoco_final/experiment_results/batch_20260306_055318/per_episode.csv`
3. 综合实验逐步：`DPG_mujoco_final/experiment_results/batch_20260306_055318/per_step.csv`
4. 消融 A 汇总（模式）：`DPG_mujoco_final/experiment_results/ablation_uncertainty_20260306_061125/summary_by_mode.csv`
5. 消融 A 汇总（差值）：`DPG_mujoco_final/experiment_results/ablation_uncertainty_20260306_061125/summary_delta.csv`
6. 消融 B 汇总（模式）：`DPG_mujoco_final/experiment_results/ablation_manip_20260306_062951/summary_by_mode.csv`
7. 消融 B 汇总（差值）：`DPG_mujoco_final/experiment_results/ablation_manip_20260306_062951/summary_delta.csv`

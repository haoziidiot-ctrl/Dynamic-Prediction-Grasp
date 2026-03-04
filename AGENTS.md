# Repository Guidelines

本文件是仓库协作与开发的简明指南，面向仿真与实机联动的动态抓取研发。

## 项目结构与模块组织
- `DPG_mujoco/`：Mujoco 仿真入口与控制器（如 `DPG_main.py`, `DPG_MPC.py`, 轨迹生成）。
- `DPG_real/`：实机 UDP/状态相关工具；`real_robot_in_mujoco/` 复用仿真流程并可接入实时底盘数据。
- `fetch_freight_mujoco/`：MJCF 资产与场景（`xml/`、`meshes/`、`images/`）。
- `CLAUDE.md`：架构说明与阶段性记录，便于理解整体路线。

## 构建、测试与开发命令
本仓库没有统一构建脚本，直接运行入口文件。
- `python DPG_mujoco/DPG_main.py`：运行仿真 MPC 主流程。
- `python DPG_real/real_robot_in_mujoco/DPG_main.py`：仿真接入实时底盘轨迹（需机器人在线）。
- 如需离线运行，可在 `DPG_real/real_robot_in_mujoco/DPG_main.py` 中将 `USE_REAL_BASE = False`。
- `python -m mujoco.viewer --mjcf fetch_freight_mujoco/xml/scene.xml`：打开场景查看器。
- `pip install mujoco`：安装 Mujoco 依赖（见 `fetch_freight_mujoco/README.md`）。

## 编码风格与命名约定
- Python 使用 4 空格缩进，控制器中普遍使用类型注解。
- 函数/变量用 `snake_case`，类用 `CamelCase`。
- 文件名保持 `DPG_*.py` 风格，避免随意改名导致入口丢失。
- 未配置格式化工具；如需统一格式，建议使用 `black` 并保持最小改动。

## 测试指南
- 当前没有自动化测试目录或框架。
- 若新增测试，建议使用 `pytest`，命名为 `test_*.py`，并集中在 `tests/` 目录。
- 运行命令：`pytest`。

## 提交与 PR 规范
- 当前目录未检测到 `.git` 历史，无法推断既有提交规范。
- 建议提交信息使用祈使句短标题（如 “Add base trajectory clamp”），正文写明改动理由与验证命令。
- PR 建议包含：简要说明、影响模块、验证步骤；涉及 MJCF 或可视化请附截图。

## 配置与安全注意事项
- 实机通信参数在 `DPG_real/get_robot_status.py`（`ROBOT_IP`, `ROBOT_PORT`, `AUTH_CODE`），运行前务必核对。
- `DPG_real/real_robot_in_mujoco/real_base_udp_kf.py` 默认使用 `fetch_freight_mujoco/xml/scene.xml` 初始化底盘原点，切换场景需同步修改。

## 优化与创新点（建议补充）
- 实时数据融合与时延补偿：对 UDP 数据加时间戳，采用多速率 KF/EKF 或延迟对齐。
- 预测—控制解耦增强：将轨迹模型抽象为可插拔接口，便于切换 CV/CTRV 等运动模型。
- 鲁棒约束 MPC：明确关节/速度/力矩约束，引入碰撞余量与能耗权重，并支持 warm-start。
- 仿真—实机一致性评估：增加日志与指标（跟踪误差、成功率、延迟），形成可复现实验流程。

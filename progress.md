# Progress

- Added stochastic MPC parameters and default enablement in `DPG_mujoco_final copy/DPG_main.py`.
- Added chance-constraint backoff and tube-style execution correction in `DPG_mujoco_final copy/DPG_MPC.py`.
- Added explicit closed-loop covariance propagation in `DPG_mujoco_final copy/DPG_track_ball_in_robot.py`.
- Added reusable benchmark script `DPG_mujoco_final copy/compare_stochastic_mpc.py`.
- Verified syntax with `python -m py_compile`.
- Verified short headless smoke test; `stochastic_cov_mode` correctly switches to `closed_loop`.
- Found and fixed a regression in `DPG_mujoco_final copy/DPG_MPC.py`:
  - stochastic-disabled QP path was incorrectly passing through `_limit_twist_cmd(...)`
  - stochastic-disabled covariance lookup no longer matched the original time base
- Re-ran headless validation:
  - base MPC behavior is restored and matches the original `DPG_mujoco_final`
  - stochastic version is stable when `stochastic_use_tube_feedback=False`
  - default main entry now keeps chance backoff enabled but disables tube feedback by default
- Re-ran multi-seed comparison:
  - both basic and stochastic now achieve `success_rate=1.0`
  - stochastic provides safety margin and slightly smaller minimum/final error
  - stochastic is slightly slower on this scenario and does not provide a speed win

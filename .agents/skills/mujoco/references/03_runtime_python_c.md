# Runtime APIs: C and Python

## C runtime lifecycle

Typical flow:

1. Build `mjModel`:
- `mj_loadXML(...)` or `mj_loadModel(...)` or `mj_compile(spec, ...)`
2. Allocate state workspace:
- `mj_makeData(model)`
3. Step simulation:
- `mj_step(model, data)` in loop
4. Clean up:
- `mj_deleteData(data)` then `mj_deleteModel(model)`

## Control timing

- `mj_step` computes forward dynamics then integrates one step.
- If control needs up-to-date intermediate quantities, use:
  - `mj_step1` -> set control -> `mj_step2`
- Caveat: `mj_step1/2` split is for single-step integrators (not full RK4 behavior).

## State/control arrays to remember

- State core: `qpos`, `qvel`, `act`
- Control input: `ctrl`
- External force inputs: `qfrc_applied`, `xfrc_applied`
- Sensor output: `sensordata`

## Python minimal pattern

```python
import mujoco

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)

for _ in range(1000):
    # Optional control assignment before step
    # data.ctrl[:] = ...
    mujoco.mj_step(model, data)
```

## Python viewer notes

Viewer modes:

- managed blocking: `viewer.launch(...)`
- standalone app: `python -m mujoco.viewer --mjcf=...`
- passive non-blocking: `viewer.launch_passive(...)`

Passive mode essentials:

- call `viewer.sync()` regularly
- use `viewer.lock()` before mutating shared model/data state
- custom debug geoms can be added to `viewer.user_scn`

## Common API distinction

- `mjModel`: mostly static model definition + options.
- `mjData`: mutable simulation state and intermediate results.

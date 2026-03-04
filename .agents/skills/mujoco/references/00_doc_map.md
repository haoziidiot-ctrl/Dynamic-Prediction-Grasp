# MuJoCo Official Doc Map

Primary entry points:

- Homepage: https://mujoco.org/
- Docs root: https://mujoco.readthedocs.io/en/latest/

Core sections for daily work:

1. Overview
- https://mujoco.readthedocs.io/en/latest/overview.html
- Use for: terminology (`mjSpec`, `mjModel`, `mjData`), scope, modeling philosophy.

2. Modeling
- https://mujoco.readthedocs.io/en/latest/modeling.html
- Use for: kinematic tree, defaults classes, coordinate frames, solver/contact parameter meaning.

3. XML Reference (MJCF)
- https://mujoco.readthedocs.io/en/latest/XMLreference.html
- Use for: exact element/attribute syntax.
- Most relevant groups:
  - `mujoco`, `option`, `compiler`, `size`, `asset`
  - `(world)body`, `contact`, `equality`, `tendon`, `actuator`, `sensor`, `keyframe`, `default`, `extension`

4. Computation
- https://mujoco.readthedocs.io/en/latest/computation/index.html
- Use for: equations, constraints, solver and contact interpretation.

5. Programming / Simulation
- https://mujoco.readthedocs.io/en/latest/programming/simulation.html
- Use for: load/compile APIs, `mj_step`, `mj_step1/mj_step2`, state-control handling.

6. Programming / Model Editing (mjSpec)
- https://mujoco.readthedocs.io/en/latest/programming/modeledit.html
- Use for: creating/editing model structures in code (`mjs_add*`, `mj_compile`, `mj_recompile`).

7. Python
- https://mujoco.readthedocs.io/en/latest/python.html
- Use for: Python binding usage, viewer modes, `MjModel`/`MjData` patterns.

8. API Reference
- https://mujoco.readthedocs.io/en/latest/APIreference/index.html
- Use for: exact types, functions, globals.

Quick routing hints:

- "XML 怎么写 / 属性啥意思" -> XML Reference
- "为什么不稳定 / 参数怎么调" -> Modeling + Computation
- "循环怎么写 / 控制何时更新" -> Programming/Simulation
- "Python viewer 怎么用" -> Python
- "代码动态加元素" -> Model Editing

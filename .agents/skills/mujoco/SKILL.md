---
name: mujoco
description: MuJoCo expert skill for MJCF syntax, model construction, simulation loops, Python/C APIs, and model editing. Use this whenever a user asks about MuJoCo XML elements or attributes (body/joint/geom/site, actuator/sensor/contact/equality/tendon/default), creating or debugging MuJoCo models, building simulation control loops, using `mujoco.viewer`, or editing models with `mjSpec`.
---

# MuJoCo Skill

Use this skill when the user needs precise MuJoCo guidance, especially for MJCF authoring and simulation implementation.

## Source Policy

- Treat official docs as source of truth:
  - `https://mujoco.org/`
  - `https://mujoco.readthedocs.io/en/latest/`
- Prefer `latest` docs unless the user specifies a fixed version.
- If behavior may be version-dependent, state the assumption explicitly.

## What This Skill Covers

- MJCF structure and element syntax.
- Element creation and wiring: `worldbody/body/joint/geom/site`.
- Constraint and control subsystems: `contact`, `equality`, `tendon`, `actuator`, `sensor`.
- Runtime simulation loops (`mj_step`, `mj_step1/mj_step2`) in C/Python.
- Python bindings and viewer usage.
- Programmatic model editing with `mjSpec` (`mjs_add*`, compile/save/recompile).
- Practical debugging and stability tuning.

## Workflow

1. Classify the question:
   - `syntax/modeling`
   - `runtime api`
   - `debug/perf`
   - `mjspec editing`
2. Load only the relevant reference file(s) from `references/`.
3. Answer with this structure:
   - What to write/do
   - Why it works in MuJoCo
   - Minimal snippet
   - Common mistakes
   - How to verify
4. When user asks for implementation, modify project files directly and validate.

## Reference Files

- `references/00_doc_map.md`: Official doc map and routing.
- `references/01_mjcf_core.md`: Core MJCF grammar and element semantics.
- `references/02_elements_controls_constraints.md`: Actuator/sensor/contact/equality/tendon details.
- `references/03_runtime_python_c.md`: Simulation loop, state/control, Python viewer/API.
- `references/04_mjspec_model_editing.md`: Programmatic model editing workflow.
- `references/05_debugging_and_tuning.md`: Diagnostics, stability, and tuning checklist.
- `references/06_templates_and_recipes.md`: Reusable templates and quick recipes.

## Local Lookup Command

Use the helper script when you need quick topic routing or keyword search:

```bash
python .agents/skills/mujoco/scripts/mujoco_lookup.py --list
python .agents/skills/mujoco/scripts/mujoco_lookup.py "position actuator"
python .agents/skills/mujoco/scripts/mujoco_lookup.py "contact pair anisotropic friction" --snippets 8
```

The script returns:
- best-matching topics
- canonical docs URLs
- matching lines in local references

## Output Requirements

When answering MuJoCo questions, prefer concrete operational output:

- Include exact element/function names.
- Provide a minimal working snippet whenever possible.
- Avoid vague advice like "tune solver" without naming candidate parameters.
- For debugging, include a short ordered checklist.

## Safety and Accuracy Rules

- Do not invent attributes or element names not in MJCF.
- Distinguish compile-time options (`compiler`) from runtime options (`option`/`mjModel.opt`).
- Distinguish model description (`mjModel`) from simulation state (`mjData`).
- State when a suggestion is heuristic versus guaranteed by docs.

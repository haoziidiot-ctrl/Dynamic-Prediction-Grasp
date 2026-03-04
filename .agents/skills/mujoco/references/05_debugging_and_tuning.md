# Debugging and Tuning Checklist

## 1) Model authoring issues

- Missing names on objects later referenced by actuator/sensor.
- Wrong frame assumptions (local vs world).
- Using `limited/range` inconsistently with `compiler/autolimits` policy.
- Inertial mismatch when relying on `inertiafromgeom` unexpectedly.

## 2) Collision behavior issues

- Wrong `contype/conaffinity` masks causing missing/excess contacts.
- Overly broad candidate contact set increasing cost/noise.
- Need explicit `contact/pair` for critical collision pairs.

## 3) Instability / divergence

First knobs to inspect:

- `option/timestep`
- `option/integrator` (`implicit` or `implicitfast` often stabilizes stiff systems)
- solver settings (`solver`, `iterations`, `ls_iterations`)
- joint damping and actuator gains

## 4) Control pipeline mistakes

- Forgetting to update `ctrl` each step.
- Computing feedback on stale intermediates when not using callback or `step1/step2`.
- Mixing state dimensions (`nq` vs `nv`).

## 5) Memory-related runtime errors

- `size/memory` too small for contact-rich scenes (for `mjData` variable allocations).
- Increase memory budget before large-contact experiments.

## 6) Verification routine

1. Compile cleanly from a minimal XML.
2. Disable actuators and validate passive dynamics.
3. Re-enable actuator blocks one by one.
4. Add constraints/sensors incrementally.
5. Track key arrays (`qpos`, `qvel`, `ctrl`, `sensordata`) for sanity.

## 7) Practical quality rules

- Keep XML modular via `include` for complex models.
- Centralize repeated attributes with `default` classes.
- Maintain a small, deterministic reset keyframe for tests.

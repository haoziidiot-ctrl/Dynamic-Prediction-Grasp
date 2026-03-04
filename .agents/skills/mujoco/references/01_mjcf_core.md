# MJCF Core: Structure and Semantics

## 1) Canonical MJCF skeleton

```xml
<mujoco model="my_model">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

  <default>
    <geom type="capsule" rgba="0.7 0.7 0.7 1"/>
    <joint damping="1"/>
  </default>

  <asset>
    <mesh name="part" file="part.stl"/>
  </asset>

  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="link1" pos="0 0 0.2">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <site name="tool" pos="0.2 0 0" size="0.005"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="j1_motor" joint="j1" gear="1"/>
  </actuator>
</mujoco>
```

## 2) Kinematic tree essentials

- `worldbody` is the root frame origin.
- Nested `body` elements define parent-child kinematics directly.
- `joint` inside a `body` creates DoF between the body and its parent.
- No joint in a body means welded to parent.

## 3) Coordinates and orientations

- Tree element poses are local to parent/body frame.
- Orientation representations are converted internally to quaternion.
- `compiler/angle` controls angle unit in XML (compile-time interpretation).

## 4) Defaults classes (`default`)

- Defaults reduce repetition and improve readability.
- Defaults can be nested and inherited.
- Typical usage:
  - top-level defaults class for global style
  - `childclass` for subtree policy
  - explicit attribute on element to override

## 5) `option` vs `compiler`

- `compiler`: parse/compile-time behavior only.
- `option`: runtime simulation parameters (`mjModel.opt`).

Examples:

- compile-time: `autolimits`, `meshdir`, `texturedir`, `inertiafromgeom`
- runtime: `timestep`, `integrator`, `solver`, `iterations`, `gravity`

## 6) High-frequency element meanings

- `body`: rigid frame in tree.
- `joint`: DoF definition (`free`, `ball`, `slide`, `hinge`).
- `geom`: collision + rendering primitive/mesh, can also contribute inertia.
- `site`: marker/anchor/sensor reference (no collision/inertia).
- `asset`: reusable mesh/texture/material/model resources.

## 7) Good default authoring pattern

1. Start with minimal model that compiles.
2. Add names for all objects used by sensors/actuators/controllers.
3. Add defaults class only after base model runs.
4. Add constraints/actuators incrementally and test after each block.

# Templates and Recipes

## A) Minimal falling body

```xml
<mujoco model="falling_box">
  <worldbody>
    <geom type="plane" size="2 2 0.1"/>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
```

## B) Revolute joint with motor

```xml
<mujoco model="hinge_motor">
  <worldbody>
    <body name="link" pos="0 0 0.2">
      <joint name="j1" type="hinge" axis="0 0 1" damping="0.2"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="u1" joint="j1" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
```

## C) Position + velocity servo (PD-style)

```xml
<actuator>
  <position name="j_pos" joint="j1" kp="100"/>
  <velocity name="j_vel" joint="j1" kv="10"/>
</actuator>
```

## D) Contact pair override

```xml
<contact>
  <pair name="grip_pair" geom1="finger_left" geom2="object" friction="1.0 0.02 0.001"/>
</contact>
```

## E) Sensor examples

```xml
<sensor>
  <jointpos joint="j1"/>
  <jointvel joint="j1"/>
  <force site="fts_site"/>
  <rangefinder site="rf_site"/>
</sensor>
```

## F) Python loop with direct control

```python
import mujoco

m = mujoco.MjModel.from_xml_path("robot.xml")
d = mujoco.MjData(m)

while d.time < 5.0:
    # Example: hold first actuator command at 0.1
    if m.nu > 0:
        d.ctrl[0] = 0.1
    mujoco.mj_step(m, d)
```

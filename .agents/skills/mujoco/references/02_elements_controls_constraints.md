# Controls, Constraints, and Key Elements

## Contact

- Group element: `<contact>`
- Typical children:
  - `<pair geom1="..." geom2="..." .../>`
  - `<exclude body1="..." body2="..."/>`

Key points:

- Dynamic contacts infer parameters from geoms.
- `contact/pair` explicitly defines pair properties and can represent anisotropic friction.
- Use explicit pairs to tightly control collision set in complex tasks.

## Equality constraints

- Group element: `<equality>`
- Types include:
  - `connect`
  - `weld`
  - `joint`
  - `tendon`
  - `flex`, `flexvert`, `distance` (version-dependent availability)

Use cases:

- Keep frames connected, lock transforms, enforce mechanical coupling.

## Tendons

- Group element: `<tendon>`
- Main types:
  - `spatial` (path through sites/geoms/pulleys)
  - `fixed` (linear combination of joint coordinates)

Use cases:

- Transmission paths for actuators.
- Length limits, spring/damper/friction effects.
- Coupled mechanisms via equality constraints.

## Actuators

- Group element: `<actuator>`
- Base type: `general` (full control over dyn/gain/bias settings)
- Common shortcuts:
  - `motor`
  - `position`
  - `velocity`
  - `intvelocity`
  - `damper`
  - `cylinder`
  - `muscle`
  - `adhesion`

Practical notes:

- MuJoCo actuators are SISO.
- PD behavior is usually composed by separate position + velocity actuators.
- `intvelocity` is useful for integrated velocity servo behavior with activation dynamics.

## Sensors

- Group element: `<sensor>`
- Outputs are concatenated into `mjData.sensordata`.
- Sensors do not directly modify simulation dynamics.

Common sensor classes:

- Site-attached: `touch`, `accelerometer`, `gyro`, `force`, `torque`, `rangefinder`
- Joint/tendon/actuator readouts: `jointpos`, `jointvel`, `tendonpos`, `actuatorfrc`
- Frame/subtree sensors: `framepos`, `framequat`, `framelinvel`, `subtreecom`

## Keyframes

- Group element: `<keyframe>`
- Useful for reset/library states and reproducible initialization.

## Defaults

- Group element: `<default>`
- Supports per-element defaults (`default/joint`, `default/geom`, `default/motor`, etc.).
- Values are copied during element initialization, not continuously linked.

## Extensions / Plugins

- Group element: `<extension>`
- `extension/plugin` declares required engine plugins.
- Explicit plugin instance + config can be used when sharing plugin config across elements.

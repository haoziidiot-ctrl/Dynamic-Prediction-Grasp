# Programmatic Model Editing with mjSpec

## Why mjSpec

`mjSpec` provides a structured editable representation in one-to-one correspondence with MJCF.
It supports parse-edit-compile workflows and dynamic model updates.

## Standard workflow

1. Create or parse spec:
- `mj_makeSpec()`
- `mj_parseXML(...)`
2. Edit spec fields/elements.
3. Compile:
- `mj_compile(spec, vfs)`
4. Optionally save:
- `mj_saveXML(...)` / `mj_saveXMLString(...)`

## Element creation pattern

- Use constructor helpers, not raw allocation.
- Example pattern:
  - find parent (`mjs_findBody(spec, "world")`)
  - create child (`mjs_addGeom`, `mjs_addBody`, `mjs_addJoint`, ...)
  - set attributes on returned struct

## Defaults in mjSpec

- Defaults classes are supported.
- Defaults are applied when element is created.
- Updating defaults later does not retroactively modify already created elements.

## Recompile options

- `mj_compile`: build fresh model from spec.
- `mj_recompile`: update an existing `mjModel` + `mjData` in place while preserving state when possible.

## Ownership and memory

- Elements belong to the spec.
- Free spec via `mj_deleteSpec`.
- Free compiled model via `mj_deleteModel`.

## Good use cases

- Auto-generating MJCF structures from higher-level templates.
- Online model edits (adding/removing substructures).
- Tooling: converters, model refactoring, and automatic defaults injection.

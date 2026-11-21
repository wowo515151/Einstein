# Migrating from Lib4

This guide helps translate common calls from the locally inspected Python Lib4 API to the JavaScript library. It does not certify complete behavior parity. Consult the [port matrix](port-matrix.md) for known gaps and the [numerical limits](limits.md) before relying on results.

## Imports, arrays, and units

Import named exports from `src/relativity/index.js`; see the runnable [examples](examples.md). Use finite JavaScript numbers and rectangular arrays in place of PyTorch tensors. Coordinates are `[ct,x,y,z]`, with signature `(-,+,+,+)`. The default speed of light and permittivity use SI; supply consistent values explicitly for other units. `new RelativityContext({units:"natural",c:1})` provides factories with natural-unit constants. `setUnits` creates a new context; it does not convert previously created values.

Leading array dimensions are supported and broadcast only for operations that document them. Tensor axes have length four; pass `indexTypes` to `Tensor4` when leading batch axes are present. JavaScript arrays are not PyTorch tensors: dtype, device placement, general PyTorch autograd, CUDA, and PyTorch operators are not provided. Selected field, geometry and ODE operations offer explicit forward differentiation when callbacks use dual-number methods. `toDevice` changes context metadata only. The optional asynchronous [GPU batch APIs](gpu.md) have separate Float32 and CPU-fallback semantics.

## Names and accessors

Python snake_case methods generally map to camelCase. See the [instance field map](lib4-fields.md) for renamed fields and mutation differences. JavaScript uses method calls for several values exposed as Python properties:

| Lib4 name | JavaScript equivalent | Notes |
| --- | --- | --- |
| `Vector4.norm_squared()` | `vector.normSquared()` | Scalar or compatible batch result. |
| `Event4.interval_to(other)` | `event.intervalTo(other)` | Returns `SpacetimeInterval`; `event.displacementTo(other)` returns component arrays. |
| `SpacetimeInterval.interval_squared` | `interval.intervalSquared()` or `interval.squared` | Method and getter expose the squared interval. |
| `FourVelocity.from_velocity3(v,c)` | `FourVelocity.fromVelocity3(v,c)` | The physical velocity must be subluminal. |
| `FourVelocity.gamma` | `velocity.gamma()` | Call `validate()` for externally supplied four-velocities. |
| `FourMomentum.from_mass_velocity(m,v,c)` | `FourMomentum.fromMassVelocity(m,v,c)` | The result stores `(E/c,p)`. |
| `FourMomentum.energy`, `momentum3` | `momentum.energy()`, `momentum.momentum3()` | JavaScript accessors are methods. |
| `FourCurrent.charge_density`, `spatial_current` | `current.chargeDensity()`, `current.spatialCurrent()` | JavaScript accessors are methods. |
| `EMFieldTensor.from_EB(E,B,...)` | `EMFieldTensor.fromEB(E,B,{c,epsilon0})` | Options are a JavaScript object. |
| `Tensor4.lower_index(i)` | `tensor.lowerIndex(i)` | Tensor `rank` remains a read-only property. |
| `Connection4.covariant_derivative(...)` | `connection.covariantDerivative(...)` | Check [geometry](geometry.md) for supported variance and finite-difference behavior. |

## Result shapes and exceptions

- `EMFieldTensor.invariants()` returns an object with `magneticMinusElectricOverC2` and `electricDotMagnetic`; Lib4 returns a tuple. Each JavaScript value retains the field batch shape. `Bivector4.invariants()` instead returns `{quadratic,pseudoscalar}`, also with batched values.
- `SpatialSlice.embed3D()` returns an `Event4` with position and metadata, whereas Lib4 returns a tensor. `Event4.transform()` returns an `Event4`; `PoincareTransform.applyEvent()` returns an object with a `position` array.
- `Geodesic4.integrate()` defaults to sample records containing `parameter`, `event`, and `tangent`. Pass `{format:"components"}` for separate parameter, event, and tangent histories. Boundary shooting in component format also returns the fitted tangent.
- `MaxwellEquations.evolve()` returns `[electricTimes,electricHistory,magneticTimes,magneticHistory]`. Residual methods return named objects; magnetic history is half-step staggered. See [physics](physics.md) for array shapes and periodic boundary assumptions.
- `RetardedField.evaluate()` and `sample()` return the retarded four-potential, not field strengths. Source callbacks return spatial positions and velocities; see the [retarded-field example](examples.md#retarded-field-model).
- JavaScript invalid inputs and unsupported calculations throw `TypeError` or `RangeError`; error classes and messages are not a one-to-one copy of Python exceptions. Dense arrays, grid sizes, batches, and numerical work have explicit limits.

The JavaScript `Tensor4.transform` requires an invertible matrix even for all-upper tensors, as the inspected Lib4 implementation does. The scalar and batched APIs use finite real numbers only; complex tensors and arbitrary PyTorch device/dtype semantics are outside the current port. Other deliberate differences and unresolved comparisons are recorded in the [port matrix](port-matrix.md).

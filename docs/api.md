# API Reference

Import public names from `src/relativity/index.js`. The [examples](examples.md) show executable imports and common operations. These tables guide users to principal methods; follow the topic pages for shapes, conventions, options, numerical limits, and differences from Lib4.

For an index of inspected Lib4 class methods and properties, see the [Lib4 public API inventory](lib4-method-inventory.md). The [instance field map](lib4-fields.md) maps all 70 annotated or constructor-assigned public class/field pairs found in the 28 inspected classes; it excludes dynamically created, inherited, and framework-provided attributes. These inventories map source surfaces, not behavioral parity. Behavioral evidence and intentional differences are recorded in the [port matrix](port-matrix.md).

Coordinates use `[ct,x,y,z]` and the default Minkowski metric has signature `(-,+,+,+)`. Values use plain JavaScript numeric arrays. Many operations support compatible leading batch dimensions, but this is not a general PyTorch replacement; see each topic before relying on broadcasting. Invalid dimensions, non-finite values, singular matrices, superluminal velocities, and invalid causal normalization can throw.

## Algebra

See [Algebra](algebra.md) for component layouts, batching, and metric conventions.

| Type | Construction and operations |
| --- | --- |
| `Vector4` | Construct from four components or a batch. `add`, `subtract`, `scale`, `dot`, and `normSquared` perform arithmetic; `causalType` and `causalName` classify; `normalizeTimelike` and `normalizeSpacelike` require the corresponding causal type. `outer`, `wedge`, `transform`, `lowerIndex`, and `toArray` provide tensor and conversion operations. |
| `Covector4` | `add`, `scale`, and `pair` operate on covectors; `raiseIndex` uses a metric and `transform` applies a transformation. |
| `Matrix4` | `identity` constructs an identity matrix. `add`, `multiply`, `apply`, `transpose`, `determinant`, and `inverse` provide matrix operations. |
| `Tensor4` | `outer`, `contract`, `permuteIndices`, `raiseIndex`, `lowerIndex`, `symmetrize`, `antisymmetrize`, and `transform` operate on tensors. |
| `Bivector4` | `wedge` constructs a bivector from vectors. `dual`, `contract`, `transform`, and `invariants` operate on it. Invariant values retain the batch shape inside the returned object. |
| `Metric4` | `minkowski` constructs the flat metric. `dot`, `intervalSquared`, `lowerIndex`, `raiseIndex`, `determinant`, and `inverseMetric` expose metric operations; `g` and `inverseComponents` expose components. |
| `LeviCivita4` | `symbol`, `epsilon`, `volumeForm`, and `dual` supply permutation and volume-form operations. |
| `LorentzTransform` | `identity`, `boost`, `fromRapidity`, and `rotation` construct transformations. `validate`, `compose`, `inverse`, `applyVector`, and `applyTensor` check and apply them. |
| `PoincareTransform` | `identity`, `compose`, and `inverse` handle affine transformations; `applyEvent` applies translation and `applyVector` applies the linear part. |

## Spacetime

See [Spacetime types](spacetime.md) for frame metadata, causal conventions, worldline interpolation, and batch limits.

| Type | Construction and operations |
| --- | --- |
| `Event4` | Construct from a four-position. `displacementTo`, `intervalTo`, and `causalRelation` compare events; `transform` returns a transformed event. |
| `SpacetimeInterval` | `squared`, `intervalSquared`, and `tolerance` expose the interval and classification tolerance. `isTimelike`, `isSpacelike`, `isNull`, `properTime`, and `properDistance` classify or measure it. |
| `LightCone` | Construct from an apex and options including `branch`, `c`, and `atol`. `contains`, `onSurface`, `sample`, `intersectWorldline`, and `intersectHypersurface` describe the cone. Hyperplane offsets are relative to its apex. |
| `SpatialSlice` | Construct from an origin and future timelike normal. `projectTo3D`, `embed3D`, `contains`, `intersectWorldline`, and `intersectLightcone` work with the slice. |
| `WorldlineSegment` and `Worldline` | `evaluate` and `tangent` evaluate Hermite segments or a continuous piecewise worldline at a parameter. `Worldline.call` is also available. |
| `ODEWorldline` | `integrate` uses the supplied four-velocity and returns parameter and event histories. |
| `InertialFrame` | `toGlobalTransform`, `fromGlobalTransform`, and `relativeTransform` construct frame maps. `toGlobal`, `fromGlobal`, `transformEvent`, and `transformVector` apply them. |

## Mechanics and fields

See [Physics and fields](physics.md) for units, grid layouts, result shapes, and numerical limitations.

| Type | Construction and operations |
| --- | --- |
| `FourVelocity` | `fromVelocity3(v,c)` constructs a value from a subluminal three-velocity. `gamma()`, `toVelocity3()`, `relativeVelocity(other)`, and `validate()` inspect it. |
| `FourMomentum` | `fromMassVelocity(m,v,c)` constructs momentum. `energy()`, `momentum3()`, `invariantMass()`, and `boost` compute derived values. |
| `FourCurrent` | `fromChargeCurrent(rho,j,{c})` constructs a current. `chargeDensity()`, `spatialCurrent()`, and `transform` inspect or transform it. |
| `EMFieldTensor` | `fromEB(E,B,{c,epsilon0})` constructs a field. `electricField`, `magneticField`, `dual`, `transform`, `lorentzForce`, and `energyMomentumTensor` operate on it. `invariants()` returns named values `magneticMinusElectricOverC2` and `electricDotMagnetic`. |
| `MaxwellEquations` | `evolve(E,B,{spacing,dt,steps,current})` returns `[electricTimes,electricHistory,magneticTimes,magneticHistory]`. `gridCoordinates` provides staggered sample positions, while `collocateFields` and `collocateHistory` align staggered samples at nodes and electric times; `maxwellResiduals`, `constraintResiduals`, and `energyMomentumTensor` provide checks and derived quantities. |
| `RetardedField` | Construct from position and velocity functions, charge, a speed bound, and options. `retardedTime` finds the source time; `retardedPotential`, `evaluate`, and `sample` return four-potential values. |

## Curved geometry

See [Differential geometry](geometry.md) for metric callbacks, finite-difference steps, index conventions, and integration limits. Numerical convergence must be checked for the application.

| Type | Operations |
| --- | --- |
| `DifferentialOperators4` | `partial`, `jacobian`, `gradient`, `divergence`, `exteriorDerivative`, `hodgeDual`, and `dAlembertian` evaluate field derivatives and related operators. `partialForward`, `jacobianForward`, `gradientForward`, `divergenceForward`, and `exteriorDerivativeForward` offer explicit first-order dual-number differentiation; `tensorHessianForward` supports explicit second-order rectangular fields; `hessianForward` and `dAlembertianForward` support scalar fields. |
| `Connection4` | `christoffel`, `covariantDerivativeVector`, `covariantDerivative`, `parallelTransport`, and `geodesicAcceleration` use the metric connection; `christoffelForward` accepts an explicitly dual-compatible metric, and `{metricDerivative:"forward"}` selects it for connection workflows. `covariantDerivativeForward` differentiates an explicitly dual-compatible field. |
| `Curvature4` | `riemann`, `ricci`, `ricciScalar`, `einsteinTensor`, and `kretschmannScalar` compute curvature quantities. |
| `Geodesic4` | `integrate` produces samples along an initial-value geodesic; `solveBoundary` shoots for an endpoint. The `components` format returns separate parameter, event, and tangent histories. |

## Context and acceleration

`RelativityContext` groups a metric, units, `c`, and `epsilon0` with factories for the common algebra, spacetime, and physics values. Its `setUnits` and `toDevice` return configured contexts; they do not convert existing values or transfer JavaScript arrays to a device. See [Physics and fields](physics.md).

`minkowskiDotBatch` offers an asynchronous CPU or optional WebGPU path for flat batches of four-vectors. `spacetimeIntervalSquaredBatch` applies the same backend to displacements between paired event batches. `lorentzTransformBatch` applies one validated Lorentz matrix to a flat vector batch through four screened dot batches. `WebGPUBackend` exposes `supported`, `initialize`, `minkowskiDotBatch`, and `dispose` for explicit backend reuse. See [GPU and CPU backends](gpu.md) for fallback behavior, precision, and ownership.

`samplePlaneWaveBatch` samples normalized electric and magnetic plane-wave fields at flat batches of four-coordinate events, using the CPU or the optional WebGPU backend. Its wave model, supported GPU envelope, return values, and fallback behavior are described in [GPU and CPU backends](gpu.md).

`samplePlaneWaveGrid` builds a bounded rectangular spatial grid and samples the same model at one coordinate time, returning flat events and fields with z varying fastest. See [GPU and CPU backends](gpu.md) for its options and limits.

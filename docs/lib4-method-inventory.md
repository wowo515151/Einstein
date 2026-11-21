# Lib4 Public API Inventory

This inventory was generated from the locally inspected `Lightlike/Lib4` Python source tree on 2026-11-21. It maps 168 named public methods and properties and seven selected data fields across 28 public classes, plus nine Python operator/call protocol methods, to corresponding JavaScript API names. It also records the two top-level Lib4 exports. The separate [instance field map](lib4-fields.md) covers the declared and constructor-assigned public fields. It does not copy or execute Python code.

This is a name and surface inventory, not proof of behavioral parity. See the [port and test matrix](port-matrix.md) for per-module evidence, intentional differences, numerical limits, and unresolved behavior. In particular, Lib4's `Vector4.device` and `Vector4.dtype` expose PyTorch tensor metadata and have no equivalent for JavaScript number arrays. `RelativityContext.device` is only a descriptive setting and does not transfer arrays to a device.

| Lib4 module | Class | Lib4 public surface → JavaScript surface |
| --- | --- | --- |
| `bivector4.py` | `Bivector4` | `wedge()` → `wedge()`<br>`dual()` → `dual()`<br>`contract()` → `contract()`<br>`transform()` → `transform()`<br>`invariants()` → `invariants()` |
| `connection4.py` | `Connection4` | `christoffel()` → `christoffel()`<br>`covariant_derivative_vector()` → `covariantDerivativeVector()`<br>`covariant_derivative()` → `covariantDerivative()`<br>`parallel_transport()` → `parallelTransport()`<br>`geodesic_acceleration()` → `geodesicAcceleration()` |
| `covector4.py` | `Covector4` | `add()` → `add()`<br>`scale()` → `scale()`<br>`pair()` → `pair()`<br>`raise_index()` → `raiseIndex()`<br>`transform()` → `transform()` |
| `curvature4.py` | `Curvature4` | `riemann()` → `riemann()`<br>`ricci()` → `ricci()`<br>`ricci_scalar()` → `ricciScalar()`<br>`einstein_tensor()` → `einsteinTensor()`<br>`kretschmann_scalar()` → `kretschmannScalar()` |
| `differential_operators4.py` | `DifferentialOperators4` | `partial()` → `partial()`<br>`jacobian()` → `jacobian()`<br>`gradient()` → `gradient()`<br>`divergence()` → `divergence()`<br>`exterior_derivative()` → `exteriorDerivative()`<br>`hodge_dual()` → `hodgeDual()`<br>`d_alembertian()` → `dAlembertian()` |
| `em_field_tensor.py` | `EMFieldTensor` | `from_EB()` → `fromEB()`<br>`electric_field()` → `electricField()`<br>`magnetic_field()` → `magneticField()`<br>`invariants()` → `invariants()`<br>`dual()` → `dual()`<br>`transform()` → `transform()`<br>`lorentz_force()` → `lorentzForce()`<br>`energy_momentum_tensor()` → `energyMomentumTensor()` |
| `event4.py` | `Event4` | `displacement_to()` → `displacementTo()`<br>`interval_to()` → `intervalTo()`<br>`causal_relation()` → `causalRelation()`<br>`transform()` → `transform()` |
| `four_current.py` | `FourCurrent` | `from_charge_current()` → `fromChargeCurrent()`<br>`charge_density` (property) → `chargeDensity()`<br>`spatial_current` (property) → `spatialCurrent()`<br>`transform()` → `transform()` |
| `four_momentum.py` | `FourMomentum` | `p` (field) → `p`<br>`from_mass_velocity()` → `fromMassVelocity()`<br>`energy` (property) → `energy()`<br>`momentum3` (property) → `momentum3()`<br>`invariant_mass()` → `invariantMass()`<br>`boost()` → `boost()` |
| `four_velocity.py` | `FourVelocity` | `u` (field) → `u`<br>`from_velocity3()` → `fromVelocity3()`<br>`gamma` (property) → `gamma()`<br>`to_velocity3()` → `toVelocity3()`<br>`relative_velocity()` → `relativeVelocity()`<br>`validate()` → `validate()` |
| `geodesic4.py` | `Geodesic4` | `integrate()` → `integrate()`<br>`solve_boundary()` → `solveBoundary()` |
| `inertial_frame.py` | `InertialFrame` | `to_global_transform` (property) → `toGlobalTransform`<br>`from_global_transform` (property) → `fromGlobalTransform`<br>`to_global()` → `toGlobal()`<br>`from_global()` → `fromGlobal()`<br>`transform_event()` → `transformEvent()`<br>`transform_vector()` → `transformVector()`<br>`relative_transform()` → `relativeTransform()` |
| `levi_civita4.py` | `LeviCivita4` | `symbol()` → `symbol()`<br>`epsilon()` → `epsilon()`<br>`volume_form()` → `volumeForm()`<br>`dual()` → `dual()` |
| `light_cone.py` | `LightCone` | `on_surface()` → `onSurface()`<br>`contains()` → `contains()`<br>`intersect_worldline()` → `intersectWorldline()`<br>`intersect_hypersurface()` → `intersectHypersurface()`<br>`sample()` → `sample()` |
| `lorentz_transform.py` | `LorentzTransform` | `identity()` → `identity()`<br>`boost()` → `boost()`<br>`from_rapidity()` → `fromRapidity()`<br>`rotation()` → `rotation()`<br>`compose()` → `compose()`<br>`inverse()` → `inverse()`<br>`apply_vector()` → `applyVector()`<br>`apply_tensor()` → `applyTensor()`<br>`validate()` → `validate()` |
| `matrix4.py` | `Matrix4` | `identity()` → `identity()`<br>`add()` → `add()`<br>`multiply()` → `multiply()`<br>`transpose()` → `transpose()`<br>`inverse()` → `inverse()`<br>`determinant()` → `determinant()`<br>`apply()` → `apply()` |
| `maxwell_equations.py` | `MaxwellEquations` | `evolve()` → `evolve()`<br>`maxwell_residuals()` → `maxwellResiduals()`<br>`constraint_residuals()` → `constraintResiduals()`<br>`energy_momentum_tensor()` → `energyMomentumTensor()` |
| `metric4.py` | `Metric4` | `minkowski()` → `minkowski()`<br>`inverse` (property) → `inverse` (read-only property)<br>`determinant()` → `determinant()`<br>`inverse_metric()` → `inverseMetric()`<br>`dot()` → `dot()`<br>`interval_squared()` → `intervalSquared()`<br>`lower_vector()` → `lowerVector()`<br>`raise_covector()` → `raiseCovector()`<br>`lower_index()` → `lowerIndex()`<br>`raise_index()` → `raiseIndex()` |
| `ode_worldline.py` | `ODEWorldline` | `integrate()` → `integrate()` |
| `poincare_transform.py` | `PoincareTransform` | `identity()` → `identity()`<br>`apply_event()` → `applyEvent()`<br>`apply_vector()` → `applyVector()`<br>`compose()` → `compose()`<br>`inverse()` → `inverse()` |
| `relativity_context.py` | `RelativityContext` | `create_vector()` → `createVector()`<br>`create_event()` → `createEvent()`<br>`create_covector()` → `createCovector()`<br>`create_tensor()` → `createTensor()`<br>`create_spacetime_interval()` → `createSpacetimeInterval()`<br>`create_inertial_frame()` → `createInertialFrame()`<br>`create_light_cone()` → `createLightCone()`<br>`create_spatial_slice()` → `createSpatialSlice()`<br>`create_four_velocity()` → `createFourVelocity()`<br>`four_velocity_from_velocity3()` → `fourVelocityFromVelocity3()`<br>`create_four_momentum()` → `createFourMomentum()`<br>`four_momentum_from_mass_velocity()` → `fourMomentumFromMassVelocity()`<br>`create_em_field_tensor()` → `createEMFieldTensor()`<br>`create_four_current()` → `createFourCurrent()`<br>`create_maxwell_equations()` → `createMaxwellEquations()`<br>`create_retarded_field()` → `createRetardedField()`<br>`to_device()` → `toDevice()`<br>`set_units()` → `setUnits()` |
| `retarded_field.py` | `RetardedField` | `retarded_time()` → `retardedTime()`<br>`retarded_potential()` → `retardedPotential()`<br>`evaluate()` → `evaluate()`<br>`sample()` → `sample()` |
| `spacetime_interval.py` | `SpacetimeInterval` | `interval_squared` (property) → `intervalSquared()` / `squared`<br>`is_timelike()` → `isTimelike()`<br>`is_spacelike()` → `isSpacelike()`<br>`is_null()` → `isNull()`<br>`proper_time()` → `properTime()`<br>`proper_distance()` → `properDistance()` |
| `spatial_slice.py` | `SpatialSlice` | `project_to_3d()` → `projectTo3D()`<br>`embed_3d()` → `embed3D()`<br>`contains()` → `contains()`<br>`intersect_worldline()` → `intersectWorldline()`<br>`intersect_lightcone()` → `intersectLightcone()` |
| `tensor4.py` | `Tensor4` | `rank` (property) → `rank` (read-only property)<br>`outer()` → `outer()`<br>`lower_index()` → `lowerIndex()`<br>`raise_index()` → `raiseIndex()`<br>`permute_indices()` → `permuteIndices()`<br>`contract()` → `contract()`<br>`symmetrize()` → `symmetrize()`<br>`antisymmetrize()` → `antisymmetrize()`<br>`transform()` → `transform()` |
| `vector4.py` | `Vector4` | `shape` (property) → `shape` (read-only property)<br>`device` (property) → No direct JavaScript-array equivalent<br>`dtype` (property) → No direct JavaScript-array equivalent<br>`add()` → `add()`<br>`subtract()` → `subtract()`<br>`scale()` → `scale()`<br>`dot()` → `dot()`<br>`norm_squared()` → `normSquared()`<br>`causal_type()` → `causalType()`<br>`normalize_timelike()` → `normalizeTimelike()`<br>`normalize_spacelike()` → `normalizeSpacelike()`<br>`outer()` → `outer()`<br>`wedge()` → `wedge()`<br>`transform()` → `transform()`<br>`lower_index()` → `lowerIndex()` |
| `worldline.py` | `Worldline` | `evaluate()` → `evaluate()`<br>`tangent()` → `tangent()` |
| `worldline_segment.py` | `WorldlineSegment` | `start_event` (field) → `startEvent` (coordinate snapshot)<br>`end_event` (field) → `endEvent` (coordinate snapshot)<br>`start_four_velocity` (field) → `startFourVelocity` (coordinate snapshot)<br>`end_four_velocity` (field) → `endFourVelocity` (coordinate snapshot)<br>`parameter_span` (field) → `parameterSpan` (copy)<br>`evaluate()` → `evaluate()`<br>`tangent()` → `tangent()` |

## Operator and callable protocols

The Python protocols below map to explicit JavaScript methods; JavaScript does not overload arithmetic or call syntax for these objects. Python's generated `Vector4.__repr__` is intentionally excluded because it is a display representation, not a numerical operation, and there is no matching custom JavaScript representation method.

| Lib4 protocol | JavaScript named equivalent |
| --- | --- |
| `Vector4.__add__` | `Vector4.add()` |
| `Vector4.__sub__` | `Vector4.subtract()` |
| `Vector4.__mul__` | `Vector4.scale()` |
| `Vector4.__rmul__` | `Vector4.scale()` |
| `Covector4.__add__` | `Covector4.add()` |
| `Covector4.__mul__` | `Covector4.scale()` |
| `Covector4.__rmul__` | `Covector4.scale()` |
| `Matrix4.__matmul__` | `Matrix4.multiply()` for matrices; `Matrix4.apply()` for vectors |
| `Worldline.__call__` | `Worldline.call()` |

## Top-level exports

| Lib4 export | JavaScript export |
| --- | --- |
| `SPEED_OF_LIGHT` | `SPEED_OF_LIGHT` |
| `minkowski_dot(a,b)` | `minkowskiDot(a,b)` |

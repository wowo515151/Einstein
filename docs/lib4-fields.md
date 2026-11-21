# Lib4 Instance Field Map

This map covers public class-level annotated fields and public attributes assigned directly in constructors of the 28 public classes in the locally inspected Lib4 source. A static syntax-tree comparison on 2026-11-21 found all 70 such class/field pairs represented in the source-field column; it did not import or execute Lib4. It is a source-to-surface map, not a claim that mutations, types, validation, or computations are behaviorally identical. Attributes created dynamically outside constructors and inherited or framework-provided attributes are not included. Python tensors and JavaScript arrays have different dtype, device, and mutation semantics.

Names in the right column refer to JavaScript instances; multiple names on a row correspond in order. Consult the [port matrix](port-matrix.md) for method and numerical limitations and the [method inventory](lib4-method-inventory.md) for callable surfaces.

| Lib4 class | Public source fields | JavaScript mapping and qualifications |
| --- | --- | --- |
| `Vector4`, `Covector4`, `Matrix4`, `Bivector4` | `components` | `components` |
| `Tensor4` | `components`, `index_types` | `components`, `indexTypes`; the initial index-type array is frozen. |
| `Metric4` | `g` | `g` aliases `components`; assignment checks the existing batch shape. |
| `LeviCivita4` | `metric`, `orientation` | `metric`, `orientation` |
| `LorentzTransform` | `matrix` | `matrix`; instance is shallowly frozen. |
| `PoincareTransform` | `lorentz`, `translation` | `lorentz`, `translation`; instance is shallowly frozen. |
| `Event4` | `position`, `frame`, `c` | `position`, `frame`, `c`; instance is shallowly frozen. |
| `SpacetimeInterval` | `displacement`, `c`, `atol` | `displacement`, `c`, `atol`; instance is shallowly frozen. |
| `LightCone` | `apex`, `branch`, `c`, `atol` | `apex`, `branch`, `c`, `atol`; instance is shallowly frozen. |
| `SpatialSlice` | `origin`, `timelike_normal`, `spatial_basis`, `c`, `atol` | `origin`, `normal`, `basis`, `c`, `atol`; batched `origin` is an array rather than an Event4. |
| `InertialFrame` | `origin`, `orientation`, `relative_velocity`, `c` | `origin`, `orientation`, `relativeVelocity`, `c`; instance is shallowly frozen. |
| `WorldlineSegment` | `start_event`, `end_event`, `start_four_velocity`, `end_four_velocity`, `parameter_span` | `startEvent`, `endEvent`, `startFourVelocity`, `endFourVelocity`, `parameterSpan`; getters return detached arrays and setters validate replacements. `start` and `end` retain event metadata. |
| `Worldline` | `segments` | `segments`; the initial array copy is frozen, while segment objects remain mutable. Existing joins are not revalidated after mutation. |
| `ODEWorldline` | `initial_event`, `four_velocity` | `initialEvent`, `fourVelocity`; the constructor wraps raw initial events in Event4. Later replacements may be Event4 or coordinate arrays, and integration validates them. |
| `FourVelocity` | `u`, `c` | `u`, `c`; `u` aliases `components`. |
| `FourMomentum` | `p`, `c` | `p`, `c`; `p` aliases `components`. |
| `FourCurrent` | `components`, `c` | `components`, `c`; instance is shallowly frozen. |
| `EMFieldTensor` | `components`, `c`, `epsilon0` | `components`, `c`, `epsilon0`; instance is shallowly frozen. |
| `MaxwellEquations` | `c`, `epsilon0` | `c`, `epsilon0`; instance is shallowly frozen. |
| `RetardedField` | `source_worldline`, `source_velocity`, `charge`, `max_speed`, `c`, `epsilon0`, `max_iterations`, `tolerance` | `sourceWorldline`, `sourceVelocity`, `charge`, `maxSpeed`, `c`, `epsilon0`, `maxIterations`, `tolerance` |
| `DifferentialOperators4`, `Connection4`, `Curvature4` | `metric` | `metric`; JavaScript also exposes numerical differentiation configuration. |
| `Geodesic4` | `connection`, `initial_event`, `initial_tangent` | `connection`, `initialEvent`, `initialTangent`; constructor event coordinates are arrays with Event4 metadata stored separately. A replacement may be an Event4 or a coordinate array. |
| `RelativityContext` | `metric`, `c`, `dtype`, `device`, `units`, `epsilon0` | `metric`, `c`, no dtype equivalent, `device`, `units`, `epsilon0`. The JavaScript device label does not transfer arrays. |

Shallow freezing prevents field reassignment, not mutation of contained arrays. Field presence or a matching name is not evidence of complete Lib4 parity; numerical and mutation contracts still require comparison and testing.

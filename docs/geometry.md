# Differential Geometry

Pass a metric function `g(x)` returning a nonsingular 4×4 covariant matrix to `DifferentialOperators4`, `Connection4`, or `Curvature4`. `DifferentialOperators4` also accepts a constant 4×4 metric or defaults to Minkowski space. The default derivatives use centered finite differences with configurable positive `step`; they divide by the actual representable distance between stencil points and reject a step that cannot move to distinct finite points on both sides. Scalar derivative samples and results must remain finite, so a stencil that crosses a field's coordinate-domain boundary fails explicitly.

## Differential operators

Each callback accepts a single four-coordinate array. The position argument `x` may have supported leading batch dimensions; the returned arrays retain them. Derivative axes appear last in partial and Jacobian results.

| Call | Returns |
| --- | --- |
| `new DifferentialOperators4(metric,{step,metricDerivative})` | An operator set. The metric may be a function, a constant matrix, or omitted for Minkowski; `step` defaults to `1e-5`. Set `metricDerivative:"forward"` for a dual-compatible metric callback. |
| `ops.partial(field,x)`, `ops.jacobian(field,x)` | Derivatives of a scalar or rectangular array-valued field, with a final axis of four coordinate derivatives. Output arrays must be finite, nonempty, acyclic, and rank eight or less, subject to the work and component bounds. |
| `ops.gradient(scalarField,x)` | A metric-raised four-component gradient. |
| `ops.partialForward(field,x)`, `ops.jacobianForward(field,x)` | First derivatives using explicit `Dual4` operations, with the same derivative-axis and batch conventions. |
| `ops.gradientForward(scalarField,x)` | Metric-raised scalar gradient using explicit first-order forward differentiation. |
| `ops.divergence(vectorField,x)`, `ops.divergenceForward(vectorField,x)` | The covariant divergence as a number or batch; the forward variant differentiates a dual-compatible field and uses the configured metric derivative method. |
| `ops.exteriorDerivative(form,x)`, `ops.exteriorDerivativeForward(form,x)` | Components of the exterior derivative of a covariant form; the latter uses an explicitly dual-compatible callback. |
| `ops.hodgeDual(twoForm,x)` | Covariant dual two-form components; `x` is required for a position-dependent metric. |
| `ops.tensorHessianForward(field,x)` | Coordinate second derivatives of a bounded rectangular field; two coordinate axes follow the output axes. |
| `ops.hessianForward(scalarField,x)` | Coordinate Hessian from an explicit second-order field callback, as a matrix or batch. |
| `ops.dAlembertian(scalarField,x)`, `ops.dAlembertianForward(scalarField,x)` | The scalar covariant wave operator; the forward variant uses explicit second-order field derivatives and the configured metric connection. |

### Explicit forward differentiation

`forwardJacobian4(field,point)` evaluates one scalar or rectangular field at a four-coordinate point and returns `{value,jacobian}`. The callback receives four immutable `Dual4` values. Use `add`, `subtract`, `multiply`, `divide`, `negate`, `sin`, `cos`, `sinh`, `cosh`, `exp`, `log`, `sqrt`, and `pow` methods to propagate first derivatives; constants may be passed as numbers. A numeric output is treated as a constant. Outputs are bounded to 16,384 scalar components and rank eight; invalid domains and non-finite values or derivatives reject. Implicit numeric coercion, including JavaScript arithmetic and `Math` functions applied directly to a dual value, throws rather than silently discarding derivatives.

```js
import { DifferentialOperators4 } from "../src/relativity/index.js";
const ops = new DifferentialOperators4();
const field = x => x[0].pow(2).add(x[1].sin());
const gradient = ops.gradientForward(field, [2, 0, 0, 0]);
// Minkowski raised gradient: [-4, 1, 0, 0]
```

`partialForward`, `jacobianForward`, `gradientForward`, `divergenceForward`, and `exteriorDerivativeForward` also accept supported coordinate batches. This opt-in first-order API does not differentiate ordinary JavaScript callbacks automatically or provide general PyTorch autograd. `Connection4.christoffelForward` separately accepts a dual-compatible metric callback for first-order metric derivatives. Constructing a connection with `{metricDerivative:"forward"}` uses that path in its ordinary Christoffel, acceleration, covariant-derivative, parallel-transport, and geodesic workflows. `covariantDerivativeForward` and `covariantDerivativeVectorForward` explicitly differentiate a dual-compatible field; combine them with forward metric derivatives to avoid finite stencils for both parts. Ordinary covariant derivatives use finite field differences; `Curvature4` separately supports opt-in forward metric derivatives. Accessing `.value` explicitly or branching on it can discard derivative information; use such operations only when mathematically appropriate.

### Explicit second-order differentiation

`forwardSecondOrder4(field,point)` returns `{value,jacobian,hessian}` for a scalar or rectangular output of rank at most eight and at most 3,120 leaves; the coordinate derivative axes follow the output axes. `tensorHessianForward` returns the Hessian and supports bounded coordinate batches. `forwardHessian4(field,point)` evaluates an explicitly written scalar field with `Jet4` inputs and returns `{value,gradient,hessian}`. `Jet4` uses the same arithmetic and elementary-method names as `Dual4`, while propagating a 4×4 Hessian. `hessianForward` and `dAlembertianForward` accept supported coordinate batches; the latter combines the field Hessian with the configured metric connection. With `metricDerivative:"forward"` and a dual-compatible metric, both field derivatives and first metric derivatives avoid finite stencils. The square root requires a positive value, and powers at zero reject when their second derivative is unsupported. Numeric conversion, non-finite values and unsupported outputs reject. Intermediate overflow or underflow can still affect a representable result.

This path supports bounded rectangular second derivatives, scalar wave operators, and jet-compatible metric curvature; it does not provide higher orders or general PyTorch autograd.

## Connection and curvature

| Call | Returns |
| --- | --- |
| `new Connection4(metric,{step,metricDerivative})` | A connection for a metric callback; `metricDerivative` is `"finite"` by default or `"forward"` for an explicitly dual-compatible metric. |
| `connection.christoffel(x)`, `connection.christoffelForward(x)` | Christoffel components with shape `(...,4,4,4)`; the latter differentiates an explicitly dual-compatible metric. Non-finite results reject. |
| `connection.christoffelDerivativeForward(x)` | Coordinate derivatives `(...,derivative,upper,lower,lower)` from a `Jet4`-compatible metric; bounded coordinate batches and non-finite checks apply. |
| `connection.covariantDerivativeForward(field,x,indexTypes)`, `connection.covariantDerivativeVectorForward(field,x)` | Covariant field derivatives from an explicitly dual-compatible field; metric derivatives follow the connection setting. |
| `connection.covariantDerivative(field,x,indexTypes)` | Field components with a final derivative axis; `indexTypes` defaults to `["up"]` and can be `[]` for a scalar. Every tensor slot must have an explicit `"up"` or `"down"` entry. Non-finite corrected derivatives reject. |
| `connection.covariantDerivativeVector(field,x)` | The vector convenience form of the covariant derivative. |
| `connection.geodesicAcceleration(x,tangent)` | Four acceleration components, broadcasting compatible positions and tangents; non-finite results reject. |
| `connection.parallelTransport(vector,path,{stepsPerSegment})` | Transported vector components at each supplied path point; the default is eight RK4 steps per segment. |
| `new Curvature4(metric,{step,metricDerivative})` | Curvature calculations for a metric callback; set `metricDerivative:"forward"` for a callback written with `Jet4` operations. |
| `curvature.riemann(x)`, `curvature.ricci(x)` | Riemann `(...,4,4,4,4)` or Ricci `(...,4,4)` components. |
| `curvature.ricciScalar(x)`, `curvature.kretschmannScalar(x)` | Scalar invariants or batches of scalars. |
| `curvature.einsteinTensor(x)` | Einstein tensor components shaped `(...,4,4)`. |

## Geodesics

| Call | Returns |
| --- | --- |
| `new Geodesic4(connection,initialEvent,initialTangent)` | An initial-value geodesic; event and tangent batches must match. |
| `geodesic.integrate(span,{steps,format})` | Sample records by default, or `[parameters,events,tangents]` for `format:"components"`. |
| `geodesic.endpointSensitivityForward(span,{steps})` | Endpoint, final tangent and the 4×4 derivative of endpoint position with respect to initial tangent, from variational equations and a jet-compatible `Connection4` metric. |
| `geodesic.solveBoundary(target,span,{steps,maxIterations,tolerance,damping,format,jacobianMethod})` | A shot trajectory to the target, or an error if it does not converge. Set `jacobianMethod:"forward"` for variational endpoint sensitivities; component format also returns the fitted initial tangent. |

See the runnable [flat-space geodesic example](examples.md#geodesic-initial-value-problem) before adapting it to a curved metric. Validate a chosen step size and integration resolution for the metric and path in use.

`DifferentialOperators4` provides partial derivatives, Jacobians of scalar and tensor-valued fields, metric-raised scalar gradient, vector divergence, exterior derivatives of covariant forms, Hodge duals of covariant two-forms, and the scalar d'Alembertian. Pointwise derivative operations accept leading coordinate batches and cap outputs at 65,536 components. Their aggregate estimated batch work is capped at 1,000,000 units; estimates account for per-point callback/contraction work, including 300 units per d'Alembertian point, and are checked before traversing the rest of the batch. Jacobians and exterior derivatives also enforce a 65,536-component finite-difference work limit. Tensor-valued callbacks must return acyclic rectangular arrays of rank at most eight. Hodge duals accept compatible leading batches of two-forms and positions, use the Lib4 covariant orientation convention with `ε_{0123}=-sqrt(|det g|)` (including metrics with positive determinant), return covariant form components, and cap output at 65,536 components; supply `x` when the metric depends on position. `Connection4.christoffel(x)` computes Levi-Civita symbols and accepts leading point batches up to 65,536 output components. Covariant derivatives support scalar, vector, covector, and mixed tensor fields across leading point batches, with the same output limit; the direct `covariantDerivativeVector` convenience method is tested against the `up`-variance dispatcher. `geodesicAcceleration` broadcasts compatible leading position and tangent batches and caps its output at 65,536 components. Path-based parallel transport uses fixed-step RK4, broadcasts compatible leading batches on paths and initial vectors, and returns each sample's values at the supplied path points. Total batch transport is capped at 10,000 substeps and 65,536 history components.

`Curvature4` uses finite differences by default; with `metricDerivative:"forward"`, all curvature methods use explicit first and second derivatives of a jet-compatible metric callback, avoiding finite stencils. The callback must return a 4×4 rectangular matrix of finite numbers or `Jet4` values; use the `Jet4` arithmetic methods for coordinate dependence. Intermediate overflow and underflow can still affect results. `Curvature4` derives Riemann, Ricci, Ricci scalar, Einstein tensor, and Kretschmann scalar using the documented `R_ab=R^c_acb` convention and accepts leading point batches for each operation. Curvature outputs are capped at 65,536 scalar components and aggregate estimated work at 4,000,000 operation units; Kretschmann contraction charges 70,000 units per point and is preflighted before metric callbacks. Kretschmann contraction first lowers the Riemann tensor's leading index, then raises all four indices for `R_abcd R^abcd`. `Geodesic4.integrate` accepts matching scalar or leading-batch `(...,4)` initial events and tangents, uses fixed-step RK4, and caps the history at 1,000,000 scalar components. Its default result is a per-step sample array; pass `{format:"components"}` to receive `[parameters, events, tangents]` histories, preserving the Lib4 values in a JavaScript-friendly API. Boundary shooting accepts one unbatched geodesic, uses damped Gauss–Newton with finite-difference endpoint Jacobians, defaults to 12 correction iterations and tolerance 1e-9, permits up to 2,000 steps per integration, and bounds the full work—including residual, Jacobian, and final integrations—to 12,000 RK4 steps. It throws when the target is not reached within tolerance; singular or poorly conditioned paths can fail to converge. Boundary shooting with `{format:"components"}` appends the fitted initial tangent to those three histories. Parallel transport has a 10,000 total substep limit. A synthetic curved-coordinate check uses `g_11 = exp(2x^1)`, whose one-dimensional spatial geodesic has `x^1(λ) = ln(1 + vλ)`, to check endpoint accuracy as fixed RK4 steps are refined. This is a narrow analytic regression, not a general curved-path accuracy guarantee.

Finite-difference quotients use scaled subtraction if the difference between finite samples overflows, and reject an unrepresentable derivative. Boundary shooting also divides its numerical endpoint Jacobian by the actual tangent perturbation spacing and rejects perturbations that collapse at large tangent magnitudes. Always check step-size convergence against analytic cases before using curved-space outputs.

When the initial geodesic event is an `Event4`, sample events retain its frame and `c`, including after boundary shooting. Component-format histories contain coordinate arrays without event metadata.

Metric callbacks used by geometry operations must return one 4×4 metric for each point. A returned `Metric4` is checked against its constructed batch shape before use; construct a new metric if its batch shape needs to change.

Gradient, divergence, exterior derivative and d’Alembertian reject non-finite final results. Extreme-scale cancellation can still lose a representable result; rescale coordinates and compare converged calculations.

Riemann, Ricci, Einstein and Kretschmann calculations reject non-finite final components or contractions. At extreme scales, intermediate overflow can still prevent a representable final result; rescaling and convergence checks remain necessary.

Finite-difference stencils use a weighted three-point derivative when floating-point rounding makes the two offsets unequal. This covers pointwise derivatives, connection and curvature derivatives, exterior derivatives, and the boundary solver's endpoint Jacobian. A pointwise Jacobian takes one additional center sample when needed and its work limit accounts for it; the other paths reuse their existing center value. Intermediate overflow can still prevent a representable derivative. This does not replace convergence checks or automatic differentiation.

The opt-in forward endpoint sensitivity integrates geodesic and variational equations together with fixed-step RK4. It accepts matching leading batches of initial events and tangents, and requires an unmodified standard `Connection4` configured with `metricDerivative:"forward"`, and a metric callback compatible with both `Dual4` and `Jet4` operations. This ensures the trajectory and its sensitivity use the same connection. Aggregate endpoint sensitivity output is limited to 65,536 scalar components and work to 12,000 sample-steps. Boundary shooting remains unbatched. The result is a numerical sensitivity of the discretized trajectory, not an adaptive error estimate. The solver retains the existing bounded shooting-work limit and defaults to finite-difference endpoint Jacobians.

A flat-space regression checks that forward boundary sensitivities can converge when a fixed tangent finite-difference step rounds away at a large initial tangent. The example requires a suitably scaled damping value; ill-conditioned or singular shooting problems can still fail.

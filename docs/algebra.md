# Algebra Types

`Metric4.g` aliases `components`. Assigning `g` accepts a finite 4×4 matrix with the existing batch shape; inverse operations use its current values and reject a singular matrix.

The exported `minkowskiDot(a,b)` helper computes the `(-,+,+,+)` product and broadcasts compatible leading dimensions for raw four-vector arrays. It accepts rectangular finite batches up to rank eight and caps the number of scalar results at one million.

For algebra operations with an optional metric, omitting the metric or passing `null` selects the default Minkowski metric. This also applies to the optional metric in `new LeviCivita4(metric)`. Raw matrices are checked when wrapped as a metric; subsequent mutations of a `Metric4` require care as described below.

## `Vector4`

Construct with four finite components `(ct,x,y,z)` or a rectangular batch ending in four components. The read-only `shape` property returns a frozen array of dimensions, including leading batch dimensions. `add`, `subtract`, `scale`, `dot`, `normSquared`, causal classification, normalization, `wedge`, `transform`, `lowerIndex`, and `outer` preserve compatible leading batch dimensions. A scalar or batched metric broadcasts with vector batches, and a single or batched transformation can be applied to vectors. `causalType()` returns Lib4-compatible numeric codes: `Vector4.CAUSAL_TIMELIKE` (`-1`), `Vector4.CAUSAL_NULL` (`0`), and `Vector4.CAUSAL_SPACELIKE` (`1`). Use `causalName()` when a readable label is needed. Both accept an optional metric and absolute squared-interval tolerance. The tolerance is unit- and scale-dependent, and near-null values can be sensitive to floating-point cancellation. Classification throws if finite input components overflow to a non-finite squared interval, rather than labeling the invalid result as null. Normalization rescales a vector when its squared norm overflows, becomes subnormal, or underflows to zero with zero tolerance; it rejects indeterminate scaled norms, including exact cancellation, and remains subject to floating-point conditioning. The default dot product uses `Metric4.minkowski()`. Lib4's PyTorch `device` and `dtype` properties have no JavaScript-array equivalent.

| Call | Returns |
| --- | --- |
| `new Vector4(components)` | A vector value; `components` ends in four coordinates. |
| `v.add(other)`, `v.subtract(other)` | A new `Vector4`; compatible leading batches broadcast. |
| `v.scale(factor)` | A new `Vector4`; factors broadcast over the complete component shape. |
| `v.dot(other, metric)`, `v.normSquared(metric)` | A number or batch of numbers. The metric argument is optional. Both diagonal and general metrics use an exact Float64-input fallback for extreme intermediate products or non-finite results, as described for diagonal products in [array helpers](arrays.md); ordinary finite calculations remain subject to cancellation. |
| `v.causalType(metric, atol)`, `v.causalName(metric, atol)` | Numeric causal codes or names, preserving batches. Both arguments are optional. |
| `v.normalizeTimelike(metric, atol)`, `v.normalizeSpacelike(metric, atol)` | A new `Vector4`, or an error when the requested normalization is invalid. |
| `v.outer(other)`, `v.wedge(other)` | A `Tensor4` or `Bivector4`, respectively. |
| `v.transform(matrix)`, `v.lowerIndex(metric)` | A transformed `Vector4` or lowered `Covector4`. |
| `v.toArray()`, `v.shape` | A copied component array or a frozen dimension array. |

## `Covector4` and `Metric4`

`Covector4.pair(vector)`, addition, and scaling broadcast compatible leading dimensions. `raiseIndex(metric)` broadcasts covectors with scalar or batched metrics; `transform(matrix)` broadcasts covectors with scalar or batched matrices and applies the inverse transpose per sample. `Metric4` accepts a single 4×4 matrix or a batch ending in 4×4 matrices. It provides `dot`, `intervalSquared`, `lowerVector`, `raiseCovector`, per-sample determinant, inverse access (`metric.inverse` or `inverseMetric()`), and tensor index conversion. Metrics and operation outputs are bounded to one million scalar components; batch shapes must be NumPy/PyTorch style compatible, and each metric must be symmetric and nonsingular.

| Call | Returns |
| --- | --- |
| `new Covector4(components)` | A covector value ending in four components. |
| `a.add(other)`, `a.scale(factor)`, `a.transform(matrix)` | A new `Covector4`; scale factors broadcast over the complete component shape, and transformation uses the inverse transpose. |
| `a.pair(vector)`, `a.raiseIndex(metric)` | A number or batched numbers, or a raised `Vector4`. |
| `new Metric4(components, {validate,atol})`, `Metric4.minkowski({batchShape})` | A metric value. Constructor and factory arguments are optional. |
| `g.dot(a,b)`, `g.intervalSquared(a)`, `g.determinant()` | Numbers or batches of numbers. |
| `g.lowerVector(v)`, `g.raiseCovector(a)` | A `Covector4` or `Vector4`, respectively. |
| `g.lowerIndex(value,axis)`, `g.raiseIndex(value,axis)` | A `Tensor4` for tensor input, or component arrays for vector/covector input. Supply the axis for tensors. |
| `g.inverseMetric()`, `g.inverse`, `g.inverseComponents` | An inverse `Metric4` from the method or property, or raw inverse component arrays. |

## `Matrix4`, `Tensor4`, and `Bivector4`

`Tensor4.transform` requires every applied transformation matrix to be invertible, including for all-upper tensors, matching the local Lib4 contract. A singular matrix in a broadcast batch rejects the call.

`Tensor4.rank` is a read-only property derived from `indexTypes`, matching Lib4's rank property.

Symmetrization and antisymmetrization halve terms separately when their intermediate sum overflows, preserving representable averages for some large finite components.

| Call | Returns |
| --- | --- |
| `new Matrix4(components)`, `Matrix4.identity({batchShape})` | A matrix value; the optional batch shape creates repeated identities. |
| `m.add(other)`, `m.multiply(other)`, `m.transpose()`, `m.inverse()` | A new `Matrix4`. Matrix multiplication applies `other` first. |
| `m.apply(vector)`, `m.determinant()` | A `Vector4`, or a number/batch of numbers. `apply` rejects `Covector4`. |
| `new Tensor4(components,indexTypes)` | A tensor with explicit `up`/`down` slots. Omitted, `null`, or empty index types infer all-upper slots only when every dimension is a tensor axis of length four. |
| `t.outer(other)`, `t.permuteIndices(order)` | A new `Tensor4` with combined or reordered index types. |
| `t.lowerIndex(axis,metric)`, `t.raiseIndex(axis,metric)` | A new `Tensor4`; the metric defaults to Minkowski. |
| `t.contract(axis1,axis2)` | A rank-reduced `Tensor4`, or scalar/batched numbers when no tensor slots remain. |
| `t.symmetrize(axis1,axis2)`, `t.antisymmetrize(axis1,axis2)`, `t.transform(matrix)` | A new `Tensor4`; the exchange axes default to zero and one. |
| `new Bivector4(components,{validate,atol})`, `Bivector4.wedge(a,b)` | An antisymmetric `Bivector4` from components or a wedge of vectors; validation is enabled by default. |
| `b.contract(covector)`, `b.dual(metric,orientation)`, `b.transform(matrix)` | A `Vector4` from contraction, or a new `Bivector4` from dual or transformation. |
| `b.invariants(metric)` | An object with `quadratic` and `pseudoscalar`, each scalar or batch-shaped. |

Bivector duals and invariants lower both indices with `G F Gᵀ`. A `Metric4` constructed with `{validate:false}` can contain a non-symmetric matrix for Lib4 compatibility, but such a matrix is not a physical metric in this model.

`Matrix4` provides broadcasted batch addition, multiplication, transpose, inverse, determinant, and vector application. Its batches are bounded to one million scalar components. `Tensor4` records an index type per tensor axis (`up`/`down`) separately from leading batch axes. It provides broadcasting outer product, permutation of tensor axes, contraction, symmetrization, antisymmetrization, raising/lowering, and transformation. Tensor rank is bounded to nine, nested input rank to sixteen, and storage to one million scalar components. Contraction requires one upper and one lower axis and returns either a batch of scalars or a rank-reduced `Tensor4` with the batch shape and remaining index types preserved. `Bivector4` stores an antisymmetric rank-two object and broadcasts wedge, contraction, dual, transformation, and metric invariants over compatible leading dimensions. The Hodge dual and invariants broadcast a scalar or batched `Metric4`; component shape and antisymmetry are validated, and dense batch results are bounded to one million scalar components. `Bivector4.invariants()` returns `{quadratic, pseudoscalar}`, with each value preserving the broadcast batch shape, matching Lib4. This replaces the earlier JavaScript array-of-objects result for batches. Invariants reject non-finite contractions rather than returning invalid scalar results. `LeviCivita4.symbol()` returns the orientation-independent permutation tensor; canonical, odd-permutation, and repeated-index values are directly tested. `LeviCivita4.epsilon()` preserves metric batch axes, subject to the same component bound.

## `LeviCivita4`

| Call | Returns |
| --- | --- |
| `new LeviCivita4(metric,orientation)` | A metric-aware helper; the metric defaults to Minkowski and orientation to `1`. |
| `LeviCivita4.symbol()` | The orientation-independent four-index permutation symbol as arrays. |
| `epsilon.epsilon(variance)` | Metric-scaled four-index components for `"down"` or `"up"` variance; default is `"down"`. |
| `epsilon.volumeForm()` | The covariant volume-form components, equivalent to `epsilon.epsilon("down")`. |
| `epsilon.dual(bivector)` | A `Bivector4` using the helper's metric and orientation. |

## Three.js value adapters

`three-adapter.js` exports `fromThreeVector3`/`toThreeVector3`, `fromThreeVector4`/`toThreeVector4`, and `fromThreeMatrix4`/`toThreeMatrix4`. Supply the Three.js namespace only to a `toThree...` call; the core module does not import or require Three.js. Vector3 values map directly to spatial `[x,y,z]` triples. Vector4 slots map directly to `[x,y,z,w]`; the adapter does not reinterpret those slots as `(ct,x,y,z)`, so callers must deliberately arrange physical components. Three.js `Matrix4.elements` is converted from column-major storage to the library's row-major rows, and conversion back calls Three.js `Matrix4.set()` with row-major arguments. Conversions are copies, accept only single values (no batches), and reject malformed or non-finite components.

## `LorentzTransform` and `PoincareTransform`

Construct Lorentz maps with `identity`, `boost(beta)`, `fromRapidity(rapidity,direction)`, or `rotation(rotation3)`. The boost uses the cancellation-resistant `gamma²/(gamma+1)` spatial coefficient. Rotation accepts `{atol}` (default `1e-10`) and applies that value as both absolute and relative tolerance when checking orthogonality, matching Lib4. The constructor checks matrix shape and finite components; pass `{validate:true}` to also reject matrices that fail the Lorentz condition. `validate({atol,rtol})` checks `Lambdaᵀ eta Lambda = eta` using absolute plus relative tolerance (both default to `1e-9`). Each factory accepts compatible leading batch dimensions; composition, inversion, vector application, and rank-two tensor application preserve or broadcast those dimensions. Validation returns one Boolean for the entire batch. Matrix storage and each result are bounded to one million scalar components. `compose` applies the other transform first; `inverse` uses the Lorentz closed form `eta Lambdaᵀ eta`, which assumes the matrix is Lorentz, so validate manually supplied matrices before group operations. A `PoincareTransform` requires a `LorentzTransform`, adds a finite scalar or batched four-position translation, and maps events supplied as an `Event4`, `{position}`, or raw four-vector, along with vectors. Affine apply, composition, and inverse broadcast compatible transform, translation, and event batches. PyTorch dtype/device and autograd behavior are not represented.

| Call | Returns |
| --- | --- |
| `new LorentzTransform(matrix,{validate,atol,rtol})` | A transform from 4×4 components; opt into Lorentz validation with `validate:true`. |
| `LorentzTransform.identity({batchShape})`, `LorentzTransform.boost(beta)` | Identity or passive boost transforms; `beta` is the dimensionless three-velocity `v/c`. |
| `LorentzTransform.fromRapidity(rapidity,direction)`, `LorentzTransform.rotation(rotation3,{atol})` | A transform from rapidity and spatial direction, or a proper 3×3 spatial rotation. |
| `L.validate({atol,rtol})` | One Boolean indicating whether every matrix in the batch meets the Lorentz condition. |
| `L.compose(other)`, `L.inverse()` | A new `LorentzTransform`; composition applies `other` first. |
| `L.applyVector(v)` | A new `Vector4`. |
| `L.applyTensor(t,indexTypes)` | A transformed `Tensor4` for tensor input, or component arrays for raw rank-two input; raw input defaults to two upper indices. |
| `new PoincareTransform(lorentz,translation)`, `PoincareTransform.identity({batchShape})` | An affine transform or a batch of identities. Constructor arguments are optional. |
| `P.applyEvent(event)`, `P.applyVector(v)` | An object containing a transformed `position` array, or a `Vector4` without translation. |
| `P.compose(other)`, `P.inverse()` | A new affine transform; composition applies `other` first. |

For raw rank-two input, `applyTensor` uses the Lorentz inverse convention `ηLᵀη` for lower indices, as Lib4 does. The constructor does not validate a matrix unless requested, so call `validate()` when accepting an external transform; the inverse convention assumes a Lorentz matrix. The `Tensor4` input extension delegates to the general tensor transformation, which requires an invertible matrix.

`Matrix4` derives batch dimensions from its current component arrays for each operation. If callers reshape mutable components, they must remain rectangular with trailing 4×4 dimensions and within the documented bounds.

`Metric4` retains its constructed batch shape: reshaping its mutable component arrays is rejected by metric-aware algebra operations. Construct a new metric for a different batch shape.

`Tensor4` retains its constructed batch shape. If mutable component arrays are reshaped, its operations reject the changed shape; construct a new tensor for different dimensions.

`Metric4` and `Tensor4` expose read-only cached batch shape and rank metadata. `Tensor4.indexTypes` may be reassigned, but operations validate its entries and compatibility with the components.

`LorentzTransform` operations reject a change to the constructed batch dimensions of its mutable matrix; construct a new transform for a different shape.

## JSON interchange for value objects

`serializeRelativity(value)` returns a versioned JSON string; `deserializeRelativity(json)` restores the corresponding value through its public constructor. Version 1 supports `Vector4`, `Covector4`, `Matrix4`, `Metric4`, `Tensor4`, `Bivector4`, `LeviCivita4`, `LorentzTransform`, `PoincareTransform`, `Event4`, `SpacetimeInterval`, `LightCone`, `SpatialSlice`, `InertialFrame`, `WorldlineSegment`, `Worldline`, `RelativityContext`, `MaxwellEquations`, `FourVelocity`, `FourMomentum`, `FourCurrent`, and `EMFieldTensor`. It preserves component batches, tensor index types, event positions and simple frame labels, and relevant physical constants. `Event4` and the supported types containing events accept only null or string frame labels; object-valued frames are rejected rather than silently losing identity. Inputs are validated again on both serialization and restoration: invalid mutable components, unsupported types, and invalid or unsupported records throw. Metric, bivector, electromagnetic and Lorentz values created with validation disabled must pass their standard validation to use this format. A serialized context preserves its device label as a string; restoring the label does not create or transfer GPU resources. JSON strings are limited to 32 million characters, and a serialized worldline is limited to 65,536 segments. Segment joins are validated when restoring a worldline. Restoring a spatial slice normalizes its normal again, which may cause small floating-point changes.

```js
import { Vector4, serializeRelativity, deserializeRelativity } from "../src/relativity/index.js";
const text = serializeRelativity(new Vector4([1, 2, 3, 4]));
const vector = deserializeRelativity(text); // Vector4 with the same components
```

This format does not serialize functions, object-valued frame metadata, callbacks, GPU resources, or the remaining value and solver types. It is independent of ordinary `JSON.stringify` on class instances.

For `Vector4.scale` and `Covector4.scale`, factor arrays follow trailing-axis broadcasting across the full `(...,4)` component shape, as in Lib4. For example, `[2,3,4,5]` scales individual components; `[[2],[3]]` scales two vectors separately. Earlier local versions treated all factor axes as batch axes, so replace a two-vector factor `[2,3]` with `[[2],[3]]`. Output is limited to 1,000,000 components.

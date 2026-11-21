# Spacetime Types

## Events and intervals

| Call | Returns |
| --- | --- |
| `new Event4(position, frame, c)` | An event with optional frame label and positive speed of light; `position` ends in four coordinates. |
| `event.displacementTo(other)` | Component arrays for `other.position - event.position`, with compatible batches broadcast. |
| `event.intervalTo(other,{c,atol})` | A `SpacetimeInterval`; options are optional. |
| `event.causalRelation(other,{atol})` | An oriented causal label or batch of labels. |
| `event.transform(transform)` | A new `Event4` under a Lorentz or Poincare transformation; its frame label is reset. |
| `new SpacetimeInterval(displacement,{c,atol})` | An interval for a four-displacement or batch. |
| `interval.squared`, `interval.intervalSquared()`, `interval.tolerance()` | Squared interval or classification tolerance, as a number or batch. |
| `interval.isTimelike()`, `interval.isSpacelike()`, `interval.isNull()` | A Boolean or batch of Booleans. |
| `interval.properTime()`, `interval.properDistance()` | A number or batch; all samples must have the required causal class. |

`Event4` stores one four-position or a rectangular batch ending in four-components, a frame label, and positive `c` constant. `displacementTo` returns the broadcasted displacement and requires another `Event4`; `intervalTo` and oriented `causalRelation` (`future_timelike`, `past_timelike`, `future_null`, `past_null`, `spacelike`, or `coincident`) build on it. Scalar and compatible batched Lorentz and Poincare transforms apply across event batches; transformed events reset the frame label, matching Lib4. `SpacetimeInterval` preserves leading batch dimensions, uses scale-aware `atol` classification, rejects non-finite interval or tolerance scales, and returns batched proper times or distances only when every sample has the matching causal class.

`LightCone(apex,{branch,c,atol})` supports `future`, `past`, or `both`; `contains`, branch-aware `onSurface`, affine worldline intersections, affine hyperplane intersections, and surface sampling are available. Containment/surface checks broadcast compatible event batches against the apex. Sampling broadcasts compatible apex, direction, and radius batches; hyperplane intersection broadcasts apex and direction batches, returning batch-shaped `events` and `hitMask`. Its covector and offset specify `normal · (event − apex) = offset`, so the offset is relative to each apex rather than the global origin. `intersectWorldline` requires a scalar apex and rejects non-finite quadratic coefficients, discriminants, or intersection events; finite inputs at extreme scales can exceed these intermediate limits. Sampling and hypersurface intersection each accept at most 65,536 directions, and generated results are capped at one million components. Causal and surface checks use scale-aware interval tolerances. `SpatialSlice(origin,futureTimelikeNormal)` normalizes each future timelike normal and constructs an orthonormal spatial basis. Compatible leading batches broadcast across origins, normals, bases, events, and coordinates. Projection and containment return batch-shaped values; embedding returns an `Event4` whose position can be batched and whose `c` and frame metadata follow an `Event4` origin, including batched origins. Worldline intersections return batch-shaped parameters and events, while `intersectLightcone(cone)` returns batch-shaped `{center,radius,hit}` for the cone/slice sphere, including branch-aware misses. Batch outputs are capped at one million components. Containment uses a scale-aware tolerance, and a parallel affine worldline raises.

## Cones and spatial slices

| Call | Returns |
| --- | --- |
| `new LightCone(apex,{branch,c,atol})` | A cone; the options object is optional and branch defaults to `both`. |
| `cone.onSurface(event)`, `cone.contains(event)` | A Boolean or compatible batch of Booleans. |
| `cone.sample(directions,spatialDistance)` | Four-coordinate arrays on the specified future or past branch. |
| `cone.intersectWorldline(origin,direction)` | An array of zero, one, or two event-coordinate arrays for an affine worldline; the apex and line are unbatched. |
| `cone.intersectHypersurface(normal,offset,directions)` | `{events,hitMask}` with compatible leading batches; the offset is measured from the apex. |
| `new SpatialSlice(origin,normal,{basis,c,atol})` | A slice with a normalized future-directed timelike normal and optional spatial basis. |
| `slice.projectTo3D(event)`, `slice.embed3D(coords)` | Spatial coordinate arrays, or an `Event4` from embedding. |
| `slice.contains(event,{atol})` | A Boolean or batch of Booleans. |
| `slice.intersectWorldline(origin,direction)` | `{parameter,event}`, with an `Event4` and batch-shaped parameters. |
| `slice.intersectLightcone(cone)` | `{center,radius,hit}` describing the intersection sphere or a miss. |

`WorldlineSegment` requires single four-vector endpoint events and tangents, uses cubic Hermite interpolation over an increasing parameter span with finite duration, and accepts scalar or batched parameter queries. Endpoints must share `c` and frame metadata; raw endpoints inherit metadata from an `Event4` endpoint. Segment and piecewise-worldline evaluations preserve that metadata, and a `Worldline` rejects joins that mix `c` or frames. `Worldline` also requires adjacent segments to meet in parameter, event, and four-velocity; it selects a segment per query and uses the first or last segment to extrapolate outside the total span. `ODEWorldline(initialEvent,fourVelocity)` integrates `d(event)/d(parameter)` and returns `[parameters,events]`; `integrateRK4` and its worldline wrapper use fixed-step RK4 and preserve leading state dimensions. The generic `integrateRK4(rhs, initialState, start, end, steps)` accepts a finite scalar or rectangular numeric array and returns `[parameter, state]` records, including both endpoints. Derivative output must match the initial state shape. Bounds, derivatives, and intermediate states must remain finite; a nonzero interval must permit distinct ordered floating-point step times and interior midpoints. The solver uses the actual representable step endpoints and preflights result storage against the one-million-component limit. The solver rejects midpoint asymmetry greater than 1e-8 of the step width; smaller rounding differences can still reduce accuracy, so shift or rescale large parameter values and check convergence. `InertialFrame` accepts proper scalar or batched 3×3 spatial rotations, subluminal scalar or batched physical relative velocities, positive `c`, and a scalar or batched global origin. Its `toGlobal`, `fromGlobal`, `transformEvent`, and `transformVector` methods plus the relative-frame transform broadcast compatible leading dimensions. Direct regressions compare event and vector transforms with the corresponding affine maps. It composes the rotation with the passive boost and exposes inverse global transforms.

## Worldlines, integration, and frames

| Call | Returns |
| --- | --- |
| `new WorldlineSegment(start,end,u0,u1,parameterSpan)` | A cubic Hermite segment; the parameter span defaults to `[0,1]`. |
| `new Worldline(segments)` | A piecewise worldline whose segment joins must agree. |
| `segment.evaluate(t)`, `worldline.evaluate(t)` | An `Event4`; scalar and bounded batch parameter queries are supported. `worldline.call(t)` is an alias. |
| `segment.tangent(t)`, `worldline.tangent(t)` | A `Vector4` containing derivatives with respect to the parameter. |
| `new ODEWorldline(initialEvent,fourVelocity).integrate(span,{steps})` | `[parameters,events]` coordinate histories; the callback receives `(parameter,eventCoordinates)`. |
| `integrateRK4(rhs,initialState,start,end,steps)` | An array of `[parameter,state]` pairs using fixed-step RK4. |
| `new InertialFrame({origin,orientation,relativeVelocity,c})` | A frame; the options object is optional and velocity uses the same units as `c`. |
| `frame.toGlobalTransform`, `frame.fromGlobalTransform`, `frame.relativeTransform(other)` | `PoincareTransform` values; the first two are getters. |
| `frame.toGlobal(position)`, `frame.fromGlobal(position)`, `frame.transformEvent(event)` | Objects containing transformed `position` arrays. |
| `frame.transformVector(vector)` | A transformed `Vector4` without translation. |

Interval, event and light-cone surface/containment classification apply the tolerance before forming extreme squared scales, so a representable final tolerance need not overflow during calculation. Spatial-slice containment also scales its Euclidean tolerance before comparison. A non-finite final tolerance or slice distance raises an error; near-null classifications remain sensitive to the chosen tolerance and floating-point conditioning.

The scale-aware tolerance uses coordinate components, whose Euclidean magnitude changes under Lorentz boosts. Consequently, classifications close to the tolerance boundary can differ between frames even though the exact interval is invariant. Use an appropriate tolerance and inspect the squared interval for sensitive cases.

`SpatialSlice.batchShape` and `LightCone.apexBatchShape` are frozen cached metadata. Component arrays remain shallowly mutable as described in the API conventions. Cone apex coordinate values may change, but its batch dimensions must stay the same; reconstruct the cone after reshaping the apex. Slice origin, normal, and basis arrays must likewise retain their constructed batch dimensions; reconstruct the slice after reshaping them. The normal must remain future-directed and normalized, and the basis must remain orthonormal and orthogonal to it.

Slice–cone intersections use a scaled branch tolerance and reject non-finite final classification or hit-center values. A branch miss retains the documented `hit: false` and NaN center/radius representation.

Light-cone worldline intersections scale the quadratic coefficients before solving, allowing some large finite intersections that previously overflowed. With zero tolerance, tiny nonzero offsets and directions are scaled as well; with nonzero tolerance, extremely small direction scales can still underflow and are rejected. Events are constructed from the scaled roots without forming the affine parameter, which can recover representable events even when that parameter is outside the Float64 range. Near-degenerate roots and event construction remain subject to floating-point limits.

`WorldlineSegment.span` and `Worldline.segments` use frozen copies of their input arrays, matching the immutable contents of the corresponding Lib4 tuples. The objects themselves remain mutable and their properties may be replaced; callers are responsible for preserving valid spans and joins after replacement.

# Array and Matrix Helpers

These named exports are available from `src/relativity/index.js`. They are low-level JavaScript helpers; unless stated otherwise, callers must supply compatible, finite arrays of the documented shape. They do not reproduce PyTorch tensor, device, or autograd behavior. For shape-validated relativity values and bounded batching, prefer the [algebra types](algebra.md).

## Constants and validation

| Export | Meaning |
| --- | --- |
| `C`, `SPEED_OF_LIGHT` | Speed of light in vacuum, `299792458` metres per second. |
| `VACUUM_PERMITTIVITY` | Default SI vacuum permittivity used by the library. |
| `ETA` | Frozen Minkowski diagonal `[-1,1,1,1]`. |
| `finiteNumber(value,name)` | Returns the value if it is a finite JavaScript number; otherwise throws. The optional name is used in errors. |
| `vector(value,size,name)` | Copies and validates a one-dimensional numeric array or typed array; size defaults to four. |
| `matrix(value,n,name)` | Copies and validates an `n` by `n` numeric array; `n` defaults to four. |

## Vector helpers

| Call | Meaning |
| --- | --- |
| `add(a,b)`, `sub(a,b)` | Elementwise sum or difference. |
| `scale(a,k)` | Elementwise multiplication by a scalar. |
| `dot(a,b)` | Euclidean component sum `Σ aᵢbᵢ`; it does **not** use the Minkowski signature. |
| `norm(a)` | Euclidean magnitude computed with `Math.hypot`. |
| `cross(a,b)` | Three-dimensional cross product of two spatial triples. |
| `minkowskiDot(a,b,metric)` | Minkowski or supplied diagonal-metric product of four-vectors. The optional diagonal defaults to `ETA`; compatible leading vector batches broadcast, subject to the documented component limit. |
| `close(a,b,tol)` | Boolean comparison using `tol * max(1,abs(a),abs(b))`; tolerance defaults to `1e-10`. |

## Matrix helpers

| Call | Meaning |
| --- | --- |
| `zeros(n)`, `identity(n)` | New `n` by `n` zero or identity arrays. |
| `transpose(a)` | Transposed nonempty rectangular matrix array. |
| `mmul(a,b)`, `mvec(a,v)` | Matrix multiplication or matrix-vector multiplication. |
| `determinant(a)` | Determinant using pivoted elimination; the result can underflow or overflow. |
| `determinantRoot(a)` | `{sign,root}`, where `root` is the square root of the absolute determinant; singular or unrepresentable volume scales throw. |
| `inverse(a,tolerance)` | Inverse via pivoted elimination; `tolerance` defaults to zero. Singular pivots or non-finite inverse results throw. |

These raw helpers do not generally validate matching dimensions or prevent non-finite intermediate results. `finiteNumber`, `vector`, `matrix`, and `minkowskiDot` perform the validation described above. Matrix inversion and determinant calculations remain sensitive to conditioning; see [numerical limits](limits.md).

`minkowskiDot` uses exact accumulation of the supplied Float64 values when products or their sum overflow, underflow, or become subnormal. This can recover a finite result when extreme terms cancel. A result outside the Float64 range is returned as signed infinity; ordinary finite intermediate calculations remain subject to floating-point cancellation.

The unchecked `weightedDot4` and `metricDot4` implementation helpers are no longer exported from the public entry point. Use `minkowskiDot` or `Vector4.dot` for validated inputs.

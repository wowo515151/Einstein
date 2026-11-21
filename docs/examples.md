# Examples

The import paths below assume the example module is placed in this `docs/` directory. Adjust the relative path when placing a module elsewhere.

## Minkowski interval

```js
import { Event4 } from "../src/relativity/index.js";
const a = new Event4([0, 0, 0, 0]);
const b = new Event4([10, 6, 0, 0]);
console.log(a.intervalTo(b).isTimelike());
```

## Euclidean and Minkowski products

The low-level `dot` helper is Euclidean; use `minkowskiDot` for the default spacetime signature. See [array helpers](arrays.md) for their input assumptions.

```js
import { dot, minkowskiDot } from "../src/relativity/index.js";
const components = [2, 1, 0, 0];
console.log(dot(components, components), minkowskiDot(components, components)); // 5, -3
```

## Lorentz boost

```js
import { LorentzTransform, Vector4 } from "../src/relativity/index.js";
const boost = LorentzTransform.boost([0.5, 0, 0]);
const event = boost.applyVector(new Vector4([2, 1, 0, 0]));
```

## Retarded field model

Construct `RetardedField(positionAtSeconds, velocityAtSeconds, charge, maxSpeed)` and call `evaluate([ct,x,y,z])` or `retardedPotential([ct,x,y,z])` for the contravariant four-potential `(φ/c,A)`. The trajectory must remain below the declared speed bound. The point-charge potential is singular on the source worldline.

```js
import { RetardedField } from "../src/relativity/index.js";
const field = new RetardedField(() => [0, 0, 0], () => [0, 0, 0], 1, 0, {
  c: 1, epsilon0: 1 / (4 * Math.PI),
});
console.log(field.retardedPotential([2, 1, 0, 0])); // approximately [1, 0, 0, 0]
```

## Momentum and electromagnetic invariants

Use consistent units for the speed of light and field components. These examples use `c = 1`.

```js
import { FourMomentum, EMFieldTensor } from "../src/relativity/index.js";
const momentum = FourMomentum.fromMassVelocity(2, [0.6, 0, 0], 1);
const field = EMFieldTensor.fromEB([1, 0, 0], [0, 1, 0], { c: 1 });
console.log(momentum.invariantMass()); // approximately 2
console.log(field.invariants()); // B² − E²/c² = 0, E·B = 0
```

## Batched vectors

Leading dimensions broadcast when compatible. Each batch element below is a separate four-vector.

```js
import { Vector4 } from "../src/relativity/index.js";
const events = new Vector4([[2, 1, 0, 0], [3, 0, 0, 0]]);
console.log(events.normSquared()); // [-3, -9]
```

## Curved-coordinate connection

This smooth metric is a numerical example; check convergence with other step sizes for an application.

```js
import { Connection4 } from "../src/relativity/index.js";
const metric = x => [[-1, 0, 0, 0], [0, Math.exp(2 * x[0]), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const connection = new Connection4(metric, { step: 1e-4 });
console.log(connection.christoffel([0, 0, 0, 0])[1][0][1]); // approximately 1
```

## Geodesic initial-value problem

In flat coordinates, a geodesic with constant initial tangent is a straight line. The default result contains records with a parameter, an `Event4`, and a tangent. See [Differential geometry](geometry.md) for component histories, curved metrics, and convergence limits.

```js
import { Connection4, Geodesic4 } from "../src/relativity/index.js";
const flatMetric = () => [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const path = new Geodesic4(new Connection4(flatMetric), [0, 0, 0, 0], [1, 0.5, 0, 0]);
const samples = path.integrate([0, 1], { steps: 10 });
console.log(samples.at(-1).event.position); // approximately [1, 0.5, 0, 0]
```

## Maxwell evolution and residuals

This uniform field is unchanged by the periodic vacuum update. The grid shape is `[Nx,Ny,Nz,3]`; each dimension must contain at least two cells. Magnetic history uses half timesteps. See [Physics and fields](physics.md) for the CFL condition, current grids, and constraint residuals.

```js
import { MaxwellEquations } from "../src/relativity/index.js";
const grid = value => Array.from({ length: 2 }, () => Array.from({ length: 2 }, () => Array.from({ length: 2 }, () => value.slice())));
const solver = new MaxwellEquations({ c: 1, epsilon0: 1 });
const options = { spacing: [1, 1, 1], dt: 0.1, steps: 2 };
const [electricTimes, electric, magneticTimes, magnetic] = solver.evolve(grid([1, 0, 0]), grid([0, 0, 0]), options);
const residuals = solver.maxwellResiduals(electric, magnetic, options);
console.log(electricTimes, magneticTimes); // approximately [0, 0.1, 0.2], [0.05, 0.15]
console.log(residuals.ampere[0][0][0][0]); // [0, 0, 0]
```

## CPU batch dot products

The asynchronous batch interface can explicitly use the Float64 CPU reference path. See [GPU backends](gpu.md) for optional WebGPU and its precision limits.

```js
import { minkowskiDotBatch } from "../src/relativity/index.js";
const dots = await minkowskiDotBatch([1, 2, 0, 0, 2, 1, 0, 0], [1, 2, 0, 0, 2, 1, 0, 0], { backend: "cpu" });
console.log(dots); // [3, -3]
```

## Batch plane-wave fields

Sample the normalized vacuum wave at two events. Use the `auto` backend to permit WebGPU with CPU fallback; see [GPU backends](gpu.md) for its precision limits.

```js
import { samplePlaneWaveBatch } from "../src/relativity/index.js";
const fields = await samplePlaneWaveBatch([0, 0, 0, 0, 0, 0, 0, Math.PI / 2], { waveNumber: 1, backend: "cpu" });
console.log(fields[0]); // electric [1,0,0], magnetic [0,1,0]
```

## Plane-wave spatial grid

Generate a small rectangular grid at `ct = 0` and sample it with the CPU reference backend. The z coordinate varies fastest; see [GPU backends](gpu.md) for limits and optional WebGPU sampling.

```js
import { samplePlaneWaveGrid } from "../src/relativity/index.js";
const grid = await samplePlaneWaveGrid({ shape: [1, 1, 2], spacing: [1, 1, Math.PI / 2], backend: "cpu" });
console.log(Array.from(grid.events)); // [0,0,0,0, 0,0,0,π/2]
console.log(grid.fields.map(field => field.electric[0])); // approximately [1, 0]
```

# Einstein Relativity Visualizations

This static page contains ten deterministic, bounded visualization models. It uses the project's relativity library for Minkowski causal classification, past and future light-cone event sampling, Lorentz boosts and collinear velocity composition, electromagnetic field transformations, charged-particle integration, and three-dimensional null-geodesic lensing. The viewer overlays connected tubes for worldlines and paths, vector glyphs for transformed fields, an indexed mesh for the sampled spatial grid, and a sphere mesh for the light-cone slice. In the Minkowski scene, switch between the lab and moving observer frames to see both worldlines transformed by the library. In the Lorentz scene, compare both observer simultaneity slices and measure the rod length in the selected frame. The velocity-addition and aberration scene pairs the library-computed composition of two collinear velocities with a Lorentz-transformed star field. Other scenes are labeled as pedagogical or synthetic where they are not exact solutions.

Open `index.html` in a browser that supports JavaScript modules. The page does not request network resources. ThreeViz source is included at `../src/three-viz/`, with its MIT notice at `../THREEVIZ-LICENSE`.

## Offline renderer requirement

Three.js r128 and the matching OrbitControls script are bundled locally in `vendor/`; the MIT license is retained in `vendor/THREE-LICENSE.txt`. The page registers both as `globalThis.EINSTEIN_THREE_RUNTIME` before loading the visualization module. This keeps the static page offline and gives the copied ThreeViz core its expected runtime. The viewer passes the runtime through the `createViewerWithRuntime` seam. A browser still needs JavaScript-module and WebGL support.

The gravitational-lensing scene traces eight null rays through a softened, static weak-field metric. Their impact points span two radii and eight directions in three-dimensional space; each curve is integrated with `Connection4` and `Geodesic4`, with its per-sample metric norm included to inspect null drift. Tests verify inward deflection, finite tangents, the 312-sample bound, and near-zero metric norm. This is a pedagogical lensing bundle, not an exact vacuum solution or calibrated astrophysical model:

```text
node --test Relativity/visualizations/scene-models.test.mjs
```

The curved-space grid is the declared embedded profile `h(r) = A exp(−a r²)`, with `A = 0.55β` and `a = 0.38`. Its induced radial metric is `g_rr = 1 + (dh/dr)²`; its Gaussian curvature is `K = h′h″ / (r(1 + h′²)²)`, using the analytic center limit `K(0) = 4a²A²`. Surface color encodes signed curvature and the overlaid wire grid shows the embedded metric spacing. These overlays describe this pedagogical two-dimensional spatial surface, not a four-dimensional spacetime solution.

Every scene includes colored 3D x/y/z axes and matching visible labels. The accessible axis legend describes what each coordinate means for the selected visualization; coordinates use normalized display units unless stated otherwise.

In the twin scene, the time control moves from departure to reunion and updates both elapsed proper-time readouts and the two markers on the worldlines. The small depth separation between paths is a display aid for the idealized out-and-back trip.

The animated field explorer uses a normalized vacuum plane wave, `Eₓ = Bᵧ = cos(k(z − ct))`, with `k = 1 + β`, `c = 1`, and `ct = 2π × playbackTime / k`. The paired vectors are constructed through the relativity library’s electromagnetic tensor. A 256-point decorative grid and the field glyphs form its time-varying slice; translucent planes mark the analytic `Eₓ = ±0.6` isosurfaces. The bounded scene contains 320 sample rows. Tests check the displayed vectors, isosurfaces, and the source-free Maxwell evolution equations by finite differences. The surface and observer paths are synthetic visualization aids, not measurements.

Two spatially projected inertial worldlines and their selected-time markers are overlaid on the field explorer. They are synthetic observer paths, not trajectories driven by the displayed field.

The light-cone time control runs from the past branch through the apex at its midpoint to the future branch. The radius is zero at the apex and equals the absolute normalized time separation elsewhere.

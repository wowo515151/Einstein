# Einstein Relativity

An offline-first JavaScript library and set of interactive 3D demonstrations for special and general relativity. The library is a clean implementation of the mathematical behavior represented by the local Lib4 API; no Python source or bytecode is copied.

## Help

Start at the [help index](docs/README.md) for conventions, API references, equation references, GPU behavior, examples, visualizations, and limitations.

For a static site, use the [HTML help index](docs/README.html). After editing Markdown help, regenerate its HTML with `node scripts/build-help.mjs`.

The current local [development-time estimate](DEVELOPMENT_TIME.md) is provisional and will be updated before delivery.

## Quick start

```js
import { Vector4, LorentzTransform, LightCone, Event4 } from "./src/relativity/index.js";
const event = new Vector4([1, 1, 0, 0]); // (ct, x, y, z)
console.log(event.normSquared());
const boosted = LorentzTransform.boost([0.4, 0, 0]).applyVector(event);
const cone = new LightCone(new Event4([0, 0, 0, 0]), { branch: "future" });
```

Run `node --test` from this directory for the offline unit suite. No build step or network package installation is required for the mathematical library.

To view the static site locally, serve this directory with a local HTTP server. For example, if Python is installed, run `python -m http.server 8000 --bind 127.0.0.1` from this directory, then open `http://127.0.0.1:8000/` in a browser. Stop the server with Ctrl+C. The site also works from a static hosting path; live rendering has not yet been verified in this environment.

## GPU and rendering

`WebGPUBackend` provides optional Float32 batch dot products and normalized plane-wave field sampling. Batch convenience calls default to `auto`, which may use WebGPU; specify `backend: "cpu"` for the Float64 reference path. Devices are requested lazily and unsupported cases fall back to CPU. GPU results can differ from CPU values, and physical GPU execution and accuracy remain unverified. See the [GPU guide](docs/gpu.md). The copied ThreeViz core is under `src/three-viz/` and retains its MIT license in `THREEVIZ-LICENSE`. The visualizations bundle the locally available Three.js r128 runtime and matching OrbitControls script with their MIT notice; the page makes no runtime network requests. A browser with JavaScript-module and WebGL support is required for rendering.

The [third-party provenance record](docs/provenance.md) identifies local modifications, bundled notices, fingerprints, and the unresolved Lib4 license gap.

## Status

The API covers the Lib4 class families, with bounded broadcasting for algebra/field value objects and leading-batch support across selected spacetime, geometry, and Maxwell operations. Three.js value adapters are isolated from the mathematical core. Several Python-specific behaviors, broader method-level parity, and GPU accuracy remain open. See the [port matrix](docs/port-matrix.md) and test results before using numerical outputs in research.

Built by [Executive51.biz](https://Executive51.biz).

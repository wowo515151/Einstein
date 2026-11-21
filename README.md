# Einstein Relativity

An offline-first JavaScript library and set of interactive 3D demonstrations for special and general relativity. The library is a clean implementation of the mathematical behavior represented by the local Lib4 API; no Python source or bytecode is copied.

**Live visualizations:** [Open the Einstein GitHub Pages site](https://wowo515151.github.io/Einstein/).

## Help

Start at the [help index](docs/README.md) for conventions, API references, equation references, GPU behavior, examples, visualizations, and limitations.

For a static site, use the [HTML help index](docs/README.html). After editing Markdown help, regenerate its HTML with `node scripts/build-help.mjs`. Help fingerprints normalize CRLF to LF so Windows and Git line endings produce the same pages.

The [development-time log](DEVELOPMENT_TIME.md) records an unreconciled historical estimate and later measured elapsed windows; the total development time is unknown.

## Quick start

```js
import { Vector4, LorentzTransform, LightCone, Event4 } from "./src/relativity/index.js";
const event = new Vector4([1, 1, 0, 0]); // (ct, x, y, z)
console.log(event.normSquared());
const boosted = LorentzTransform.boost([0.4, 0, 0]).applyVector(event);
const cone = new LightCone(new Event4([0, 0, 0, 0]), { branch: "future" });
```

Run `node --test` from this directory for the offline unit suite. No build step or network package installation is required for the mathematical library.

The package entry point is `src/relativity/index.js`. When this repository is installed as a package in a Node.js project, its name is `einstein-relativity-local` and consumers can use `import { Vector4 } from "einstein-relativity-local"`. The copied ThreeViz entry point is also exposed as `einstein-relativity-local/three-viz`; rendering still requires a compatible Three.js runtime and browser environment. Browser applications can import the same entry points by URLs relative to their hosted files. The package is marked private and is not configured for publication to a package registry.

To view the static site locally, serve this directory with a local HTTP server. For example, if Python is installed, run `python -m http.server 8000 --bind 127.0.0.1` from this directory, then open `http://127.0.0.1:8000/` in a browser. Stop the server with Ctrl+C. On 2026-11-21, isolated headless Chrome rendered all ten scene routes from a loopback server; each displayed a 3D canvas and a “3D scene ready” status, and the screenshots were visually reviewed. Manual browser interaction, assistive-technology behavior, and physical GPU rendering remain unverified.

## GPU and rendering

`WebGPUBackend` provides optional Float32 batch dot products and normalized plane-wave field sampling. Batch convenience calls default to `auto`, which may use WebGPU; specify `backend: "cpu"` for the Float64 reference path. Devices are requested lazily and unsupported cases fall back to CPU. On 2026-11-21, isolated headless Chrome executed both product GPU paths on an Intel Gen12LP adapter: a dot product returned `-3`, and one plane-wave sample matched its analytic electric and magnetic fields. This is a hardware smoke check, not a benchmark or broad accuracy study; GPU results can differ from CPU values, and 3D rendering on physical GPU hardware remains unverified. See the [GPU guide](docs/gpu.md). The copied ThreeViz core is under `src/three-viz/` and retains its MIT license in `THREEVIZ-LICENSE`. The visualizations bundle the locally available Three.js r128 runtime and matching OrbitControls script with their MIT notice; the page makes no runtime network requests. A browser with JavaScript-module and WebGL support is required for rendering.

The [third-party provenance record](docs/provenance.md) identifies local modifications, bundled notices, fingerprints, and the project operator's authorization to use Lib4's behavior. The repository MIT license covers Einstein files only; no Lib4 Python source is included.

## Status

Project development is paused at the user’s request as of 2026-11-21. The last local verification passed 640/640 tests; the public Pages build for source commit bcfba1c65fbc1031c69f8dfe4cd4e6d3651abb33 reports built. All ten live animations have been individually reviewed. The JavaScript library focuses on useful reusable operations for the project and its visualizations; complete feature-by-feature Lib4 parity is not required. Python/PyTorch behavior, broader metric-family coverage, broad GPU accuracy, physical device-loss behavior, physical-GPU rendering, and assistive-technology behavior remain limited or unverified. See the [port matrix](docs/port-matrix.md) and test results before using numerical outputs in research.

Built by [Executive51.biz](https://Executive51.biz).

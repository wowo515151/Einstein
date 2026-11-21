# ThreeViz Integration

The complete ThreeViz `src/` core is copied to `src/three-viz/`; `THREEVIZ-LICENSE` retains its MIT notice. The visualization page bundles the existing local Three.js r128 distribution and matching OrbitControls script in `visualizations/vendor/`; `THREE-LICENSE.txt` preserves their MIT notice. `visualizations/index.html` registers this local runtime before the application module loads. The page makes no runtime network requests. A browser needs JavaScript-module and WebGL support to render scenes; model data and scientific readouts remain available when rendering cannot start. The Minkowski scene lets visitors switch between laboratory and moving-observer coordinates; both inertial worldlines are transformed by the relativity library, and their intervals remain invariant. Its time control moves markers along both worldlines at the selected laboratory time; the Lorentz scene similarly marks rod-end events at a selected rod-rest time. In a moving frame those two source-simultaneous events need not be simultaneous. The Lorentz scene shows the rod-end worldlines plus rest-frame and moving-observer simultaneity slices; it reports the length measured on the selected frame's simultaneous slice and displays the other frame's tilted slice. The particle and lensing scenes mark the latest sampled position on each trajectory; at time zero the lensing rays begin at their initial events.

Applicable scene meshes and field geometry use the copied ThreeViz car-paint material helper and its shared procedural reflection environment. Tests cover palette generation, glossy material defaults and overrides, and shared-environment cleanup. These checks exercise the renderer helpers without claiming a live visual or GPU-hardware review.

Selecting a plotted sample or using the sample inspector pauses playback so the selected coordinates remain visible until the scene data changes.

If a scene update fails before replacing its previous point data, selecting a retained point reports its displayed coordinates and notes when they differ from the current sample data.

Renderer failures during the point update propagate so the prior geometry and overlay can be restored. A later render failure after the update has committed is reported but is not rolled back; real WebGL failure behavior has not been verified.

Direct time and animation-frame updates use the last committed observer frame and model parameter, and preserve prior point data and marker rotation if the point update fails before commit.

The field explorer's wave and its synthetic inertial observer markers share the coordinate-time mapping `ct = 2π time / (1 + beta)`, with normalized playback time from zero to one. The markers move at the declared subluminal speeds over the same wave period; their paths are spatial projections, not trajectories driven by the field.

The `fieldIsosurfaces` helper computes both positive and negative electric-field levels over the requested spatial extent, with playback time interpreted periodically. It rejects requests producing more than 4,096 surfaces or whose extent cannot be represented safely.

Focus the scene and use the arrow keys to orbit or plus and minus to zoom; pointer orbit and zoom are also available when OrbitControls loads. Click or tap a displayed sample to inspect its coordinates, or use the numbered sample inspector with a keyboard. Inspecting by number pauses animation and works from scene data even when 3D rendering is unavailable. Camera drags, multitouch gestures, and secondary mouse clicks do not select a sample; the selection readout clears when the scene data changes.

The public `three-adapter.js` module converts Three.js `Vector3`, `Vector4`, and `Matrix4` values without importing Three.js into the mathematical core. Pass the Three.js namespace to outbound conversions. Vector slots are copied in order; Matrix4 storage is transposed between Three.js column-major `elements` and the library's row-major nested arrays. These adapters handle single values, not batches. See [algebra types](algebra.md#threejs-value-adapters) for the exact component conventions.

The page's animation timer is managed by `visualizations/playback.js`; its injected scheduler makes start, pause, reduced-motion changes, unavailable state, cyclic time, and disposal testable without launching a browser. The page supports model playback and readout updates even when rendering is unavailable, pauses on reduced-motion changes, and disposes the timer on unload. The in-app browser restricted local file navigation, and automatic approval review rejected a separate Chrome visual-review attempt. Live browser rendering remains unverified.

Viewer setup is asynchronous. The app snapshots the requested scene and controls before mounting, then reapplies the current scene, time, observer frame, and parameter if they changed while the renderer was initializing. A failed post-mount update disposes the partially initialized viewer.

Reset scene pauses playback and restores the default time, model parameter, observer frame, and camera view.

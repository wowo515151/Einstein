# Third-Party Code and Provenance

This page records the local files reviewed for this project. It does not establish rights for material whose provenance or license remains unresolved.

## ThreeViz core

`src/three-viz/` contains the 27 files from the local 3DViz project's `library/three-viz/src/` tree. The source project's index identifies the repository-root license as MIT; the copied notice is [THREEVIZ-LICENSE](../THREEVIZ-LICENSE). In the local audit on 2026-11-21, the source and copied notices had identical SHA-256 hashes, and all 27 source files were present with no extra files. Two files differed from the local source snapshot because of project changes:

- `src/three-viz/core/mount.js`: selection gesture handling and render-failure propagation.
- `src/three-viz/plotters/point-cloud-plotter.js`: selectable point data and update rollback behavior.

The rest of the core was byte-identical to the local source at that audit. This statement describes that comparison date, not future changes.

## Three.js and OrbitControls

The visualization loads existing local copies of Three.js r128 and its matching OrbitControls script from `visualizations/vendor/`; it does not fetch them at runtime. The retained MIT notice is [THREE-LICENSE.txt](../visualizations/vendor/THREE-LICENSE.txt). Their audited SHA-256 fingerprints are:

| Local file | SHA-256 |
| --- | --- |
| `three-r128.min.js` | `9274BBCEC8D96168626C732B5D31C775AA8CFB7EAA0599BEC0C175908A2C1CE2` |
| `OrbitControls-r128.js` | `02BB4ADE710F3E607329E37A21F098BC3AC70EB6E33DAF8A65E79F4DB785E7B2` |

The fingerprints identify the bundled local copies. The official archive was not obtained through the blocked route, so these hashes are not a claim of independent upstream verification. Runtime version and local geometry checks do not replace a live browser or hardware review.

## Lib4

The JavaScript library implements mathematical behavior from the inspected local Lib4 package; Python source and bytecode are not copied into this repository. No license file was found in the inspected Lib4 directory. Resolve its provenance and any applicable rights before external publication or source-level parity claims. See the [port matrix](port-matrix.md) for current coverage and known differences.

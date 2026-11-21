import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import * as relativity from "../src/relativity/index.js";

const inventory = readFileSync(new URL("../docs/lib4-method-inventory.md", import.meta.url), "utf8");
const rows = [...inventory.matchAll(/^\| `([^`]+)` \| `([^`]+)` \| (.*) \|$/gm)];
const unsupported = "No direct JavaScript-array equivalent";
const mappings = [];
const unsupportedSurfaces = [];

for (const [, moduleName, className, cell] of rows) {
  const Constructor = relativity[className];
  assert.equal(typeof Constructor, "function", `${className} is exported for ${moduleName}`);
  for (const entry of cell.split("<br>")) {
    const [sourceSurface, javascriptSurface] = entry.split("→").map((part) => part.trim());
    assert.ok(sourceSurface && javascriptSurface, `well-formed mapping in ${moduleName}: ${entry}`);
    if (javascriptSurface.includes(unsupported)) {
      const sourceName = sourceSurface.match(/`([^`]+)`/)?.[1];
      unsupportedSurfaces.push(`${className}.${sourceName}`);
      continue;
    }
    for (const match of javascriptSurface.matchAll(/`([^`]+)`/g)) {
      for (const candidate of match[1].split("/").map((name) => name.trim())) {
        if (candidate.endsWith("()")) {
          mappings.push({ moduleName, className, sourceSurface, name: candidate.slice(0, -2), callable: true });
        } else {
          mappings.push({ moduleName, className, sourceSurface, name: candidate, callable: false });
        }
      }
    }
  }
}

test("Lib4 method inventory covers every public class surface", () => {
  assert.equal(rows.length, 28, "public classes in the inspected Lib4 inventory");
  const sourceSurfaceCount = rows.reduce((count, [, , , cell]) => count + cell.split("<br>").length, 0);
  assert.equal(sourceSurfaceCount, 175, "inspected methods, properties and selected data fields");
  assert.equal(mappings.length, 174, "mapped JavaScript surfaces, including the squared alias");
  assert.deepEqual(unsupportedSurfaces.sort(), ["Vector4.device", "Vector4.dtype"]);
});

test("Lib4 operator and callable protocols map to named JavaScript methods", () => {
  const protocols = [
    ["Vector4", "__add__", "add"],
    ["Vector4", "__sub__", "subtract"],
    ["Vector4", "__mul__", "scale"],
    ["Vector4", "__rmul__", "scale"],
    ["Covector4", "__add__", "add"],
    ["Covector4", "__mul__", "scale"],
    ["Covector4", "__rmul__", "scale"],
    ["Matrix4", "__matmul__", "multiply"],
    ["Matrix4", "__matmul__", "apply"],
    ["Worldline", "__call__", "call"],
  ];
  assert.equal(protocols.length, 10, "nine Python protocol methods, with matrix multiplication's two operand types");
  assert.match(inventory, /Python's generated `Vector4\.__repr__` is intentionally excluded/);
  for (const [className, pythonProtocol, javascriptMethod] of protocols) {
    assert.ok(inventory.includes(`| \`${className}.${pythonProtocol}\` |`), `${className}.${pythonProtocol} is inventoried`);
    assert.ok(inventory.includes(`\`${className}.${javascriptMethod}()\``), `${className}.${javascriptMethod} is documented`);
    assert.equal(typeof relativity[className].prototype[javascriptMethod], "function", `${className}.${javascriptMethod} exists`);
  }
});

test("Lib4 module-level exports map to public JavaScript exports", () => {
  assert.match(inventory, /\| `SPEED_OF_LIGHT` \| `SPEED_OF_LIGHT` \|/);
  assert.match(inventory, /\| `minkowski_dot\(a,b\)` \| `minkowskiDot\(a,b\)` \|/);
  assert.equal(relativity.SPEED_OF_LIGHT, 299792458);
  assert.equal(typeof relativity.minkowskiDot, "function");
  assert.equal(relativity.minkowskiDot([1, 0, 0, 0], [1, 0, 0, 0]), -1);
});

for (const mapping of mappings) {
  test(`${mapping.moduleName}: ${mapping.className}.${mapping.sourceSurface}`, () => {
    const Constructor = relativity[mapping.className];
    const descriptor = Object.getOwnPropertyDescriptor(Constructor.prototype, mapping.name)
      ?? Object.getOwnPropertyDescriptor(Constructor, mapping.name);
    assert.ok(descriptor, `${mapping.className}.${mapping.name} exists`);
    if (mapping.callable) {
      assert.equal(typeof descriptor.value, "function", `${mapping.className}.${mapping.name} is callable`);
    }
  });
}

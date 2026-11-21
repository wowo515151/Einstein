import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir, access } from "node:fs/promises";
import { createHash } from "node:crypto";
import { generatePage } from "../scripts/build-help.mjs";

test("static help has a page for each Markdown document and local links resolve", async () => {
  const docs = new URL("../docs/", import.meta.url);
  const markdown = (await readdir(docs)).filter(name => name.endsWith(".md"));
  const generatorHash = createHash("sha256").update((await readFile(new URL("../scripts/build-help.mjs", import.meta.url), "utf8")).replaceAll("\r\n", "\n")).digest("hex");
  for (const name of markdown) {
    const page = new URL(name.replace(/\.md$/, ".html"), docs);
    const html = await readFile(page, "utf8");
    const source = await readFile(new URL(name, docs), "utf8");
    const hash = createHash("sha256").update(source.replaceAll("\r\n", "\n")).digest("hex");
    assert.ok(html.includes(`name="source-sha256" content="${hash}"`), `${name} HTML is stale`);
    assert.ok(html.includes(`name="generator-sha256" content="${generatorHash}"`), `${name} HTML uses an outdated generator`);
    assert.equal(html, generatePage(name, source), `${name} HTML differs from generated output`);
    assert.match(html, /<main>[\s\S]*<\/main>/);
    for (const [, href] of html.matchAll(/href="([^"]+)"/g)) {
      if (/^https:\/\//.test(href)) continue;
      assert.doesNotMatch(href, /\.md(?:#|$)/);
      const [target, fragment] = href.split("#");
      const resolved = target ? new URL(target, page) : page;
      await access(resolved);
      if (fragment) assert.ok((await readFile(resolved, "utf8")).includes(`id="${fragment}"`), `${href} has no matching heading`);
    }
  }
  const references = await readFile(new URL("references.html", docs), "utf8");
  assert.match(references, /c\(t-tᵣ\)=\|x-xₛ\(tᵣ\)\|/);
});

test("help output and source fingerprints do not depend on Windows line endings", () => {
  const markdown = "# Sample\n\nA paragraph.\n";
  assert.equal(generatePage("sample.md", markdown), generatePage("sample.md", markdown.replaceAll("\n", "\r\n")));
});

test("help renders inventory line breaks while escaping other HTML", () => {
  const html = generatePage("sample.md", "# Sample\n\n| API | Mapping |\n| --- | --- |\n| `one`<br>`two` | <script>alert(1)</script> |\n");
  assert.match(html, /<code>one<\/code><br><code>two<\/code>/);
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.doesNotMatch(html, /<script>/);
});

test("RetardedField help documents iteration and tolerance defaults and bounds", async () => {
  const physics = await readFile(new URL("../docs/physics.md", import.meta.url), "utf8");
  assert.match(physics, /`maxIterations` defaults to `96` and must be an integer from `1` to `4096`/);
  assert.match(physics, /`tolerance` defaults to `1e-12` and must be positive and finite/);
});

test("Geodesic4 help documents integration-step defaults and bounds", async () => {
  const geometry = await readFile(new URL("../docs/geometry.md", import.meta.url), "utf8");
  assert.match(geometry, /`steps` defaults to `100` and must be an integer from `1` to `10,000`/);
});

test("ODEWorldline help documents integration-step defaults and implementation bounds", async () => {
  const spacetime = await readFile(new URL("../docs/spacetime.md", import.meta.url), "utf8");
  const row = spacetime.split("\n").find(line => line.startsWith("| `new ODEWorldline"));
  assert.ok(row, "ODEWorldline integration API row exists");
  assert.match(row, /`steps` defaults to `100`/);
  assert.match(row, /JavaScript requires an integer from `1` to `10,000`/);
  assert.match(row, /Lib4 rejects values below `1` and has no matching upper cap/);
});

test("LightCone and SpatialSlice help documents constructor defaults and bounds", async () => {
  const spacetime = await readFile(new URL("../docs/spacetime.md", import.meta.url), "utf8");
  const cone = spacetime.split("\n").find(line => line.startsWith("| `new LightCone"));
  const slice = spacetime.split("\n").find(line => line.startsWith("| `new SpatialSlice"));
  assert.ok(cone, "LightCone constructor API row exists");
  assert.match(cone, /`branch` defaults to `both`, `c` to `C`, and `atol` to `1e-10`/);
  assert.match(cone, /`c` must be finite and positive, and `atol` finite and nonnegative/);
  assert.ok(slice, "SpatialSlice constructor API row exists");
  assert.match(slice, /`basis` defaults to `null`, `c` to `C`, and `atol` to `1e-9`/);
  assert.match(slice, /`c` must be finite and positive, and `atol` finite and nonnegative/);
});

test("Geodesic4 boundary-shooting help documents controls and work bounds", async () => {
  const geometry = await readFile(new URL("../docs/geometry.md", import.meta.url), "utf8");
  const matrix = await readFile(new URL("../docs/port-matrix.md", import.meta.url), "utf8");
  const row = geometry.split("\n").find(line => line.startsWith("| `geodesic.solveBoundary"));
  assert.ok(row, "boundary-shooting API row exists");
  assert.match(row, /`steps` defaults to `100` and must be an integer from `1` to `2,000`/);
  assert.match(row, /`maxIterations` defaults to `12`, accepts `iterations` as an alias, and must be an integer from `0` to `12`/);
  assert.match(row, /`tolerance` defaults to `1e-9` and `damping` to `1e-10`; both must be positive and finite/);
  assert.match(row, /`jacobianMethod` defaults to `"finite"`/);
  assert.match(row, /`steps\*\(9\*maxIterations\+2\) <= 12,000`/);
  assert.match(matrix, /JavaScript requires finite positive boundary `tolerance` and `damping`; Lib4 checks only that both compare greater than zero/);
});

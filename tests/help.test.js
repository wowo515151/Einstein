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

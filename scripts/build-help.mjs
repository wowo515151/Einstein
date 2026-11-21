import { readdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";

const docs = new URL("../docs/", import.meta.url);
const generatorHash = createHash("sha256").update(await readFile(new URL(import.meta.url))).digest("hex");
const escape = text => text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
function inline(text) {
  text = text.replaceAll("\\|", "|");
  const parts = text.split(/(`[^`]+`|\[[^\]]+\]\([^)]+\))/g);
  return parts.map(part => {
    if (part.startsWith("`") && part.endsWith("`")) return `<code>${escape(part.slice(1, -1))}</code>`;
    const link = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(part);
    if (link) {
      const href = link[2].replace(/\.md(?=#|$)/, ".html");
      if (!/^(?:https:\/\/|[A-Za-z0-9._/#-]+$)/.test(href)) return escape(part);
      return `<a href="${escape(href)}">${escape(link[1])}</a>`;
    }
    return escape(part).replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>").replace(/\*([^*]+)\*/g, "<em>$1</em>");
  }).join("");
}
function render(markdown) {
  const lines = markdown.replaceAll("\r\n", "\n").split("\n"), out = [];
  const headingCounts = new Map();
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.startsWith("```")) {
      i++; const code = []; while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]);
      if (i === lines.length) throw new Error("Unclosed code fence");
      i++; out.push(`<pre><code>${escape(code.join("\n"))}</code></pre>`); continue;
    }
    const heading = /^(#{1,6})\s+(.+)$/.exec(line);
    if (heading) {
      const base = heading[2].replace(/`/g, "").toLowerCase().replace(/[^\p{L}\p{N}_\-\s]/gu, "").replace(/\s+/g, "-");
      const count = headingCounts.get(base) ?? 0; headingCounts.set(base, count + 1);
      const id = count ? `${base}-${count}` : base;
      out.push(`<h${heading[1].length} id="${escape(id)}">${inline(heading[2])}</h${heading[1].length}>`); i++; continue;
    }
    if (/^\|/.test(line)) {
      const rows = []; while (i < lines.length && /^\|/.test(lines[i])) rows.push(lines[i++].slice(1, -1).split(/(?<!\\)\|/).map(cell => cell.trim()));
      if (rows.length < 2 || !rows[1].every(cell => /^:?-+:?$/.test(cell))) throw new Error("Invalid Markdown table");
      if (rows.some(row => row.length !== rows[0].length)) throw new Error("Inconsistent Markdown table columns");
      const cells = (row, tag) => `<tr>${row.map(cell => `<${tag}>${inline(cell)}</${tag}>`).join("")}</tr>`;
      out.push(`<div class="table-wrap"><table><thead>${cells(rows[0], "th")}</thead><tbody>${rows.slice(2).map(row => cells(row, "td")).join("")}</tbody></table></div>`); continue;
    }
    if (/^- /.test(line)) {
      const items = []; while (i < lines.length && /^- /.test(lines[i])) items.push(`<li>${inline(lines[i++].slice(2))}</li>`);
      out.push(`<ul>${items.join("")}</ul>`); continue;
    }
    const paragraph = []; while (i < lines.length && lines[i].trim() && !/^(?:#{1,6}\s|```|\||- )/.test(lines[i])) paragraph.push(lines[i++]);
    out.push(`<p>${inline(paragraph.join(" "))}</p>`);
  }
  return out.join("\n");
}
export function generatePage(name, markdown) {
  const title = /^# (.+)$/m.exec(markdown)?.[1] ?? name;
  const sourceHash = createHash("sha256").update(markdown).digest("hex");
  return `<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="source-sha256" content="${sourceHash}"><meta name="generator-sha256" content="${generatorHash}"><title>${escape(title)}</title><link rel="stylesheet" href="help.css"></head><body><nav><a href="README.html">Help index</a> · <a href="../index.html">Visualizations</a></nav><main>${render(markdown)}</main></body></html>\n`;
}
if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  for (const name of (await readdir(docs)).filter(name => name.endsWith(".md"))) {
    const markdown = await readFile(new URL(name, docs), "utf8");
    await writeFile(new URL(name.replace(/\.md$/, ".html"), docs), generatePage(name, markdown));
  }
}

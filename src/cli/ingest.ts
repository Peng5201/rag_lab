/**
 * 入库前处理管线：物料目录 → 解析 → 归位 → <文档>/RAG-Lab/knowledge
 *
 *   npm run ingest -- <物料目录> [--out <目录>] [--no-vlm] [--max-images 5]
 *
 * 三层边界：
 *   物料层  原图 / 视频 / docx 原件，不进 knowledge
 *   解析层  assetParsers：docx→Markdown、图片→文字、视频→引用
 *   知识层  knowledge/*.md、*.csv —— RAG 只吃这里
 *
 * 归位是「判断」不是「规则」：图片解析结果该补充到哪个章节，交给 LLM
 * （图片信息常常同时属于两个字段，规则做不出「一份信息归多处」）。
 */
import { copyFile, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

import {
  detectKind,
  imageBlock,
  parseImage,
  parseJsonObject,
  parseOfficeDocument,
  parseUnknownDocument,
  referenceVideo,
  type ParsedImage,
  type VisionOptions,
} from "../ingest/parsers.js";
import { defaultKnowledgeDir } from "../knowledge/paths.js";
import { createChatModel, describeConfig, resolveConfig } from "../llm/providers.js";
import type { ChatModelType } from "../llm/model.js";

/* ------------------------------ CLI ------------------------------ */

/** 默认知识库目录：<文档>/RAG-Lab/knowledge，与代码仓库分离 */
const PROJECT_KNOWLEDGE = defaultKnowledgeDir();

interface CliOptions {
  srcDir: string;
  outDir: string;
  useVlm: boolean;
  maxImages: number;
  /** PDF 页数上限（默认不限） */
  maxPages?: number;
  vlmModel?: string;
}

function parseCli(argv: string[]): CliOptions {
  const positional = argv.filter((a) => !a.startsWith("--"));
  const pick = (name: string) => argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1];
  const srcDir = positional[0];
  if (!srcDir) {
    console.error("用法：npm run ingest -- <物料目录> [--out <目录>] [--no-vlm] [--max-images 5] [--max-pages 30]");
    process.exit(1);
  }
  return {
    srcDir,
    // 相对路径按当前目录解析；默认写到系统「文档」下的 RAG-Lab/knowledge
    outDir: pick("out") ? path.resolve(pick("out") as string) : PROJECT_KNOWLEDGE,
    useVlm: !argv.includes("--no-vlm"),
    maxImages: Number(pick("max-images") ?? Infinity),
    maxPages: pick("max-pages") ? Number(pick("max-pages")) : undefined,
    vlmModel: pick("vlm-model"),
  };
}

/* --------------------------- 归位（LLM） --------------------------- */

const PLACE_PROMPT = ChatPromptTemplate.fromMessages([
  [
    "human",
    "[TASK:place] 下面是文档《{doc}》的章节标题，以及一张图片解析出的信息。\n" +
      "请判断这些信息应该补充到哪几个章节：从给出的章节标题原样选取，**最多两个**最相关的；" +
      "都不合适就返回空数组。\n" +
      '只返回 JSON：{{"sections": ["章节标题"]}}\n\n章节：\n{sections}\n\n图片信息：\n{info}',
  ],
]);

/** 章节名归一化：模型常把标点、空格、全半角写法改掉，严格等值匹配会全落空 */
function normalizeTitle(s: string): string {
  return s.replace(/[\s\p{P}]/gu, "").toLowerCase();
}

function matchSections(picked: string[], sections: string[]): string[] {
  const out: string[] = [];
  for (const sec of sections) {
    const norm = normalizeTitle(sec);
    const hit = picked.some((p) => {
      const q = normalizeTitle(p);
      if (!q) return false;
      return q === norm || (Math.min(q.length, norm.length) >= 4 && (norm.includes(q) || q.includes(norm)));
    });
    if (hit && !out.includes(sec)) out.push(sec);
  }
  return out.slice(0, 2);
}

async function askPlacement(
  model: ChatModelType,
  docName: string,
  sections: string[],
  img: ParsedImage,
): Promise<string[]> {
  if (!sections.length) return [];
  const chain = PLACE_PROMPT.pipe(model).pipe(new StringOutputParser());
  const raw: string = await chain.invoke({
    doc: docName,
    sections: sections.map((s, i) => `${i + 1}. ${s}`).join("\n"),
    info: `${img.caption ? `概括：${img.caption}\n` : ""}${img.parsedText}`.slice(0, 3000),
  });
  const list = parseJsonObject(raw).sections;
  const picked = (Array.isArray(list) ? list : []).map(String).map((s) => s.trim());
  return matchSections(picked, sections);
}

/* --------------------------- Markdown 组装 --------------------------- */

function headingLevel(line: string): number {
  const m = /^(#{1,6})\s+/.exec(line);
  return m ? m[1].length : 0;
}

function headingText(line: string): string {
  return line.replace(/^#{1,6}\s+/, "").trim();
}

/** 在指定章节末尾（下一个同级或更高级标题前）插入内容 */
function insertIntoSection(markdown: string, title: string, block: string): string {
  const lines = markdown.split("\n");
  const idx = lines.findIndex((l) => headingLevel(l) > 0 && headingText(l) === title);
  if (idx < 0) return markdown;
  const level = headingLevel(lines[idx]);
  let end = lines.length;
  for (let i = idx + 1; i < lines.length; i++) {
    const lv = headingLevel(lines[i]);
    if (lv > 0 && lv <= level) {
      end = i;
      break;
    }
  }
  lines.splice(end, 0, "", block);
  return lines.join("\n");
}

/** 把内容插到图片占位符下方（归位失败时的兜底） */
function insertAfterImage(markdown: string, ref: string, block: string): string {
  const lines = markdown.split("\n");
  const idx = lines.findIndex((l) => l.includes(ref));
  if (idx < 0) return `${markdown}\n\n${block}`;
  lines.splice(idx + 1, 0, "", block);
  return lines.join("\n");
}


function manifestTable(images: ParsedImage[], comments: string[]): string {
  const rows = images.map((img) => {
    const file = img.ref.split("/").pop() ?? img.ref;
    const kind = /\.(mp4|mov|avi|mkv|webm)$/i.test(img.ref) ? "视频" : "图片";
    return `| ${file} | ${kind} | ${img.status} | ${img.caption || "—"} |`;
  });
  if (comments.length) rows.push(`| 批注 ×${comments.length} | 批注 | parsed | 见下 |`);
  return [
    "## 附：物料清单",
    "",
    "| 物料 | 类型 | 状态 | 说明 |",
    "| --- | --- | --- | --- |",
    ...rows,
  ].join("\n");
}

/* ------------------------------ 主流程 ------------------------------ */

/** 简单并发池：限制同时在飞的任务数，保持完成顺序无关（结果写回各自的 img） */
async function mapLimit<T>(items: T[], limit: number, fn: (item: T) => Promise<void>): Promise<void> {
  const queue = [...items];
  const workers = Array.from({ length: Math.max(1, Math.min(limit, queue.length)) }, async () => {
    while (queue.length) await fn(queue.shift() as T);
  });
  await Promise.all(workers);
}

/** 视觉解析走 OpenAI 兼容端点（多模态模型）；未开识图或没配端点就返回 undefined 跳过 */
function visionOptions(
  cli: CliOptions,
  cfg: ReturnType<typeof resolveConfig>,
): VisionOptions | undefined {
  if (!cli.useVlm || !cfg.baseUrl) return undefined;
  return {
    baseUrl: cfg.baseUrl,
    apiKey: cfg.apiKey,
    model: cli.vlmModel ?? cfg.vlmModel,
  };
}

/** 递归收集物料文件；name 是相对路径，保留子目录结构，避免同名文件互相覆盖 */
async function collectFiles(dir: string, prefix = ""): Promise<{ full: string; name: string }[]> {
  const out: { full: string; name: string }[] = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...(await collectFiles(full, rel)));
    } else {
      out.push({ full, name: rel });
    }
  }
  return out.sort((a, b) => a.name.localeCompare(b.name));
}

/** 写出知识单元，子目录自动创建 */
async function writeOut(outDir: string, relPath: string, content: string): Promise<void> {
  const target = path.join(outDir, relPath);
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, content);
}

async function main() {
  const cli = parseCli(process.argv.slice(2));
  const cfg = resolveConfig();
  const model = createChatModel(cfg);
  const vision = visionOptions(cli, cfg);
  console.log(`[ingest] 物料目录 ${cli.srcDir}`);
  console.log(`[ingest] 输出目录 ${cli.outDir}`);
  console.log(`[ingest] ${describeConfig(cfg)}  vlm=${vision ? `openai:${vision.model}` : "关闭"}`);

  await mkdir(cli.outDir, { recursive: true });
  const mediaRoot = path.join(cli.outDir, "_media");
  const files = await collectFiles(cli.srcDir);
  const manifest: Record<string, unknown>[] = [];
  let vlmUsed = 0;

  for (const { full, name } of files) {
    const kind = detectKind(name);
    const base = name.replace(/\.[^.]+$/, "");

    // 1) 表格：结构化数据按行存，检索走「字段: 值」，不转 md 表格
    if (kind === "table") {
      await mkdir(path.dirname(path.join(cli.outDir, name)), { recursive: true });
      await copyFile(full, path.join(cli.outDir, name));
      manifest.push({ file: name, kind, status: "copied" });
      console.log(`  [表格] ${name} → 原样复制（走行级 KV）`);
      continue;
    }

    // 2) 视频：不解析，只留引用（信息密度低、逐帧解析成本高）
    if (kind === "video") {
      const ref = referenceVideo(full);
      manifest.push({ file: name, kind, status: "referenced", caption: ref.caption });
      console.log(`  [视频] ${name} → 只留引用，不入库（需依附文档）`);
      continue;
    }

    // 3) 图片：视觉解析 → 一个图一个知识单元
    if (kind === "image") {
      const file = path.basename(name);
      const ref = `_media/${base}/${file}`;
      await mkdir(path.join(mediaRoot, base), { recursive: true });
      await copyFile(full, path.join(mediaRoot, base, file));
      const img =
        vision && vlmUsed < cli.maxImages
          ? ((vlmUsed += 1), await parseImage(full, ref, vision))
          : { ref, caption: "", parsedText: "", status: "skipped" as const };
      await writeOut(
        cli.outDir,
        `${base}.md`,
        [`# ${base}`, "", imageBlock(img), "", manifestTable([img], [])].join("\n"),
      );
      manifest.push({ file: name, kind, status: img.status, caption: img.caption });
      console.log(`  [图片] ${name} → ${img.status}${img.caption ? `（${img.caption.slice(0, 30)}）` : ""}`);
      continue;
    }

    // 4) 文档：office 文档走 officeparser（本地解析、不耗模型），未知格式走多模态兜底
    const mediaDir = path.join(mediaRoot, base);
    const title = path.basename(name).replace(/\.[^.]+$/, "");
    let markdown = "";
    let images: ParsedImage[] = [];
    let comments: string[] = [];

    if (kind === "office") {
      // docx / pptx / xlsx / pdf 统一读取；只有 PDF 扫描版会回退到多模态
      const parsed = await parseOfficeDocument(full, {
        mediaDir,
        docKey: base,
        vision,
        maxPages: cli.maxPages,
        onProgress: (done, total) => console.log(`       · ${done}/${total} 块`),
      });
      if (!parsed) {
        manifest.push({ file: name, kind, status: "unreadable", note: "文本层为空，需配 chat 走多模态" });
        console.log(`  [跳过] ${name} → 抽不出文本（扫描件需配 chat 走多模态解析）`);
        continue;
      }
      markdown = parsed.markdown.startsWith("#") ? parsed.markdown : `# ${title}\n\n${parsed.markdown}`;
      images = parsed.images;
      comments = parsed.comments;
      // 只有 PDF 扫描页是「已解析」（parsePdf 内部走完视觉）；docx 抽出的图还是 skipped，交给下面的循环
      vlmUsed += parsed.images.filter((i) => i.status === "parsed").length;
      console.log(
        `  [office] ${name} → ${parsed.markdown.length} 字` +
          `${images.length ? ` / ${images.length} 图` : ""}${comments.length ? ` / ${comments.length} 批注` : ""}`,
      );
    } else if (kind === "multimodal") {
      // 未知格式兜底：抽可读文本 → 大模型整理成知识单元
      if (!vision) {
        manifest.push({ file: name, kind, status: "needs-vision" });
        console.log(`  [跳过] ${name} → 未知格式需大模型解析，请先配置 chat（run.ps1 配置块）`);
        continue;
      }
      const parsed = await parseUnknownDocument(full, { docKey: base, vision });
      if (!parsed) {
        manifest.push({ file: name, kind, status: "unreadable", note: "抽不出可用内容" });
        console.log(`  [跳过] ${name} → 抽不出可用内容（二进制格式请先转 .docx / .pdf）`);
        continue;
      }
      markdown = parsed.markdown.startsWith("#") ? parsed.markdown : `# ${title}\n\n${parsed.markdown}`;
      images = parsed.images;
      vlmUsed += parsed.images.length;
      console.log(`  [兜底] ${name} → 交给大模型整理`);
    } else {
      markdown = (await readFile(full, "utf-8")).trim();
      console.log(`  [md]   ${name} → 原样入库`);
    }

    // 图片解析（原件已落在 outDir/_media/<doc>/）；并发 3 路
    // 只解析还没解析过的：office 抽出的图是 skipped，PDF 扫描页在 parsePdf 里已解析完
    if (vision) {
      const budget = Math.max(0, cli.maxImages - vlmUsed);
      const targets = images.filter((img) => img.status === "skipped").slice(0, budget);
      vlmUsed += targets.length;
      await mapLimit(targets, 3, async (img) => {
        const local = path.join(mediaDir, path.basename(img.ref));
        Object.assign(img, await parseImage(local, img.ref, vision));
        console.log(`       · ${path.basename(img.ref)} → ${img.status}`);
      });
    }

    // 归位：图片信息该进哪个章节，交给 LLM 判断
    const sections = [...markdown.matchAll(/^#{1,6}\s+(.+)$/gm)].map((m) => m[1].trim());
    for (const img of images) {
      const file = path.basename(img.ref);
      // PDF 的页已按页序就位，不重复归位
      if (img.placed) {
        manifest.push({ file, kind: "image", status: img.status, placed: [] });
        continue;
      }
      // 解析失败/跳过的也要留档：知识库的完整性本身是可信度的一部分
      if (img.status !== "parsed" || (!img.parsedText && !img.caption)) {
        manifest.push({ file, kind: "image", status: img.status, placed: [] });
        continue;
      }
      const block = imageBlock(img);
      // 归位是增强、不是必需：chat 模型不可用时降级到图片原位，别让整个入库挂掉
      let targets: string[] = [];
      try {
        targets = await askPlacement(model, name, sections, img);
      } catch (e) {
        console.error(`       · 归位失败，降级到图片原位：${(e as Error).message.split("\n")[0]}`);
      }
      if (targets.length) {
        for (const t of targets) markdown = insertIntoSection(markdown, t, block);
        console.log(`       · 归位 → ${targets.join(" / ")}`);
      } else {
        // LLM 没找到更合适的章节，就放回图片原位——别丢，也别塞到不相干的章节
        markdown = insertAfterImage(markdown, img.ref, block);
        console.log(`       · 归位 → 图片原位（LLM 未给出更合适的章节）`);
      }
      manifest.push({ file, kind: "image", status: img.status, placed: targets });
    }

    // 汇总输出
    const parts = [markdown, "", "---", "", manifestTable(images, comments)];
    if (comments.length) {
      parts.push("", "## 附：批注", "", ...comments.map((c, i) => `${i + 1}. ${c}`));
    }
    await writeOut(cli.outDir, `${base}.md`, parts.join("\n"));
  }

  await writeFile(path.join(cli.outDir, "_manifest.json"), JSON.stringify(manifest, null, 2));
  console.log(`\n[ingest] 完成：${files.length} 个物料，视觉解析 ${vlmUsed} 张`);
  console.log(`[ingest] 下一步：RAG_KNOWLEDGE_DIR=${cli.outDir} npm run eval`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

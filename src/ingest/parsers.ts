/**
 * 物料解析层 —— 把多模态物料转成可入库的文本
 *
 * 边界（重要）：
 *   物料层存原件 → 解析层产出 parsed_text + 状态 → 知识层只留结构化文本 + 引用
 * 这里只负责「解析」，不负责「归位」——图片信息该进哪个章节是语义判断，
 * 由 cli/ingest.ts 交给 LLM，不用规则硬猜。
 *
 * 读取统一走 officeparser：docx / pptx / xlsx / pdf 一个入口，
 * 保留标题层级、幻灯片/页/工作表边界、表格结构。
 *
 * 分派规则：
 *   .docx/.pptx/.xlsx/.pdf → parseOfficeDocument（officeparser）
 *   图片                   → parseImage（多模态模型抽文字与关键事实）
 *   视频                   → 只留引用（信息密度低、逐帧解析成本高）
 *   .md / .txt / .csv      → 已是干净文本，直接入库
 *   其他/未知               → 多模态兜底：抽得动就交模型，抽不动就跳过
 */
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import officeParser from "officeparser";

import { openAIChat, toDataUrl } from "../llm/openai.js";

export type AssetKind = "doc" | "table" | "image" | "video" | "office" | "multimodal";

export interface Asset {
  path: string;
  name: string;
  kind: AssetKind;
}

export interface ParsedImage {
  /** 相对知识库的引用路径，如 _media/投稿/image1.png */
  ref: string;
  /** 一句话概括 */
  caption: string;
  /** 抽出的文字与关键事实 */
  parsedText: string;
  status: "parsed" | "failed" | "skipped";
  /** 已按顺序就位（如 PDF 的页）：不再做章节归位，也不再重复解析 */
  placed?: boolean;
}

export interface ParsedDoc {
  name: string;
  /** 正文 Markdown，图片以 ![alt](ref) 保留原始位置 */
  markdown: string;
  images: ParsedImage[];
  comments: string[];
}

const IMAGE_EXT = /\.(png|jpe?g|gif|webp|bmp)$/i;
const VIDEO_EXT = /\.(mp4|mov|avi|mkv|webm)$/i;
const OFFICE_EXT = /\.(docx|pptx|xlsx|pdf)$/i;

export function detectKind(name: string): AssetKind {
  if (/\.csv$/i.test(name)) return "table";
  if (IMAGE_EXT.test(name)) return "image";
  if (VIDEO_EXT.test(name)) return "video";
  if (/\.(md|txt)$/i.test(name)) return "doc";
  if (OFFICE_EXT.test(name)) return "office";
  // 其余（.doc / .ppt / 未知扩展名）→ 多模态兜底
  return "multimodal";
}

/* ------------------------------------------------------------------ */
/* 块 → Markdown                                                       */
/* ------------------------------------------------------------------ */

type Block = {
  type: string;
  text?: string;
  children?: Block[];
  metadata?: Record<string, any>;
  formatting?: { bold?: boolean; size?: string };
};

function fontSize(b: Block): number {
  const raw = b.formatting?.size ?? firstRun(b)?.formatting?.size;
  return raw ? parseFloat(String(raw)) || 0 : 0;
}

function firstRun(b: Block): Block | undefined {
  return (b.children ?? []).find((c) => c.type === "text");
}

function styleOf(b: Block): string | undefined {
  const own = b.metadata?.style;
  if (own) return String(own);
  for (const c of b.children ?? []) {
    const s = c.metadata?.style ?? firstRun(c)?.metadata?.style;
    if (s) return String(s);
  }
  return undefined;
}

/**
 * 标题层级判定：
 *   1) Word/WPS 样式 —— `2..7` = 标题 1..6（WPS 导出用数字 ID），`Heading1`/`标题 1` 也认
 *   2) 无样式时按字号 + 加粗兜底（很多文档的标题是手动排的，不走样式）
 */
function headingLevel(b: Block): number | null {
  const style = styleOf(b);
  if (style) {
    const named = /(?:heading|标题)\s*(\d)/i.exec(style);
    if (named) return Number(named[1]);
    if (/^(\d)$/.test(style)) {
      const n = Number(style);
      return n >= 2 && n <= 7 ? n - 1 : null; // 1 = 正文
    }
    if (/^main_?title$|^title$/i.test(style)) return 1;
  }

  const size = fontSize(b);
  const bold = (b.children ?? []).some((c) => c.formatting?.bold || firstRun(c)?.formatting?.bold);
  if (bold && size >= 16) return 1;
  if (bold && size >= 13.5) return 2;
  return null;
}

function cellText(cell: Block): string {
  return (cell.children ?? [])
    .map((c) => (c.type === "table" ? "" : c.text ?? cellText(c)))
    .filter(Boolean)
    .join(" ")
    .trim();
}

function tableToMd(b: Block): string {
  const rows = (b.children ?? []).filter((r) => r.type === "row");
  const cells = rows.map((r) => (r.children ?? []).map((c) => cellText(c).replace(/\|/g, "\\|")));
  if (!cells.length) return "";
  const width = Math.max(...cells.map((r) => r.length));
  const line = (r: string[]) => `| ${[...r, ...Array(width - r.length).fill("")].join(" | ")} |`;
  return [line(cells[0]), line(Array(width).fill("---")), ...cells.slice(1).map(line)].join("\n");
}

function blockToMd(b: Block, nextIndex: () => number): string {
  switch (b.type) {
    case "slide":
    case "page":
    case "sheet": {
      const label =
        b.type === "slide"
          ? `幻灯片 ${nextIndex()}`
          : b.type === "page"
            ? `第 ${nextIndex()} 页`
            : String(b.metadata?.sheetName ?? `工作表 ${nextIndex()}`);
      const inner = (b.children ?? []).map((c) => blockToMd(c, nextIndex)).filter(Boolean).join("\n\n");
      return inner ? `## ${label}\n\n${inner}` : `## ${label}`;
    }
    case "table":
      return tableToMd(b);
    case "list": {
      const ordered = b.metadata?.listType === "ordered";
      const marker = ordered ? `${b.metadata?.itemIndex ?? 1}. ` : "- ";
      return `${marker}${(b.text ?? "").trim()}`;
    }
    case "heading":
    case "paragraph":
    default: {
      const text = (b.text ?? "").trim();
      if (!text) return "";
      const level = headingLevel(b);
      return level ? `${"#".repeat(level)} ${text}` : text;
    }
  }
}

function blocksToMarkdown(blocks: Block[]): string {
  let counter = 0;
  const nextIndex = () => ++counter;
  const md = blocks
    .map((b) => blockToMd(b, nextIndex))
    .filter(Boolean)
    .join("\n\n");
  // 列表项之间不要空行
  return md.replace(/\n\n(- |[0-9]+\. )/g, "\n$1").replace(/\n{3,}/g, "\n\n").trim();
}

/** 容器型块（幻灯片/页）截断，避免大文档无节制消耗 */
function limitContainers(blocks: Block[], max: number): Block[] {
  const containers = blocks.filter((b) => b.type === "slide" || b.type === "page");
  if (containers.length <= max) return blocks;
  const keep = new Set(containers.slice(0, max));
  return blocks.filter((b) => !(b.type === "slide" || b.type === "page") || keep.has(b));
}

/* ------------------------------------------------------------------ */
/* officeparser：docx / pptx / xlsx / pdf 统一读取                      */
/* ------------------------------------------------------------------ */

export interface OfficeParseOptions {
  mediaDir: string;
  docKey: string;
  /** PDF 文本层为空时（扫描版）回退到逐页渲染 + 多模态 */
  vision?: VisionOptions;
  /** 页/幻灯片数上限（默认不限） */
  maxPages?: number;
  onProgress?: (done: number, total: number) => void;
}

export async function parseOfficeDocument(filePath: string, opts: OfficeParseOptions): Promise<ParsedDoc | null> {
  const name = path.basename(filePath);
  const result = (await officeParser.parseOffice(filePath, {})) as {
    type: string;
    content?: Block[];
    warnings?: unknown[];
  };

  const blocks = opts.maxPages ? limitContainers(result.content ?? [], opts.maxPages) : (result.content ?? []);
  const markdown = blocksToMarkdown(blocks);
  opts.onProgress?.(blocks.length, (result.content ?? []).length);

  // PDF 扫描版没有文本层：有视觉能力就回退逐页渲染，没有就明确放弃
  const isPdf = result.type === "pdf" || /\.pdf$/i.test(filePath);
  if (isPdf && markdown.replace(/\s/g, "").length < 80) {
    if (!opts.vision) return null;
    return parsePdf(filePath, {
      mediaDir: opts.mediaDir,
      docKey: opts.docKey,
      vision: opts.vision,
      maxPages: opts.maxPages,
      onProgress: opts.onProgress,
    });
  }
  if (!markdown.trim()) return null;

  const isDocx = /\.docx$/i.test(filePath);
  return {
    name,
    markdown,
    images: isDocx ? await extractDocxMedia(filePath, { mediaDir: opts.mediaDir, docKey: opts.docKey }) : [],
    comments: isDocx ? await extractComments(filePath) : [],
  };
}

/** docx 里的批注单独抽出来——转文本会整段丢掉，而它常是结论确认 */
async function extractComments(filePath: string): Promise<string[]> {
  try {
    const { default: JSZip } = await import("jszip");
    const zip = await JSZip.loadAsync(await readFile(filePath));
    const xml = await zip.file("word/comments.xml")?.async("string");
    if (!xml) return [];
    const out: string[] = [];
    for (const block of xml.match(/<w:comment [\s\S]*?<\/w:comment>/g) ?? []) {
      const text = [...block.matchAll(/<w:t[^>]*>([\s\S]*?)<\/w:t>/g)].map((m) => m[1]).join("");
      if (text.trim()) out.push(text.trim());
    }
    return out;
  } catch {
    return [];
  }
}

/**
 * docx 嵌入图片从 word/media/ 直接抽字节（officeparser 的 extractAttachments 只给名单不给数据）。
 * 位置信息在这里丢了——所以这些图统一标记为 skipped，交给 cli/ingest.ts 的
 * 「图片解析 + LLM 归位」链路，由模型决定该进哪个章节。
 */
async function extractDocxMedia(filePath: string, opts: { mediaDir: string; docKey: string }): Promise<ParsedImage[]> {
  try {
    const { default: JSZip } = await import("jszip");
    const zip = await JSZip.loadAsync(await readFile(filePath));
    const names = Object.keys(zip.files).filter((f) => /^word\/media\//.test(f) && !f.endsWith("/"));
    if (!names.length) return [];

    await mkdir(opts.mediaDir, { recursive: true });
    const out: ParsedImage[] = [];
    let i = 0;
    for (const n of names) {
      const ext = path.extname(n).slice(1) || "png";
      const file = `image${++i}.${ext}`;
      await writeFile(path.join(opts.mediaDir, file), await zip.file(n)!.async("nodebuffer"));
      out.push({ ref: `_media/${opts.docKey}/${file}`, caption: "", parsedText: "", status: "skipped" });
    }
    return out;
  } catch {
    return [];
  }
}

/* ------------------------------------------------------------------ */
/* 视觉解析：图片 / PDF 扫描页                                          */
/* ------------------------------------------------------------------ */

export interface VisionOptions {
  /** OpenAI 兼容端点（多模态模型） */
  baseUrl: string;
  apiKey?: string;
  model: string;
  timeoutMs?: number;
}

const IMAGE_MIME: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
};

const IMAGE_PROMPT =
  "这是一张来自技术支持工单或排查文档的图片（可能是控制台截图、报错截图、界面截图、流程图）。" +
  "请提取其中对问题排查有用的信息，只返回 JSON，不要任何其他文字：\n" +
  '{{"caption": "一句话概括这张图说明什么", "text": "逐字读出可见的关键文字：报错信息、接口路径、状态码、版本号、设备信息、日志、配置项；若是图表或流程图，描述结构与结论"}}';

export async function parseImage(filePath: string, ref: string, opts: VisionOptions): Promise<ParsedImage> {
  const base = { ref, caption: "", parsedText: "", status: "failed" as const };

  for (let attempt = 1; attempt <= 2; attempt += 1) {
    try {
      const base64 = (await readFile(filePath)).toString("base64");
      const mime = IMAGE_MIME[path.extname(filePath).toLowerCase()] ?? "image/png";
      const raw = await openAIChat(
        opts.baseUrl,
        opts.apiKey ?? "no-key-required",
        opts.model,
        [
          {
            role: "user",
            content: [
              { type: "text", text: IMAGE_PROMPT },
              { type: "image_url", image_url: { url: toDataUrl(mime, base64) } },
            ],
          },
        ],
        { temperature: 0.1, timeoutMs: opts.timeoutMs ?? 180_000 },
      );

      const parsed = parseJsonObject(raw);
      const caption = String(parsed.caption ?? "").trim();
      const parsedText = String(parsed.text ?? raw).trim();
      if (!caption && !parsedText) throw new Error("解析结果为空（模型没返回 caption / text）");
      return { ref, caption, parsedText, status: "parsed" };
    } catch (e) {
      const msg = (e as Error).message.split("\n")[0];
      if (attempt < 2) {
        console.error(`  [图片解析重试] ${path.basename(filePath)}: ${msg}`);
        continue;
      }
      console.error(`  [图片解析失败] ${path.basename(filePath)}: ${msg}`);
    }
  }
  return base;
}

/** 纯文本也走同一个模型：产出形态与图片解析一致（caption + text） */
export async function parseText(content: string, ref: string, opts: VisionOptions): Promise<ParsedImage> {
  for (let attempt = 1; attempt <= 2; attempt += 1) {
    try {
      const raw = await openAIChat(
        opts.baseUrl,
        opts.apiKey ?? "no-key-required",
        opts.model,
        [{ role: "user", content: `${TEXT_PROMPT}\n\n${content.slice(0, 6000)}` }],
        { temperature: 0.1, timeoutMs: opts.timeoutMs ?? 180_000 },
      );
      const parsed = parseJsonObject(raw);
      const caption = String(parsed.caption ?? "").trim();
      const parsedText = String(parsed.text ?? raw).trim();
      if (!caption && !parsedText) throw new Error("解析结果为空");
      return { ref, caption, parsedText, status: "parsed" };
    } catch (e) {
      const msg = (e as Error).message.split("\n")[0];
      if (attempt < 2) {
        console.error(`  [文本解析重试] ${ref}: ${msg}`);
        continue;
      }
      console.error(`  [文本解析失败] ${ref}: ${msg}`);
    }
  }
  return { ref, caption: "", parsedText: "", status: "failed" };
}

const TEXT_PROMPT =
  "这是一份文档（幻灯片/章节）的原始文本。请提取其中对问题排查或知识检索有用的信息，" +
  "只返回 JSON，不要任何其他文字：\n" +
  '{{"caption": "一句话概括这一页讲什么", "text": "整理后的关键信息（保留术语、编号、步骤、结论）"}}';

export interface PdfParseOptions {
  mediaDir: string;
  docKey: string;
  vision: VisionOptions;
  maxPages?: number;
  onProgress?: (done: number, total: number) => void;
}

/**
 * PDF 扫描版兜底：逐页渲染成图送视觉模型。
 * 只在文本层为空时才走这里（解析慢、按页计费）。
 */
export async function parsePdf(filePath: string, opts: PdfParseOptions): Promise<ParsedDoc> {
  const pdfjs = (await import("pdfjs-dist/legacy/build/pdf.mjs")) as any;
  const data = new Uint8Array(await readFile(filePath));
  const doc = await pdfjs.getDocument({ data, isEvalSupported: false, useSystemFonts: true }).promise;
  const total: number = doc.numPages;
  const limit = Math.min(total, opts.maxPages ?? total);

  await mkdir(opts.mediaDir, { recursive: true });
  const images: ParsedImage[] = [];
  const blocks: string[] = [];

  for (let i = 1; i <= limit; i += 1) {
    const page = await doc.getPage(i);
    const viewport = page.getViewport({ scale: 2 });
    const { createCanvas } = await import("@napi-rs/canvas");
    const canvas = createCanvas(viewport.width, viewport.height);
    await page.render({ canvasContext: canvas.getContext("2d"), viewport, canvas }).promise;

    const file = `page${i}.png`;
    const target = path.join(opts.mediaDir, file);
    await writeFile(target, canvas.toBuffer("image/png"));

    const parsed = await parseImage(target, `_media/${opts.docKey}/${file}`, opts.vision);
    images.push({ ...parsed, placed: true });
    blocks.push(imageBlock(parsed, `第 ${i} 页`));
    opts.onProgress?.(i, total);
  }

  if (limit < total) blocks.push(`> （只解析了前 ${limit} 页，共 ${total} 页）`);
  return { name: path.basename(filePath), markdown: blocks.join("\n\n"), images, comments: [] };
}

/* ------------------------------------------------------------------ */
/* 兜底：未知格式 / 视频                                                */
/* ------------------------------------------------------------------ */

const JUNK_CHARS = new RegExp("[\\u0000-\\u0008\\u000b\\u000c\\u000e-\\u001f\\u007f-\\u009f\\ufffd]", "g");

/**
 * 未知格式兜底：抽可读文本 → 交给模型整理成知识单元。
 * 抽不出就返回 null（宁可不入库，也不写乱码）。
 */
export async function parseUnknownDocument(
  filePath: string,
  opts: { docKey: string; vision: VisionOptions },
): Promise<ParsedDoc | null> {
  const text = await extractReadableText(filePath);
  if (!text) return null;
  const name = path.basename(filePath);
  const parsed = await parseText(text, `${name}#p1`, opts.vision);
  return {
    name,
    markdown: imageBlock(parsed, name),
    images: [{ ...parsed, placed: true }],
    comments: [],
  };
}

/** 未知格式的兜底：解码得出可读文本才用，否则放弃——宁可不入库，也不写乱码 */
export async function extractReadableText(filePath: string): Promise<string | null> {
  const raw = (await readFile(filePath)).toString("utf-8");
  const junk = (raw.match(JUNK_CHARS) ?? []).length;
  const text = raw.replace(JUNK_CHARS, "").replace(/\n{3,}/g, "\n\n").trim();
  if (junk / Math.max(raw.length, 1) > 0.1 || text.length < 40) return null;
  return text;
}

/**
 * 视频不做全量抽取：5 分钟录屏按 1fps 抽 300 帧，逐帧解析成本爆炸而信息密度极低。
 * 例外是「视频为唯一信息源」时才抽关键帧，那属于后续增强。
 */
export function referenceVideo(filePath: string, caption = ""): ParsedImage {
  return {
    ref: path.basename(filePath),
    caption: caption || `复现录屏（未解析，见 ${path.basename(filePath)}）`,
    parsedText: "",
    status: "skipped",
  };
}

/* ------------------------------------------------------------------ */
/* 通用工具                                                            */
/* ------------------------------------------------------------------ */

/** 解析结果 → Markdown 引用块（保留来源，便于回溯原图/原页） */
export function imageBlock(img: ParsedImage, label?: string): string {
  const title = label ?? `图解 ${img.ref.split("/").pop() ?? img.ref}`;
  const body = [
    img.caption ? `**${title}**：${img.caption}` : `**${title}**`,
    img.parsedText,
    `（来源: ${img.ref}）`,
  ]
    .filter(Boolean)
    .join("\n");
  return body
    .split("\n")
    .map((l) => `> ${l}`)
    .join("\n");
}

/**
 * 模型偶尔会包 ```json 围栏、补说明、或在字符串里输出未转义换行（JSON.parse 直接失败）。
 * 先严格解析，失败退到宽松抽取，保证 caption / text 不丢。
 */
export function parseJsonObject(raw: string): Record<string, unknown> {
  const cleaned = raw.replace(/```(?:json)?/gi, "").trim();
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start >= 0 && end > start) {
    try {
      return JSON.parse(cleaned.slice(start, end + 1)) as Record<string, unknown>;
    } catch {
      /* 未转义换行等，退到宽松抽取 */
    }
  }

  const out: Record<string, unknown> = {};
  for (const m of cleaned.matchAll(/"(\w+)"\s*:\s*"([\s\S]*?)"\s*(?=[,}])/g)) {
    out[m[1]] = m[2].replace(/\\n/g, "\n").replace(/\\"/g, '"').trim();
  }
  return out;
}

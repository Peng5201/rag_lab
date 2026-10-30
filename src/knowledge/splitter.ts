/**
 * 差异化分块 —— RAG 效果的第一决定因素
 *
 * 通用 TextSplitter 按固定长度切，在工单/表格场景有两个致命问题：
 *   1. 表格行截断：一行被切两半，后半段丢表头，字段语义无法还原
 *   2. 案例碎片化：一条完整工单（工单号+现象+根因+方案）被切碎
 *
 * 对策：按文档类型差异化
 *   - 表格：每行一个 chunk，KV 格式，保留表头语义
 *   - 文档：用 RecursiveCharacterTextSplitter 按自然边界切
 */
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { existsSync } from "node:fs";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { defaultKnowledgeDir } from "./paths.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** 知识库目录：RAG_KNOWLEDGE_DIR → <文档>/RAG-Lab/knowledge → 项目内 knowledge/（兼容旧位置） */
export function findKnowledgeDir(): string {
  const override = process.env.RAG_KNOWLEDGE_DIR;
  if (override) return override; // 显式指定就认它，不存在时报错也指向它

  const preferred = defaultKnowledgeDir();
  if (existsSync(preferred)) return preferred;

  const root = path.join(__dirname, "..", ".."); // src/knowledge → 项目根
  const local = path.join(root, "knowledge");
  if (existsSync(local)) return local;

  const python = path.join(root, "..", "_rag_lab", "knowledge");
  if (existsSync(python)) return python;

  return preferred; // 都没有时指向默认位置，报错提示才对得上
}

/** 读取目录下全部 .csv / .md，按文档类型差异化分块（递归子目录，跳过 _ 开头的私有目录） */
export async function loadAll(dir: string): Promise<Document[]> {
  if (!existsSync(dir)) {
    throw new Error(
      `知识库目录不存在：${dir}\n` +
        `先入库：npm run ingest -- <物料目录> --out=${dir}\n` +
        `或用 RAG_KNOWLEDGE_DIR 指向已有的知识库目录（放 .csv 工单表 + .md 手册）`,
    );
  }
  return walk(dir, "");
}

async function walk(dir: string, prefix: string): Promise<Document[]> {
  const docs: Document[] = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.name.startsWith("_")) continue; // _media、_manifest.json 等不进索引
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      docs.push(...(await walk(p, rel)));
    } else if (entry.name.endsWith(".csv")) {
      docs.push(...(await loadCsvAsChunks(p, rel)));
    } else if (entry.name.endsWith(".md")) {
      docs.push(...(await loadMarkdownAsChunks(p, rel)));
    }
  }
  return docs;
}

export interface ChunkSource {
  type: "doc" | "table";
  source: string;
  row?: number;
  data?: Record<string, string>;
}

/** 表格类：每行一个 chunk，拼成「字段: 值」 */
export function chunkTableRow(source: ChunkSource): Document {
  const kv = Object.entries(source.data ?? {})
    .filter(([, v]) => String(v).trim())
    .map(([k, v]) => `${k}: ${v}`)
    .join("\n");

  return new Document({
    pageContent: kv,
    metadata: {
      source: source.source,
      row: source.row,
      type: "table",
      chunkId: `${source.source}#row${source.row}`,
    },
  });
}

/** 文档类：用 LangChain 的递归字符分割器，按自然边界切 */
export async function chunkDocument(
  content: string,
  source: string,
  opts: { chunkSize?: number; chunkOverlap?: number } = {},
): Promise<Document[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: opts.chunkSize ?? 512,
    chunkOverlap: opts.chunkOverlap ?? 64,
    // 自定义分隔符：优先按段落 → 换行 → 句号，保证语义完整
    separators: ["\n\n", "\n", "。", "！", "？", "；", " ", ""],
  });

  const parts = await splitter.splitText(content);
  return parts.map(
    (text, i) =>
      new Document({
        pageContent: text,
        metadata: {
          source,
          type: "doc",
          chunkId: `${source}#${i}`,
        },
      }),
  );
}

/** CSV 加载 + 差异化分块 */
export async function loadCsvAsChunks(path: string, source: string): Promise<Document[]> {
  const raw = await readFile(path, "utf-8");
  const lines = raw.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];

  const headers = lines[0].split(",");
  return lines.slice(1).map((line, row) => {
    const values = line.split(",");
    const data: Record<string, string> = {};
    headers.forEach((h, i) => (data[h.trim()] = (values[i] ?? "").trim()));
    return chunkTableRow({ type: "table", source, row, data });
  });
}

/** Markdown 加载 + 差异化分块 */
export async function loadMarkdownAsChunks(path: string, source: string): Promise<Document[]> {
  const content = await readFile(path, "utf-8");
  return chunkDocument(content, source);
}

/**
 * 评测指标（TS 版，对应 Python 版的 rag-touchstone）
 *
 * 与 LangChain 无关，自研实现 —— 生产上我们的评测管线也是自研而非 ragas，
 * 因为需要控制指标口径与可插拔性来做技术选型对比。
 *
 * LLM 指标与规则指标成对出现、互相校验：
 *   Context Precision / Recall（LLM，吃同义表述） vs NonLLM 版（纯规则，只看词面）
 *   Faithfulness（LLM） vs Answer Correctness（规则相似度）
 */
import type { Document } from "@langchain/core/documents";
import { isSupported, matchesAny } from "./chain.js";
import type { ChatModelType } from "../llm/model.js";

const THRESHOLD = 0.75;

function similarity(a: string, b: string): number {
  // 归一化字符级 Dice，够用且零依赖
  const bigrams = (s: string) => {
    const out = new Set<string>();
    for (let i = 0; i < s.length - 1; i++) out.add(s.slice(i, i + 2));
    return out;
  };
  const A = bigrams(a);
  const B = bigrams(b);
  if (!A.size || !B.size) return 0;
  let hits = 0;
  for (const g of A) if (B.has(g)) hits++;
  return (2 * hits) / (A.size + B.size);
}

function isMatch(a: string, bList: string[]): boolean {
  return bList.some((b) => similarity(a, b) >= THRESHOLD);
}

function extractEntities(text: string): Set<string> {
  const ents = new Set<string>();
  for (const re of [
    /[A-Z]{2,}[_-]?\d+/g,
    /v?\d+\.\d+(?:\.\d+){0,2}/g,
    /[a-zA-Z]{3,}(?:\.[a-zA-Z]+)*/g,
    /[一-鿿]{2,6}(?:失败|异常|错误|问题|配置|权限)/g,
  ]) {
    for (const m of text.match(re) ?? []) ents.add(m);
  }
  return ents;
}

// ---------- 检索质量 ----------
//
// LLM 版与 NonLLM 版口径一致（都是「检索到的 vs 该检索到的」），
// 区别只在匹配方式：LLM 判语义等价，规则判词面重合。

/** Context Precision（LLM）：给的结果有多少是有用的 */
export async function contextPrecision(
  retrieved: string[],
  relevant: string[],
  model: ChatModelType,
): Promise<number> {
  if (!retrieved.length || !relevant.length) return 0;
  const flags = await Promise.all(retrieved.map((r) => matchesAny(r, relevant, model)));
  return flags.filter(Boolean).length / retrieved.length;
}

/** Context Recall（LLM）：该找到的有没有都找到 */
export async function contextRecall(
  retrieved: string[],
  relevant: string[],
  model: ChatModelType,
): Promise<number> {
  if (!relevant.length) return 0;
  const flags = await Promise.all(relevant.map((rel) => matchesAny(rel, retrieved, model)));
  return flags.filter(Boolean).length / relevant.length;
}

/** NonLLM Precision：纯规则版精确率，便宜、稳定，但只看词面 */
export function nonLlmPrecision(retrieved: string[], relevant: string[]): number {
  if (!retrieved.length) return 0;
  return retrieved.filter((r) => isMatch(r, relevant)).length / retrieved.length;
}

/** NonLLM Recall：纯规则版召回率 */
export function nonLlmRecall(retrieved: string[], relevant: string[]): number {
  if (!relevant.length) return 0;
  return relevant.filter((rel) => isMatch(rel, retrieved)).length / relevant.length;
}

export function contextEntityRecall(retrieved: string[], relevant: string[]): number {
  const relEnts = extractEntities(relevant.join(" "));
  if (!relEnts.size) return 0;
  const retEnts = extractEntities(retrieved.join(" "));
  let hits = 0;
  for (const e of relEnts) if (retEnts.has(e)) hits++;
  return hits / relEnts.size;
}

// ---------- 生成质量 ----------

export function splitClaims(answer: string): string[] {
  return answer
    .split(/[。！？；\n]/)
    .map((s) => s.trim())
    .filter((s) => s.length > 4);
}

/** Faithfulness（LLM）：有没有编造——答案的每条原子陈述能否被上下文支撑 */
export async function faithfulness(answer: string, contexts: string[], model: ChatModelType): Promise<number> {
  const claims = splitClaims(answer);
  if (!claims.length) return 0;
  const flags = await Promise.all(claims.map((c) => isSupported(c, contexts, model)));
  return flags.filter(Boolean).length / claims.length;
}

export function answerRelevancy(question: string, answer: string): number {
  // 规则版：问题关键词在答案中的覆盖率
  const qWords = question.match(/[一-鿿]{2,}|[a-zA-Z0-9_]{2,}/g) ?? [];
  if (!qWords.length) return 0;
  const hit = qWords.filter((w) => answer.includes(w)).length;
  return hit / qWords.length;
}

export function answerCorrectness(answer: string, reference: string): number {
  return similarity(answer, reference);
}

// ---------- 汇总 ----------

export interface EvalRow {
  context_precision: number;
  context_recall: number;
  context_entity_recall: number;
  nonllm_precision: number;
  nonllm_recall: number;
  faithfulness: number;
  answer_relevancy: number;
  answer_correctness: number;
}

export function averageRows(rows: EvalRow[]): EvalRow {
  const keys = Object.keys(rows[0] ?? {}) as (keyof EvalRow)[];
  const out = {} as EvalRow;
  for (const k of keys) {
    out[k] = rows.reduce((s, r) => s + r[k], 0) / Math.max(rows.length, 1);
  }
  return out;
}

export function printReport(title: string, report: Record<string, number>): void {
  console.log(`\n===== ${title} =====`);
  for (const [k, v] of Object.entries(report)) {
    console.log(`  ${k.padEnd(26)} ${v.toFixed(3)}`);
  }
}

export function printCompare(results: Record<string, Record<string, number>>): void {
  const names = Object.keys(results);
  const metrics = Object.keys(results[names[0]] ?? {});
  const w = 16;
  console.log(`\n${"=".repeat(26 + w * names.length)}`);
  console.log(`${"指标".padEnd(26)}${names.map((n) => n.padStart(w)).join("")}`);
  console.log("=".repeat(26 + w * names.length));
  for (const m of metrics) {
    const vals = names.map((n) => results[n][m] ?? 0);
    const best = Math.max(...vals);
    const line = vals
      .map((v, i) => `${v.toFixed(3)}${v === best && names.length > 1 ? " *" : "  "}`.padStart(w))
      .join("");
    console.log(`${m.padEnd(26)}${line}`);
  }
  console.log("=".repeat(26 + w * names.length));
}

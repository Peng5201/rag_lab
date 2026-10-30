/**
 * 端到端示例：建库 → 检索 → 生成 → Gold → 评测对比
 *
 *   npm install
 *   npm start                    # Mock 模式，离线跑通
 *   npm run eval                 # DeepSeek / 任何 OpenAI 兼容端点
 */
import type { Document } from "@langchain/core/documents";

import { findKnowledgeDir, loadAll } from "../knowledge/splitter.js";
import { buildRetriever } from "../retrieval/hybridRetriever.js";
import { ask, isRelevant, type ChatModelType } from "../rag/chain.js";
import { progressiveSearch } from "../retrieval/search.js";
import { createChatModel, createEmbeddings, describeConfig, resolveConfig, type ProviderKind } from "../llm/providers.js";
import {
  answerCorrectness,
  answerRelevancy,
  averageRows,
  contextEntityRecall,
  contextPrecision,
  contextRecall,
  faithfulness,
  nonLlmPrecision,
  nonLlmRecall,
  printCompare,
  printReport,
  type EvalRow,
} from "../rag/evaluate.js";

const QUERIES = [
  "表格转 PDF 后字体重叠跑版怎么办？",
  "下拉列表点击没反应，无法弹出选项",
  "手机上编辑文档时虚拟键盘收起",
];

/** 检索系统签名：任何被测系统都长这样（可插拔，对应文档的 RAGSystem.retrieve） */
type RetrieveFn = (q: string, k: number) => Promise<Document[]>;
type SystemMap = Record<string, RetrieveFn>;
type GoldItem = { query: string; contexts: string[]; answer: string };

/**
 * 三阶段自动出卷
 *   Phase 1 素材采集：多系统取并集，最大化召回（借鉴 TREC Pooling）
 *   Phase 2 上下文提取：LLM 逐片判断相关性，过滤噪声
 *   Phase 3 答案生成：严格基于筛选后上下文生成参考答案（grounded，不编造）
 */
async function buildGold(queries: string[], systems: SystemMap, model: ChatModelType): Promise<GoldItem[]> {
  const gold: GoldItem[] = [];

  for (const q of queries) {
    // Phase 1：并集采样 —— 单系统的召回盲区由其他系统补上
    const pool = new Map<string, Document>();
    for (const retrieve of Object.values(systems)) {
      for (const doc of await retrieve(q, 5)) {
        const id = (doc.metadata.chunkId as string) ?? doc.pageContent.slice(0, 32);
        pool.set(id, doc);
      }
    }
    const candidates = [...pool.values()];

    // Phase 2：LLM 逐片相关性筛选
    const flags = await Promise.all(candidates.map((c) => isRelevant(q, c.pageContent, model)));
    const relevant = candidates.filter((_, i) => flags[i]);

    // Phase 3：相关片段不足时退回候选 top3，宁可少答也不编造
    const ctx = relevant.length ? relevant : candidates.slice(0, 3);
    const answer = await ask(q, ctx, model);

    gold.push({ query: q, contexts: ctx.map((d) => d.pageContent), answer });
    console.log(`[gold] ${q.slice(0, 22)}... -> 并集 ${candidates.length} / 相关 ${relevant.length}`);
  }
  return gold;
}

async function evaluateSystem(
  name: string,
  retrieve: RetrieveFn,
  gold: GoldItem[],
  model: ChatModelType,
): Promise<Record<string, number>> {
  console.log(`  [评测] ${name}`);
  const rows: EvalRow[] = [];

  for (const item of gold) {
    const hits = await retrieve(item.query, 5);
    const retrieved = hits.map((d) => d.pageContent);
    const answer = await ask(item.query, hits, model);

    rows.push({
      context_precision: await contextPrecision(retrieved, item.contexts, model),
      context_recall: await contextRecall(retrieved, item.contexts, model),
      context_entity_recall: contextEntityRecall(retrieved, item.contexts),
      nonllm_precision: nonLlmPrecision(retrieved, item.contexts),
      nonllm_recall: nonLlmRecall(retrieved, item.contexts),
      faithfulness: await faithfulness(answer, item.contexts, model),
      answer_relevancy: answerRelevancy(item.query, answer),
      answer_correctness: answerCorrectness(answer, item.answer),
    });
  }
  return averageRows(rows) as unknown as Record<string, number>;
}

/**
 * 命令行参数 → Provider 配置
 *   --provider=mock|openai   （--real 等价于 --provider=openai）
 *   --chat-model=xxx  --embed-model=xxx
 */
function parseArgs(argv: string[]): Parameters<typeof resolveConfig>[0] {
  const pick = (name: string) => argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1];
  const provider = pick("provider") as ProviderKind | undefined;

  return {
    kind: provider ?? (argv.includes("--real") ? "openai" : undefined),
    chatModel: pick("chat-model"),
    embeddingModel: pick("embed-model"),
  };
}

async function main() {
  const cfg = resolveConfig(parseArgs(process.argv.slice(2)));
  console.log(`[配置] ${describeConfig(cfg)}`);

  // 1. 向量化（mock→Hash / openai→OpenAIEmbeddings）
  const embeddings = createEmbeddings(cfg);

  // 2. 建库（差异化分块）
  const dir = findKnowledgeDir();
  const docs = await loadAll(dir);
  console.log(`[建库] 知识库 ${dir}`);
  console.log(`       chunk ${docs.length} 个`);
  docs.slice(0, 3).forEach((d) =>
    console.log(`       · ${String(d.metadata.chunkId).padEnd(28)} ${d.pageContent.slice(0, 36).replace(/\n/g, " / ")}...`),
  );

  // 3. 构建混合检索
  const { retriever, vectorStore } = await buildRetriever(docs, embeddings);
  const model: ChatModelType = createChatModel(cfg);

  // 4. 被测系统集合（Gold 取并集、评测对比都用它；新增一个系统只要加一行）
  const systems: SystemMap = {
    VectorOnly: (q, k) => vectorStore.similaritySearch(q, k),
    BM25Only: (q, k) => retriever.keywordSearch(q, k),
    Hybrid: (q, k) => retriever.hybridSearch(q, k),
    Progressive: (q, k) => progressiveSearch(retriever, q, model, k),
  };

  // 5. 检索示例
  console.log("\n[检索示例]");
  for (const q of QUERIES.slice(0, 2)) {
    console.log(`\n  Q: ${q}`);
    const hits = await progressiveSearch(retriever, q, model, 3);
    hits.forEach((d) =>
      console.log(`     ${String(d.metadata.source).padEnd(18)} | ${d.pageContent.slice(0, 40).replace(/\n/g, " / ")}`),
    );
  }

  // 6. 生成示例
  console.log("\n[生成示例]");
  const q0 = QUERIES[0];
  const ctx0 = await retriever.hybridSearch(q0, 3);
  const ans0 = await ask(q0, ctx0, model);
  console.log(`  Q: ${q0}`);
  console.log(`  A: ${ans0.slice(0, 160)}...`);

  // 7. Gold + 评测对比
  console.log("\n[构建 Gold Dataset]（多系统并集采样 + LLM 相关性筛选）");
  const gold = await buildGold(QUERIES, systems, model);

  console.log("\n[多系统对比 —— 技术选型场景]");
  const results: Record<string, Record<string, number>> = {};
  for (const [name, retrieve] of Object.entries(systems)) {
    results[name] = await evaluateSystem(name, retrieve, gold, model);
  }
  printCompare(results);

  printReport("Hybrid 检索", results.Hybrid);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

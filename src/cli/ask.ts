/**
 * 单次问答：检索 + 生成，不跑评测
 *
 *   npm run ask -- "表格转 PDF 后字体重叠跑版怎么办？"
 *   npm run ask -- "下拉列表没反应" --strategy=progressive --k=5
 *
 * 与 index.ts 的区别：index 是「自检 + 评测」（建 Gold、跑 8 指标），
 * 这里只回答一个问题，几秒出结果。
 */
import type { Document } from "@langchain/core/documents";

import { findKnowledgeDir, loadAll } from "../knowledge/splitter.js";
import { buildRetriever, type HybridRetriever } from "../retrieval/hybridRetriever.js";
import { ask } from "../rag/chain.js";
import { progressiveSearch } from "../retrieval/search.js";
import { createChatModel, createEmbeddings, describeConfig, resolveConfig } from "../llm/providers.js";
import type { ChatModelType } from "../llm/model.js";
import type { InMemoryVectorStore } from "../retrieval/vectorStore.js";

type Strategy = "hybrid" | "progressive" | "vector" | "bm25";

interface CliArgs {
  question: string;
  k: number;
  strategy: Strategy;
}

function parseCli(argv: string[]): CliArgs {
  const pick = (name: string) => argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1];
  return {
    question: argv.filter((a) => !a.startsWith("--")).join(" ").trim(),
    k: Number(pick("k") ?? 5),
    strategy: (pick("strategy") ?? "hybrid") as Strategy,
  };
}

async function retrieve(
  retriever: HybridRetriever,
  vectorStore: InMemoryVectorStore,
  question: string,
  k: number,
  strategy: Strategy,
  model: ChatModelType,
): Promise<Document[]> {
  switch (strategy) {
    case "progressive":
      return progressiveSearch(retriever, question, model, k);
    case "vector":
      return vectorStore.similaritySearch(question, k);
    case "bm25":
      return retriever.keywordSearch(question, k);
    default:
      return retriever.hybridSearch(question, k);
  }
}

async function main() {
  const args = parseCli(process.argv.slice(2));
  if (!args.question) {
    console.error('用法：npm run ask -- "你的问题" [--strategy=hybrid|progressive|vector|bm25] [--k=5]');
    process.exit(1);
  }

  const cfg = resolveConfig();
  console.log(`[配置] ${describeConfig(cfg)}  策略=${args.strategy}  k=${args.k}`);

  if (cfg.kind === "mock") {
    console.log(
      [
        "",
        "⚠️  当前是 mock 模式 —— 没配任何模型，下面的「答案」是按规则拼的占位文案，不是真的。",
        "    要真答案，先配 chat（向量默认走本地，不用配）：",
        '      $env:LLM_BASE_URL   = "https://api.deepseek.com"',
        '      $env:LLM_API_KEY    = "sk-xxx"',
        '      $env:LLM_CHAT_MODEL = "deepseek-chat"',
        '    或用一键脚本：.\\run.ps1 ask "你的问题"',
        "",
        "    另：mock 的 HashEmbeddings 只看词面，检索排序也不可信。",
        "",
      ].join("\n"),
    );
  }

  // 索引每次重建（内存库）；生产会持久化，这里几秒无所谓
  const t0 = Date.now();
  const docs = await loadAll(findKnowledgeDir());
  const { retriever, vectorStore } = await buildRetriever(docs, createEmbeddings(cfg));
  const model = createChatModel(cfg);
  const buildMs = Date.now() - t0;

  const t1 = Date.now();
  const hits = await retrieve(retriever, vectorStore, args.question, args.k, args.strategy, model);
  const answer = hits.length ? await ask(args.question, hits, model) : "知识库里没有检索到相关内容。";
  const qaMs = Date.now() - t1;

  console.log(`\n[索引] ${docs.length} chunk（${buildMs}ms）   [检索+生成] ${qaMs}ms`);
  console.log(`\nQ: ${args.question}\n`);
  console.log(answer);
  console.log("\n[来源]");
  hits.forEach((d, i) =>
    console.log(`  [${i + 1}] ${String(d.metadata.source ?? "未知")} — ${d.pageContent.slice(0, 48).replace(/\n/g, " / ")}…`),
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

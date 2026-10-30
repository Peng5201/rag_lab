/**
 * 检索策略层 —— 查询改写 + 三轮递进检索
 *
 * RAG 效果 70% 在分块和检索策略，不在模型。三轮各取所长：
 *   第一轮 q1 完整描述   → hybrid，发挥语义召回
 *   第二轮 q2 关键词组合 → hybrid，覆盖不同表述
 *   第三轮 q3 精确术语   → 纯 BM25，错误码 / 配置项名精确兜底
 * 配合术语同义词映射（见 hybridRetriever.ts 的 SYNONYMS），
 * 消除「用户叫法 ≠ 知识库写法」的召回盲区。
 */
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import type { Document } from "@langchain/core/documents";

import type { ChatModelType } from "../llm/model.js";
import { normalizeTerms, type HybridRetriever } from "./hybridRetriever.js";

/** 多视角查询改写：一条含糊的工单描述只用它检索会漏召回 */
export async function rewriteQueries(question: string, model: ChatModelType): Promise<string[]> {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "human",
      "[TASK:rewrite] 把下面的问题改写成 3 个不同角度的检索 query，只返回 JSON：" +
        '{{"query_1": "完整自然语言描述", "query_2": "核心关键词组合", "query_3": "精确术语"}}\n问题：{question}',
    ],
  ]);
  const chain = prompt.pipe(model).pipe(new StringOutputParser());
  const raw: string = await chain.invoke({ question });

  try {
    const data = JSON.parse(raw);
    return [data.query_1 ?? question, data.query_2 ?? "", data.query_3 ?? ""];
  } catch {
    return [question, question, question];
  }
}

/** 三轮递进检索：语义 → 不同表述 → 精确兜底，按权重合并去重 */
export async function progressiveSearch(
  retriever: HybridRetriever,
  question: string,
  model: ChatModelType,
  k = 5,
): Promise<Document[]> {
  const [q1, q2, q3] = await rewriteQueries(question, model);
  const rounds: Array<[Document[], number]> = [
    [await retriever.hybridSearch(normalizeTerms(q1), k), 1],
    [await retriever.hybridSearch(normalizeTerms(q2), k), 0.8],
    [await retriever.keywordSearch(q3, k), 0.6],
  ];

  const all = new Map<string, { doc: Document; score: number }>();
  for (const [hits, weight] of rounds) {
    hits.forEach((doc, i) => {
      const id = (doc.metadata.chunkId as string) ?? doc.pageContent.slice(0, 32);
      const prev = all.get(id);
      const score = weight * (1 - i / k);
      if (prev) prev.score += score;
      else all.set(id, { doc, score });
    });
  }
  return [...all.values()].sort((a, b) => b.score - a.score).slice(0, k).map((v) => v.doc);
}

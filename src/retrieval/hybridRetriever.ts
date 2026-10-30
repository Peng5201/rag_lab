/**
 * 混合检索 Retriever —— 继承 LangChain 的 BaseRetriever
 *
 * 这是"用框架"与"有深度"的结合点：
 *   - 遵循 BaseRetriever 契约，可直接插进任何 LangChain Chain
 *   - 融合算法自己写（向量 + BM25 分数归一化加权）
 *
 * LangChain 的 VectorStore 不直接支持混合检索，
 * 所以这里组合 InMemoryVectorStore（向量）+ 自写 BM25Index（关键词）。
 */
import { BaseRetriever, type BaseRetrieverInput } from "@langchain/core/retrievers";
import type { CallbackManagerForRetrieverRun } from "@langchain/core/callbacks/manager";
import { Document } from "@langchain/core/documents";
import { InMemoryVectorStore } from "./vectorStore.js";
import type { Embeddings } from "@langchain/core/embeddings";

import { BM25Index } from "./bm25.js";

/** 术语同义词映射：消除「用户叫法 ≠ 知识库写法」的召回盲区 */
export const SYNONYMS: Record<string, string> = {
  区域权限: "保护工作表",
  保护区域: "保护工作表",
  数据有效性: "下拉列表",
  多级列表: "下拉列表",
  跑版: "格式错乱",
  显示异常: "格式错乱",
  打不开: "无法加载",
  文件损坏: "无法加载",
};

export function normalizeTerms(query: string): string {
  let q = query;
  for (const [user, std] of Object.entries(SYNONYMS)) {
    q = q.split(user).join(std);
  }
  return q;
}

export interface HybridRetrieverInput extends BaseRetrieverInput {
  vectorStore: InMemoryVectorStore;
  bm25: BM25Index<Document>;
  /** 向量权重，1-alpha 为 BM25 权重 */
  alpha?: number;
  k?: number;
}

export class HybridRetriever extends BaseRetriever {
  lc_namespace = ["rag", "retrievers"];

  private vectorStore: InMemoryVectorStore;
  private bm25: BM25Index<Document>;
  private alpha: number;
  private k: number;

  constructor(input: HybridRetrieverInput) {
    super(input);
    this.vectorStore = input.vectorStore;
    this.bm25 = input.bm25;
    this.alpha = input.alpha ?? 0.5;
    this.k = input.k ?? 5;
  }

  /** LangChain 契约：实现这个方法即可成为标准 Retriever */
  async _getRelevantDocuments(
    query: string,
    _runManager: CallbackManagerForRetrieverRun,
  ): Promise<Document[]> {
    return this.hybridSearch(query, this.k);
  }

  /** 分数归一化 + 加权融合 */
  async hybridSearch(query: string, k: number): Promise<Document[]> {
    const q = normalizeTerms(query);

    const [vecHits, bmHits] = await Promise.all([
      this.vectorStore.similaritySearchWithScore(q, k * 2),
      Promise.resolve(this.bm25.search(q, k * 2)),
    ]);

    // 归一化：两种检索分数尺度不同（余弦 0~1，BM25 无上界）
    const normalize = (scores: number[]): number[] => {
      if (!scores.length) return [];
      const lo = Math.min(...scores);
      const hi = Math.max(...scores);
      return hi === lo ? scores.map(() => 1) : scores.map((s) => (s - lo) / (hi - lo));
    };

    const vecNorm = normalize(vecHits.map(([, s]) => s));
    const bmNorm = normalize(bmHits.map((h) => h.score));

    const merged = new Map<string, { doc: Document; score: number }>();
    vecHits.forEach(([doc,], i) => {
      const id = (doc.metadata.chunkId as string) ?? doc.pageContent.slice(0, 32);
      merged.set(id, { doc, score: this.alpha * vecNorm[i] });
    });
    bmHits.forEach((hit, i) => {
      const id = (hit.doc.metadata.chunkId as string) ?? hit.doc.pageContent.slice(0, 32);
      const prev = merged.get(id);
      const add = (1 - this.alpha) * bmNorm[i];
      if (prev) prev.score += add;
      else merged.set(id, { doc: hit.doc, score: add });
    });

    return [...merged.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((v) => v.doc);
  }

  /** 第三轮兜底：跳过向量，纯 keyword 精确匹配（错误码、版本号、配置项名） */
  async keywordSearch(query: string, k: number): Promise<Document[]> {
    return this.bm25.search(normalizeTerms(query), k).map((h) => h.doc);
  }
}

/** 构建双索引 + 混合 Retriever */
export async function buildRetriever(
  docs: Document[],
  embeddings: Embeddings,
  opts: { alpha?: number; k?: number } = {},
): Promise<{ retriever: HybridRetriever; vectorStore: InMemoryVectorStore; bm25: BM25Index<Document> }> {
  const vectorStore = new InMemoryVectorStore(embeddings);
  await vectorStore.addDocuments(docs);

  const bm25 = new BM25Index<Document>();
  bm25.add(docs, (d) => d.pageContent);

  const retriever = new HybridRetriever({ vectorStore, bm25, ...opts });
  return { retriever, vectorStore, bm25 };
}

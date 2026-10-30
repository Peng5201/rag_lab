/**
 * InMemoryVectorStore —— 继承 LangChain 的 VectorStore 抽象
 *
 * 为什么自己写：
 *   1. @langchain/community 与当前 core 版本存在依赖冲突
 *   2. VectorStore 的抽象就两个方法，实现它正好展示对框架契约的理解
 *
 * 实现了 LangChain 要求的两个抽象方法：
 *   - addVectors(vectors, documents)
 *   - similaritySearchVectorWithScore(query, k)
 *
 * 其余能力（addDocuments / similaritySearch / fromDocuments）
 * 全部由基类根据这两个方法自动派生 —— 这就是抽象的价值。
 */
import { VectorStore } from "@langchain/core/vectorstores";
import type { Embeddings } from "@langchain/core/embeddings";
import type { Document } from "@langchain/core/documents";

export class InMemoryVectorStore extends VectorStore {
  declare FilterType: (doc: Document) => boolean;

  private vectors: number[][] = [];
  private docs: Document[] = [];
  /** 首次写入时确定，用于拦截「换 embedding 模型却没重建库」的静默算错 */
  private dim = 0;

  constructor(embeddings: Embeddings) {
    super(embeddings, {});
  }

  /** LangChain 契约：向量库类型标识（基类序列化会用到） */
  _vectorstoreType(): string {
    return "in-memory";
  }

  /**
   * 写入文档：基类只提供检索辅助，写入由子类实现
   * （embedding + 落库 两步在这里完成）
   */
  async addDocuments(documents: Document[]): Promise<void> {
    const vectors = await this.embeddings.embedDocuments(
      documents.map((d) => d.pageContent),
    );
    await this.addVectors(vectors, documents);
  }

  /** LangChain 契约 1：写入向量 */
  async addVectors(vectors: number[][], documents: Document[]): Promise<void> {
    for (let i = 0; i < vectors.length; i++) {
      const v = vectors[i];
      if (this.dim === 0) this.dim = v.length;
      else if (v.length !== this.dim) {
        throw new Error(
          `向量维度不一致：库里是 ${this.dim}，新写入 ${v.length}。` +
            `换了 embedding 模型（如 Hash 512 → bge-m3 1024）必须重建向量库`,
        );
      }
      this.vectors.push(v);
      this.docs.push(documents[i]);
    }
  }

  /** LangChain 契约 2：按向量检索（余弦相似度，向量已归一化） */
  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: this["FilterType"],
  ): Promise<[Document, number][]> {
    return this.rawSearch(query, k, filter);
  }

  get size(): number {
    return this.docs.length;
  }

  /** 便捷：直接用原始向量检索（供混合检索复用） */
  rawSearch(queryVec: number[], k: number, filter?: this["FilterType"]): [Document, number][] {
    this.checkDim(queryVec);
    const scored = this.vectors.map((vec, i) => ({ doc: this.docs[i], score: dot(vec, queryVec) }));
    const filtered = filter ? scored.filter((s) => filter(s.doc)) : scored;
    return filtered
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((s) => [s.doc, s.score]);
  }

  /** query 与 documents 必须同模型同维度，否则点积会静默算错 */
  private checkDim(queryVec: number[]): void {
    if (this.dim && queryVec.length !== this.dim) {
      throw new Error(
        `查询向量维度 ${queryVec.length} 与库中 ${this.dim} 不一致：` +
          `query 和 documents 必须用同一个 embedding 模型`,
      );
    }
  }
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * 向量化：继承 LangChain 的 Embeddings 抽象
 *
 * - HashEmbeddings: 零依赖，离线跑通流程（字符 n-gram + hash）
 * - 真实向量走 providers.ts 的 OpenAIEmbeddings（任何 OpenAI 兼容向量端点）
 *
 * 关键点：只要实现 Embeddings 接口，就能插进 LangChain 任意组件。
 *
 * 契约：所有实现都返回 **L2 归一化** 的单位向量，
 * 这样向量库里点积 = 余弦相似度，换实现不需要改检索逻辑。
 */
import { Embeddings, type EmbeddingsParams } from "@langchain/core/embeddings";

/** L2 归一化，使点积 = 余弦相似度 */
export function l2Normalize(vec: number[]): number[] {
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return norm > 0 ? vec.map((v) => v / norm) : vec;
}

export class HashEmbeddings extends Embeddings {
  private dim: number;

  constructor(params: EmbeddingsParams & { dim?: number } = {}) {
    super(params);
    this.dim = params.dim ?? 512;
  }

  private ngrams(text: string): string[] {
    // 中文按字、英文按词，混合切分
    const tokens = text.toLowerCase().match(/[一-鿿]|[a-zA-Z0-9_]+/g) ?? [];
    const grams: string[] = [];
    for (const n of [2, 3]) {
      for (let i = 0; i + n <= tokens.length; i++) {
        grams.push(tokens.slice(i, i + n).join(""));
      }
    }
    return grams.length ? grams : tokens;
  }

  private hash(text: string): number {
    // FNV-1a，稳定且快
    let h = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }

  private encodeOne(text: string): number[] {
    const vec = new Array<number>(this.dim).fill(0);
    for (const gram of this.ngrams(text)) {
      vec[this.hash(gram) % this.dim] += 1;
    }
    // L2 归一化，使点积 = 余弦相似度
    return l2Normalize(vec);
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return texts.map((t) => this.encodeOne(t));
  }

  async embedQuery(text: string): Promise<number[]> {
    return this.encodeOne(text);
  }
}
